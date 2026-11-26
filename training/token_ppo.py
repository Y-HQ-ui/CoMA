"""Masked on-policy PPO with physical-time discount and a learned value head."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn


@dataclass
class PPOConfig:
    learning_rate: float = 1e-5
    clip: float = .2
    value_clip: float = .2
    gamma_per_second: float = .99
    gae_lambda_per_second: float = .95
    entropy_coef: float = .01
    value_coef: float = .5
    max_grad_norm: float = 1.
    epochs: int = 4
    target_kl: float = .03
    weight_decay: float = 0.

    def __post_init__(self):
        if not all(math.isfinite(float(v)) for v in asdict(self).values()):
            raise ValueError("nonfinite PPO configuration")
        if self.learning_rate <= 0 or self.epochs < 1 or self.max_grad_norm <= 0 or self.target_kl <= 0:
            raise ValueError("positive learning rate, PPO epochs, gradient bound and KL target required")
        if not 0 < self.gamma_per_second <= 1 or not 0 < self.gae_lambda_per_second <= 1:
            raise ValueError("discount and GAE lambda must be in (0,1]")
        if min(self.clip, self.value_clip, self.entropy_coef, self.value_coef, self.weight_decay) < 0:
            raise ValueError("PPO loss coefficients cannot be negative")


def masked_mean(values, mask):
    mask=mask.to(values)
    if mask.sum()<=0: raise ValueError("empty action mask")
    return (values*mask).sum()/mask.sum()


def temporal_credit(windows, token_intervals, mask):
    """Distribute each window reward by intersection duration; conserve reward mass.

    token_intervals [B,T,2] must describe real event spans (or codec-frame spans).
    Header/padding tokens have mask=False. No interpolation invents token timing.
    """
    spans=torch.as_tensor(token_intervals,dtype=torch.float32)
    mask=torch.as_tensor(mask,dtype=torch.bool,device=spans.device)
    if spans.shape[:-1]!=mask.shape or spans.shape[-1]!=2 or not torch.isfinite(spans).all() or (spans[...,1]<spans[...,0]).any():
        raise ValueError("invalid token intervals")
    rewards=torch.zeros_like(spans[...,0])
    for b,batch_windows in enumerate(windows):
        for start,end,score in batch_windows:
            if not end>start or not all(torch.isfinite(torch.tensor(x)) for x in (start,end,score)):
                raise ValueError("invalid reward window")
            overlap=(torch.minimum(spans[b,:,1],torch.tensor(end,device=spans.device))-torch.maximum(spans[b,:,0],torch.tensor(start,device=spans.device))).clamp_min(0)*mask[b]
            if overlap.sum()<=0: raise ValueError("reward window has no eligible token action")
            rewards[b]+=score*overlap/overlap.sum()
    return rewards


def time_gae(rewards, values, mask, elapsed, *, gamma=.99, gae_lambda=.95):
    """Terminal episodes only. gamma/lambda are per-second, not per byte/token."""
    if not (rewards.shape==values.shape==mask.shape==elapsed.shape): raise ValueError("GAE shapes differ")
    if not 0<gamma<=1 or not 0<gae_lambda<=1 or (elapsed<0).any() or not all(torch.isfinite(x).all() for x in (rewards, values, elapsed)):
        raise ValueError("invalid time discount or nonfinite rollout")
    advantages=torch.zeros_like(rewards)
    for b in range(rewards.shape[0]):
        ids=mask[b].nonzero(as_tuple=True)[0]
        carry=values.new_tensor(0.); next_value=values.new_tensor(0.)
        for t in reversed(ids.tolist()):
            discount=gamma**elapsed[b,t]; lam=gae_lambda**elapsed[b,t]
            delta=rewards[b,t]+discount*next_value-values[b,t]
            carry=delta+discount*lam*carry
            advantages[b,t]=carry; next_value=values[b,t]
    return advantages,advantages+values


@dataclass
class TokenRollout:
    payload: dict
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    mask: torch.Tensor
    elapsed_seconds: torch.Tensor
    policy_version: int


class TokenPPOTrainer:
    def __init__(self, policy, config=None):
        self.policy=policy
        self.config=config or PPOConfig()
        self.parameters=[p for p in policy.parameters() if p.requires_grad]
        if not self.parameters: raise ValueError("policy has no trainable parameters")
        self.optimizer=torch.optim.AdamW(self.parameters,lr=self.config.learning_rate,weight_decay=self.config.weight_decay)
        self.version=0

    def warm_start(self, payload):
        self.policy.eval()  # dropout disabled for consistent likelihoods; gradients remain enabled
        logp,_,mask=self.policy.statistics(payload)
        loss=-masked_mean(logp,mask)
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=nn.utils.clip_grad_norm_(self.parameters,self.config.max_grad_norm)
        if not torch.isfinite(norm): raise ValueError("nonfinite warm-start gradient")
        self.optimizer.step();self.version+=1
        return {"loss":float(loss.detach()),"grad_norm":float(norm),"policy_version":self.version}

    @torch.no_grad()
    def record(self,payload,rewards,elapsed_seconds):
        self.policy.eval()
        logp,value,mask=self.policy.statistics(payload)
        rewards=torch.as_tensor(rewards,device=logp.device,dtype=logp.dtype)
        elapsed=torch.as_tensor(elapsed_seconds,device=logp.device,dtype=logp.dtype)
        if rewards.shape!=logp.shape or elapsed.shape!=logp.shape: raise ValueError("credit must match action grid")
        return TokenRollout(payload,logp.detach(),value.detach(),rewards,mask,elapsed,self.version)

    def update(self,rollouts):
        if not rollouts or any(r.policy_version!=self.version for r in rollouts):
            raise ValueError("PPO requires current-policy rollouts; stale/offline rollouts are rejected")
        cfg=self.config;prepared=[]
        for r in rollouts:
            advantages,returns=time_gae(r.rewards,r.old_values,r.mask,r.elapsed_seconds,gamma=cfg.gamma_per_second,gae_lambda=cfg.gae_lambda_per_second)
            prepared.append((r,advantages,returns))
        flat=torch.cat([a[r.mask] for r,a,_ in prepared])
        mean,std=flat.mean(),flat.std(unbiased=False)
        stats=[]
        for epoch in range(cfg.epochs):
            for r,adv,returns in prepared:
                new_logp,values,mask,entropy=self.policy.statistics(r.payload,entropy=True)
                if not torch.equal(mask,r.mask): raise ValueError("action mask changed during PPO")
                normalized=(adv-mean)/(std+1e-8) if flat.numel()>1 and std>1e-8 else adv
                log_ratio=new_logp-r.old_log_probs
                ratio=torch.exp(log_ratio)
                kl=masked_mean((ratio-1)-log_ratio,mask)
                if not torch.isfinite(kl): raise ValueError("nonfinite PPO ratio")
                if kl>cfg.target_kl:
                    self.version+=1
                    return {"updates":stats,"stopped":"target_kl","policy_version":self.version}
                actor=-masked_mean(torch.minimum(ratio*normalized,ratio.clamp(1-cfg.clip,1+cfg.clip)*normalized),mask)
                clipped=r.old_values+(values-r.old_values).clamp(-cfg.value_clip,cfg.value_clip)
                critic=.5*masked_mean(torch.maximum((values-returns).square(),(clipped-returns).square()),mask)
                ent=masked_mean(entropy,mask)
                loss=actor+cfg.value_coef*critic-cfg.entropy_coef*ent
                self.optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=nn.utils.clip_grad_norm_(self.parameters,cfg.max_grad_norm)
                if not torch.isfinite(norm): raise ValueError("nonfinite PPO gradient")
                self.optimizer.step()
                stats.append({"epoch":epoch,"actor_loss":float(actor.detach()),"value_loss":float(critic.detach()),"entropy":float(ent.detach()),"kl":float(kl.detach()),"grad_norm":float(norm)})
        self.version+=1
        return {"updates":stats,"stopped":"epoch_budget","policy_version":self.version}

    def save(self,path,metadata):
        from pathlib import Path
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        trainable={n:p.detach().cpu() for n,p in self.policy.named_parameters() if p.requires_grad}
        torch.save({"parameters":trainable,"optimizer":self.optimizer.state_dict(),"ppo_config":asdict(self.config),"policy_version":self.version,"metadata":metadata},path)

    def load(self,path,*,restore_optimizer=True):
        state=torch.load(path,map_location=next(self.policy.parameters()).device,weights_only=True)
        expected={n for n,p in self.policy.named_parameters() if p.requires_grad}
        if set(state["parameters"])!=expected: raise ValueError("checkpoint trainable parameters do not match policy")
        current=dict(self.policy.named_parameters())
        with torch.no_grad():
            for n,value in state["parameters"].items():current[n].copy_(value)
        if restore_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
            self.config=PPOConfig(**state["ppo_config"])
        self.version=state["policy_version"]
        return state["metadata"]
