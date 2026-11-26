"""Native MuPT and delayed-codebook MusicGen likelihood/sampling adapters."""
from __future__ import annotations
import torch
from torch import nn
from audio.temporal_control import musicgen_temporal_context


class LoRALinear(nn.Module):
    def __init__(self,base,rank=8,scale=16):
        super().__init__();self.base=base.requires_grad_(False);self.scale=scale/rank
        self.a=nn.Linear(base.in_features,rank,bias=False,device=base.weight.device,dtype=torch.float32)
        self.b=nn.Linear(rank,base.out_features,bias=False,device=base.weight.device,dtype=torch.float32)
        nn.init.normal_(self.a.weight,std=.02);nn.init.zeros_(self.b.weight)
    def forward(self,x):return self.base(x)+(self.b(self.a(x.float()))*self.scale).to(x.dtype)


def install_lora(model,rank=8):
    model.requires_grad_(False)
    targets=[(name,module) for name,module in model.named_modules() if isinstance(module,nn.Linear) and name.rsplit(".",1)[-1] in {"q_proj","v_proj"}]
    if not targets:raise ValueError("no q_proj/v_proj layers for LoRA")
    for name,module in targets:
        parent,_,attr=name.rpartition(".");setattr(model.get_submodule(parent),attr,LoRALinear(module,rank))
    return [name for name,_ in targets]


class SymbolicTokenPolicy(nn.Module):
    def __init__(self,model,rank=8):
        super().__init__();self.model=model;self.lora_layers=install_lora(model,rank)
        device=next(model.parameters()).device
        self.value=nn.Linear(model.config.hidden_size,1,device=device,dtype=torch.float32)
    def statistics(self,payload,entropy=False):
        ids=payload["tokens"]
        out=self.model(input_ids=ids[:,:-1],attention_mask=payload.get("attention_mask",torch.ones_like(ids))[:,:-1],output_hidden_states=True,use_cache=False)
        dist=torch.distributions.Categorical(logits=out.logits.float())
        logp=dist.log_prob(ids[:,1:]);value=self.value(out.hidden_states[-1].float()).squeeze(-1)
        mask=payload["action_mask"].bool()
        result=(logp,value,mask)
        return (*result,dist.entropy()) if entropy else result
    @torch.no_grad()
    def sample(self,prompt_tokens,max_new_tokens,eos_token_id):
        self.eval()
        tokens=self.model.generate(input_ids=prompt_tokens,max_new_tokens=max_new_tokens,do_sample=True,temperature=1.,top_k=0,top_p=1.,eos_token_id=eos_token_id,pad_token_id=eos_token_id)
        mask=torch.zeros_like(tokens[:,:-1],dtype=torch.bool);mask[:,prompt_tokens.shape[1]-1:]=True
        return {"tokens":tokens,"action_mask":mask}


def delayed_audio_tokens(codes,pad_id,audio_channels=1):
    """[B,C,F] -> [B*C,F+delay+1], with BOS/EOS delay padding masked out."""
    if codes.ndim!=3:raise ValueError("codes must be [B,C,F]")
    b,c,f=codes.shape
    delays=torch.arange(c,device=codes.device)//audio_channels
    length=f+int(delays.max())+2
    tokens=torch.full((b,c,length),pad_id,dtype=torch.long,device=codes.device)
    valid=torch.zeros_like(tokens,dtype=torch.bool)
    for i,d in enumerate(delays.tolist()):
        tokens[:,i,d+1:d+1+f]=codes[:,i];valid[:,i,d+1:d+1+f]=True
    return tokens.reshape(b*c,length),valid.reshape(b*c,length)[:,1:]


class AudioTokenPolicy(nn.Module):
    def __init__(self,model,adapter):
        super().__init__();self.model=model.requires_grad_(False);self.adapter=adapter
        self.value=nn.Linear(model.decoder.config.hidden_size,1,device=next(model.parameters()).device,dtype=torch.float32)
    def statistics(self,payload,entropy=False):
        tokens=payload["tokens"];c=self.model.decoder.num_codebooks;b=tokens.shape[0]//c
        with musicgen_temporal_context(self.model,self.adapter,payload["features"]):
            out=self.model(**payload["text_inputs"],decoder_input_ids=tokens[:,:-1],output_hidden_states=True,use_cache=False)
        dist=torch.distributions.Categorical(logits=out.logits.float())
        valid=payload["codebook_mask"].bool()
        targets=tokens[:,1:].clamp_max(out.logits.shape[-1]-1)
        logp=(dist.log_prob(targets)*valid).reshape(b,c,-1).sum(1)
        mask=valid.reshape(b,c,-1).any(1)
        value=self.value(out.decoder_hidden_states[-1].float()).squeeze(-1)
        result=(logp,value,mask)
        ent=(dist.entropy()*valid).reshape(b,c,-1).sum(1)
        return (*result,ent) if entropy else result
    @torch.no_grad()
    def sample(self,text_inputs,features,frames):
        self.eval();c=self.model.decoder.num_codebooks;channels=self.model.decoder.config.audio_channels
        device=next(self.model.parameters()).device
        # Template has no ground truth; only the structural codebook-delay mask is used.
        template,mask=delayed_audio_tokens(torch.zeros(1,c,frames,dtype=torch.long,device=device),self.model.decoder.config.pad_token_id,channels)
        tokens=template[:,:1]
        encoder=self.model.text_encoder(**text_inputs,return_dict=True)
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs=BaseModelOutput(last_hidden_state=encoder.last_hidden_state)
        past=None
        with musicgen_temporal_context(self.model,self.adapter,features):
            for t in range(template.shape[1]-1):
                out=self.model(encoder_outputs=encoder_outputs,attention_mask=text_inputs.get("attention_mask"),decoder_input_ids=tokens if past is None else tokens[:,-1:],past_key_values=past,use_cache=True)
                selected=torch.distributions.Categorical(logits=out.logits[:,-1].float()).sample()
                selected=torch.where(mask[:,t],selected,template[:,t+1])
                tokens=torch.cat((tokens,selected[:,None]),dim=1);past=out.past_key_values
        return {"tokens":tokens,"codebook_mask":mask,"features":features,"text_inputs":text_inputs}
    @torch.no_grad()
    def decode(self,payload):
        tokens=payload["tokens"][:,1:];mask=payload["codebook_mask"]
        c=self.model.decoder.num_codebooks;codes=tokens[mask].reshape(1,c,-1)
        if self.model.decoder.config.audio_channels!=1:raise ValueError("training decoder currently supports mono MusicGen only")
        result=self.model.audio_encoder.decode(codes.unsqueeze(0),[None])
        return result.audio_values
