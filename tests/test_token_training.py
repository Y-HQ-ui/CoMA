import tempfile
from pathlib import Path
import unittest
import torch
from transformers import LlamaConfig,LlamaForCausalLM,MusicgenConfig,MusicgenDecoderConfig,MusicgenForConditionalGeneration,T5Config,EncodecConfig
from training.policies import SymbolicTokenPolicy,AudioTokenPolicy,delayed_audio_tokens
from training.token_ppo import TokenPPOTrainer,PPOConfig,temporal_credit,time_gae
from audio.temporal_control import TemporalControlAdapter,AdapterConfig,musicgen_temporal_context


class TokenTrainingTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(7);torch.set_num_threads(1)
    def symbolic(self):
        return SymbolicTokenPolicy(LlamaForCausalLM(LlamaConfig(vocab_size=24,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64)),rank=2)
    def audio(self):
        cfg=MusicgenConfig.from_sub_models_config(T5Config(d_model=16,d_ff=32,num_layers=1,num_heads=2,vocab_size=24),
            EncodecConfig(hidden_size=16,num_filters=8,codebook_size=32,sampling_rate=32000,upsampling_ratios=[8,5,4,4]),
            MusicgenDecoderConfig(vocab_size=32,hidden_size=16,ffn_dim=32,num_hidden_layers=1,num_attention_heads=2,num_codebooks=4,pad_token_id=32,bos_token_id=32))
        return AudioTokenPolicy(MusicgenForConditionalGeneration(cfg),TemporalControlAdapter(AdapterConfig(16,bottleneck=8)))

    def test_symbolic_sft_and_ppo_update_generator_adapter(self):
        policy=self.symbolic();trainer=TokenPPOTrainer(policy,PPOConfig(learning_rate=.01,epochs=2,target_kl=100))
        payload={"tokens":torch.tensor([[1,2,3,4,5]]),"action_mask":torch.tensor([[False,True,True,True]])}
        frozen={n:p.detach().clone() for n,p in policy.named_parameters() if not p.requires_grad}
        before={n:p.detach().clone() for n,p in policy.named_parameters() if p.requires_grad and "value" not in n}
        trainer.warm_start(payload)
        self.assertTrue(any(not torch.equal(before[n],p) for n,p in policy.named_parameters() if n in before))
        r=trainer.record(payload,torch.tensor([[0.,0.,0.,1.]]),torch.tensor([[0.,0.,0.,2.]]))
        result=trainer.update([r]);self.assertGreater(len(result["updates"]),0)
        with self.assertRaises(ValueError):trainer.update([r])
        for n,p in policy.named_parameters():
            if n in frozen:self.assertTrue(torch.equal(frozen[n],p))
        with tempfile.TemporaryDirectory() as d:
            trainer.save(Path(d)/"policy.pt",{"family":"symbolic"});trainer.load(Path(d)/"policy.pt")

    def test_audio_adapter_zero_init_gradient_and_sampling(self):
        policy=self.audio();policy.eval()
        codes=torch.randint(0,32,(1,4,6));tokens,mask=delayed_audio_tokens(codes,32)
        payload={"tokens":tokens,"codebook_mask":mask,"features":torch.randn(1,6,29),"text_inputs":{"input_ids":torch.tensor([[1,2]]),"attention_mask":torch.ones(1,2,dtype=torch.long)}}
        baseline=policy.model(**payload["text_inputs"],decoder_input_ids=tokens[:,:-1]).logits
        with musicgen_temporal_context(policy.model,policy.adapter,payload["features"]):
            controlled=policy.model(**payload["text_inputs"],decoder_input_ids=tokens[:,:-1]).logits
        torch.testing.assert_close(baseline,controlled)
        trainer=TokenPPOTrainer(policy,PPOConfig(learning_rate=.01,epochs=1,target_kl=100))
        trainer.warm_start(payload)
        self.assertGreater(policy.adapter.network[-1].weight.detach().norm(),0)
        sampled=policy.sample(payload["text_inputs"],payload["features"],6)
        self.assertEqual(sampled["tokens"].shape,tokens.shape)
        lp,values,valid=policy.statistics(sampled)
        self.assertEqual(lp.shape,values.shape);self.assertTrue(torch.isfinite(lp).all())
        rewards=torch.zeros_like(lp);rewards[:,-1]=1
        rollout=trainer.record(sampled,rewards,torch.full_like(lp,.02))
        trainer.update([rollout])
        self.assertTrue(all(p.grad is None for p in policy.model.parameters()))

    def test_reward_credit_conservation_and_mask(self):
        intervals=torch.tensor([[[0.,.2],[.2,.4],[.4,.6],[.6,.8]]]);mask=torch.tensor([[False,True,True,True]])
        reward=temporal_credit([[(.2,.8,3.)]],intervals,mask)
        torch.testing.assert_close(reward,torch.tensor([[0.,1.,1.,1.]]))
        with self.assertRaises(ValueError):temporal_credit([[(0,.1,1)]],intervals,mask)

    def test_temporal_adapter_cached_logits_match_full_teacher_forcing(self):
        policy=self.audio();policy.eval()
        torch.nn.init.normal_(policy.adapter.network[-1].weight,std=.03)
        features=torch.randn(1,6,29)
        text={'input_ids':torch.tensor([[1,2]]),'attention_mask':torch.ones(1,2,dtype=torch.long)}
        ids=torch.randint(0,32,(4,5));ids[:,0]=32
        with musicgen_temporal_context(policy.model,policy.adapter,features):
            full=policy.model(**text,decoder_input_ids=ids).logits
            previous=policy.model(**text,decoder_input_ids=ids[:,:-1],use_cache=True)
            cached=policy.model(**text,decoder_input_ids=ids[:,-1:],past_key_values=previous.past_key_values,use_cache=True).logits
        torch.testing.assert_close(full[:,-1],cached[:,-1],atol=1e-5,rtol=1e-5)

    def test_seconds_gae_and_terminal_do_not_bootstrap_padding(self):
        r=torch.tensor([[0.,1.,0.]]);v=torch.tensor([[0.,0.,999.]])
        mask=torch.tensor([[True,True,False]]);dt=torch.tensor([[2.,1.,0.]])
        adv,_=time_gae(r,v,mask,dt,gamma=.5,gae_lambda=1)
        torch.testing.assert_close(adv,torch.tensor([[.25,1.,0.]]))

    def test_delay_pattern_is_invertible(self):
        codes=torch.arange(12).reshape(1,4,3);tokens,mask=delayed_audio_tokens(codes,99)
        torch.testing.assert_close(tokens[:,1:][mask].reshape(1,4,3),codes)

    def test_audio_checkpoint_rejects_wrong_base_model(self):
        adapter=TemporalControlAdapter(AdapterConfig(16,bottleneck=8))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'adapter.pt';adapter.save(path,metadata={'base_model':{'path':'/original/musicgen'}})
            with self.assertRaisesRegex(ValueError,'different MusicGen base'):
                TemporalControlAdapter.load(path,expected_base_model='/different/musicgen')

if __name__=="__main__":unittest.main()
