"""Model-backed artifact generation used by frozen refinement and interventions."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from audio.symbolic_renderer import generate_blueprint_score
from audio.abc_to_render_controls import blueprint_to_render_controls
from audio.musicgen_adapter import render_controls_to_prompts
from audio.musicgen_renderer import render_musicgen_prompts
from reward.refinement import Candidate
from reward.artifact_scoring import score_artifacts


@dataclass
class RealizationBackend:
    output_dir: str
    mupt_model_dir: str = ""
    musicgen_model_dir: str = ""
    mupt_device: str = "cpu"
    musicgen_device: str = "cpu"
    adapter_checkpoint: str = ""
    consistency_enabled: bool = True
    candidate_count: int = 0
    independent_sampling: bool = False
    planner: object = None
    max_new_tokens: int = 384
    symbolic_policy_checkpoint: str = ""
    local_audio_window_seconds: float = 0.
    scorer: object = None

    def generate(self,blueprint,*,seed,budget,symbolic_outputs=None):
        root=Path(self.output_dir)/f"candidate_{self.candidate_count:04d}"
        self.candidate_count+=1;root.mkdir(parents=True,exist_ok=False)
        if symbolic_outputs is None:
            symbolic_outputs=generate_blueprint_score(blueprint,output_dir=str(root/"symbolic"),model_dir=self.mupt_model_dir,
                device=self.mupt_device,seed=seed,budget=budget,max_new_tokens=self.max_new_tokens,
                policy_checkpoint=self.symbolic_policy_checkpoint)
        controls=blueprint_to_render_controls(blueprint,symbolic_outputs=symbolic_outputs)
        audio=render_musicgen_prompts(render_controls_to_prompts(controls),model_dir=self.musicgen_model_dir,
            output_dir=str(root/"audio"),device=self.musicgen_device,seed=seed,controls=controls,
            symbolic_outputs=symbolic_outputs,adapter_checkpoint=self.adapter_checkpoint,budget=budget)
        evaluation=(self.scorer or score_artifacts)(blueprint,symbolic_outputs,audio,consistency_enabled=self.consistency_enabled)
        for name,data in (("blueprint",blueprint.to_dict()),("symbolic_outputs",symbolic_outputs),("score",evaluation)):
            (root/f"{name}.json").write_text(json.dumps(data,ensure_ascii=False,indent=2))
        return Candidate(blueprint,symbolic_outputs,audio,evaluation["rewards"],str(root))

    def regenerate(self,scope,current,seed,budget):
        if scope=="audio" and self.local_audio_window_seconds>0:
            return self.repair_audio_window(current,seed,budget)
        blueprint=current.blueprint
        if scope=="blueprint" and not self.independent_sampling:
            if self.planner is not None:
                if getattr(self.planner, "manages_budget", False):
                    blueprint=self.planner(current,seed,budget)
                else:
                    budget.consume("composer")
                    blueprint=self.planner(current,seed)
            else:
                if self.scorer is not None:
                    raise ValueError("paper blueprint repair requires a Composer callback")
                from reward.critic import diagnose_failure
                from reward.repair import repair_blueprint_deterministic
                diagnosis=diagnose_failure(current.rewards,global_threshold=.55,component_threshold=.45)
                blueprint=repair_blueprint_deterministic(blueprint,diagnosis,scope="blueprint",num_passes=1)
        return self.generate(blueprint,seed=seed,budget=budget,
                             symbolic_outputs=current.symbolic_outputs if scope=="audio" else None)

    def repair_audio_window(self,current,seed,budget):
        """Generate a context segment then commit only the selected local mask.

        This is waveform splicing with a fade, not native MusicGen inpainting.
        The full context segment counts as one generation call and is retained.
        """
        import copy
        import numpy as np
        import soundfile as sf
        from reward.artifact_scoring import weakest_audio_window
        from audio.timeline import replace_audio_interval
        from audio.temporal_control import build_temporal_features
        from audio.musicgen_renderer import MusicGenRendererSession
        root=Path(self.output_dir)/f"candidate_{self.candidate_count:04d}"
        self.candidate_count+=1;root.mkdir(parents=True,exist_ok=False)
        window=weakest_audio_window(current.blueprint,current.symbolic_outputs,current.audio_manifest,
                                   window_seconds=self.local_audio_window_seconds)
        controls=blueprint_to_render_controls(current.blueprint,symbolic_outputs=current.symbolic_outputs)
        control=copy.deepcopy(controls[window["segment_index"]])
        prompt=render_controls_to_prompts([control])[0]
        origin=control["start_time"];duration=control["end_time"]-origin
        control["start_time"],control["end_time"]=0.,duration
        for item in control.get("render_windows",[]):
            item["start_time"]-=origin;item["end_time"]-=origin
        events=[]
        for item in current.symbolic_outputs:
            for event in item.get("events",[]):
                end=event["onset"]+event["duration"]
                if end>origin and event["onset"]<origin+duration:
                    event=dict(event,onset=max(0,event["onset"]-origin))
                    event["duration"]=min(duration,end-origin)-event["onset"];events.append(event)
        features=build_temporal_features([control],events,duration=duration)
        renderer=MusicGenRendererSession.get(model_dir=self.musicgen_model_dir,device=self.musicgen_device)
        budget.consume("musicgen_context_for_local_patch")
        proposal_path=root/"context_proposal.wav"
        renderer.render_one(prompt,output_path=str(proposal_path),seed=seed,duration=duration,
            temporal_features=features,adapter_checkpoint=self.adapter_checkpoint)
        original,sr=sf.read(current.audio_manifest["soundtrack_path"])
        proposal,proposal_sr=sf.read(proposal_path)
        if original.ndim!=1 or proposal.ndim!=1 or sr!=proposal_sr:raise ValueError("local repair requires aligned mono sample rates")
        segment_left=round(origin*sr);segment_right=round((origin+duration)*sr)
        if len(proposal)!=segment_right-segment_left:raise ValueError("local context proposal duration differs")
        candidate=original.copy();candidate[segment_left:segment_right]=proposal
        left,right=round(window["start"]*sr),round(window["end"]*sr)
        repaired=replace_audio_interval(original,candidate,start_sample=left,end_sample=right,fade_samples=round(.01*sr))
        subtype=sf.info(current.audio_manifest["soundtrack_path"]).subtype
        path=root/"soundtrack.wav";sf.write(path,repaired,sr,subtype=subtype)
        readback,_=sf.read(path)
        if not np.array_equal(readback[:left],original[:left]) or not np.array_equal(readback[right:],original[right:]):
            raise RuntimeError("local repair changed samples outside its mask")
        manifest=copy.deepcopy(current.audio_manifest)
        manifest.update(soundtrack_path=str(path),seed=seed,segment_paths=[],generation_calls=1,
            local_repair={**window,"start_sample":left,"end_sample":right,"outside_mask_bit_identical":True,
                "method":"whole_context_generation_then_10ms_internal_splice", "proposal":str(proposal_path)},
            manifest_path=str(root/"manifest.json"))
        for index,segment in enumerate(current.blueprint.segments):
            segment_path=root/f"segment_{index:04d}.wav"
            sf.write(segment_path,readback[round(segment.start_time*sr):round(segment.end_time*sr)],sr,subtype=subtype)
            manifest["segment_paths"].append(str(segment_path))
        evaluation=(self.scorer or score_artifacts)(current.blueprint,current.symbolic_outputs,manifest,consistency_enabled=self.consistency_enabled)
        for name,value in (("manifest",manifest),("score",evaluation),("blueprint",current.blueprint.to_dict()),("symbolic_outputs",current.symbolic_outputs)):
            (root/f"{name}.json").write_text(json.dumps(value,indent=2))
        return Candidate(current.blueprint,current.symbolic_outputs,manifest,evaluation["rewards"],str(root))
