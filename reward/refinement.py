"""Frozen-parameter inference refinement with explicit dependency and budget rules."""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter
import copy
import math
import random
import time
from .critic import diagnose_failure


class BudgetExceeded(RuntimeError):pass


@dataclass
class GenerationBudget:
    max_calls: int | None
    max_seconds: float | None = None
    calls: int = 0
    stages: list = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)
    def __post_init__(self):
        if self.max_calls is not None and (type(self.max_calls) is not int or self.max_calls < 0):
            raise ValueError("max_calls must be a nonnegative integer or None")
        if self.max_seconds is not None and (not math.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ValueError("max_seconds must be finite and positive")
    def check(self):
        if self.max_calls is not None and self.calls>=self.max_calls:raise BudgetExceeded("generation_call_budget")
        if self.max_seconds is not None and time.monotonic()-self.started>=self.max_seconds:
            raise BudgetExceeded("wall_clock_budget")
    def consume(self,stage):
        self.check();self.calls+=1;self.stages.append({"stage":stage,"call":self.calls,"elapsed_seconds":time.monotonic()-self.started})


@dataclass
class Candidate:
    blueprint: object
    symbolic_outputs: list
    audio_manifest: dict
    rewards: dict
    artifact_dir: str = ""


def refine_inference(initial,regenerate,*,budget,strategy="routed",max_iterations=5,
                     min_gain=1e-4,patience=2,seed=123,global_threshold=.55,component_threshold=.45,
                     force_scope=None,decision_callback=None,stopping="adaptive"):
    """regenerate(scope,current,seed,budget) returns a fully evaluated Candidate.

    Each backend must consume the budget BEFORE every model call, including
    retries. No optimization callback exists in this inference API.
    """
    strategies={"none","best_of_n","random","full","routed"}
    if strategy not in strategies:raise ValueError(f"strategy must be one of {sorted(strategies)}")
    if max_iterations<0 or patience<=0 or min_gain<0:raise ValueError("invalid refinement stopping rule")
    if stopping not in {"adaptive", "fixed", "convergence"}:raise ValueError("unknown stopping rule")
    if force_scope not in {None,"blueprint","symbolic","audio"}:raise ValueError("invalid forced scope")
    rng=random.Random(seed);current=copy.deepcopy(initial);history=[];stale=0;stop="iteration_budget"
    if strategy=="none":return current,{"strategy":strategy,"history":[],"stop":"no_refinement","calls":budget.calls,
        "stopping":stopping,"proposals":0,"completed_proposals":0,"acceptances":0,
        "calls_by_stage":dict(Counter(row["stage"] for row in budget.stages)),
        "elapsed_seconds":time.monotonic()-budget.started,"budget_stages":budget.stages,
        "max_calls":budget.max_calls,"max_seconds":budget.max_seconds,"parameters_updated":False}
    for i in range(max_iterations):
        diagnosis=(decision_callback(current) if decision_callback is not None else
                   diagnose_failure(current.rewards,global_threshold=global_threshold,component_threshold=component_threshold))
        if diagnosis.get("human_veto"):
            stop="human_veto";break
        if stopping=="adaptive" and strategy=="routed" and not diagnosis["repair_required"] and force_scope is None:
            stop="thresholds_satisfied";break
        scope=force_scope or (diagnosis["source_level"] if strategy=="routed" else rng.choice(["blueprint","symbolic","audio"]) if strategy=="random" else "blueprint")
        before=budget.calls
        try:
            budget.check()
            # best-of-N samples independent initial conditions; full uses feedback from the incumbent.
            proposed=regenerate(scope,copy.deepcopy(initial if strategy=="best_of_n" else current),seed+i+1,budget)
            if not all(math.isfinite(float(v)) for v in proposed.rewards.values()):raise ValueError("nonfinite candidate rewards")
            if scope in {"audio","symbolic"} and proposed.blueprint.to_dict()!=current.blueprint.to_dict():
                raise ValueError("downstream repair changed its preserved blueprint")
            if scope=="audio" and proposed.symbolic_outputs!=current.symbolic_outputs:
                raise ValueError("audio repair changed its preserved score")
            gain=proposed.rewards["R_total"]-current.rewards["R_total"]
            accepted=gain>min_gain
            history.append({"iteration":i,"scope":scope,"diagnosis":diagnosis,"accepted":accepted,"gain":gain,
                            "before":dict(current.rewards),"after":dict(proposed.rewards),"calls":budget.calls-before,"artifact_dir":proposed.artifact_dir,
                            "local_repair":proposed.audio_manifest.get("local_repair")})
            if accepted:current=proposed;stale=0
            else:stale+=1
            if stopping!="fixed" and (strategy=="routed" or stopping=="convergence") and stale>=patience:stop="no_improvement_patience";break
        except BudgetExceeded as exc:
            stop=str(exc);history.append({"iteration":i,"scope":scope,"accepted":False,"incomplete":True,"reason":stop,"calls":budget.calls-before});break
        except Exception as exc:
            history.append({"iteration":i,"scope":scope,"accepted":False,"error":str(exc),"calls":budget.calls-before});stale+=1
            if stopping!="fixed" and stale>=patience:stop="failure_patience";break
    return current,{"strategy":strategy,"history":history,"stop":stop,"calls":budget.calls,
                    "stopping":stopping,"proposals":len(history),"acceptances":sum(bool(h.get("accepted")) for h in history),
                    "completed_proposals":sum("after" in h for h in history),
                    "calls_by_stage":dict(Counter(row["stage"] for row in budget.stages)),
                    "elapsed_seconds":time.monotonic()-budget.started,"budget_stages":budget.stages,
                    "max_calls":budget.max_calls,"max_seconds":budget.max_seconds,"parameters_updated":False,
                    "scope":"representation_level; local waveform preservation only when candidate local_repair evidence is recorded"}
