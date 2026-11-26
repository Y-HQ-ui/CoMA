from __future__ import annotations

import json
from typing import Any, Dict, Optional


class LLMStructuredCritic:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_config(cls, config) -> Optional["LLMStructuredCritic"]:
        if not getattr(config, "llm_critic_enabled", False):
            return None
        if not config.llm_critic_base_url or not config.llm_critic_api_key or not config.llm_critic_model:
            return None
        return cls(
            base_url=config.llm_critic_base_url,
            api_key=config.llm_critic_api_key,
            model=config.llm_critic_model,
            timeout=config.llm_critic_timeout,
        )

    def _request(self, system_prompt: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        import requests

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM critic did not return a JSON object")
        return json.loads(content[start : end + 1])

    def diagnose(
        self,
        *,
        reward_dict: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result = self._request(
            "You are the CoMA structured music critic. Diagnose whether failure comes from blueprint, symbolic, or audio. Return JSON only with repair_required, source_level, dominant_failure, deficits, confidence, rationale, and optional actions. Deficits must contain non-negative visual, musicality, consistency, and symbolic values.",
            {"rewards": reward_dict, "context": context or {}},
        )
        source_level = str(result.get("source_level", "audio")).lower()
        if source_level not in {"blueprint", "symbolic", "audio"}:
            source_level = "audio"
        result["source_level"] = source_level
        result["repair_required"] = bool(result.get("repair_required", True))
        result["confidence"] = float(max(0.0, min(1.0, result.get("confidence", 0.5))))
        deficits = result.get("deficits")
        if isinstance(deficits, dict):
            result["deficits"] = {
                name: max(0.0, float(deficits.get(name, 0.0)))
                for name in ("visual", "musicality", "consistency", "symbolic")
            }
        return result

    def suggest_weights(
        self,
        *,
        R_vis: float,
        R_mus: float,
        R_cons: float,
        H_human: Optional[Dict[str, Any]] = None,
        H_llm: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._request(
            "You are the CoMA adaptive reward controller. Return JSON only with alpha_vis (3 positive numbers), beta_mus (4 positive numbers), gamma_cons (positive number), focus, and rationale. Increase weights for the weakest reliable component while preserving all objectives.",
            {
                "R_vis": R_vis,
                "R_mus": R_mus,
                "R_cons": R_cons,
                "human_feedback": H_human,
                "previous_llm_feedback": H_llm,
            },
        )


__all__ = ["LLMStructuredCritic"]
