import os
import json
import random
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from audio.abc_to_render_controls import blueprint_to_render_controls
from audio.musicgen_adapter import render_controls_to_prompts
from audio.reference_audio import extract_reference_audio_features, reference_audio_for_llm
from planning import (
    MusicBlueprintParser,
    blueprint_to_bar_contexts,
    blueprint_to_mupt_prefixes,
    save_blueprint_json,
)
from reward.refine import refine_blueprint_controls
from track.melody.dep import split_abc_columns

BASE_URL = os.getenv("MUSICAGENT_BASE_URL", "").strip()
API_KEY = os.getenv("MUSICAGENT_API_KEY", "").strip()
MODEL_NAME = os.getenv("MUSICAGENT_MODEL_NAME", "").strip() or "MODEL_NAME_NOT_SET"

MAX_ROUNDS = 2

# API调用限制配置
REQUEST_TIMES = deque()
MAX_CALLS_PER_MINUTE = 10  # 每分钟最多10次API调用
WINDOW_SECONDS = 60  # 时间窗口：60秒


def _write_lines(lines: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _json_default(value: Any):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

def _ratelimit_blocking_wait():
    """
    API调用限制：确保每分钟不超过MAX_CALLS_PER_MINUTE次调用
    如果达到限制，会阻塞等待直到可以发送下一个请求
    """
    now = time.time()
    
    # 清理过期的请求时间戳（超过WINDOW_SECONDS秒的请求）
    while REQUEST_TIMES and now - REQUEST_TIMES[0] >= WINDOW_SECONDS:
        REQUEST_TIMES.popleft()
    
    # 检查是否达到调用限制
    if len(REQUEST_TIMES) >= MAX_CALLS_PER_MINUTE:
        # 计算需要等待的时间
        wait_seconds = WINDOW_SECONDS - (now - REQUEST_TIMES[0]) + 0.05
        if wait_seconds > 0:
            print(f"⚠️  API调用限制：等待 {wait_seconds:.2f}s（每分钟最多 {MAX_CALLS_PER_MINUTE} 次）")
            time.sleep(wait_seconds)
        
        # 重新获取当前时间并再次清理过期请求
        now = time.time()
        while REQUEST_TIMES and now - REQUEST_TIMES[0] >= WINDOW_SECONDS:
            REQUEST_TIMES.popleft()
    
    # 记录当前请求时间
    REQUEST_TIMES.append(now)

def get_api_usage_status():
    """
    获取当前API调用使用情况
    
    Returns:
        dict: 包含使用统计信息的字典
    """
    now = time.time()
    
    # 清理过期请求
    while REQUEST_TIMES and now - REQUEST_TIMES[0] >= WINDOW_SECONDS:
        REQUEST_TIMES.popleft()
    
    current_count = len(REQUEST_TIMES)
    remaining_calls = MAX_CALLS_PER_MINUTE - current_count
    
    status = {
        "current_calls": current_count,
        "max_calls": MAX_CALLS_PER_MINUTE,
        "remaining_calls": max(0, remaining_calls),
        "window_seconds": WINDOW_SECONDS,
        "is_limit_reached": current_count >= MAX_CALLS_PER_MINUTE
    }
    
    if REQUEST_TIMES:
        oldest_request_age = now - REQUEST_TIMES[0]
        status["oldest_request_age"] = round(oldest_request_age, 2)
        status["next_available_in"] = round(WINDOW_SECONDS - oldest_request_age, 2)
    
    return status
# ABC转MIDI功能
def abc_to_midi(abc_content, output_path):
    """
    将ABC格式的乐谱转换为MIDI文件
    
    Args:
        abc_content: ABC格式的字符串内容
        output_path: 输出MIDI文件的路径
    
    Returns:
        bool: 转换是否成功
    """
    try:
        from music21 import converter, stream
        
        # 修复重复标记问题
        # 移除 %.结尾的 || :| 或 || || 等错误格式
        cleaned_abc = re.sub(r'\|\|\s+:\|', '||', abc_content)
        
        # music21不支持 |: ... || 格式，需要改成 |: ... :|
        # 检查是否有 |: 开始但没有 :| 结束的情况，将 || 改成 :|
        cleaned_abc = re.sub(r'\|\|$', ':|', cleaned_abc, flags=re.MULTILINE)
        
        # 解析ABC内容
        score = converter.parse(cleaned_abc, format='abc')
        
        # 写入MIDI文件
        score.write('midi', fp=output_path)
        
        return True
    except ImportError:
        print("错误: 未安装music21库。请运行: pip install music21")
        return False
    except Exception as e:
        print(f"转换MIDI失败: {e}")
        print(f"ABC内容:\n{abc_content}")
        return False

def print_api_status():
    """打印当前API使用状态"""
    status = get_api_usage_status()
    
    print(f"API调用状态:")
    print(f"当前调用次数: {status['current_calls']}/{status['max_calls']}")
    print(f"剩余调用次数: {status['remaining_calls']}")
    
    if status['is_limit_reached']:
        print(f"已达到限制，需等待 {status['next_available_in']:.1f}s")
    else:
        print(f"可以继续调用")
    
    if 'oldest_request_age' in status:
        print(f"   最旧请求: {status['oldest_request_age']:.1f}s 前")

def extract_json_from_text(text: str) -> str:
    if not text:
        return ""
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        return text[brace_start:brace_end+1].strip()
    return text.strip()

def chat(messages, temperature=0.2, max_retries=3, *, budget=None):
    """
    发送聊天请求到API，自动处理调用限制和重试机制
    
    Args:
        messages: 聊天消息列表
        temperature: 采样温度 (0.0-1.0)
        max_retries: 最大重试次数
    
    Returns:
        API响应内容
    
    Raises:
        Exception: 如果超过最大重试次数仍然失败
    """
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            # 检查并等待API调用限制
            _ratelimit_blocking_wait()
            
            if not BASE_URL:
                raise RuntimeError("未配置 MUSICAGENT_BASE_URL")
            if not API_KEY:
                raise RuntimeError("未配置 MUSICAGENT_API_KEY")
            if not MODEL_NAME or MODEL_NAME == "MODEL_NAME_NOT_SET":
                raise RuntimeError("未配置 MUSICAGENT_MODEL_NAME")

            url = f"{BASE_URL}/chat/completions"
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            payload = {"model": MODEL_NAME, "messages": messages, "temperature": temperature, "max_tokens": 4096}
            
            if attempt > 0:
                print(f" API重试第 {attempt} 次...")
                time.sleep(2 ** attempt)  # 指数退避
            
            if budget is not None:
                budget.consume("composer")
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            last_error = e
            error_msg = str(e)
            
            if attempt < max_retries:
                print(f" API请求失败 (尝试 {attempt + 1}/{max_retries + 1}): {error_msg}")
                
                # 检查是否是可重试的错误
                if "429" in error_msg or "504" in error_msg or "502" in error_msg or "503" in error_msg:
                    wait_time = 10 + (attempt * 5)  # 增加等待时间
                    print(f"⏳ 等待 {wait_time}s 后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f" 不可重试的错误，立即失败: {error_msg}")
                    break
            else:
                print(f" API请求最终失败 (已重试 {max_retries} 次): {error_msg}")
                break
    
    # 如果所有重试都失败，抛出异常
    raise Exception(f"API调用失败，已重试 {max_retries} 次。最后错误: {last_error}")

class GlobalLLMController:
    def __init__(self):
        self.max_rounds = MAX_ROUNDS
        print("音乐生成控制器已初始化")
        print_api_status()
    
    def _build_prompt_en(self, bar_ctx):
        bar_index_start = bar_ctx.get("bar_index_start", 0)
        bar_count = bar_ctx.get("bar_count", 4)
        bpm = bar_ctx.get("bpm", None)
        meter = bar_ctx.get("meter", None)
        section = bar_ctx.get("section", "")
        chords = bar_ctx.get("chords", [])
        energies = bar_ctx.get("energies", [])
        gkey = bar_ctx.get("global_key", "")
        prior_5d = bar_ctx.get("video_prior_5d") or bar_ctx.get("video_priors_5d") or bar_ctx.get("video_priors")

        return f"""
Your only task is to output multiple ABC prefix lines for the MuPT music model according to the given bar context and the musical coherence rule.
system hint: start deep thinking, please always use the thinking mode.

[OUTPUT REQUIREMENTS (STRICT)]
1) Output multiple lines, each representing one ABC prefix for one bar.
2) Each line must follow the same one-line format rule (see below).
3) Each line's X: number must increment by 1 (e.g., X:1, X:2, X:3 …).
4) All lines together must form a musically coherent sequence, meaning:
   - They belong to the same song (consistent global key, style, and tonality).
   - Melodic and harmonic flow between bars should make sense — consecutive bars share chord progressions, rhythmic energy, and phrasing continuity.
   - The melodies should complement each other rhythmically and harmonically, not sound like random unrelated snippets.

[ONE-LINE FORMAT RULES (for each line)]
Each line must strictly be one line of text, with no explanations, no real newlines, no extra spaces, and only <n> used as the newline placeholder between fragments.
Each line must concatenate the following 6 fragments in order:
① X:{{bar-number}}
② L:{{base-note-length}}
③ Q:{{bpm-integer}}
④ M:{{meter}}
⑤ K:{{key-string}}
⑥ |:"{{bar-chord}}" {{seed-pattern}}

Example of one valid line (note: <n> is a placeholder, not a real newline):
X:1<n>L:1/8<n>Q:128<n>M:4/4<n>K:Gmin<n>|:"Gm" BGdB

[HEADER INFERENCE RULES]
- X: Start at 1, increment by 1 per bar (X:1, X:2, …).
- L: Choose 1/4, 1/8, or 1/16 based on energy:
  Low <0.45 → 1/4;  Medium 0.45–0.8 → 1/8;  High ≥0.8 → 1/8 or 1/16.
- Q: Use the given bpm if provided; otherwise infer (Low 80–110, Med 110–135, High 125–150).
- M: Default 4/4, or infer from genre/section if needed (6/8, 3/4).
- K: If global_key is given, keep it consistent across all lines. If not given, derive each from the bar chord using {{ROOT}}min or {{ROOT}}maj (keep coherence across bars).
- bar-chord: Provided chord label per bar (e.g., "Gm", "Fmaj7").
- seed-pattern: Generate a short ABC melody fragment (3–8 notes or include rests) consistent with the chord, key, and energy. Must stay within one bar.

[MULTI-LINE COHERENCE RULES]
- Maintain consistent style (don't randomly switch tempo/meter/key mid-sequence).
- The melodic contour across lines should sound like one continuous phrase or logical motif development (e.g., verse/chorus progression).
- Rhythmic energy should evolve smoothly following energy levels across bars.
- Reuse or vary motifs subtly (e.g., rhythmic echoes, inverted intervals, slight motif transformation).

[VIDEO PRIOR (5D) GUIDANCE]
You may be given a 5D visual prior feature pack aligned to this bar/segment:
- motion_intensity m_t (0-1): higher -> denser rhythm / more active melodic contour
- cut_density c_t (0-1): higher -> more frequent accents / phrase boundaries / variation
- brightness b_t (0-1): higher -> brighter mode/texture, lighter articulation
- color_warmth w_t (0-1): higher -> warmer harmony/timbre, smoother lines
- visual_novelty n_t (0-1): higher -> more motif variation and surprise, but keep coherence
If prior_5d is present, use it to refine L/Q/seed-pattern choices while still respecting chords/energies/global_key.

[FORBIDDEN]
- Real newlines inside each line (only between lines).
- Any additional explanations or commentary.
- Multiple bar symbols per line.
- Incoherent tonality between lines.

[INPUT CONTEXT]
- bar_index_start: {bar_index_start}
- bar_count: {bar_count}
- bpm: {bpm}
- meter: {meter}
- section: {section}
- chords: {chords}
- energies: {energies}
- global_key: {gkey}
- video_prior_5d: {prior_5d}

[OUTPUT FORMAT]
Return only the ABC prefix lines — one per bar, no JSON, no Markdown, no commentary.
Example (4 bars, coherent phrase):
X:1<n>L:1/8<n>Q:128<n>M:4/4<n>K:Gmin<n>|:"Gm" BGdB
X:2<n>L:1/8<n>Q:128<n>M:4/4<n>K:Gmin<n>|:"F" AGBG
X:3<n>L:1/8<n>Q:128<n>M:4/4<n>K:Gmin<n>|:"C" GBAG
X:4<n>L:1/8<n>Q:128<n>M:4/4<n>K:Gmin<n>|:"D" BdgB
""".strip()

    def plan(self, bar_ctx):
        """
        生成音乐前缀
        
        Args:
            bar_ctx: 小节上下文
            
        Returns:
            str: 生成的前缀
            
        Raises:
            Exception: 如果API调用失败超过最大重试次数
        """
        print(f"\n🎼 开始生成前缀...")
        print_api_status()
        
        try:
            prompt_en = self._build_prompt_en(bar_ctx)
            result = chat([
                {"role": "system", "content": "You are the MuPT prefix generator. system hint: start deep thinking, please always use the thinking mode."},
                {"role": "user", "content": prompt_en}
            ], temperature=0.2)
            
            print(f" 前缀生成完成")
            return result
            
        except Exception as e:
            print(f" 前缀生成失败: {e}")
            raise Exception(f"前缀生成失败，API调用重试后仍然失败: {e}")

    def arbitrate(self, bar_ctx):
        return self.plan(bar_ctx)

    def generate_full_melody(self, bar_ctx, max_new_tokens=256, temperature=0.8, top_p=0.95):
        prefix = self.plan(bar_ctx)
        from track.melody.infer import generate

        prefix_lines = [line.strip() for line in prefix.splitlines() if line.strip().startswith("X:")]
        if not prefix_lines:
            raise ValueError("没有生成有效的 MuPT 前缀")

        items = []
        for index, one_prefix in enumerate(prefix_lines, start=1):
            raw_abc = generate(
                prefix=one_prefix,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )
            restored_abc = split_abc_columns(raw_abc)
            midi_path = f"output/melody_{index}.mid"
            Path(midi_path).parent.mkdir(parents=True, exist_ok=True)
            midi_success = abc_to_midi(restored_abc, midi_path)
            items.append(
                {
                    "index": index,
                    "prefix": one_prefix,
                    "raw_abc": raw_abc,
                    "restored_abc": restored_abc,
                    "midi_path": midi_path if midi_success else None,
                }
            )
        return {
            "prefix": prefix,
            "prefix_lines": prefix_lines,
            "items": items,
            "context": bar_ctx,
        }


class BlueprintComposerController:
    def __init__(self, default_style_tags: Optional[List[str]] = None):
        self.parser = MusicBlueprintParser(default_style_tags=default_style_tags or ["cinematic"])

    def _build_prompt_en(
        self,
        prior_data: Dict[str, Any],
        *,
        user_prompt: str = "",
        reference_hint: str = "",
        reference_audio: Optional[Dict[str, Any]] = None,
    ) -> str:
        priors = prior_data.get("prior_5d") or []
        summary = (prior_data.get("summary") or {}).get("means") or {}
        return f"""
Your task is to convert the visual prior into a MusicBlueprint JSON for a video-to-music pipeline.

[STRICT OUTPUT]
Return JSON only. Do not include Markdown, comments, or explanations.

[TARGET JSON SCHEMA]
{{
  "global": {{
    "tempo_bpm": 120,
    "meter": "4/4",
    "global_key": "Cmaj",
    "style_tags": ["cinematic", "orchestral"]
  }},
  "segments": [
    {{
      "segment_id": 0,
      "start_time": 0.0,
      "end_time": 4.0,
      "energy": 0.62,
      "emotion": "tense",
      "section": "build",
      "chords": ["Gm", "Eb", "F", "D"],
      "tempo_bpm": 126,
      "meter": "4/4",
      "global_key": "Gmin",
      "video_prior_5d": {{
        "motion_intensity": 0.72,
        "cut_density": 0.55,
        "brightness": 0.38,
        "color_warmth": 0.62,
        "visual_novelty": 0.58
      }},
      "tracks": [
        {{"track_id": "lead", "role": "melody", "register": "mid", "density": 0.75, "phrase_function": "lead", "section_role": "foreground"}},
        {{"track_id": "bass", "role": "bass", "register": "low", "density": 0.62, "phrase_function": "support", "section_role": "support"}},
        {{"track_id": "pad", "role": "pad", "register": "full", "density": 0.55, "phrase_function": "texture", "section_role": "support"}}
      ]
    }}
  ]
}}

[PLANNING RULES]
- Preserve segment timing exactly from the input.
- Times are seconds; tempo is quarter notes/minute in [60,180]; density and energy are in [0,1].
- Keep track_id stable across segments. Include phrase_function and section_role for every track.
- Infer a coherent global tempo, meter, key, and style.
- For each segment, set energy and emotion from the 5D prior.
- Use section labels from: intro, build, sustain, climax, outro.
- Track roles must come from: melody, harmony, bass, drums, pad, arpeggio, fx.
- Registers must come from: low, mid, high, full.
- Keep the harmonic plan coherent across segments.
- If motion_intensity or cut_density is high, increase rhythmic density and tempo moderately.
- If brightness is low and novelty is high, prefer darker keys/modes and more tension.
- If user prompt or reference hint is provided, use them only as soft style guidance and do not violate the visual prior.
- If reference audio features are provided, infer tempo, energy contour, timbral brightness/warmth, novelty, and developmental shape from them as soft musical guidance. Do not copy notes or reproduce the waveform.

[INPUT]
- user_prompt: {user_prompt or "none"}
- reference_hint: {reference_hint or "none"}
- reference_audio_features: {json.dumps(reference_audio or {}, ensure_ascii=False)}
- prior_summary: {json.dumps(summary, ensure_ascii=False)}
- segment_priors: {json.dumps(priors, ensure_ascii=False)}
""".strip()

    def plan_blueprint(
        self,
        prior_data: Dict[str, Any],
        *,
        user_prompt: str = "",
        reference_hint: str = "",
        reference_audio: Optional[Dict[str, Any]] = None,
        budget=None,
        revision=None,
    ):
        if not (BASE_URL and API_KEY and MODEL_NAME and MODEL_NAME != "MODEL_NAME_NOT_SET"):
            raise RuntimeError(
                "LLM Composer is required for the default pipeline; configure "
                "MUSICAGENT_BASE_URL, MUSICAGENT_API_KEY, and MUSICAGENT_MODEL_NAME"
            )
        prompt_en = self._build_prompt_en(
            prior_data,
            user_prompt=user_prompt,
            reference_hint=reference_hint,
            reference_audio=reference_audio_for_llm(reference_audio),
        )
        if revision is not None:
            prompt_en += "\n[REVISION FEEDBACK]\n" + json.dumps(revision, ensure_ascii=False)
        from planning.validation import validate_raw_blueprint, bind_visual_priors
        messages = [{"role": "system", "content": "You are a structured music planner. Output valid JSON only."},
                    {"role": "user", "content": prompt_en}]
        self.last_trace = {"model": MODEL_NAME, "temperature": 0.2, "prompt": prompt_en, "attempts": []}
        for attempt in range(3):
            raw_text = chat(messages, temperature=0.2, **({"budget": budget} if budget is not None else {}))
            try:
                raw_json = json.loads(extract_json_from_text(raw_text))
                validate_raw_blueprint(raw_json)
                bind_visual_priors(raw_json, prior_data)
                blueprint = self.parser.parse(raw_json, source="llm_blueprint")
                self.last_trace["attempts"].append({"response": raw_text, "valid": True})
                return blueprint
            except (ValueError, TypeError, KeyError) as exc:
                self.last_trace["attempts"].append({"response": raw_text, "valid": False, "error": str(exc)})
                messages.extend([{"role": "assistant", "content": raw_text},
                                 {"role": "user", "content": f"Validation failed: {exc}. Return a corrected complete JSON."}])
        raise ValueError("Composer failed blueprint validation after three attempts")


def build_pipeline_artifacts(
    blueprint,
    *,
    output_dir: str,
    refine_with_rl: bool = False,
    rl_episodes: int = 20,
) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_blueprint_json(blueprint, str(out_dir / "blueprint.json"))

    bar_contexts = blueprint_to_bar_contexts(blueprint)
    _write_json(out_dir / "bar_contexts.json", bar_contexts)

    prefixes = blueprint_to_mupt_prefixes(blueprint)
    _write_lines(prefixes, out_dir / "mupt_prefixes.txt")

    render_controls = blueprint_to_render_controls(blueprint)
    _write_json(out_dir / "render_controls.json", render_controls)

    prompts = render_controls_to_prompts(render_controls)
    _write_lines(prompts, out_dir / "musicgen_prompts.txt")

    artifacts = {
        "blueprint": str(out_dir / "blueprint.json"),
        "bar_contexts": str(out_dir / "bar_contexts.json"),
        "mupt_prefixes": str(out_dir / "mupt_prefixes.txt"),
        "render_controls": str(out_dir / "render_controls.json"),
        "musicgen_prompts": str(out_dir / "musicgen_prompts.txt"),
    }

    if refine_with_rl:
        refinement = refine_blueprint_controls(blueprint, num_episodes=rl_episodes)
        _write_json(out_dir / "rl_refinement.json", refinement)
        _write_json(out_dir / "render_controls_refined.json", refinement["refined_controls"])
        _write_lines(refinement["refined_prompts"], out_dir / "musicgen_prompts_refined.txt")
        artifacts["rl_refinement"] = str(out_dir / "rl_refinement.json")
        artifacts["render_controls_refined"] = str(out_dir / "render_controls_refined.json")
        artifacts["musicgen_prompts_refined"] = str(out_dir / "musicgen_prompts_refined.txt")

    return artifacts


def plan_blueprint_from_prior_file(
    *,
    prior_json_path: str,
    output_dir: str,
    user_prompt: str = "",
    reference_hint: str = "",
    reference_audio_path: str = "",
    style_tags: Optional[List[str]] = None,
    refine_with_rl: bool = False,
    rl_episodes: int = 20,
) -> Dict[str, Any]:
    prior_data = json.loads(Path(prior_json_path).read_text(encoding="utf-8"))
    reference_audio = (
        extract_reference_audio_features(reference_audio_path)
        if reference_audio_path
        else None
    )
    controller = BlueprintComposerController(default_style_tags=style_tags or ["cinematic"])
    blueprint = controller.plan_blueprint(
        prior_data,
        user_prompt=user_prompt,
        reference_hint=reference_hint,
        reference_audio=reference_audio,
    )
    artifacts = build_pipeline_artifacts(
        blueprint,
        output_dir=output_dir,
        refine_with_rl=refine_with_rl,
        rl_episodes=rl_episodes,
    )
    pipeline_log = {
        "prior_json": prior_json_path,
        "output_dir": output_dir,
        "num_segments": len(blueprint.segments),
        "blueprint_source": blueprint.source,
        "reference_audio": reference_audio_path,
        "artifacts": artifacts,
    }
    if reference_audio is not None:
        reference_path = Path(output_dir) / "reference_audio_features.json"
        _write_json(reference_path, reference_audio_for_llm(reference_audio))
        pipeline_log["artifacts"]["reference_audio_features"] = str(reference_path)
    log_path = Path(output_dir) / "pipeline_log.json"
    _write_json(log_path, pipeline_log)
    return pipeline_log



def sanitize_abc(abc: str) -> str:
    s = abc

    # 1) 统一小调写法
    s = re.sub(r'^(K:\s*)Gmin\b', r'\1Gm', s, flags=re.MULTILINE)

    # 2) :: -> :|:
    s = s.replace('::', ':|:')

    # 3) 若开头有 |: 但末尾没 :| 则补上
    if '|:' in s and ':|' not in s:
        s = s.rstrip() + ' :|'

    # 4) 去掉悬空引号（如 _A" -> _A）
    s = re.sub(r'(_?[A-Ga-g][,\'"]?\d*)(?="\s*(\||$))', r'\1', s)

    # 5) 和弦标记两侧留空格
    s = re.sub(r'([_^=]?[A-Ga-g][,\']?\d*)(\"[^\"]+\")', r'\1 \2', s)
    s = re.sub(r'(\"[^\"]+\")([_^=]?[A-Ga-g])', r'\1 \2', s)

    return s

def demo_generate():
    """
    保留旧demo入口，但不在默认执行路径触发（避免过期API导致直接运行失败）。
    """
    ctrl = GlobalLLMController()

    ctx = {
        "bar_index_start": 1,
        "bar_count": 4,
        "bpm": 128,
        "meter": "4/4",
        "section": "Chorus",
        "chords": ["Gm", "F", "C", "D"],
        "energies": [0.75, 0.8, 0.7, 0.85],
        "global_key": "Gmin",
    }

    try:
        result = ctrl.generate_full_melody(ctx, max_new_tokens=512, temperature=0.8)

        print("\n=== 生成结果 ===")
        print(f"前缀(多行):\n{result['prefix']}")
        if "items" in result:
            for it in result["items"]:
                print(f"\n--- 条目 {it['index']} ---")
                print(f"前缀: {it['prefix']}")
                print("原始ABC:\n" + it["raw_abc"])
                print("后处理ABC:\n" + it["restored_abc"])
                if it.get("midi_path"):
                    print(f"MIDI文件: {it['midi_path']}")

        output_file = "generated_melody.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"上下文: {result['context']}\n\n")
            f.write(f"前缀(多行):\n{result['prefix']}\n\n")
            if "items" in result:
                for it in result["items"]:
                    f.write(f"--- 条目 {it['index']} ---\n")
                    f.write(f"前缀: {it['prefix']}\n")
                    f.write(f"原始ABC:\n{it['raw_abc']}\n")
                    f.write(f"后处理ABC:\n{it['restored_abc']}\n")
                    if it.get("midi_path"):
                        f.write(f"MIDI文件: {it['midi_path']}\n")
                    f.write("\n")

        print(f"\n结果已保存到: {output_file}")

        if "items" in result:
            print(f"\n🎵 生成的MIDI文件:")
            for it in result["items"]:
                if it.get("midi_path"):
                    print(f"  条目 {it['index']}: {it['midi_path']}")

    except Exception as e:
        print(f"生成过程中出现错误: {e}")
        print("尝试只生成前缀...")
        try:
            prefix_line = ctrl.plan(ctx)
            print(f"前缀: {prefix_line}")
        except Exception as e2:
            print(f"前缀生成也失败: {e2}")


def main(argv=None) -> int:
    """
    主入口：默认不触发任何过期API调用。
    需要显式子命令才会执行对应逻辑。
    """
    import argparse

    parser = argparse.ArgumentParser(prog="musicagent")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("generate-demo", help="运行旧的音乐生成demo（会调用网络API，可能已过期）")
    blueprint_cmd = sub.add_parser("plan-blueprint", help="从视频先验文件生成MusicBlueprint并导出后续控制产物")
    blueprint_cmd.add_argument("--prior-json", required=True, help="extract_prior_5d 输出的 JSON 文件")
    blueprint_cmd.add_argument("--output-dir", default="output/blueprint_pipeline")
    blueprint_cmd.add_argument("--user-prompt", default="", help="可选文字提示")
    blueprint_cmd.add_argument("--reference-hint", default="", help="可选参考信息")
    blueprint_cmd.add_argument("--reference-audio", default="", help="可选参考音频文件")
    blueprint_cmd.add_argument("--style-tag", action="append", dest="style_tags", default=None, help="可重复指定风格标签")
    blueprint_cmd.add_argument("--refine-with-rl", action="store_true")
    blueprint_cmd.add_argument("--rl-episodes", type=int, default=20)
    sub.add_parser("reward-status", help="仅检查reward强化模块是否已接入且可用（不训练）")

    args = parser.parse_args(argv)

    if args.cmd == "generate-demo":
        demo_generate()
        return 0

    if args.cmd == "plan-blueprint":
        log = plan_blueprint_from_prior_file(
            prior_json_path=args.prior_json,
            output_dir=args.output_dir,
            user_prompt=args.user_prompt,
            reference_hint=args.reference_hint,
            reference_audio_path=args.reference_audio,
            style_tags=args.style_tags,
            refine_with_rl=args.refine_with_rl,
            rl_episodes=args.rl_episodes,
        )
        print(json.dumps(log, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "reward-status":
        from reward.integration import get_integration_status

        status = get_integration_status()
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
