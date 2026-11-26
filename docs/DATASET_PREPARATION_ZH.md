# CoMA 本地数据准备

入口是 `python -m training.prepare_dataset`。脚本只处理本地文件，不下载数据，不生成虚构标注，不覆盖已有输出目录。各子命令均支持 `--help`。

## 1. 无真实数据时验证

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.verify_dataset \
  --output-dir output/dataset_check
```

生成三组明确标记为 synthetic 的 15 秒双轨 MIDI/音频和视频，运行 Slakh、MusicPro、CineParse 三种准备流程，重新解析 ABC 和训练清单。结果写入 `verification.json`。这些文件只用于软件测试，不能算作论文数据或性能结果。

## 2. Slakh2100 Redux

预期目录结构：

```text
datasets/slakh2100/
  train/Track00001/all_src.mid
  train/Track00001/mix.flac
  validation/Track01290/all_src.mid
  validation/Track01290/mix.flac
  test/Track01560/all_src.mid
  test/Track01560/mix.flac
```

`mix.wav`、`valid/`、`val/` 也支持。保留官方 train/validation/test 分区；跨分区重复 MIDI/音频哈希会报错。

```bash
python -m training.prepare_dataset slakh \
  --source datasets/slakh2100 --output-dir data/slakh
```

默认按论文提取非重叠 15 秒窗口，每个源曲目最多 32 个。每个窗口至少两轨、八个 onset groups，所有轨道活动时间的并集至少 10 秒。train/validation/test 的窗口上限为 20000/2000/1000，超出后按 seed 2026 抽样；不足时记录实际数量，不重复样本凑数。所有过滤原因和实际数量记录在 `dataset_report.json`。抽样未选入 manifest 的已提取文件仍保留在输出目录。

产物：

- `audio_{train,validation,test}.jsonl`：音频 SFT 数据。
- `symbolic_{train,validation,test}.jsonl`：符号 SFT 数据。
- 每个窗口的 `audio.wav`、`score.mid`、`symbolic_outputs.json`、`blueprint.json` 和每声部的 `.abc`。

音频转为 32 kHz 单声道。MIDI 读取 tempo map、音符 velocity、program 和 CC64 延音，跨窗口音符裁剪到边界。音频条件保存原始秒级事件。ABC 在 1/96 四分音符网格量化，将同起止音符合为和弦，并将交错复音分成单独声部训练样本，避免将多个声部误当作单轨生成结果。

**预处理边界**：这里的蓝图由谱面构造，只供 SFT；C 是带显式升降号的记谱参考，不是调性推断。音色角色只做 program/打击乐通道的简单映射。它们都不是人工标注或 Composer 输出，不能直接作为真实视频 PPO 条件。原论文的精确策展名单和原始 MIDI→ABC 转换规则仍未恢复。

## 3. MusicPro 视频/音乐对

准备 JSONL 索引，一行一个样本。文件路径相对索引文件解析。电影和音乐作品 ID 必须使用跨数据集一致的真实身份，不能每个 clip 新建一个 ID 来规避分组。

```json
{"sample_id":"clip001","dataset":"MusicPro-7k","movie_id":"movie001","music_work_id":"work001","video":"videos/clip001.mp4","audio":"audio/clip001.wav","input_condition":"V"}
```

```bash
python -m training.prepare_dataset musicpro \
  --index datasets/musicpro/index.jsonl \
  --output-dir data/musicpro --seed 2026 --extract-visual
```

脚本以同电影、同音乐作品和相同文件哈希建立连通分组，按约 80%/10%/10% 划分。已有 `split` 时保留它；同一连通组跨分区会失败。分组约束优先于数量，因此不能保证恰好 5934/742/742，实际数量和目标数量均记录。SHA-256 只识别完全相同的文件；音频近重复仍需真实 acoustic fingerprint 和规范作品 ID。

`--extract-visual` 用原生视频帧时间提取 OpenCV 实测曲线，写 `prior.json` 和 `reward_inputs.json`。它不生成 CineParse 监督标签或视频情感标签。输入应为论文要求的已对齐 15 秒视频/音乐对；此入口不自动裁剪长视频，也不执行 MT3 或 LLM。

主要产物是 `all.jsonl` 和 `train/validation/test.jsonl`。索引还可以提供：

| 字段 | 用途 |
|---|---|
| `blueprint` | 已由视频条件 Composer 生成的蓝图 JSON |
| `symbolic_outputs` | 音频 SFT 的对齐音符 JSON，MusicPro 应由固定转录模型生成 |
| `prefix`、`duration_seconds` | 符号 PPO 的显式 ABC 条件和片段时长 |
| `reward_inputs` | 已有的视频实测曲线和独立标注，未使用 `--extract-visual` 时保留 |
| `emotion_target` | 下方格式的视频 VA 标注，提取视觉时写入 reward_inputs |

满足字段要求的行分别导出 `audio_train.jsonl`、`symbolic_ppo_train.jsonl` 和 `guidance_train.jsonl`。不满足时相应清单为空，不能据此启动训练。`guidance_*` 和 `symbolic_ppo_*` 不带目标音频、目标 ABC 或目标事件路径。

视频情感目标格式：

```json
{"times":[1.5,2.0,2.5],"values":[[0.2,0.3],[0.4,0.6],[0.3,0.5]],"source":"silent_video_annotations","used_target_audio":false}
```

`values` 按 `[valence, arousal]` 排列，时间为片段秒数。示例只有三个点，实际应覆盖完整 15 秒片段。不能从隐藏目标音轨构造这个目标。

## 4. CineParse 人工标注

```json
{"sample_id":"visual001","dataset":"visual_corpus","movie_id":"movie001","video":"videos/clip001.mp4","start_time":0.0,"end_time":2.0,"prior_5d":[0.2,0.1,0.5,0.4,0.3],"label_source":"human_annotations","label_protocol":"five_descriptors_v1","hit_times":[0.5]}
```

五维顺序是 motion、cut、brightness、warmth、novelty，也支持与 `vedio.cineparse.LABELS` 一致的命名字典。窗口时间和 hit_times 都相对于所引用视频文件；hit 必须在窗口内。缺失 hit 标注时省略字段，不要用空列表代替未知标注。

```bash
python -m training.prepare_dataset cineparse \
  --index datasets/visual/annotations.jsonl --output-dir data/visual \
  --exclude-manifest data/musicpro/all.jsonl
```

按电影/作品/文件分组；`--exclude-manifest` 进一步将 MusicPro 验证/测试身份从视觉训练集排除，并记录排除项。输出可直接传给 `vedio.train_cineparse --train-manifest ... --validation-manifest ...`。

## 5. 混合音频 SFT 与训练衔接

```bash
python -m training.prepare_dataset mix-audio \
  --slakh-manifest data/slakh/audio_train.jsonl \
  --musicpro-manifest data/musicpro/audio_train.jsonl \
  --output-dir data/audio_mixture
```

两个来源的样本数相等。较小来源按打乱的循环过采样，重复行具有新的 sample_id，同时保留 `source_sample_id`；不会把重复采样伪装为更多独立曲目。这里只验证训练文件可读，不替代转录模型来源审核。

`training.run_staged` 现在可分别设置 `symbolic_manifest`（SFT）、`symbolic_ppo_manifest`（视频 PPO）、`audio_manifest`（SFT）和 `guidance_manifest`（供冻结符号策略生成）。旧配置未给后两个字段时仍沿用旧行为。示例见 `configs/staged_training.example.json`。音频 PPO 的 `symbolic_guidance` 必须由 `training.prepare_guidance` 生成并绑定检查点哈希，不能直接使用目标 MIDI。
