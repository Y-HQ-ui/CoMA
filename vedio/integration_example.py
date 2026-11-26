#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频分析 + 音乐生成集成示例
Video Analysis + Music Generation Integration Example

演示完整的流程：
1. 分析视频内容
2. 提取特征
3. 生成音乐生成指导
4. 调用音乐生成器生成音乐
"""

import sys
from pathlib import Path
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vedio.lens_detect import VideoAnalyzer
from vedio.music_guidance import MusicGuidanceGenerator, load_analysis_from_json
from main import GlobalLLMController


def analyze_video(video_path: str, output_dir: str = "output") -> dict:
    """
    分析视频并返回结果
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
    
    Returns:
        dict: 分析结果
    """
    print("=" * 80)
    print("步骤1: 视频分析")
    print("=" * 80)
    
    analyzer = VideoAnalyzer(video_path, output_dir=output_dir)
    segment = analyzer.analyze(sample_rate=1)  # 每秒采样1帧
    analyzer.save_results(segment, "video_analysis.json")
    
    # 返回JSON格式的结果
    return load_analysis_from_json(f"{output_dir}/video_analysis.json")


def generate_music_context(analysis_result: dict) -> list:
    """
    根据视频分析生成音乐生成上下文
    
    Args:
        analysis_result: 视频分析结果
    
    Returns:
        list: 音乐生成上下文列表
    """
    print("\n" + "=" * 80)
    print("步骤2: 生成音乐指导")
    print("=" * 80)
    
    guidance = MusicGuidanceGenerator(analysis_result)
    contexts = guidance.generate_bar_context(segment_duration=4.0)
    
    print(f"\n生成了 {len(contexts)} 个小节的上下文")
    print(f"\n前3个小节示例：")
    for i, ctx in enumerate(contexts[:3]):
        print(f"\n小节 {i+1}:")
        print(f"  BPM: {ctx['bpm']}")
        print(f"  情绪: {ctx['section']}")
        print(f"  和弦: {ctx['chords']}")
        print(f"  能量: {ctx['energies']}")
        print(f"  调性: {ctx['global_key']}")
    
    # 保存上下文
    output_file = "output/music_contexts.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(contexts, f, indent=2, ensure_ascii=False)
    
    print(f"\n上下文已保存到: {output_file}")
    
    return contexts


def generate_music(contexts: list, max_segments: int = None):
    """
    根据上下文生成音乐
    
    Args:
        contexts: 音乐生成上下文列表
        max_segments: 最大生成段落数（None表示全部）
    """
    print("\n" + "=" * 80)
    print("步骤3: 生成音乐")
    print("=" * 80)
    
    controller = GlobalLLMController()
    
    if max_segments:
        contexts = contexts[:max_segments]
    
    print(f"\n将为 {len(contexts)} 个段落生成音乐...")
    
    for i, ctx in enumerate(contexts):
        print(f"\n--- 段落 {i+1}/{len(contexts)} ---")
        try:
            result = controller.generate_full_melody(
                ctx, 
                max_new_tokens=512, 
                temperature=0.8
            )
            
            print(f"✓ 段落 {i+1} 生成完成")
            if 'items' in result:
                for item in result['items']:
                    if item.get('midi_path'):
                        print(f"  MIDI: {item['midi_path']}")
            
        except Exception as e:
            print(f"✗ 段落 {i+1} 生成失败: {e}")
            continue


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("视频驱动音乐生成系统")
    print("Video-Driven Music Generation System")
    print("=" * 80 + "\n")
    
    # 配置
    video_path = "input/sample_video.mp4"  # 替换为实际视频路径
    
    if not Path(video_path).exists():
        print(f"⚠️  视频文件不存在: {video_path}")
        print("\n请按以下步骤操作：")
        print("1. 准备一个视频文件")
        print("2. 将其放在 input/ 目录下")
        print("3. 更新此脚本中的 video_path 变量")
        print("\n或者使用命令行参数：")
        print("  python integration_example.py <video_path>")
        return
    
    try:
        # 步骤1: 分析视频
        analysis_result = analyze_video(video_path)
        
        # 步骤2: 生成音乐上下文
        contexts = generate_music_context(analysis_result)
        
        # 步骤3: 生成音乐（可选，限制前3个段落用于测试）
        generate_music(contexts, max_segments=3)
        
        print("\n" + "=" * 80)
        print("✨ 流程完成！")
        print("=" * 80)
        print("\n输出文件：")
        print("  - output/video_analysis.json: 视频分析结果")
        print("  - output/music_contexts.json: 音乐生成上下文")
        print("  - output/melody_*.mid: 生成的MIDI文件")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 支持命令行参数
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = None
    
    if video_path:
        print(f"使用视频: {video_path}")
        # 这里需要修改main函数以接受参数
        # 为简化，可以直接替换main函数中的video_path
    
    main()

