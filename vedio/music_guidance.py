#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频特征到音乐生成指导
Video Features to Music Generation Guidance

将视频分析结果转换为音乐生成的上下文参数
"""

import json
import numpy as np
from typing import Dict, List
from pathlib import Path


class MusicGuidanceGenerator:
    """音乐生成指导生成器"""
    
    def __init__(self, analysis_result: Dict):
        self.analysis = analysis_result
    
    def generate_bar_context(self, segment_duration: float = 4.0) -> List[Dict]:
        """
        基于视频分析生成小节上下文
        
        Args:
            segment_duration: 每个小节的时长（秒）
        
        Returns:
            List[Dict]: 小节上下文列表
        """
        contexts = []
        
        num_segments = int(self.analysis.duration / segment_duration)
        
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, self.analysis.duration)
            
            # 获取该时间段内的情绪
            emotion = self._get_emotion_for_time(start_time, end_time)
            
            # 估算BPM
            bpm = self._estimate_bpm()
            
            # 估算拍号
            meter = self._estimate_meter()
            
            # 估算能量
            energy = self._estimate_energy(start_time, end_time)
            
            # 估算和弦（根据情绪和能量简化）
            chord = self._estimate_chord(emotion, energy)
            
            # 估算调性
            key = self._estimate_key(emotion)
            
            context = {
                "bar_index_start": i + 1,
                "bar_count": 1,
                "bpm": int(bpm),
                "meter": meter,
                "section": self._estimate_section(i, num_segments),
                "chords": [chord],
                "energies": [energy],
                "global_key": key
            }
            
            contexts.append(context)
        
        return contexts
    
    def _get_emotion_for_time(self, start_time: float, end_time: float) -> str:
        """获取时间段的主要情绪"""
        # 简化：返回整体情绪
        scores = self.analysis.semantic_features.emotion_scores
        dominant_emotion = max(scores.items(), key=lambda x: x[1])[0]
        
        emotion_map = {
            'energetic': 'upbeat',
            'calm': 'peaceful',
            'mysterious': 'dark',
            'romantic': 'warm'
        }
        
        return emotion_map.get(dominant_emotion, 'neutral')
    
    def _estimate_bpm(self) -> float:
        """估算BPM"""
        if hasattr(self.analysis.audio_features, 'bpm'):
            return float(self.analysis.audio_features.bpm)
        return 120.0  # 默认BPM
    
    def _estimate_meter(self) -> str:
        """估算拍号"""
        # 简化：默认4/4
        return "4/4"
    
    def _estimate_energy(self, start_time: float, end_time: float) -> float:
        """估算能量值（0-1）"""
        # 简化：根据情绪估算
        emotion = self._get_emotion_for_time(start_time, end_time)
        
        energy_map = {
            'upbeat': 0.8,
            'peaceful': 0.4,
            'dark': 0.6,
            'warm': 0.7,
            'neutral': 0.5
        }
        
        return energy_map.get(emotion, 0.5)
    
    def _estimate_chord(self, emotion: str, energy: float) -> str:
        """估算和弦"""
        # 简化和弦选择
        if emotion == 'upbeat':
            if energy > 0.7:
                return "C"  # 大和弦
            return "Am"  # 小调
        elif emotion == 'dark':
            return "Em"  # 小调
        elif emotion == 'warm':
            return "F"
        else:
            return "Gm"  # 小调
    
    def _estimate_key(self, emotion: str) -> str:
        """估算调性"""
        key_map = {
            'upbeat': 'C',
            'peaceful': 'Am',
            'dark': 'Em',
            'warm': 'F',
            'neutral': 'Gm'
        }
        return key_map.get(emotion, 'C')
    
    def _estimate_section(self, index: int, total: int) -> str:
        """估算音乐段落"""
        if index < total * 0.25:
            return "Intro"
        elif index < total * 0.5:
            return "Verse"
        elif index < total * 0.75:
            return "Chorus"
        else:
            return "Outro"
    
    def generate_prompt_text(self) -> str:
        """
        生成LLM提示文本
        
        Returns:
            str: 音乐生成提示
        """
        emotion = max(
            self.analysis.semantic_features.emotion_scores.items(),
            key=lambda x: x[1]
        )[0]
        
        bpm = self._estimate_bpm()
        scene = self.analysis.semantic_features.scene_type
        
        prompt = f"""
基于以下视频特征生成音乐：

**视频信息：**
- 情绪基调: {emotion}
- BPM建议: {int(bpm)}
- 场景类型: {scene}
- 视频时长: {self.analysis.duration:.1f}秒
- 镜头数量: {len(self.analysis.shot_boundaries)}
- 关键时间点: {len(self.analysis.hit_points)}个

**音乐风格建议：**
- 主色调: {emotion}
- 推荐能量范围: {self._get_energy_range()}
- 推荐拍号: {self._estimate_meter()}

请根据这些特征生成适配的音乐。
"""
        return prompt.strip()
    
    def _get_energy_range(self) -> str:
        """获取能量范围描述"""
        avg_energy = np.mean([self._estimate_energy(i, i+1) 
                             for i in range(int(self.analysis.duration))])
        
        if avg_energy > 0.7:
            return "高能量 (0.7-0.9)"
        elif avg_energy > 0.4:
            return "中等能量 (0.4-0.7)"
        else:
            return "低能量 (0.2-0.4)"


def load_analysis_from_json(json_path: str) -> Dict:
    """从JSON加载分析结果"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    # 示例用法
    analysis_json = "output/video_analysis.json"
    
    if not Path(analysis_json).exists():
        print(f"请先运行 lens_detect.py 生成分析结果")
        exit(1)
    
    data = load_analysis_from_json(analysis_json)
    
    # 创建指导生成器
    # guidance = MusicGuidanceGenerator(data)
    
    print("视频到音乐生成指导接口已准备就绪")


