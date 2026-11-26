#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频内容分析与音乐生成指导系统
Video Content Analysis for Music Generation Guidance

功能：
1. 视频帧抽帧、音频提取
2. 镜头切分检测
3. 音频特征提取（BPM、节拍、能量）
4. 视觉特征分析（颜色、亮度、运动）
5. 场景/情绪/动作识别
6. 时间对齐与特征融合
7. 输出特征JSON供LLM使用
"""

import os
import json
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import cv2
import librosa


@dataclass
class VideoFrame:
    """视频帧数据"""
    timestamp: float
    frame_number: int
    image: np.ndarray


@dataclass
class ShotBoundary:
    """镜头边界"""
    start_time: float
    end_time: float
    shot_id: int
    cut_type: str  # 'hard', 'fade', 'dissolve'


@dataclass
class AudioFeatures:
    """音频特征"""
    bpm: float
    tempo: float
    energy: np.ndarray
    onsets: np.ndarray
    beats: np.ndarray
    mfcc: np.ndarray
    spectral_centroid: np.ndarray
    zero_crossing_rate: np.ndarray


@dataclass
class VisualFeatures:
    """视觉特征"""
    brightness: float
    color_dominance: Dict[str, float]  # HSV dominant color
    motion_intensity: float
    color_histogram: np.ndarray


@dataclass
class SemanticFeatures:
    """语义特征"""
    emotion_scores: Dict[str, float]
    scene_type: str
    action_type: str
    visual_keywords: List[str]


@dataclass
class VideoSegment:
    """视频片段"""
    start_time: float
    end_time: float
    duration: float
    shot_boundaries: List[ShotBoundary]
    audio_features: AudioFeatures
    visual_features: List[VisualFeatures]
    semantic_features: SemanticFeatures
    hit_points: List[float]  # 关键时间点


class VideoAnalyzer:
    """视频分析器"""
    
    def __init__(self, video_path: str, output_dir: str = "output"):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 缓存目录
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # 视频信息
        self.fps = None
        self.duration = None
        self.width = None
        self.height = None
        
    def extract_video_info(self) -> Dict:
        """提取视频基本信息"""
        print(f"📹 分析视频: {self.video_path}")
        
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,duration',
            '-of', 'json',
            str(self.video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        
        if info['streams']:
            stream = info['streams'][0]
            self.width = int(stream['width'])
            self.height = int(stream['height'])
            
            # 计算FPS
            fps_parts = stream['r_frame_rate'].split('/')
            self.fps = float(fps_parts[0]) / float(fps_parts[1])
            
            # 获取duration
            cmd2 = ['ffprobe', '-v', 'error', '-show_entries', 
                   'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
                   str(self.video_path)]
            result2 = subprocess.run(cmd2, capture_output=True, text=True)
            self.duration = float(result2.stdout.strip())
        
        print(f"  分辨率: {self.width}x{self.height}")
        print(f"  FPS: {self.fps:.2f}")
        print(f"  时长: {self.duration:.2f}秒")
        
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'duration': self.duration
        }
    
    def extract_frames(self, sample_rate: int = 1) -> List[VideoFrame]:
        """
        提取视频帧
        
        Args:
            sample_rate: 每秒采样帧数，1表示每秒1帧
        """
        print(f"\n🎞️  提取视频帧 (采样率: {sample_rate} fps)...")
        
        frames = []
        cap = cv2.VideoCapture(str(self.video_path))
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 根据采样率决定是否保留这一帧
            if frame_count % int(self.fps / sample_rate) == 0:
                timestamp = frame_count / self.fps
                frames.append(VideoFrame(
                    timestamp=timestamp,
                    frame_number=frame_count,
                    image=frame
                ))
            
            frame_count += 1
        
        cap.release()
        print(f"  提取了 {len(frames)} 帧")
        
        return frames
    
    def extract_audio(self) -> Tuple[np.ndarray, int]:
        """
        提取音频
        
        Returns:
            audio_data: 音频数据
            sample_rate: 采样率
        """
        print(f"\n🔊 提取音频...")
        
        # 使用ffmpeg提取音频
        audio_path = self.cache_dir / "audio.wav"
        cmd = [
            'ffmpeg', '-i', str(self.video_path),
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '22050', '-ac', '1',
            '-y', str(audio_path)
        ]
        subprocess.run(cmd, capture_output=True)
        
        # 使用librosa加载
        audio, sr = librosa.load(str(audio_path), sr=22050)
        print(f"  音频长度: {len(audio)/sr:.2f}秒, 采样率: {sr}Hz")
        
        return audio, sr
    
    def detect_shot_boundaries(self, frames: List[VideoFrame], 
                               threshold: float = 0.3) -> List[ShotBoundary]:
        """
        检测镜头切分
        
        Args:
            frames: 视频帧列表
            threshold: 切分阈值
        """
        print(f"\n🎬 检测镜头切分 (阈值: {threshold})...")
        
        boundaries = []
        shot_id = 0
        start_time = 0
        
        for i in range(1, len(frames)):
            # 计算相邻帧的差异
            frame1 = cv2.cvtColor(frames[i-1].image, cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frames[i].image, cv2.COLOR_BGR2GRAY)
            
            # 使用帧差分
            diff = cv2.absdiff(frame1, frame2)
            diff_score = np.mean(diff) / 255.0
            
            # 如果差异超过阈值，认为是镜头切换
            if diff_score > threshold:
                # 判断切分类型（简化版：假设都是hard cut）
                boundaries.append(ShotBoundary(
                    start_time=start_time,
                    end_time=frames[i].timestamp,
                    shot_id=shot_id,
                    cut_type='hard'
                ))
                shot_id += 1
                start_time = frames[i].timestamp
        
        # 添加最后一个镜头
        if boundaries and boundaries[-1].end_time < self.duration:
            boundaries.append(ShotBoundary(
                start_time=boundaries[-1].end_time,
                end_time=self.duration,
                shot_id=shot_id,
                cut_type='hard'
            ))
        
        print(f"  检测到 {len(boundaries)} 个镜头")
        
        return boundaries
    
    def extract_audio_features(self, audio: np.ndarray, sr: int) -> AudioFeatures:
        """
        提取音频特征
        
        Args:
            audio: 音频数据
            sr: 采样率
        """
        print(f"\n🎵 提取音频特征...")
        
        # BPM和节拍检测
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        # tempo可能返回数组，提取标量
        bpm = float(tempo) if isinstance(tempo, (list, np.ndarray)) else tempo
        
        # 能量包络
        energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        
        # 起始点检测
        onsets = librosa.onset.onset_detect(y=audio, sr=sr)
        onsets_time = librosa.frames_to_time(onsets, sr=sr)
        
        # MFCC特征
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        # 频谱质心
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        
        # 过零率
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        print(f"  BPM: {bpm:.1f}")
        print(f"  节拍数: {len(beats)}")
        print(f"  起始点数: {len(onsets)}")
        
        return AudioFeatures(
            bpm=bpm,
            tempo=tempo,
            energy=energy,
            onsets=onsets_time,
            beats=beats,
            mfcc=mfcc,
            spectral_centroid=spectral_centroid,
            zero_crossing_rate=zcr
        )
    
    def extract_visual_features(self, frames: List[VideoFrame]) -> List[VisualFeatures]:
        """
        提取视觉特征
        
        Args:
            frames: 视频帧列表
        """
        print(f"\n🎨 提取视觉特征...")
        
        visual_features = []
        
        for frame in frames:
            img = frame.image
            
            # 亮度
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            # 颜色直方图（HSV）
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            
            # 主色分析（简化版）
            color_histogram = np.concatenate([hist_h.flatten(), 
                                            hist_s.flatten(), 
                                            hist_v.flatten()])
            
            # 运动强度（使用光流，此处简化）
            motion_intensity = 0.0  # 简化处理
            
            visual_features.append(VisualFeatures(
                brightness=brightness,
                color_dominance={},  # 简化
                motion_intensity=motion_intensity,
                color_histogram=color_histogram
            ))
        
        print(f"  提取了 {len(visual_features)} 个视觉特征")
        
        return visual_features
    
    def extract_semantic_features(self, frames: List[VideoFrame]) -> SemanticFeatures:
        """
        提取语义特征（场景、情绪、动作）
        
        注意：这里使用简化版本，实际应用可以使用CLIP、TimeSformer等模型
        """
        print(f"\n🧠 提取语义特征...")
        
        # 基于亮度和颜色简单推断情绪
        avg_brightness = np.mean([np.mean(cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY)) 
                                 for f in frames])
        
        # 简单的情绪推断（可扩展为CLIP零样本分类）
        emotion_scores = {
            'energetic': 0.5,
            'calm': 0.5,
            'mysterious': 0.3,
            'romantic': 0.2
        }
        
        # 根据亮度调整
        if avg_brightness > 128:
            emotion_scores['energetic'] = 0.8
            emotion_scores['calm'] = 0.2
        else:
            emotion_scores['energetic'] = 0.2
            emotion_scores['mysterious'] = 0.8
        
        print(f"  情绪分析: {emotion_scores}")
        
        return SemanticFeatures(
            emotion_scores=emotion_scores,
            scene_type='general',
            action_type='unknown',
            visual_keywords=['video']
        )
    
    def detect_hitpoints(self, audio_features: AudioFeatures, 
                        boundaries: List[ShotBoundary]) -> List[float]:
        """
        检测关键时间点（hitpoints）
        
        融合镜头切换、音频起始点、能量峰值等
        """
        print(f"\n⏰ 检测关键时间点...")
        
        hitpoints = []
        
        # 镜头切换点
        for boundary in boundaries:
            hitpoints.append(boundary.start_time)
            hitpoints.append(boundary.end_time)
        
        # 音频起始点
        hitpoints.extend(audio_features.onsets[:10].tolist())  # 前10个
        
        # 去重并排序
        hitpoints = sorted(list(set(hitpoints)))
        
        print(f"  检测到 {len(hitpoints)} 个关键时间点")
        
        return hitpoints
    
    def align_and_resample(self, features, target_fps: float = 10.0):
        """
        时间对齐与重采样
        
        将特征对齐到固定时间步长
        """
        print(f"\n📊 时间对齐 (目标FPS: {target_fps})...")
        
        # 简化实现
        # 实际应该使用时间对齐算法
        return features
    
    def analyze(self, sample_rate: int = 1) -> VideoSegment:
        """
        完整分析流程
        
        Args:
            sample_rate: 帧采样率
        
        Returns:
            VideoSegment: 分析结果
        """
        print("=" * 60)
        print("开始视频分析")
        print("=" * 60)
        
        # 1. 视频信息
        self.extract_video_info()
        
        # 2. 提取帧
        frames = self.extract_frames(sample_rate=sample_rate)
        
        # 3. 提取音频
        audio, sr = self.extract_audio()
        
        # 4. 镜头切分
        boundaries = self.detect_shot_boundaries(frames)
        
        # 5. 音频特征
        audio_features = self.extract_audio_features(audio, sr)
        
        # 6. 视觉特征
        visual_features = self.extract_visual_features(frames)
        
        # 7. 语义特征
        semantic_features = self.extract_semantic_features(frames)
        
        # 8. 关键时间点
        hitpoints = self.detect_hitpoints(audio_features, boundaries)
        
        # 9. 构建结果
        segment = VideoSegment(
            start_time=0,
            end_time=self.duration,
            duration=self.duration,
            shot_boundaries=boundaries,
            audio_features=audio_features,
            visual_features=visual_features,
            semantic_features=semantic_features,
            hit_points=hitpoints
        )
        
        print("\n" + "=" * 60)
        print("分析完成")
        print("=" * 60)
        
        return segment
    
    def save_results(self, segment: VideoSegment, output_file: str = "video_analysis.json"):
        """
        保存分析结果
        
        Args:
            segment: 分析结果
            output_file: 输出文件名
        """
        print(f"\n💾 保存结果到: {output_file}")
        
        # 转换为可序列化的格式
        result_dict = {
            'video_info': {
                'width': self.width,
                'height': self.height,
                'fps': self.fps,
                'duration': self.duration
            },
            'segment': {
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'duration': segment.duration,
                'shot_boundaries': [asdict(b) for b in segment.shot_boundaries],
                'audio_features': {
                    'bpm': float(segment.audio_features.bpm),
                    'tempo': float(segment.audio_features.tempo),
                    'onsets': segment.audio_features.onsets.tolist(),
                    'beats': segment.audio_features.beats.tolist()
                },
                'semantic_features': {
                    'emotion_scores': segment.semantic_features.emotion_scores,
                    'scene_type': segment.semantic_features.scene_type,
                    'action_type': segment.semantic_features.action_type,
                    'visual_keywords': segment.semantic_features.visual_keywords
                },
                'hit_points': segment.hit_points
            }
        }
        
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ 已保存到: {output_path}")


def main():
    """主函数 - 示例用法"""
    # 示例：分析视频
    video_path = "input/sample_video.mp4"  # 替换为实际视频路径
    
    if not os.path.exists(video_path):
        print(f"⚠️  视频文件不存在: {video_path}")
        print("请提供有效的视频文件路径")
        return
    
    # 创建分析器
    analyzer = VideoAnalyzer(video_path, output_dir="output")
    
    # 执行分析
    segment = analyzer.analyze(sample_rate=1)
    
    # 保存结果
    analyzer.save_results(segment, "video_analysis.json")
    
    print("\n✨ 分析完成！结果已保存到 output/video_analysis.json")


if __name__ == "__main__":
    main()

