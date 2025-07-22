#!/usr/bin/env python3
"""
FEBID监控器工具模块
包含共享的工具类和函数

Author: 刘宇
Date: 2025/7
"""

from typing import Dict, List, Tuple
from config import calculate_dynamic_visualization_ranges
from base_classes import BaseMonitor
import monitor_templates as templates
import numpy as np

class ZAxisCalculator:
    """Z轴范围计算器"""

    @staticmethod
    def calculate_adaptive_z_range(config: Dict, monitor_config: Dict) -> Tuple[List[float], str]:
        """
        根据基底几何自适应计算Z轴范围

        Parameters:
        -----------
        config : Dict
            仿真配置
        monitor_config : Dict
            监控配置

        Returns:
        --------
        Tuple[List[float], str] : (新的height_range, 调整说明)
        """
        # 如果用户在 monitor_config 中指定了具体范围，优先使用
        if isinstance(monitor_config.get('height_range'), list) and monitor_config['height_range'] != 'auto':
            return monitor_config['height_range'], "使用用户指定的Z轴范围"

        # 否则动态计算
        dynamic_ranges = calculate_dynamic_visualization_ranges(config)
        return dynamic_ranges[
            'height_range'], f"动态计算范围（平衡浓度: {dynamic_ranges['equilibrium_concentration']:.3f}）"


class MonitorHTMLGenerator:
    """HTML生成器类，减少代码重复"""

    @staticmethod
    def generate_plot_javascript():
        """生成共享的绘图JavaScript代码 - 使用模板"""
        return templates.get_plot_javascript()

    @staticmethod
    def generate_control_panel(mode='realtime'):
        """生成控制面板HTML - 使用模板"""
        if mode == 'realtime':
            return templates.get_realtime_control_panel()
        else:
            return templates.get_traditional_control_panel()

    @staticmethod
    def generate_common_styles():
        """生成共享的CSS样式 - 委托给BaseMonitor"""
        return BaseMonitor.generate_common_styles()


class MonitorDataProcessor:
    """监控数据处理工具类"""

    @staticmethod
    def prepare_snapshot_data(h_surface, n_surface, pixel_idx, timestamp, beam_pos, is_running):
        """准备快照数据，优化内存使用"""

        return {
            'h_surface': h_surface.astype(np.float32).tolist(),  # 使用float32减少内存
            'n_surface': n_surface.astype(np.float32).tolist(),
            'metadata': {
                'pixel_index': int(pixel_idx),
                'timestamp': float(timestamp),
                'beam_position': [float(beam_pos[0]), float(beam_pos[1])],
                'max_height': float(np.max(h_surface)),
                'is_running': is_running
            }
        }

    @staticmethod
    def compress_history_data(history_data, max_points=1000):
        """压缩历史数据，保持关键点"""
        if len(history_data['pixel_indices']) <= max_points:
            return history_data

        # 等间隔采样
        indices = np.linspace(0, len(history_data['pixel_indices']) - 1,
                              max_points, dtype=int)

        return {
            'pixel_indices': [history_data['pixel_indices'][i] for i in indices],
            'max_heights': [history_data['max_heights'][i] for i in indices],
            'timestamps': [history_data['timestamps'][i] for i in indices]
        }
