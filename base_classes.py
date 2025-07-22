#!/usr/bin/env python3
"""
FEBID仿真基础类和工具函数
提供共享功能以减少代码重复

Author: 刘宇
Date: 2025/7
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import json
from numba import jit, prange


class BaseMonitor(ABC):
    """监控器基类，包含共享功能"""

    @staticmethod
    def generate_plot_config(width: int, height: int) -> Dict:
        """生成通用图表配置"""
        return {
            'width': width,
            'height': height,
            'margin': {'l': 45, 'r': 80, 't': 50, 'b': 45},
            'font': {'size': 11}
        }

    @staticmethod
    def generate_colorbar_config(title: str, range_vals: list) -> Dict:
        """生成优化的colorbar配置"""
        return {
            'title': {
                'text': title,
                'font': {'size': 12}
            },
            'titleside': 'right',
            'tickformat': '.2e',
            'exponentformat': 'e',
            'nticks': 8,
            'thickness': 12,
            'len': 0.8,
            'x': 1.02,
            'tickfont': {'size': 10}
        }

    @staticmethod
    def generate_common_styles() -> str:
        """生成共享的CSS样式"""
        return """
        <style>
            body { font-family: Arial, sans-serif; margin: 12px; background-color: #f5f5f5; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                      color: white; padding: 15px; border-radius: 10px; margin-bottom: 12px; text-align: center; }
            .status { background-color: #e8f5e8; border: 2px solid #4CAF50; border-radius: 8px; 
                      padding: 12px; margin: 12px 0; font-family: monospace; font-size: 13px; }
            .container { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; 
                         background-color: white; padding: 12px; border-radius: 10px; }
            .plot-container { border: 2px solid #ddd; border-radius: 8px; margin: auto; }
            .controls { grid-column: 1 / -1; text-align: center; margin: 12px 0; padding: 12px; 
                        background-color: #fff3cd; border-radius: 10px; }
            .button { background-color: #4CAF50; border: none; color: white; padding: 8px 16px; 
                      margin: 3px; cursor: pointer; border-radius: 5px; font-size: 13px; }
            .button.secondary { background-color: #17a2b8; }
            .slider { width: 80%; height: 18px; margin: 8px; }
            .status-indicator { display: inline-block; width: 12px; height: 12px; 
                               border-radius: 50%; margin-right: 8px; }
            .status-running { background-color: #4CAF50; animation: pulse 2s infinite; }
            .status-stopped { background-color: #f44336; }
            @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
        </style>
        """

    @abstractmethod
    def update_data(self, *args, **kwargs):
        """更新监控数据"""
        pass

    @abstractmethod
    def launch_viewer(self):
        """启动查看器"""
        pass


class ScanStrategy(ABC):
    """扫描策略抽象基类"""

    @abstractmethod
    def generate_path(self, x_positions: np.ndarray, y_positions: np.ndarray) -> np.ndarray:
        """生成扫描路径"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """策略描述"""
        pass


class ConfigValidator:
    """集中的配置验证器"""

    @staticmethod
    def validate_range(value: float, min_val: float, max_val: float, name: str):
        """通用范围验证"""
        if not min_val <= value <= max_val:
            raise ValueError(f"{name} must be between {min_val} and {max_val}")

    @classmethod
    def validate_geometry_config(cls, geometry: Dict) -> bool:
        """验证几何配置"""
        try:
            if geometry['X_max'] <= geometry['X_min'] or geometry['Y_max'] <= geometry['Y_min']:
                print("❌ 几何范围设置错误: max应大于min")
                return False

            if geometry['dx'] <= 0 or geometry['dy'] <= 0:
                print("❌ 网格间距必须为正值")
                return False

            return True
        except KeyError as e:
            print(f"❌ 缺少几何配置项: {e}")
            return False

    @classmethod
    def validate_scan_config(cls, scan: Dict, geometry: Dict) -> bool:
        """验证扫描配置"""
        try:
            # 扫描范围验证
            if scan['scan_x_end'] < scan['scan_x_start'] or scan['scan_y_end'] < scan['scan_y_start']:
                print("❌ 扫描范围设置错误: end应大于start")
                return False

            # 扫描范围是否在几何范围内
            if (scan['scan_x_start'] < geometry['X_min'] or scan['scan_x_end'] > geometry['X_max'] or
                    scan['scan_y_start'] < geometry['Y_min'] or scan['scan_y_end'] > geometry['Y_max']):
                print("⚠️  警告: 扫描范围超出几何范围")

            # 正值验证
            positive_params = ['pixel_size_x', 'pixel_size_y', 'dwell_time']
            for param in positive_params:
                if scan[param] <= 0:
                    print(f"❌ {param} 必须为正值")
                    return False

            # 循环参数验证
            if scan.get('loop') is not None and scan.get('subloop') is not None:
                print("❌ 不能同时设置loop和subloop参数")
                return False

            if scan.get('loop') is None and scan.get('subloop') is None:
                print("❌ 必须设置loop或subloop参数之一")
                return False

            return True
        except KeyError as e:
            print(f"❌ 缺少扫描配置项: {e}")
            return False

    @classmethod
    def validate_physical_config(cls, physical: Dict) -> bool:
        """验证物理参数配置"""
        try:
            positive_params = ['Phi', 'tau', 'sigma', 'n0', 'DeltaV', 'k', 'dx', 'dy']
            for param in positive_params:
                if physical[param] <= 0:
                    print(f"❌ 物理参数 {param} 必须为正值")
                    return False

            # D_surf可以为0（无扩散）
            if physical['D_surf'] < 0:
                print("❌ 表面扩散系数D_surf不能为负值")
                return False

            return True
        except KeyError as e:
            print(f"❌ 缺少物理配置项: {e}")
            return False


def calculate_surface_statistics(surface_data: np.ndarray, region_mask: Optional[np.ndarray] = None) -> Dict:
    """计算表面统计数据"""
    if region_mask is not None:
        data = surface_data[region_mask]
    else:
        data = surface_data.flatten()

    return {
        'max': float(np.max(data)),
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'median': float(np.median(data))
    }


@jit(nopython=True, parallel=True)
def compute_edge_mask_parallel(nx_total: int, ny_total: int, edge_layers: int) -> np.ndarray:
    """并行计算边缘掩码"""
    edge_mask = np.zeros((nx_total, ny_total), dtype=np.bool_)

    if edge_layers <= 0:
        return edge_mask

    for i in prange(nx_total):
        for j in range(ny_total):
            is_x_edge = (i < edge_layers) or (i >= nx_total - edge_layers)
            is_y_edge = (j < edge_layers) or (j >= ny_total - edge_layers)
            edge_mask[i, j] = is_x_edge or is_y_edge

    return edge_mask


@jit(nopython=True, parallel=True, fastmath=True)
def laplace_2d_parallel(arr: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """并行计算2D拉普拉斯算子"""
    ny, nx = arr.shape
    result = np.zeros_like(arr)

    dx2_inv = 1.0 / (dx * dx)
    dy2_inv = 1.0 / (dy * dy)

    for i in prange(1, ny - 1):
        for j in range(1, nx - 1):
            result[i, j] = ((arr[i, j + 1] - 2 * arr[i, j] + arr[i, j - 1]) * dx2_inv +
                            (arr[i + 1, j] - 2 * arr[i, j] + arr[i - 1, j]) * dy2_inv)

    # 边界条件（零通量）
    result[0, :] = result[1, :]
    result[-1, :] = result[-2, :]
    result[:, 0] = result[:, 1]
    result[:, -1] = result[:, -2]

    return result
