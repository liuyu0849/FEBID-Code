#!/usr/bin/env python3
"""
FEBID仿真数据结构定义
包含所有仿真所需的数据类

Author: 刘宇
Date: 2025/7
"""

from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import numpy as np

# 设置数值精度（内存优化）
FLOAT_DTYPE = np.float32  # 可根据精度需求调整为float64


@dataclass
class GaussianParams:
    """四高斯参数数据类"""
    sigma: float
    amplitude: float


@dataclass
class MaterialParams:
    """材料参数数据类"""
    gaussian1: GaussianParams
    gaussian2: GaussianParams
    gaussian3: GaussianParams
    gaussian4: GaussianParams


@dataclass
class QuadGaussianParams:
    """双材料四高斯参数"""
    substrate: MaterialParams
    deposit: MaterialParams
    z_deposit: float


@dataclass
class PhysicalParams:
    """物理参数数据类"""
    Phi: float  # 前驱体通量 [nm^-2 s^-1]
    tau: float  # 平均停留时间 [s]
    sigma: float  # 积分解离截面 [nm^2]
    n0: float  # 最大表面前驱体密度 [molecules/nm^2]
    DeltaV: float  # 有效解离前驱体分子体积 [nm^3]
    k: float  # 吸附系数
    D_surf: float  # 表面扩散系数 [nm^2/s]
    dx: float  # X方向网格间距 [nm]
    dy: float  # Y方向网格间距 [nm]


@dataclass
class ScanInfo:
    """扫描信息数据类"""
    nx_pixels: int
    ny_pixels_per_subloop: int
    base_pixels_per_subloop: int
    pixels_per_subloop: int
    total_subloops: int
    total_pixels: int
    pixel_size_x: float
    pixel_size_y: float
    edge_layers: int
    edge_repeat_times: int


@dataclass
class RectangularDefect:
    """矩形缺陷/特征定义"""
    x1: float  # 左下角x坐标 [nm]
    y1: float  # 左下角y坐标 [nm]
    x2: float  # 右上角x坐标 [nm]
    y2: float  # 右上角y坐标 [nm]
    height_offset: float  # 相对基准平面的高度偏移 [nm]
    name: str = ""  # 特征名称（可选）


@dataclass
class SubstrateGeometry:
    """基底几何配置"""
    base_height: float = 0.0  # 基准平面高度 [nm]
    rectangular_defects: List[RectangularDefect] = None

    def __post_init__(self):
        if self.rectangular_defects is None:
            self.rectangular_defects = []
