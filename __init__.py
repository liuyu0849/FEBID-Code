#!/usr/bin/env python3
"""
FEBID仿真包初始化文件 - 精简版（无监控）
删除所有监控相关导入

Author: 刘宇
Date: 2025/7
"""

__version__ = "2.0.0-simplified"
__author__ = "刘宇"
__description__ = "FEBID仿真系统 - 精简版（无监控）"

# 核心模块导入
from .substrate_geometry import (
    SubstrateGeometryGenerator, RectangularDefect, SubstrateGeometry,
    validate_substrate_geometry_config, create_example_substrate_config
)
from .base_classes import (
    ScanStrategy, ConfigValidator,
    calculate_surface_statistics, compute_edge_mask_parallel
)
from .simulation_core_main import MemoryOptimizedFEBID
from .scan_strategies import ScanPathGenerator
from .visualization_analysis import VisualizationAnalyzer
from .data_structures import (
    GaussianParams, MaterialParams, QuadGaussianParams,
    PhysicalParams, ScanInfo, FLOAT_DTYPE
)
from .config import (
    SIMULATION_CONFIG,
    validate_config,
    print_config_summary
)

# 便捷函数导入
from .main import main_with_custom_config, FEBIDSimulationRunner

# 公开API - 精简版
__all__ = [
    # 基础类
    'ScanStrategy',
    'ConfigValidator',

    # 核心类
    'MemoryOptimizedFEBID',
    'ScanPathGenerator',
    'VisualizationAnalyzer',
    'FEBIDSimulationRunner',

    # 数据结构
    'GaussianParams',
    'MaterialParams',
    'QuadGaussianParams',
    'PhysicalParams',
    'ScanInfo',
    'FLOAT_DTYPE',

    # 配置
    'SIMULATION_CONFIG',

    # 工具函数
    'validate_config',
    'print_config_summary',
    'main_with_custom_config',
    'calculate_surface_statistics',
    'compute_edge_mask_parallel',

    # 版本信息
    '__version__',
    '__author__',
    '__description__',
    'get_version',
    'get_info',
    'quick_start_example',

    # 基底几何功能
    'SubstrateGeometryGenerator',
    'RectangularDefect',
    'SubstrateGeometry',
    'validate_substrate_geometry_config',
    'create_example_substrate_config',
]


def get_version():
    """获取版本信息"""
    return __version__


def get_info():
    """获取包信息"""
    return {
        'name': 'FEBID Simulation (Simplified)',
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'features': [
            '双材料四高斯电子散射模型',
            '多循环/子循环扫描策略',
            '边缘重复扫描补偿',
            '反应-扩散方程RK4求解',
            '内存优化和进度监控',
            '精简的数据保存和可视化',
            '✨ 优化：Numba并行计算',
            '✨ 优化：表面感知通量计算',
            '✨ 优化：连续衰减模型',
            '🏗️ 自定义基底几何支持',
            '📐 矩形缺陷/凸起定义',
            '⚡ 精简版：无监控开销，更快的计算速度'
        ]
    }


def quick_start_example():
    """快速开始示例"""
    example_code = '''
# FEBID仿真快速开始示例 - 精简版（无监控）

# 方法1: 使用默认配置
from febid_simulation import FEBIDSimulationRunner

runner = FEBIDSimulationRunner()
results = runner.run()

# 方法2: 使用便捷函数
from febid_simulation import main_with_custom_config, SIMULATION_CONFIG

results = main_with_custom_config(
    sim_config=SIMULATION_CONFIG
)

# 方法3: 直接使用仿真类
from febid_simulation import MemoryOptimizedFEBID

# 自定义配置
custom_config = {
    'geometry': {'X_min': -50, 'X_max': 50, 'Y_min': -50, 'Y_max': 50, 'dx': 1, 'dy': 1},
    'scan': {
        'scan_x_start': -20, 'scan_x_end': 20, 
        'scan_y_start': -20, 'scan_y_end': 20,
        'pixel_size_x': 2, 'pixel_size_y': 2, 
        'dwell_time': 1e-6,
        'scan_strategy': 'serpentine',
        'loop': 1, 'subloop': None,
        'edge_layers': 2, 'edge_repeat_times': 3
    },
    'physical': {
        'Phi': 1.06e+2, 'tau': 1e-4, 'sigma': 0.42, 'n0': 2.8,
        'DeltaV': 0.094, 'k': 1, 'D_surf': 40000, 'dx': 1, 'dy': 1
    },
    'quad_gaussian': {
        'z_deposit': 5.0,
        'substrate': {
            'gaussian1': {'sigma': 4, 'amplitude': 0.5e+7},
            'gaussian2': {'sigma': 6, 'amplitude': 1.0e+7},
            'gaussian3': {'sigma': 8, 'amplitude': 0.8e+7},
            'gaussian4': {'sigma': 10, 'amplitude': 0.2e+7},
        },
        'deposit': {
            'gaussian1': {'sigma': 2, 'amplitude': 0.5e+7},
            'gaussian2': {'sigma': 4, 'amplitude': 1.0e+7},
            'gaussian3': {'sigma': 6, 'amplitude': 0.80e+7},
            'gaussian4': {'sigma': 8, 'amplitude': 0.20e+7},
        },
    },
    'surface_effects': {'depth_scale_factor': 5, 'slope_decay_min': 0.1, 'slope_decay_max': 10.0},
    'surface_propagation': {'enable': True},
    'numerical': {'dt': 1e-7},
    'output': {'create_plots': True, 'save_core_results': True, 'verbose': True}
}

# 创建仿真对象
febid = MemoryOptimizedFEBID(config=custom_config)

# 运行仿真
results = febid.run_simulation()

# 访问结果
print(f"最大高度: {results['h_surface'].max():.3e} nm")
print(f"仿真时间: {results['simulation_time']:.2f} 秒")
'''

    return example_code


# 启动信息
print(f"🔬 FEBID仿真包 v{__version__} 已加载 (精简版 - 无监控)")
print(f"📖 使用 help(febid_simulation.get_info) 查看详细信息")
print(f"🚀 使用 print(febid_simulation.quick_start_example()) 查看快速开始示例")
print(f"⚡ 精简版特性: 无监控开销，更快的计算速度")
