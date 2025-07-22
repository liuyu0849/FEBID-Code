#!/usr/bin/env python3
"""
FEBID仿真配置模块 - 优化版
使用ConfigValidator减少重复验证代码

Author: 刘宇
Date: 2025/7
"""

from base_classes import ConfigValidator
# 主仿真配置
SIMULATION_CONFIG = {
    'substrate_geometry': {
    'base_height': 0.0,
    'rectangular_defects': [
        {
            'name': 'Central_Hill',
            'x1': -40, 'y1': 40, 'x2': 40, 'y2': 80,
            'height_offset': 5.5  # 6nm凸起
        },

        {
            'name': 'Central_Hill',
            'x1': -40, 'y1': -80, 'x2': 40, 'y2': -40,
            'height_offset': 5.5  # 6nm凸起
        },

        #{
        #    'name': 'Edge_Trench',
        #    'x1': -40, 'y1': -40, 'x2': 40, 'y2': 40,
        #    'height_offset': -0.001  # 4nm凹陷
        #}
    ]
},

    'monitoring': {
        'enable_realtime_monitor': True,
        'use_realtime_mode': True,
        'save_interval': 10,
        'max_memory_frames': 200,
        'height_range': 'auto',  # 或者 [min, max] 覆盖自动计算
        'precursor_range': 'auto',  # 或者 [min, max] 覆盖自动计算
    },

    'geometry': {
        'X_min': -100, #单位nm
        'X_max': 100,
        'Y_min': -100,
        'Y_max': 100,
        'dx': 1, #网格参数，nm
        'dy': 1,
    },

    'numerical': {'dt': 1e-7}, #时间步长~0.1倍dwell time效率最高

    'scan': {
        'scan_x_start': -10, #单位nm
        'scan_x_end': 10,
        'scan_y_start': -10,
        'scan_y_end': 10,
        'pixel_size_x': 1,
        'pixel_size_y': 1,
        'dwell_time': 1e-6, #电子束停留时间
        'scan_strategy': 'serpentine',
        # 扫描策略: 'raster' , 'serpentine' spiral_square_in2out, spiral_square_out2in, spiral_circle_in2out, spiral_circle_out2in, in2out反过来用
        'loop': None,
        'subloop': 1, # 🔄 循环控制参数 (互斥: subloop/loop只能设置其中一个)
        'edge_layers': 0, # 最外面多少圈进行重复扫描 (0表示不启用)
        'edge_repeat_times': 0,# 边缘点重复扫描次数,一共扫描N+1次，重复N次
    },

    'physical': {
        'Phi': 1.06e+4,  # 前驱体通量 [nm^-2 s^-1]
        'tau': 1e-4,  # 平均停留时间 [s]
        'sigma': 0.42,  # 积分解离截面 [nm^2]
        'n0': 2.8,  # 最大表面前驱体密度 [molecules/nm^2]
        'DeltaV': 0.094,  # 有效解离前驱体分子体积 [nm^3]
        'k': 1,  # 吸附粘度系数
        'D_surf': 40000,  # 表面扩散系数 [nm^2/s]
        "dx":1,#忽略，设置为等同于网格参数即可
        "dy":1,#忽略，设置为等同于网格参数即可
    },

    'quad_gaussian': {
        'z_deposit': 5.0, # 沉积物阈值高度 [nm]
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

    'surface_effects': {
        'depth_scale_factor': 5,         # 深度效应
        'slope_decay_min': 0.0000001,          # 指数衰减开始斜率
        'slope_decay_max': 10.0,         # 指数衰减结束斜率
    },

    'surface_propagation': {
        'enable': True,                  # 启用表面传播
        'enable_numba': True            # 启用Numba加速
    },

    'output': {
        'create_plots': True,# 是否生成图表
        'save_core_results': True,
        'verbose': True,# 是否显示详细信息
    },


}


# config.py - 添加动态计算函数
def calculate_dynamic_visualization_ranges(sim_config: dict) -> dict:
    """根据仿真配置动态计算可视化范围"""

    # 1. 计算高度范围
    substrate_config = sim_config.get('substrate_geometry', {})
    defects = substrate_config.get('rectangular_defects', [])

    if defects:
        height_offsets = [d.get('height_offset', 0.0) for d in defects]
        min_offset = min(height_offsets + [0])  # 包含0确保基准面
        max_offset = max(height_offsets + [0])

        # 动态计算Z轴范围
        if min_offset < 0:  # 有凹陷
            z_min = min_offset * 1.2  # 留20%余量
            z_max = abs(min_offset) * 1.5  # 根据凹陷深度设置上限
        else:  # 只有凸起
            z_min = -max_offset * 0.1  # 稍微低于基准面
            z_max = max_offset * 2.0  # 凸起高度的2倍

        # 考虑预期沉积高度（根据物理参数估算）
        estimated_deposition = estimate_max_deposition_height(sim_config)
        z_max = max(z_max, max_offset + estimated_deposition * 1.2)
    else:
        # 平面基底：根据预期沉积高度
        estimated_deposition = estimate_max_deposition_height(sim_config)
        z_min = 0.0
        z_max = estimated_deposition * 1.5

    # 2. 计算前驱体浓度范围
    physical = sim_config['physical']
    # 平衡浓度计算：n_eq = k*Phi*tau*n0 / (1 + k*Phi*tau)
    k_phi_tau = physical['k'] * physical['Phi'] * physical['tau']
    n_eq = k_phi_tau * physical['n0'] / (1 + k_phi_tau)

    precursor_min = 0.0
    precursor_max = n_eq * 1.2  # 平衡浓度的1.2倍

    return {
        'height_range': [z_min, z_max],
        'precursor_range': [precursor_min, precursor_max],
        'equilibrium_concentration': n_eq
    }


def estimate_max_deposition_height(sim_config: dict) -> float:
    """估算最大沉积高度"""
    scan = sim_config['scan']
    physical = sim_config['physical']

    # 计算总扫描时间
    nx = int((scan['scan_x_end'] - scan['scan_x_start']) / scan['pixel_size_x']) + 1
    ny = int((scan['scan_y_end'] - scan['scan_y_start']) / scan['pixel_size_y']) + 1
    total_pixels = nx * ny

    # 考虑循环次数
    if scan.get('loop'):
        total_pixels *= scan['loop']

    # 最大沉积速率估算（假设电子束中心）
    max_flux = physical['Phi'] * physical['sigma']  # 简化估算
    max_deposition_rate = physical['DeltaV'] * max_flux * physical['n0']

    # 估算最大高度
    effective_time = scan['dwell_time'] * (scan.get('edge_repeat_times', 0) + 1)
    estimated_height = max_deposition_rate * effective_time * 10  # 10是经验因子

    return estimated_height



def validate_config(config: dict) -> bool:
    """验证配置参数的合理性 - 使用ConfigValidator"""
    validator = ConfigValidator()

    try:
        # 检查必要的配置节
        required_sections = ['geometry', 'scan', 'physical', 'quad_gaussian', 'numerical']
        for section in required_sections:
            if section not in config:
                print(f"❌ 缺少配置节: {section}")
                return False

        # 使用ConfigValidator进行验证
        if not validator.validate_geometry_config(config['geometry']):
            return False

        if not validator.validate_scan_config(config['scan'], config['geometry']):
            return False

        if not validator.validate_physical_config(config['physical']):
            return False

        # 验证四高斯参数
        quad_gaussian = config['quad_gaussian']
        if quad_gaussian['z_deposit'] <= 0:
            print("❌ 材料转换阈值z_deposit必须为正值")
            return False

        for material in ['substrate', 'deposit']:
            for i in range(1, 5):
                gaussian = quad_gaussian[material][f'gaussian{i}']
                if gaussian['sigma'] <= 0 or gaussian['amplitude'] <= 0:
                    print(f"❌ {material} gaussian{i}参数必须为正值")
                    return False
        # 验证基底几何配置
        from substrate_geometry import validate_substrate_geometry_config
        if not validate_substrate_geometry_config(config):
            return False

        return True

    except Exception as e:
        print(f"❌ 配置验证失败: {e}")
        return False


def validate_monitoring_ranges(monitor_config: dict, sim_config: dict) -> bool:
    """验证监控范围配置"""
    height_range = monitor_config.get('height_range')
    precursor_range = monitor_config.get('precursor_range')

    # 允许 'auto' 或具体数值
    for range_name, range_value in [('height_range', height_range),
                                    ('precursor_range', precursor_range)]:
        if range_value != 'auto' and not isinstance(range_value, list):
            print(f"❌ {range_name} 必须是 'auto' 或 [min, max] 列表")
            return False

        if isinstance(range_value, list):
            if len(range_value) != 2 or range_value[1] <= range_value[0]:
                print(f"❌ {range_name} 格式错误")
                return False

    return True

def validate_visualization_config(viz_config: dict, sim_config: dict) -> bool:
    """验证可视化配置的合理性"""
    try:
        # 检查高度范围
        if viz_config['height_range'][1] <= viz_config['height_range'][0]:
            print("❌ 高度显示范围设置错误")
            return False

        # 检查前驱体范围
        if viz_config['precursor_range'][1] <= viz_config['precursor_range'][0]:
            print("❌ 前驱体显示范围设置错误")
            return False

        # 检查前驱体范围与物理参数的一致性
        max_n0 = sim_config['physical']['n0']
        if viz_config['precursor_range'][1] < max_n0:
            print(f"⚠️  警告: 前驱体显示范围 ({viz_config['precursor_range'][1]}) 小于最大密度 ({max_n0})")

        # 检查保存间隔合理性
        if viz_config['save_interval'] <= 0:
            print("❌ 保存间隔必须为正值")
            return False

        print("✓ 可视化配置验证通过")
        return True

    except Exception as e:
        print(f"❌ 可视化配置验证失败: {e}")
        return False


def print_config_summary(config: dict, viz_config: dict = None):
    """打印配置摘要 - 优化版"""
    print("\n" + "=" * 60)
    print("🎛️  FEBID仿真配置摘要")
    print("=" * 60)

    # 几何配置
    geom = config['geometry']
    print(f"📐 几何范围: X=[{geom['X_min']}, {geom['X_max']}] nm, Y=[{geom['Y_min']}, {geom['Y_max']}] nm")
    print(f"📏 网格分辨率: dx={geom['dx']} nm, dy={geom['dy']} nm")

    # 扫描配置
    scan = config['scan']
    print(f"🎯 扫描区域: X=[{scan['scan_x_start']}, {scan['scan_x_end']}] nm, "
          f"Y=[{scan['scan_y_start']}, {scan['scan_y_end']}] nm")
    print(f"📏 扫描步长: X={scan['pixel_size_x']} nm, Y={scan['pixel_size_y']} nm")
    print(f"⏱️  停留时间: {scan['dwell_time'] * 1e6:.1f} μs, 策略: {scan['scan_strategy']}")

    # 循环参数
    if scan.get('loop') is not None:
        print(f"🔄 循环模式: {scan['loop']} 完整循环")
    elif scan.get('subloop') is not None:
        print(f"🔄 子循环模式: {scan['subloop']} 子循环")

    # 边缘增强
    if scan['edge_layers'] > 0:
        print(f"🎯 边缘增强: {scan['edge_layers']} 层, 重复 {scan['edge_repeat_times']}x")

    # 物理参数
    phys = config['physical']
    print(f"🧪 物理参数: Φ={phys['Phi']:.2f} nm⁻²s⁻¹, τ={phys['tau'] * 1e6:.1f} μs, σ={phys['sigma']:.2f} nm²")
    print(f"🔬 前驱体: n₀={phys['n0']:.1f} mol/nm², k={phys['k']:.3f}, D={phys['D_surf']:.0f} nm²/s")

    # 材料参数
    print(f"⚡ 材料阈值: {config['quad_gaussian']['z_deposit']:.1f} nm")

    # 数值参数
    print(f"🔢 时间步长: {config['numerical']['dt'] * 1e9:.1f} ns")

    # 监控配置
    monitor = config.get('monitoring', {})
    if monitor.get('enable_realtime_monitor', False):
        mode = "实时" if monitor.get('use_realtime_mode', True) else "传统"
        print(f"🖥️  监控模式: {mode}监控已启用")

        # 打印可视化范围
        height_range = monitor.get('height_range', 'auto')
        precursor_range = monitor.get('precursor_range', 'auto')
        print(f"📊 可视化范围: 高度={height_range}, 前驱体={precursor_range}")
    else:
        print(f"🖥️  实时监控: 禁用")

    # 输出配置
    output = config.get('output', {})
    print(f"📊 输出设置: 图表={'启用' if output.get('create_plots', True) else '禁用'}, "
          f"保存={'启用' if output.get('save_core_results', True) else '禁用'}")

    if viz_config:
        print(f"\n🎨 可视化配置:")
        print(f"📊 高度范围: {viz_config['height_range']} nm")
        print(f"🧪 前驱体范围: {viz_config['precursor_range']} molecules/nm²")
        print(f"💾 保存间隔: 每 {viz_config['save_interval']} 个像素")
        print(f"📸 最大帧数: {viz_config['max_memory_frames']}")

    print("=" * 60)
