#!/usr/bin/env python3
"""
FEBID仿真配置模块 - 精简版（无监控）
添加了循环和子循环间前驱体浓度重置选项
添加了自动计算稳定时间步长功能

Author: 刘宇
Date: 2025/7
"""

import numpy as np
from base_classes import ConfigValidator

# 主仿真配置
SIMULATION_CONFIG = {
    'substrate_geometry': {
        'base_height': 0.0,
        'rectangular_defects': [
            # {
            #    'name': 'Central_Hill',
            #    'x1': -50, 'y1': -90, 'x2': 50, 'y2': -50,
            #    'height_offset': 0.6  # 6nm凸起
            # },
            # {
            #    'name': 'Central_Hill',
            #    'x1': -50, 'y1': 50, 'x2': 50, 'y2': 90,
            #    'height_offset': 0.6  # 6nm凸起
            # },
        ]
    },

    'geometry': {
        'X_min': -200,  # 单位nm
        'X_max': 200,
        'Y_min': -200,
        'Y_max': 200,
        'dx': 1,  # 网格参数，nm
        'dy': 1,
    },

    'numerical': {'dt': None },  # 将自动计算稳定时间步长

    'scan': {
        'scan_x_start': -150,  # 单位nm
        'scan_x_end': 150,
        'scan_y_start': -150,
        'scan_y_end': 150,
        'pixel_size_x': 1,
        'pixel_size_y': 8,
        'dwell_time': 1e-7,  # 电子束停留时间
        'scan_strategy': 'serpentine',
        # 扫描策略: 'raster' , 'serpentine' spiral_square_in2out, spiral_square_out2in, spiral_circle_in2out, spiral_circle_out2in
        'loop': 1,
        'subloop': None,  # 循环控制参数 (互斥: subloop/loop只能设置其中一个)
        'edge_layers': 0,  # 最外面多少圈进行重复扫描 (0表示不启用)
        'edge_repeat_times': 0,  # 边缘点重复扫描次数,一共扫描N+1次，重复N次

        # 前驱体浓度重置选项（两个选项可以独立使用）：
        'reset_precursor_between_loops': False,  # 仅在完整loop之间重置浓度
        'reset_precursor_between_subloops': True,  # 在每个subloop之间重置浓度, FRT enabled
        # 注：如果两个都是True，reset_precursor_between_subloops的效果会覆盖reset_precursor_between_loops
    },

    'physical': {
        'Phi': 1.55555555e+2,
        # 前驱体通量 [nm^-2 s^-1]对于Cr(CO)6，当1e-2 Pa，Phi（16℃）= 1.044e+2, Phi（-20℃）= 1.116e+2, 变化~8% ∝ 1/√T，Phi（20℃）= 1.037e+2
        'tau': 1.2779e-06 ,  # 平均停留时间 [s]
        'sigma': 0.42,  # 0.42 = Cr(CO)6 0.25 = TEOS 解离截面 [nm^2]
        'n0': 2.8,  # 2.8 = Cr(CO)6, 2.0 = TEOS, 5.7 = XeF2 最大表面前驱体密度 [molecules/nm^2]
        'DeltaV': 0.5,  # 有效解离前驱体分子体积 [nm^3]
        'k': 0.01,  # 吸附粘度系数
        'D_surf': 0.5e+6,  # 表面扩散系数 [nm^2/s]
        'scan_correction': 3,  # scan correction系数，用于矫正沉积表面曲率带来的电子通量增加，需实验拟合。参数范围：1~3
        "dx": 1,  # 忽略，设置为等同于网格参数即可
        "dy": 1,  # 忽略，设置为等同于网格参数即可
    },

    'quad_gaussian': {
        'z_deposit': 500.0,  # 沉积物阈值高度 [nm]
        'substrate': {
            'gaussian1': {'sigma': 40, 'amplitude': 1.6768e+07},
            'gaussian2': {'sigma': 40, 'amplitude': 0.0e+7},
            'gaussian3': {'sigma': 40, 'amplitude': 0.0e+1},
            'gaussian4': {'sigma': 40, 'amplitude': 0.0e+1},
        },
        'deposit': {
            'gaussian1': {'sigma': 40, 'amplitude': 5.0e+4},
            'gaussian2': {'sigma': 40, 'amplitude': 1.0e+4},
            'gaussian3': {'sigma': 40, 'amplitude': 0.0e+5},
            'gaussian4': {'sigma': 40, 'amplitude': 0.0e+5},
        },
    },

    'surface_effects': {
        'slope_decay_min': 10e+10,  # 指数衰减开始斜率
        'enable_slope_enhancement': False,  # 新增：是否启用指数斜率增强
    },

    'surface_propagation': {
        'enable': False,  # 启用表面传播
        'enable_numba': True  # 启用Numba加速
    },

    'output': {
        'create_plots': True,  # 是否生成图表
        'save_core_results': True,
        'verbose': True,  # 是否显示详细信息
    },
}


def calculate_stable_timestep(config: dict) -> float:
    """
    自动计算稳定的时间步长

    基于反应-扩散方程的稳定性条件：
    1. 扩散稳定性（CFL条件）：dt < dx²/(4*D)
    2. 反应稳定性：dt < 1/R_max，其中R_max是总反应率

    Returns:
        float: 稳定的时间步长 [s]
    """
    phys = config['physical']
    quad = config['quad_gaussian']

    # 1. 计算扩散稳定时间步长
    dx = phys['dx']
    dy = phys['dy']
    D_surf = phys['D_surf']

    if D_surf > 0:
        # 2D扩散的CFL条件
        dt_diffusion = min(dx ** 2, dy ** 2) / (4 * D_surf)
    else:
        dt_diffusion = float('inf')

    # 2. 计算反应稳定时间步长
    # 2.1 解吸率
    desorption_rate = 1.0 / phys['tau']

    # 2.2 吸附率
    adsorption_rate = phys['k'] * phys['Phi'] / phys['n0']

    # 2.3 解离率（需要估计最大通量）
    # 计算四高斯函数在中心的最大值
    f_max = 0
    for material in ['substrate', 'deposit']:
        f_material = 0
        for i in range(1, 5):
            gaussian = quad[material][f'gaussian{i}']
            # 在r=0处的值
            f_material += gaussian['amplitude']
        f_max = max(f_max, f_material)

    dissociation_rate = phys['sigma'] * f_max

    # 总反应率
    R_max = desorption_rate + adsorption_rate + dissociation_rate
    dt_reaction = 1.0 / R_max if R_max > 0 else float('inf')

    # 3. 取最小值并应用安全系数
    dt_min = min(dt_diffusion, dt_reaction, 0.1*config['scan']['dwell_time'])
    safety_factor = 0.8333333333  # 安全系数
    dt_stable = safety_factor * dt_min

    # 打印分析结果
    print(f"\n 时间步长稳定性分析:")
    print(f"   扩散限制: dt < {dt_diffusion:.3e} s (D={D_surf:.1e} nm²/s, dx={dx} nm)")
    print(f"   反应限制: dt < {dt_reaction:.3e} s")
    print(f"   Dwell Time 限制: dt <= {0.1*config['scan']['dwell_time']}s")
    #print(f"     - 解吸率: {desorption_rate:.1f} s⁻¹")
    #print(f"     - 吸附率: {adsorption_rate:.2f} s⁻¹")
    #print(f"     - 解离率: {dissociation_rate:.1f} s⁻¹ (f_max={f_max:.1e} nm⁻²s⁻¹)")
    #print(f"   限制因素: {'扩散' if dt_diffusion < dt_reaction else '反应'}")
    print(f"   ⚡ 推荐时间步长: {dt_stable:.3e} s (安全系数={safety_factor})")

    return dt_stable


def calculate_quad_gaussian_properties(config: dict):
    """计算四高斯函数的属性：最大值f0和FWHM"""
    # 获取基底的四个高斯参数（假设使用基底参数）
    substrate = config['quad_gaussian']['substrate']

    # 计算f0 - 四高斯函数在中心的最大值
    f0 = 0
    for i in range(1, 5):
        amplitude = substrate[f'gaussian{i}']['amplitude']
        f0 += amplitude

    # 计算加权FWHM
    # FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma
    # 对于多高斯，使用幅度加权平均
    weighted_sigma = 0
    total_amplitude = 0

    for i in range(1, 5):
        sigma = substrate[f'gaussian{i}']['sigma']
        amplitude = substrate[f'gaussian{i}']['amplitude']
        weighted_sigma += sigma * amplitude
        total_amplitude += amplitude

    avg_sigma = weighted_sigma / total_amplitude if total_amplitude > 0 else 1
    fwhm = 2.355 * avg_sigma  # 2 * sqrt(2 * ln(2)) ≈ 2.355

    return f0, fwhm


def calculate_equilibrium_concentration(config: dict) -> float:
    """
    计算无电子束作用下的平衡前驱体浓度

    在无电子束时，前驱体吸附和解吸达到平衡：
    吸附速率: k * Phi * (1 - n/n0)
    解吸速率: n/tau
    平衡时: k * Phi * (1 - n_eq/n0) = n_eq/tau
    解得: n_eq = k * Phi * tau * n0 / (n0 + k * Phi * tau)

    Returns:
        float: 平衡浓度 [molecules/nm^2]
    """
    phys = config['physical']
    n_eq = phys['k'] * phys['Phi'] * phys['tau'] * phys['n0'] / (phys['n0'] + phys['k'] * phys['Phi'] * phys['tau'])
    return n_eq


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
                if gaussian['sigma'] < 0 or gaussian['amplitude'] < 0:
                    print(f"❌ {material} gaussian{i}参数必须为正值")
                    return False

        # 验证基底几何配置
        from substrate_geometry import validate_substrate_geometry_config
        if not validate_substrate_geometry_config(config):
            return False

        # 验证前驱体重置选项
        scan_config = config['scan']

        # 验证reset_precursor_between_loops
        if 'reset_precursor_between_loops' in scan_config:
            if not isinstance(scan_config['reset_precursor_between_loops'], bool):
                print("❌ reset_precursor_between_loops必须是布尔值")
                return False

        # 验证reset_precursor_between_subloops
        if 'reset_precursor_between_subloops' in scan_config:
            if not isinstance(scan_config['reset_precursor_between_subloops'], bool):
                print("❌ reset_precursor_between_subloops必须是布尔值")
                return False

        # 逻辑提示
        if scan_config.get('reset_precursor_between_loops', False) and \
                scan_config.get('reset_precursor_between_subloops', False):
            print(" 提示: 同时启用了循环间和子循环间重置，将在每个子循环间重置")

        # 自动计算稳定时间步长（如果未设置）
        if config['numerical'].get('dt') is None:
            config['numerical']['dt'] = calculate_stable_timestep(config)
            print(f"✅ 时间步长已自动设置为: {config['numerical']['dt']:.3e} s")
        else:
            # 验证用户设置的时间步长
            recommended_dt = calculate_stable_timestep(config)
            user_dt = config['numerical']['dt']
            if user_dt > recommended_dt:
                print(f"⚠️  警告: 用户设置的时间步长 {user_dt:.3e} s 大于推荐值 {recommended_dt:.3e} s")
                print(f"   可能导致数值不稳定，建议使用推荐值")

        return True

    except Exception as e:
        print(f"❌ 配置验证失败: {e}")
        return False


def print_config_summary(config: dict):
    """打印配置摘要 - 精简版"""
    # print("\n" + "=" * 60)
    # print("️  FEBID仿真配置摘要")
    print("=" * 60)

    # 几何配置
    geom = config['geometry']
    # print(f" 几何范围: X=[{geom['X_min']}, {geom['X_max']}] nm, Y=[{geom['Y_min']}, {geom['Y_max']}] nm")
    # print(f" 网格分辨率: dx={geom['dx']} nm, dy={geom['dy']} nm")

    # 扫描配置
    scan = config['scan']
    # print(f" 扫描区域: X=[{scan['scan_x_start']}, {scan['scan_x_end']}] nm, "
    # f"Y=[{scan['scan_y_start']}, {scan['scan_y_end']}] nm")
    # print(f" 扫描步长: X={scan['pixel_size_x']} nm, Y={scan['pixel_size_y']} nm")
    print(f"⏱️  停留时间: {scan['dwell_time'] * 1e6:.1f} μs, 策略: {scan['scan_strategy']}")

    # 显示时间步长信息（含稳定性分析）
    dt = config['numerical']['dt']
    if dt is not None:
        print(f" 时间步长: {dt * 1e9:.3f} ns")
        # 显示时间步长与停留时间的比例
        steps_per_dwell = int(scan['dwell_time'] / dt)
        print(f"   每像素计算步数: {steps_per_dwell} (停留时间/时间步长)")
    else:
        print(f" 时间步长: 将自动计算")

    # 计算平衡浓度
    n_eq = calculate_equilibrium_concentration(config)

    # 循环参数和前驱体重置
    if scan.get('loop') is not None:
        total_subloops = scan['loop'] * scan['pixel_size_x'] * scan['pixel_size_y']
        print(
            f" 循环模式: {scan['loop']} 完整循环 (每循环包含 {scan['pixel_size_x']}×{scan['pixel_size_y']}={scan['pixel_size_x'] * scan['pixel_size_y']} 子循环，共{total_subloops}个子循环)")

        reset_between_loops = scan.get('reset_precursor_between_loops', False)
        reset_between_subloops = scan.get('reset_precursor_between_subloops', False)

        if reset_between_loops or reset_between_subloops:
            print(f" 前驱体重置策略 (平衡浓度: {n_eq:.4e} molecules/nm²):")
            if reset_between_loops and not reset_between_subloops:
                print(f"   - 仅在完整loop间重置: {scan['loop'] - 1} 次重置（每个新loop开始时）")
            elif reset_between_subloops and not reset_between_loops:
                print(f"   - 在每个subloop间重置: {total_subloops - 1} 次重置")
            elif reset_between_loops and reset_between_subloops:
                print(f"   - 两个选项都启用: 将在每个subloop间重置（{total_subloops - 1} 次）")
                print(f"      注: reset_precursor_between_subloops覆盖了reset_precursor_between_loops的效果")

    elif scan.get('subloop') is not None:
        print(f" 子循环模式: {scan['subloop']} 子循环")
        if scan.get('reset_precursor_between_subloops', False):
            print(f" 子循环间前驱体重置: 启用 (平衡浓度: {n_eq:.4f} molecules/nm²)")
        if scan.get('reset_precursor_between_loops', False):
            print(f"⚠️  注意: reset_precursor_between_loops在subloop模式下无效")

    # 边缘增强
    if scan['edge_layers'] > 0:
        print(f" 边缘增强: {scan['edge_layers']} 层, 重复 {scan['edge_repeat_times']}x")

    # 物理参数
    phys = config['physical']
    # print(f"離 物理参数: Φ={phys['Phi']:.2f} nm⁻²s⁻¹, τ={phys['tau'] * 1e6:.1f} μs, σ={phys['sigma']:.2f} nm²")
    # print(f" 前驱体: n₀={phys['n0']:.1f} mol/nm², k={phys['k']:.3f}, D={phys['D_surf']:.0f} nm²/s")

    # 计算并输出新的无量纲参数
    f0, fwhm = calculate_quad_gaussian_properties(config)
    t_char = 1 / (phys['k'] * phys['Phi'] / phys['n0'] + 1 / phys['tau'] + phys['sigma'] * f0)
    tau_tilde = 1 + phys['sigma'] * f0 * phys['tau'] / (1 + phys['k'] * phys['Phi'] * phys['tau'] / phys['n0'])
    effective_tau = 1 / (1 / phys['tau'] + phys['k'] * phys['Phi'] / phys['n0'])
    gamma = 2 * np.sqrt(phys['D_surf'] * effective_tau) / fwhm
    if tau_tilde > (1 + 2 * gamma ** 1.5):
        crater_radius = (fwhm / 2.355) * np.sqrt(np.log(tau_tilde / (1 + 2 * gamma ** 1.5)))
    else:
        crater_radius = 0.0  # 当条件不满足时，火山口不会形成
    denominator = np.log(1.15 + tau_tilde / (0.42 + 5.28 * gamma ** 1.32))
    if denominator > 0:
        h_ratio = 0.693 / np.sqrt(denominator)
    else:
        h_ratio = float('inf')  # 当分母为负或零时

    #print(f"生长偏离高斯形状特征时间: t_char={4 * t_char:.3e} s")
    print(f"无量纲参数: τ̃={tau_tilde:.3f}, γ={gamma:.3f} (f₀={f0:.2e} nm⁻²s⁻¹, FWHM={fwhm:.2f} nm)")
    #print(f"火山口半径: r_crater={crater_radius:.2f} nm")
    print(f"平衡前驱体浓度: n_eq={n_eq:.4f} molecules/nm²")
    steady_deposition_rate = phys['DeltaV'] * phys['sigma'] * f0 * (n_eq / tau_tilde)
    print(f"稳态沉积速率: {steady_deposition_rate:.6e} nm/s")
    # print(f"高度比: h_center/h_rim={h_ratio:.3f}")
    print("=" * 60)
    #print(f"t_char = 1/(k·Φ/n₀ + 1/τ + σ·f₀)")
    print(f"τ̃ = 1 + σ·f₀·τ/(1 + k·Φ·τ/n₀)")
    print(f"γ = 2√(D·τ_eff)/FWHM, 其中 τ_eff = 1/(1/τ + k·Φ/n₀)")
    #print(f"r_crater = (FWHM/2.355)·√[ln(τ̃/(1+2γ^1.5))]")
    print(f"n_eq = k·Φ·τ·n₀/(1 + k·Φ·τ) (无电子束时的平衡浓度)")
    # print(f"拟合h_center/h_rim = 0.693/√[ln(1.15 + τ̃/(0.42+5.28·^γ1.32))]")
    # 材料参数
    # print(f"⚡ 材料阈值: {config['quad_gaussian']['z_deposit']:.1f} nm")

    # 数值参数

    # 输出配置
    output = config.get('output', {})
    # print(f" 输出设置: 图表={'启用' if output.get('create_plots', True) else '禁用'}, "
    #      f"保存={'启用' if output.get('save_core_results', True) else '禁用'}")

    print("=" * 60)
