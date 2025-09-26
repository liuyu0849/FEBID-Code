#!/usr/bin/env python3
"""
FEBID仿真主控制器 - 精简版（无监控）
添加了subloop间前驱体浓度重置功能

Author: 刘宇
Date: 2025/7
"""

import numpy as np
import time
import psutil
from typing import Dict

from data_structures import (
    GaussianParams, MaterialParams, QuadGaussianParams,
    PhysicalParams, ScanInfo, FLOAT_DTYPE
)
from base_classes import ConfigValidator
from scan_strategies import ScanPathGenerator
from visualization_analysis import VisualizationAnalyzer
from simulation_core_algorithms import (
    calculate_quad_gaussian_flux_numba,
    rk4_step_parallel,
    apply_surface_effects_numba
)


class MemoryOptimizedFEBID:
    """内存优化的FEBID仿真类 - 精简版（无监控）"""

    def __init__(self, config: Dict):
        """初始化仿真参数"""
        self.config = config
        self.physical_params = self._create_physical_params()
        self.quad_gaussian_params = self._create_quad_gaussian_params()
        self.scan_history = []
        self.start_time = None

        # 初始化辅助模块
        self.scan_generator = ScanPathGenerator(config)
        self.visualizer = VisualizationAnalyzer(config)
        self.validator = ConfigValidator()

        # 表面传播配置
        self.surface_config = config.get('surface_propagation', {})
        self.enable_surface_propagation = self.surface_config.get('enable', False)

        # 前驱体浓度重置配置
        self.reset_precursor_between_loops = config['scan'].get('reset_precursor_between_loops', False)
        self.reset_precursor_between_subloops = config['scan'].get('reset_precursor_between_subloops', False)

        # 预处理参数（默认启用Numba）
        self._prepare_numba_params()
        self._prepare_rk4_constants()

        # 计算无电子束平衡浓度
        self._calculate_equilibrium_concentration()

        # print(f"✓ 仿真器初始化完成 {'(表面感知)' if self.enable_surface_propagation else '(2D模式)'}")
        if self.reset_precursor_between_loops:
            print(f"✓ 循环间前驱体重置已启用 (平衡浓度: {self.n_equilibrium:.4f} molecules/nm²)")
        if self.reset_precursor_between_subloops:
            print(f"✓ 子循环间前驱体重置已启用 (平衡浓度: {self.n_equilibrium:.4f} molecules/nm²)")

    def _calculate_equilibrium_concentration(self):
        """计算无电子束作用下的平衡前驱体浓度"""
        # 在无电子束时，前驱体吸附和解吸达到平衡
        # k * Phi * (1 - n/n0) = n/tau
        # 解得: n_eq = k * Phi * tau * n0 / (1 + k * Phi * tau)
        p = self.physical_params
        self.n_equilibrium = p.k * p.Phi * p.tau * p.n0 / (p.n0 + p.k * p.Phi * p.tau)

    def _prepare_rk4_constants(self):
        """预计算RK4常量"""
        p = self.physical_params
        self.rk4_constants = {
            'k_phi': p.k * p.Phi,
            'tau_inv': 1.0 / p.tau,
            'sigma': p.sigma,
            'n0_inv': 1.0 / p.n0,
            'D_surf_factor': p.D_surf
        }

    def run_simulation(self) -> Dict:
        """运行主仿真"""
        print("=== FEBID仿真开始 ===")

        # 基础验证
        self._validate_core_parameters()

        # 生成扫描位置
        scan_positions, scan_info = self.scan_generator.generate_scan_positions(self.config['scan'])
        total_pixels = scan_info.total_pixels

        print(f"扫描点数: {total_pixels:,}, 停留时间: {self.config['scan']['dwell_time'] * 1e6:.1f} μs")

        # 初始化网格和表面
        x_grid, y_grid, h_surface, n_surface = self._initialize_simulation_grid()

        # 时间参数
        dt = self.config['numerical']['dt']
        dwell_time = self.config['scan']['dwell_time']
        edge_repeat_times = self.config['scan']['edge_repeat_times']

        # 显示配置
        self._display_config(scan_positions)

        print("\n开始仿真...")
        self.start_time = time.time()

        # 主仿真循环
        h_surface, n_surface = self._run_main_simulation_loop(
            scan_positions, scan_info, x_grid, y_grid, h_surface, n_surface,
            dt, dwell_time, edge_repeat_times, total_pixels
        )

        total_time = time.time() - self.start_time

        # 完成处理
        self._finalize_simulation(h_surface, n_surface, total_time, scan_info,
                                  x_grid, y_grid, total_pixels)

        return self._prepare_results(x_grid, y_grid, h_surface, n_surface,
                                     scan_positions, scan_info, total_time)

    def _validate_core_parameters(self):
        """核心参数验证"""
        if not self.validator.validate_geometry_config(self.config['geometry']):
            raise ValueError("几何配置错误")
        if not self.validator.validate_scan_config(self.config['scan'], self.config['geometry']):
            raise ValueError("扫描配置错误")
        if not self.validator.validate_physical_config(self.config['physical']):
            raise ValueError("物理参数错误")

    def _initialize_simulation_grid(self):
        """初始化仿真网格"""
        geom = self.config['geometry']
        x_grid = np.arange(geom['X_min'], geom['X_max'] + geom['dx'], geom['dx'], dtype=FLOAT_DTYPE)
        y_grid = np.arange(geom['Y_min'], geom['Y_max'] + geom['dy'], geom['dy'], dtype=FLOAT_DTYPE)

        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)

        # 生成基底表面
        if 'substrate_geometry' in self.config and self.config['substrate_geometry'].get('rectangular_defects'):
            from substrate_geometry import SubstrateGeometryGenerator
            substrate_generator = SubstrateGeometryGenerator(self.config)
            h_surface = substrate_generator.generate_substrate_surface(x_grid, y_grid)
            print(f"✓ 自定义基底已加载")
        else:
            h_surface = np.zeros_like(X_mesh, dtype=FLOAT_DTYPE)
            print("✓ 平面基底已加载")

        # 初始前驱体覆盖度（使用平衡浓度）
        n_surface = np.full_like(X_mesh, self.n_equilibrium, dtype=FLOAT_DTYPE)

        self.X_mesh = X_mesh
        self.Y_mesh = Y_mesh

        return x_grid, y_grid, h_surface, n_surface

    def _display_config(self, scan_positions):
        """显示配置信息"""
        edge_mask = scan_positions[:, 3] == 1
        basic_mask = scan_positions[:, 3] == 0

        print(f"🎯 扫描统计: 边缘点={np.sum(edge_mask)}, 基础点={np.sum(basic_mask)}")
        print(f"🌊 计算模式: {'表面感知' if self.enable_surface_propagation else '传统2D'}")

    def _run_main_simulation_loop(self, scan_positions, scan_info, x_grid, y_grid,
                                  h_surface, n_surface, dt, dwell_time,
                                  edge_repeat_times, total_pixels):
        """主仿真循环 - 支持循环和子循环间重置前驱体浓度"""
        current_subloop = 0
        pixels_in_current_subloop = 0
        rk4_const = self.rk4_constants

        # 获取扫描配置
        scan_config = self.config['scan']
        pixel_size_x = scan_config['pixel_size_x']
        pixel_size_y = scan_config['pixel_size_y']

        # 计算循环参数
        total_possible_steps = pixel_size_x * pixel_size_y  # 一个完整loop包含的subloop数
        pixels_per_complete_loop = scan_info.base_pixels_per_subloop * total_possible_steps

        # 跟踪当前位置
        current_loop_number = 0
        current_subloop_in_loop = 0  # 当前loop内的subloop索引
        last_subloop_idx = 0
        last_loop_number = 0

        # 检查是否使用loop模式
        using_loop_mode = scan_config.get('loop') is not None

        for pixel_idx in range(total_pixels):
            # 获取当前subloop索引
            current_pixel_subloop = scan_positions[pixel_idx, 2]

            # 检测subloop变化
            if current_pixel_subloop != last_subloop_idx:
                # 新的subloop开始
                if last_subloop_idx > 0:  # 不是第一个subloop
                    # 计算当前loop编号
                    if using_loop_mode:
                        loop_num = (current_pixel_subloop - 1) // total_possible_steps + 1
                        subloop_in_loop = (current_pixel_subloop - 1) % total_possible_steps + 1

                        # 检测是否是新的loop开始
                        if loop_num != last_loop_number and last_loop_number > 0:
                            # 新的完整loop开始
                            if self.reset_precursor_between_loops:
                                n_surface.fill(self.n_equilibrium)
                                print(f"🔄 Loop {loop_num} 开始，前驱体浓度已重置为平衡值")
                            else:
                                print(f"🔄 Loop {loop_num} 开始")
                            current_loop_number = loop_num
                            last_loop_number = loop_num

                        # 检测是否需要在subloop间重置（独立判断）
                        if self.reset_precursor_between_subloops:
                            n_surface.fill(self.n_equilibrium)
                            print(
                                f"  📍 Loop {loop_num}, Subloop {subloop_in_loop}/{total_possible_steps}，前驱体浓度已重置")
                    else:
                        # subloop模式
                        if self.reset_precursor_between_subloops:
                            n_surface.fill(self.n_equilibrium)
                            print(f"📍 Subloop {current_pixel_subloop} 开始，前驱体浓度已重置")
                        else:
                            print(f"📍 Subloop {current_pixel_subloop} 开始")

                last_subloop_idx = current_pixel_subloop
                current_subloop = current_pixel_subloop
                pixels_in_current_subloop = scan_info.pixels_per_subloop

            # 第一个subloop的特殊处理
            if pixel_idx == 0:
                current_subloop = 1
                pixels_in_current_subloop = scan_info.pixels_per_subloop
                last_subloop_idx = 1
                if using_loop_mode:
                    current_loop_number = 1
                    last_loop_number = 1
                    print(f"🔄 Loop 1 开始")
                else:
                    print(f"📍 Subloop 1 开始")

            # 获取扫描位置
            beam_pos_x = scan_positions[pixel_idx, 0]
            beam_pos_y = scan_positions[pixel_idx, 1]
            is_edge_repeat = bool(scan_positions[pixel_idx, 3])

            # 计算停留时间
            effective_dwell_time = dwell_time * (edge_repeat_times + 1) if is_edge_repeat else dwell_time
            steps_per_dwell = int(effective_dwell_time / dt)

            # 像素停留循环
            for step in range(steps_per_dwell):
                # 计算电子通量
                f_surface = self._calculate_flux(beam_pos_x, beam_pos_y, h_surface)

                # RK4步进
                n_surface = rk4_step_parallel(
                    n_surface, f_surface, dt,
                    rk4_const['k_phi'], rk4_const['tau_inv'],
                    rk4_const['sigma'], rk4_const['n0_inv'],
                    rk4_const['D_surf_factor'],
                    self.physical_params.dx, self.physical_params.dy,
                    h_surface  # 新增参数
                )

                # 更新高度
                scan_config = self.config['scan']
                is_single_point = (scan_config['scan_x_start'] == scan_config['scan_x_end'] == 0 and
                                   scan_config['scan_y_start'] == scan_config['scan_y_end'] == 0)

                if is_single_point:
                    scan_correction = 1.0  # 单点扫描不应用修正
                else:
                    scan_correction = self.config['physical'].get('scan_correction', 1.0)

                deposition_rate = scan_correction * self.physical_params.DeltaV * self.physical_params.sigma * f_surface * n_surface
                h_surface += deposition_rate * dt

            # 记录历史（简化版：减少记录频率）
            if pixel_idx % 100 == 0:  # 每100个点记录一次
                self.scan_history.append([
                    pixel_idx, pixel_idx + 1, beam_pos_x, beam_pos_y,
                    np.max(h_surface), current_subloop, is_edge_repeat
                ])

            pixels_in_current_subloop -= 1

            # 进度显示（简化版）
            if pixel_idx % max(1, total_pixels // 10) == 0:
                self._display_progress(pixel_idx + 1, total_pixels, h_surface,
                                       (beam_pos_x, beam_pos_y))

        return h_surface, n_surface

    def _calculate_flux(self, beam_pos_x, beam_pos_y, h_surface):
        """计算电子通量"""
        X_flat = self.X_mesh.flatten().astype(np.float32)
        Y_flat = self.Y_mesh.flatten().astype(np.float32)
        h_flat = h_surface.flatten().astype(np.float32)

        # 使用Numba并行计算
        f_surface_flat = calculate_quad_gaussian_flux_numba(
            X_flat, Y_flat, h_flat,
            beam_pos_x, beam_pos_y,
            self.sub_params_array, self.dep_params_array,
            self.quad_gaussian_params.z_deposit,
            self.enable_surface_propagation,
            self.physical_params.dx, self.physical_params.dy,
            self.X_mesh.shape
        )

        f_surface = f_surface_flat.reshape(self.X_mesh.shape)

        # 应用表面效应
        if self.enable_surface_propagation:
            f_surface = self._apply_surface_effects(f_surface, h_surface)

        return f_surface

    def _apply_surface_effects(self, f_surface, h_surface):
        """应用连续表面效应"""
        surface_params = self.config.get('surface_effects', {
            'slope_decay_min': 0.1,
            'slope_decay_max': 10.0,
            'enable_slope_enhancement': True,  # 默认启用
        })

        weight_substrate, weight_deposit = self._calculate_material_weights(h_surface)

        # 获取sigma参数
        sub = self.quad_gaussian_params.substrate
        dep = self.quad_gaussian_params.deposit

        f_surface_final = apply_surface_effects_numba(
            f_surface, h_surface, weight_substrate, weight_deposit,
            sub.gaussian1.sigma, dep.gaussian1.sigma,
            self.physical_params.dx, self.physical_params.dy,
            surface_params.get('slope_decay_min', 0.176),
            surface_params.get('slope_decay_max', 10.0),
            surface_params.get('enable_slope_enhancement', True)  # 新增参数传递
        )

        return np.maximum(f_surface_final, 0)

    def _calculate_material_weights(self, h_surface):
        """计算材料权重"""
        z_deposit = self.quad_gaussian_params.z_deposit
        weight_substrate = np.zeros_like(h_surface, dtype=FLOAT_DTYPE)
        weight_deposit = np.zeros_like(h_surface, dtype=FLOAT_DTYPE)

        # 凹陷和基准面：纯基底
        baseline_mask = (h_surface <= 0)
        weight_substrate[baseline_mask] = 1.0

        # 过渡区域：基底-沉积物混合
        transition_mask = (h_surface > 0) & (h_surface < z_deposit)
        weight_substrate[transition_mask] = (z_deposit - h_surface[transition_mask]) / z_deposit
        weight_deposit[transition_mask] = h_surface[transition_mask] / z_deposit

        # 厚沉积：纯沉积物
        deposit_mask = (h_surface >= z_deposit)
        weight_deposit[deposit_mask] = 1.0

        return weight_substrate, weight_deposit

    def _display_progress(self, pixel_idx, total_pixels, h_surface, beam_pos):
        """显示进度 - 简化版"""
        progress_pct = pixel_idx / total_pixels * 100
        max_height = np.max(h_surface)
        memory_mb = psutil.Process().memory_info().rss / 1024 ** 2

        print(f"进度: {progress_pct:.0f}% | 最大高度: {max_height:.3e} nm | "
              f"位置: ({beam_pos[0]:.1f},{beam_pos[1]:.1f}) | 内存: {memory_mb:.0f}MB")

    def _finalize_simulation(self, h_surface, n_surface, total_time, scan_info,
                             x_grid, y_grid, total_pixels):
        """完成仿真后处理"""
        # 打印结果
        self.visualizer.print_results(
            h_surface, n_surface, total_time, scan_info, x_grid, y_grid,
            self.quad_gaussian_params, self.physical_params
        )

    def _prepare_results(self, x_grid, y_grid, h_surface, n_surface,
                         scan_positions, scan_info, total_time):
        """准备结果"""
        return {
            'x_grid': x_grid,
            'y_grid': y_grid,
            'h_surface': h_surface,
            'n_surface': n_surface,
            'scan_positions': scan_positions,
            'scan_history': np.array(self.scan_history),
            'scan_info': scan_info,
            'simulation_time': total_time,
            'config': self.config
        }

    # ========================================================================
    # 工具方法
    # ========================================================================

    def _create_physical_params(self) -> PhysicalParams:
        p = self.config['physical']
        return PhysicalParams(
            Phi=p['Phi'], tau=p['tau'], sigma=p['sigma'],
            n0=p['n0'], DeltaV=p['DeltaV'], k=p['k'],
            D_surf=p['D_surf'], dx=p['dx'], dy=p['dy']
        )

    def _create_quad_gaussian_params(self):
        """创建四高斯参数"""
        qg = self.config['quad_gaussian']

        substrate = MaterialParams(
            gaussian1=GaussianParams(qg['substrate']['gaussian1']['sigma'],
                                     qg['substrate']['gaussian1']['amplitude']),
            gaussian2=GaussianParams(qg['substrate']['gaussian2']['sigma'],
                                     qg['substrate']['gaussian2']['amplitude']),
            gaussian3=GaussianParams(qg['substrate']['gaussian3']['sigma'],
                                     qg['substrate']['gaussian3']['amplitude']),
            gaussian4=GaussianParams(qg['substrate']['gaussian4']['sigma'],
                                     qg['substrate']['gaussian4']['amplitude'])
        )

        deposit = MaterialParams(
            gaussian1=GaussianParams(qg['deposit']['gaussian1']['sigma'],
                                     qg['deposit']['gaussian1']['amplitude']),
            gaussian2=GaussianParams(qg['deposit']['gaussian2']['sigma'],
                                     qg['deposit']['gaussian2']['amplitude']),
            gaussian3=GaussianParams(qg['deposit']['gaussian3']['sigma'],
                                     qg['deposit']['gaussian3']['amplitude']),
            gaussian4=GaussianParams(qg['deposit']['gaussian4']['sigma'],
                                     qg['deposit']['gaussian4']['amplitude'])
        )

        return QuadGaussianParams(substrate, deposit, qg['z_deposit'])

    def _prepare_numba_params(self):
        """预处理Numba参数"""
        sub = self.quad_gaussian_params.substrate
        dep = self.quad_gaussian_params.deposit

        self.sub_params_array = np.array([
            sub.gaussian1.sigma, sub.gaussian1.amplitude,
            sub.gaussian2.sigma, sub.gaussian2.amplitude,
            sub.gaussian3.sigma, sub.gaussian3.amplitude,
            sub.gaussian4.sigma, sub.gaussian4.amplitude
        ], dtype=np.float32)

        self.dep_params_array = np.array([
            dep.gaussian1.sigma, dep.gaussian1.amplitude,
            dep.gaussian2.sigma, dep.gaussian2.amplitude,
            dep.gaussian3.sigma, dep.gaussian3.amplitude,
            dep.gaussian4.sigma, dep.gaussian4.amplitude
        ], dtype=np.float32)

    # ========================================================================
    # 公共接口
    # ========================================================================

    def visualize_results(self, results):
        """可视化结果"""
        self.visualizer.visualize_results(results, self.quad_gaussian_params)

    def save_results(self, results):
        """保存结果"""
        self.visualizer.save_results(results, self.physical_params, self.quad_gaussian_params)
