#!/usr/bin/env python3
"""
FEBID仿真主控制器 - 精简版
删除所有备用方法，只使用Numba并行计算

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
from realtime_monitor import RealTimeWebMonitor, FixedRangeRealTimeMonitor
from scan_strategies import ScanPathGenerator
from visualization_analysis import VisualizationAnalyzer
from simulation_core_algorithms import (
    calculate_quad_gaussian_flux_numba,
    rk4_step_parallel,
    apply_surface_effects_numba
)
from config import calculate_dynamic_visualization_ranges

class MemoryOptimizedFEBID:
    """内存优化的FEBID仿真类 - 精简版（只使用Numba并行）"""

    def __init__(self, config: Dict, enable_realtime_monitor: bool = None,
                 monitor_save_interval: int = None, visualization_config: Dict = None,
                 use_realtime_mode: bool = None):
        """初始化仿真参数"""
        self.config = config
        self.physical_params = self._create_physical_params()
        self.quad_gaussian_params = self._create_quad_gaussian_params()
        self.scan_history = []
        self.start_time = None

        # 监控配置
        self._setup_monitoring(enable_realtime_monitor, monitor_save_interval,
                               use_realtime_mode)

        # 初始化辅助模块
        self.scan_generator = ScanPathGenerator(config)
        self.visualizer = VisualizationAnalyzer(config)
        self.validator = ConfigValidator()

        # 表面传播配置
        self.surface_config = config.get('surface_propagation', {})
        self.enable_surface_propagation = self.surface_config.get('enable', False)

        # 预处理参数（默认启用Numba）
        self._prepare_numba_params()
        self._prepare_rk4_constants()

        print(f"✓ 仿真器初始化完成 {'(表面感知)' if self.enable_surface_propagation else '(2D模式)'}")

    # simulation_core_main.py
    def _setup_monitoring(self, enable_realtime_monitor, monitor_save_interval,
                          use_realtime_mode):  # 删除 visualization_config 参数
        """设置监控配置"""
        monitor_config = self.config.get('monitoring', {})

        # 计算动态范围（如果需要）
        if monitor_config.get('height_range') == 'auto' or monitor_config.get('precursor_range') == 'auto':
            from config import calculate_dynamic_visualization_ranges
            dynamic_ranges = calculate_dynamic_visualization_ranges(self.config)

            if monitor_config.get('height_range') == 'auto':
                monitor_config['height_range'] = dynamic_ranges['height_range']
            if monitor_config.get('precursor_range') == 'auto':
                monitor_config['precursor_range'] = dynamic_ranges['precursor_range']

        self.enable_realtime_monitor = (enable_realtime_monitor
                                        if enable_realtime_monitor is not None
                                        else monitor_config.get('enable_realtime_monitor', False))

        self.monitor_save_interval = (monitor_save_interval
                                      if monitor_save_interval is not None
                                      else monitor_config.get('save_interval', 50))

        self.use_realtime_mode = (use_realtime_mode
                                  if use_realtime_mode is not None
                                  else monitor_config.get('use_realtime_mode', True))

        # 不再需要单独的 visualization_config
        self.monitor = None

    def _prepare_rk4_constants(self):
        """预计算RK4常量"""
        p = self.physical_params
        self.rk4_constants = {
            'k_phi_n0': p.k * p.Phi * p.n0,
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

        # 初始化监控器
        self.monitor = self._initialize_monitor(x_grid, y_grid)

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

        # 初始前驱体覆盖度
        p = self.physical_params
        n_eq = p.k * p.Phi * p.tau * p.n0 / (1 + p.k * p.Phi * p.tau)
        n_surface = np.full_like(X_mesh, n_eq, dtype=FLOAT_DTYPE)

        self.X_mesh = X_mesh
        self.Y_mesh = Y_mesh

        return x_grid, y_grid, h_surface, n_surface

    def _initialize_monitor(self, x_grid, y_grid):
        """初始化监控器"""
        if not self.enable_realtime_monitor:
            return None

        monitor_config = self.config.get('monitoring', {})

        # 确保监控配置包含可视化参数
        if 'height_range' not in monitor_config or 'precursor_range' not in monitor_config:
            print("⚠️  监控配置缺少可视化参数，使用默认值")
            monitor_config['height_range'] = [0, 8e-4]
            monitor_config['precursor_range'] = [0, 4e-4]

        monitor_class = RealTimeWebMonitor if self.use_realtime_mode else FixedRangeRealTimeMonitor

        monitor = monitor_class(
            x_grid, y_grid,
            self.config['geometry'],
            monitor_config,  # 直接传递整个监控配置
            save_interval=self.monitor_save_interval,
            simulation_config=self.config
        )

        if self.use_realtime_mode and monitor.launch_realtime_viewer():
            print("✅ 实时监控已启动")
        elif not self.use_realtime_mode:
            print("✅ 传统监控已准备")

        return monitor

    def _display_config(self, scan_positions):
        """显示配置信息"""
        edge_mask = scan_positions[:, 3] == 1
        basic_mask = scan_positions[:, 3] == 0

        print(f"🎯 扫描统计: 边缘点={np.sum(edge_mask)}, 基础点={np.sum(basic_mask)}")
        print(f"🌊 计算模式: {'表面感知' if self.enable_surface_propagation else '传统2D'}")

    def _run_main_simulation_loop(self, scan_positions, scan_info, x_grid, y_grid,
                                  h_surface, n_surface, dt, dwell_time,
                                  edge_repeat_times, total_pixels):
        """主仿真循环"""
        current_subloop = 0
        pixels_in_current_subloop = 0
        rk4_const = self.rk4_constants

        for pixel_idx in range(total_pixels):
            # 子循环管理
            if pixels_in_current_subloop == 0:
                current_subloop += 1
                pixels_in_current_subloop = scan_info.pixels_per_subloop

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
                    rk4_const['k_phi_n0'], rk4_const['tau_inv'],
                    rk4_const['sigma'], rk4_const['n0_inv'],
                    rk4_const['D_surf_factor'],
                    self.physical_params.dx, self.physical_params.dy,
                    h_surface  # 新增参数
                )

                # 更新高度
                deposition_rate = self.physical_params.DeltaV * self.physical_params.sigma * f_surface * n_surface
                h_surface += deposition_rate * dt
                n_surface = np.clip(n_surface, 0, self.physical_params.n0)

            # 更新监控
            self._update_monitoring(pixel_idx, h_surface, n_surface, beam_pos_x, beam_pos_y,
                                    current_subloop, is_edge_repeat, total_pixels)

            pixels_in_current_subloop -= 1

            # 进度显示
            if pixel_idx % max(1, total_pixels // 20) == 0:
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

    def _apply_surface_effects_fixed(self, f_surface, h_surface):
        """
        应用连续表面效应 - 修复版本
        """
        surface_params = self.config.get('surface_effects', {
            'depth_scale_factor': 5,
            'slope_decay_min': 0.1,
            'slope_decay_max': 10.0,
            'flux_min_factor': 0.01,
        })

        # 获取新参数（移除gradient_factor）
        depth_scale_factor = surface_params.get('depth_scale_factor', 5)
        slope_min = surface_params.get('slope_decay_min', 0.1)
        slope_max = surface_params.get('slope_decay_max', 10.0)
        flux_min_factor = surface_params.get('flux_min_factor', 0.01)

        # 计算材料权重
        weight_substrate, weight_deposit = self._calculate_material_weights(h_surface)

        # 修复：正确获取sigma参数
        sub = self.quad_gaussian_params.substrate
        dep = self.quad_gaussian_params.deposit

        f_surface_final = apply_surface_effects_numba(
            f_surface, h_surface, weight_substrate, weight_deposit,
            sub.gaussian1.sigma, dep.gaussian1.sigma,  # 修复：确保这些变量可用
            depth_scale_factor,
            self.physical_params.dx, self.physical_params.dy,
            slope_min, slope_max, flux_min_factor  # 新参数
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

    def _update_monitoring(self, pixel_idx, h_surface, n_surface, beam_pos_x, beam_pos_y,
                           current_subloop, is_edge_repeat, total_pixels):
        """更新监控"""
        if self.enable_realtime_monitor and self.monitor is not None:
            current_time = time.time() - self.start_time

            if self.use_realtime_mode:
                self.monitor.update_data(
                    pixel_idx, h_surface, n_surface,
                    (beam_pos_x, beam_pos_y), current_time,
                    total_pixels, is_running=True
                )
            else:
                self.monitor.record_frame(
                    pixel_idx, h_surface, n_surface,
                    (beam_pos_x, beam_pos_y), current_time
                )

        # 记录历史
        self.scan_history.append([
            pixel_idx, pixel_idx + 1, beam_pos_x, beam_pos_y,
            np.max(h_surface), current_subloop, is_edge_repeat
        ])

    def _apply_surface_effects(self, f_surface, h_surface):
        """应用连续表面效应"""
        surface_params = self.config.get('surface_effects', {
            'depth_scale_factor': 5,
            'slope_decay_min': 0.1,
            'slope_decay_max': 10.0,
        })

        weight_substrate, weight_deposit = self._calculate_material_weights(h_surface)

        # 获取sigma参数
        sub = self.quad_gaussian_params.substrate
        dep = self.quad_gaussian_params.deposit

        # 调用你现有的函数，但需要更新参数
        f_surface_final = apply_surface_effects_numba(
            f_surface, h_surface, weight_substrate, weight_deposit,
            sub.gaussian1.sigma, dep.gaussian1.sigma,
            surface_params.get('depth_scale_factor', 5),
            self.physical_params.dx, self.physical_params.dy,
            surface_params.get('slope_decay_min', 0.1),
            surface_params.get('slope_decay_max', 10.0),
        )

        return np.maximum(f_surface_final, 0)

    def _display_progress(self, pixel_idx, total_pixels, h_surface, beam_pos):
        """显示进度"""
        if not hasattr(self, '_last_display_time'):
            self._last_display_time = 0

        current_time = time.time() - self.start_time

        if (current_time - self._last_display_time) >= 2.0 or pixel_idx == total_pixels:
            progress_pct = pixel_idx / total_pixels * 100
            max_height = np.max(h_surface)
            memory_mb = psutil.Process().memory_info().rss / 1024 ** 2

            print(f"进度: {progress_pct:.1f}% | 最大高度: {max_height:.3e} nm | "
                  f"位置: ({beam_pos[0]:.1f},{beam_pos[1]:.1f}) | 内存: {memory_mb:.1f}MB")

            self._last_display_time = current_time

    def _finalize_simulation(self, h_surface, n_surface, total_time, scan_info,
                             x_grid, y_grid, total_pixels):
        """完成仿真后处理"""
        if self.enable_realtime_monitor and self.monitor is not None and self.use_realtime_mode:
            current_time = time.time() - self.start_time
            self.monitor.update_data(
                total_pixels, h_surface, n_surface, (0, 0), current_time,
                total_pixels, is_running=False
            )

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
            'config': self.config,
            'monitor': self.monitor
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

    def launch_realtime_viewer(self):
        """启动实时查看器"""
        if self.monitor is not None:
            if self.use_realtime_mode:
                if hasattr(self.monitor, 'server') and self.monitor.server:
                    print(f"🌐 实时监控: http://localhost:{self.monitor.web_port}")
                else:
                    self.monitor.launch_realtime_viewer()
            else:
                self.monitor.launch_fixed_range_viewer()
        else:
            print("⚠️ 监控器未启用")

    def stop_realtime_monitor(self):
        """停止监控器"""
        if self.monitor is not None and hasattr(self.monitor, 'stop_server'):
            self.monitor.stop_server()
            print("🛑 监控器已停止")

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'monitor') and self.monitor is not None:
            if hasattr(self.monitor, 'close'):
                self.monitor.close()
