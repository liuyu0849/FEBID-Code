#!/usr/bin/env python3
"""
FEBID仿真可视化和分析模块 - 优化版
减少代码重复，提高性能

Author: 刘宇
Date: 2025/7
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import warnings
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from data_structures import ScanInfo, PhysicalParams, QuadGaussianParams, FLOAT_DTYPE
from base_classes import calculate_surface_statistics

# 设置字体显示
try:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.default'] = 'regular'
except Exception:
    warnings.warn("Font configuration warning - using system defaults")


class VisualizationAnalyzer:
    """可视化分析器 - 优化版"""

    def __init__(self, config: Dict):
        self.config = config
        self._plot_config = {
            'figure_size': (16, 12),
            'dpi': 300,
            'colormap_height': 'viridis',
            'colormap_precursor': 'cool',
            'contour_levels': 30
        }

    def print_results(self, h_surface: np.ndarray, n_surface: np.ndarray,
                      total_time: float, scan_info: ScanInfo, x_grid: np.ndarray, y_grid: np.ndarray,
                      quad_gaussian_params: QuadGaussianParams, physical_params: PhysicalParams):
        """打印仿真结果统计 - 优化版"""
        # 使用统一的统计函数
        h_stats = calculate_surface_statistics(h_surface)
        n_stats = calculate_surface_statistics(n_surface)

        print(f"\n=== Simulation Complete - Results Analysis ===")
        print(f"Total simulation time: {total_time:.2f} seconds")
        print(f"Max deposition height: {h_stats['max']:.3e} nm")
        print(f"Mean deposition height: {h_stats['mean']:.3f} nm")
        print(f"Height std deviation: {h_stats['std']:.3f} nm")
        print(f"Relative deposition volume: {np.sum(h_surface) * physical_params.dx * physical_params.dy:.2f}")

        # 前驱体统计
        print(f"\n=== Precursor Analysis ===")
        print(f"Min concentration: {n_stats['min']:.6f} molecules/nm^2")
        print(f"Max concentration: {n_stats['max']:.6f} molecules/nm^2")
        print(f"Mean concentration: {n_stats['mean']:.6f} molecules/nm^2")
        print(f"Concentration std dev: {n_stats['std']:.6f} molecules/nm^2")

        # 材料分布 - 使用向量化操作
        self._print_material_distribution(h_surface, quad_gaussian_params.z_deposit)

        # 性能统计
        points_per_second = scan_info.total_pixels / total_time
        print(f"\nSimulation performance: {points_per_second:.0f} scan points/second")

    def _print_material_distribution(self, h_surface: np.ndarray, z_deposit: float):
        """打印材料分布统计 - 向量化版本"""
        substrate_mask = h_surface == 0
        transition_mask = (h_surface > 0) & (h_surface < z_deposit)
        deposit_mask = h_surface >= z_deposit

        total_points = h_surface.size
        substrate_points = np.sum(substrate_mask)
        transition_points = np.sum(transition_mask)
        deposit_points = np.sum(deposit_mask)

        print(f"\n=== Dual-Material Distribution ===")
        print(f"Substrate region: {substrate_points} ({substrate_points / total_points * 100:.1f}%)")
        print(f"Transition region: {transition_points} ({transition_points / total_points * 100:.1f}%)")
        print(f"Deposit region: {deposit_points} ({deposit_points / total_points * 100:.1f}%)")

    def visualize_results(self, results: Dict, quad_gaussian_params: QuadGaussianParams):
        """优化的结果可视化"""
        if not self.config['output']['create_plots']:
            print("图表生成已禁用")
            return

        print("生成核心可视化图表...")

        # 提取数据
        plot_data = self._prepare_plot_data(results)
        # 检查是否有基底几何特征
        has_substrate_features = self._check_substrate_features(plot_data)
        # 如果有基底特征，添加基底可视化
        if has_substrate_features:
            self._create_substrate_visualization(plot_data)

        # 分析数据
        edge_analysis = self._analyze_edge_enhancement_vectorized(
            plot_data['h_surface'], plot_data['scan_positions'],
            plot_data['x_grid'], plot_data['y_grid']
        )
        edge_stats = self._analyze_edge_statistics(plot_data['scan_positions'], results['scan_info'])

        # 创建图表
        self._create_visualization_plots(plot_data, edge_analysis, edge_stats,
                                         quad_gaussian_params, results)

    def _check_substrate_features(self, plot_data: Dict) -> bool:
        """检查是否有非平面基底特征"""
        h_surface = plot_data['h_surface']
        # 检查初始表面是否为零平面
        return not np.allclose(h_surface, 0.0, atol=1e-10)

    def _create_substrate_visualization(self, plot_data: Dict):
        """创建基底可视化"""
        print("📊 检测到自定义基底，生成基底几何图表...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        X_mesh = plot_data['X_mesh']
        Y_mesh = plot_data['Y_mesh']
        h_surface = plot_data['h_surface']

        # 基底3D图
        ax1 = fig.add_subplot(221, projection='3d')
        surf = ax1.plot_surface(X_mesh, Y_mesh, h_surface, cmap='terrain', alpha=0.9)
        ax1.set_title('3D Topology')
        ax1.set_xlabel('X [nm]')
        ax1.set_ylabel('Y [nm]')
        ax1.set_zlabel('Height [nm]')

        # 基底等高线
        contour = ax2.contourf(X_mesh, Y_mesh, h_surface, levels=20, cmap='terrain')
        ax2.set_title('Height Map')
        ax2.set_xlabel('X [nm]')
        ax2.set_ylabel('Y [nm]')
        ax2.set_aspect('equal')
        plt.colorbar(contour, ax=ax2)

        # X方向截面
        mid_y_idx = h_surface.shape[0] // 2
        ax3.plot(plot_data['x_grid'], h_surface[mid_y_idx, :], 'b-', linewidth=2)
        ax3.set_title('X-Direction Cross Section')
        ax3.set_xlabel('X [nm]')
        ax3.set_ylabel('Height [nm]')
        ax3.grid(True, alpha=0.3)

        # Y方向截面
        mid_x_idx = h_surface.shape[1] // 2
        ax4.plot(plot_data['y_grid'], h_surface[:, mid_x_idx], 'r-', linewidth=2)
        ax4.set_title('Y-Direction Cross Section')
        ax4.set_xlabel('Y [nm]')
        ax4.set_ylabel('Height [nm]')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('FEBID_Substrate_Geometry.png', dpi=300, bbox_inches='tight')
        print("📊 基底几何图表已保存: FEBID_Substrate_Geometry.png")
        plt.show()

    def _prepare_plot_data(self, results: Dict) -> Dict:
        """准备绘图数据"""
        h_surface = results['h_surface']
        n_surface = results['n_surface']
        x_grid = results['x_grid']
        y_grid = results['y_grid']
        scan_positions = results['scan_positions']

        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)

        return {
            'h_surface': h_surface,
            'n_surface': n_surface,
            'x_grid': x_grid,
            'y_grid': y_grid,
            'X_mesh': X_mesh,
            'Y_mesh': Y_mesh,
            'scan_positions': scan_positions
        }

    def _create_visualization_plots(self, plot_data: Dict, edge_analysis: Dict,
                                    edge_stats: Dict, quad_gaussian_params: QuadGaussianParams,
                                    results: Dict):
        """创建可视化图表 - 重构版"""
        fig = plt.figure(figsize=self._plot_config['figure_size'])

        # 1. 3D表面形貌
        ax1 = self._create_3d_surface_plot(fig, 221, plot_data)

        # 2. 2D等高线图 + 边缘点标记
        ax2 = self._create_contour_plot_with_edges(fig, 222, plot_data)

        # 3. 前驱体浓度分布
        ax3 = self._create_precursor_plot(fig, 223, plot_data)

        # 4. 核心统计信息
        ax4 = self._create_statistics_panel(fig, 224, results, edge_stats,
                                            quad_gaussian_params, edge_analysis)

        plt.tight_layout()

        # 保存图表
        output_path = 'FEBID_Results.png'
        plt.savefig(output_path, dpi=self._plot_config['dpi'], bbox_inches='tight')
        print(f"结果图表已保存: {output_path}")

        plt.show()

    def _create_3d_surface_plot(self, fig, position: int, plot_data: Dict):
        """创建3D表面图"""
        ax = fig.add_subplot(position, projection='3d')
        surf = ax.plot_surface(
            plot_data['X_mesh'], plot_data['Y_mesh'], plot_data['h_surface'],
            cmap=self._plot_config['colormap_height'], alpha=0.9
        )
        ax.set_xlabel('X Position [nm]')
        ax.set_ylabel('Y Position [nm]')
        ax.set_zlabel('Height [nm]')
        ax.set_title('3D Surface Morphology')
        plt.colorbar(surf, ax=ax, shrink=0.6, pad=0.15)
        return ax

    def _create_contour_plot_with_edges(self, fig, position: int, plot_data: Dict):
        """创建带边缘标记的等高线图"""
        ax = fig.add_subplot(position)
        contour = ax.contourf(
            plot_data['X_mesh'], plot_data['Y_mesh'], plot_data['h_surface'],
            levels=self._plot_config['contour_levels'],
            cmap=self._plot_config['colormap_height']
        )

        # 叠加边缘点标记
        edge_mask = plot_data['scan_positions'][:, 3] == 1
        if np.any(edge_mask):
            edge_positions = plot_data['scan_positions'][edge_mask]
            ax.scatter(edge_positions[:, 0], edge_positions[:, 1],
                       c='red', s=8, alpha=0.8,
                       label=f'Edge Enhanced Points ({len(edge_positions)})')
            ax.legend()

        ax.set_xlabel('X Position [nm]')
        ax.set_ylabel('Y Position [nm]')
        ax.set_title('Height Contour + Edge Enhancement')
        ax.set_aspect('equal')
        plt.colorbar(contour, ax=ax)
        return ax

    def _create_precursor_plot(self, fig, position: int, plot_data: Dict):
        """创建前驱体浓度图"""
        ax = fig.add_subplot(position)
        contour = ax.contourf(
            plot_data['X_mesh'], plot_data['Y_mesh'], plot_data['n_surface'],
            levels=20, cmap=self._plot_config['colormap_precursor']
        )
        ax.set_xlabel('X Position [nm]')
        ax.set_ylabel('Y Position [nm]')
        ax.set_title('Precursor Concentration')
        ax.set_aspect('equal')
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Density [molecules/nm^2]')
        return ax

    def _create_statistics_panel(self, fig, position: int, results: Dict,
                                 edge_stats: Dict, quad_gaussian_params: QuadGaussianParams,
                                 edge_analysis: Dict):
        """创建统计信息面板"""
        ax = fig.add_subplot(position)
        ax.axis('off')

        stats_text = self._generate_core_statistics_text(
            results, edge_stats, quad_gaussian_params, edge_analysis
        )

        ax.text(0.5, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
                fontfamily='monospace',
                verticalalignment='center',
                horizontalalignment='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        ax.set_title('Simulation Summary', fontweight='bold')
        return ax

    def _analyze_edge_enhancement_vectorized(self, h_surface: np.ndarray, scan_positions: np.ndarray,
                                             x_grid: np.ndarray, y_grid: np.ndarray) -> Dict:
        """向量化的边缘增强分析"""
        # 定义中心区域
        nx, ny = len(x_grid), len(y_grid)
        center_x_slice = slice(nx // 3, 2 * nx // 3)
        center_y_slice = slice(ny // 3, 2 * ny // 3)

        # 使用切片操作代替循环
        center_heights = h_surface[center_y_slice, center_x_slice].flatten()

        # 创建边缘掩码
        edge_mask = np.ones(h_surface.shape, dtype=bool)
        edge_mask[center_y_slice, center_x_slice] = False
        edge_heights = h_surface[edge_mask]

        # 计算统计
        edge_stats = calculate_surface_statistics(edge_heights) if len(edge_heights) > 0 else {'mean': 0}
        center_stats = calculate_surface_statistics(center_heights) if len(center_heights) > 0 else {'mean': 0}

        enhancement_ratio = edge_stats['mean'] / center_stats['mean'] if center_stats['mean'] > 0 else 0

        return {
            'edge_avg': edge_stats['mean'],
            'center_avg': center_stats['mean'],
            'max_height': np.max(h_surface),
            'enhancement_ratio': enhancement_ratio
        }

    def _analyze_edge_statistics(self, scan_positions: np.ndarray, scan_info: ScanInfo) -> Dict:
        """分析边缘统计信息 - 向量化版本"""
        edge_mask = scan_positions[:, 3] == 1
        basic_mask = ~edge_mask

        edge_points = np.sum(edge_mask)
        basic_points = np.sum(basic_mask)
        total_points = len(scan_positions)

        # 计算唯一边缘坐标 - 使用numpy的unique
        if edge_points > 0:
            edge_coords = scan_positions[edge_mask, :2]
            # 四舍五入到6位小数并获取唯一值
            rounded_coords = np.round(edge_coords, 6)
            unique_coords = np.unique(rounded_coords, axis=0)
            unique_edge_count = len(unique_coords)
        else:
            unique_edge_count = 0

        return {
            'basic_points': basic_points,
            'edge_points': edge_points,
            'total_points': total_points,
            'unique_edge_points': unique_edge_count,
            'edge_percentage': edge_points / total_points * 100 if total_points > 0 else 0
        }

    def _generate_core_statistics_text(self, results: Dict, edge_stats: Dict,
                                       quad_gaussian_params: QuadGaussianParams,
                                       edge_analysis: Dict) -> str:
        """生成核心统计信息文本 - 优化版"""
        scan_config = self.config['scan']

        # 获取扫描区域内的统计
        scan_stats = self._get_scan_area_statistics(results)

        stats_text = f"""FEBID Simulation Results Summary

    Scan Configuration:
    Strategy: {scan_config['scan_strategy']}
    Scan Step: {scan_config['pixel_size_x']} nm
    Line Step: {scan_config['pixel_size_y']} nm
    Edge Layers: {scan_config['edge_layers']}
    Edge Scan: Total {scan_config['edge_repeat_times'] + 1} Scan

    Edge Analysis:
    Basic Points: {edge_stats['basic_points']:,}
    Edge Points: {edge_stats['edge_points']:,}
    Edge Coverage: {edge_stats['edge_percentage']:.1f}%

    Deposition Results (Scan Area Only):
    Max Height: {scan_stats['max']:.2e} nm
    Mean Height: {scan_stats['mean']:.2e} nm
    Std Dev: {scan_stats['std']:.2e} nm
    Std/Mean: {scan_stats['std'] / scan_stats['mean']:.2f}

    Performance:
    Runtime: {results['simulation_time']:.1f} s
    Scan Area: {scan_config['scan_x_end'] - scan_config['scan_x_start']:.0f}×{scan_config['scan_y_end'] - scan_config['scan_y_start']:.0f} nm²
    Speed: {edge_stats['total_points'] / results['simulation_time']:.0f} pts/s"""

        return stats_text

    def _get_scan_area_statistics(self, results: Dict) -> Dict:
        """获取扫描区域内的统计数据"""
        scan_config = self.config['scan']
        h_surface = results['h_surface']
        x_grid = results['x_grid']
        y_grid = results['y_grid']

        # 创建扫描区域掩码
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
        scan_mask = (
                (X_mesh >= scan_config['scan_x_start']) &
                (X_mesh <= scan_config['scan_x_end']) &
                (Y_mesh >= scan_config['scan_y_start']) &
                (Y_mesh <= scan_config['scan_y_end'])
        )

        return calculate_surface_statistics(h_surface, scan_mask)

    def save_results(self, results: Dict, physical_params: PhysicalParams,
                     quad_gaussian_params: QuadGaussianParams):
        """优化的结果保存"""
        if not self.config['output']['save_core_results']:
            print("结果保存已禁用")
            return

        print("保存核心仿真结果...")

        # 准备核心数据
        core_data = self._prepare_core_data(results, physical_params, quad_gaussian_params)

        # 保存压缩的numpy格式
        np.savez_compressed('FEBID_Core_Results.npz', **core_data)
        print("✓ 核心结果已保存: FEBID_Core_Results.npz")

        # 保存文本摘要
        self._save_text_summary(results, self.config)
        print("✓ 摘要报告已保存: FEBID_Summary.txt")

    def _prepare_core_data(self, results: Dict, physical_params: PhysicalParams,
                           quad_gaussian_params: QuadGaussianParams) -> Dict:
        """准备要保存的核心数据"""
        return {
            'h_surface': results['h_surface'],
            'n_surface': results['n_surface'],
            'x_grid': results['x_grid'],
            'y_grid': results['y_grid'],
            'scan_positions': results['scan_positions'],
            'simulation_time': results['simulation_time'],
            'scan_info': {
                'total_pixels': results['scan_info'].total_pixels,
                'total_subloops': results['scan_info'].total_subloops,
                'edge_layers': results['scan_info'].edge_layers,
                'edge_repeat_times': results['scan_info'].edge_repeat_times
            },
            'physical_params': asdict(physical_params),
            'quad_gaussian_params': asdict(quad_gaussian_params)
        }

    def _save_text_summary(self, results: Dict, config: Dict):
        """保存文本摘要"""
        # 获取统计信息
        h_stats = calculate_surface_statistics(results['h_surface'])

        with open('FEBID_Summary.txt', 'w', encoding='utf-8') as f:
            f.write("=== FEBID仿真结果摘要 ===\n")
            f.write(f"仿真时间: {results['simulation_time']:.2f} 秒\n")
            f.write(f"最大沉积高度: {h_stats['max']:.3e} nm\n")
            f.write(f"平均沉积高度: {h_stats['mean']:.3e} nm\n")
            f.write(f"高度标准差: {h_stats['std']:.3e} nm\n")
            f.write(f"总扫描点数: {results['scan_info'].total_pixels:,}\n")
            f.write(f"扫描策略: {config['scan']['scan_strategy']}\n")
            f.write(f"边缘增强: {config['scan']['edge_repeat_times']}x\n")
            f.write("\n文件输出:\n")
            f.write("- FEBID_Results.png (可视化图表)\n")
            f.write("- FEBID_Core_Results.npz (数值结果)\n")
            f.write("- FEBID_Summary.txt (本摘要文件)\n")
