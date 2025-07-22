#!/usr/bin/env python3
"""
FEBID仿真基底几何模块
支持自定义基底形状，包括矩形缺陷、凸起等

Author: 刘宇
Date: 2025/7
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from data_structures import FLOAT_DTYPE
from base_classes import ConfigValidator


@dataclass
class RectangularDefect:
    """矩形缺陷/特征定义"""
    x1: float  # 左下角x坐标 [nm]
    y1: float  # 左下角y坐标 [nm]
    x2: float  # 右上角x坐标 [nm]
    y2: float  # 右上角y坐标 [nm]
    height_offset: float  # 相对基准平面的高度偏移 [nm] (正值=凸起, 负值=凹陷)
    name: str = ""  # 特征名称（可选）


@dataclass
class SubstrateGeometry:
    """基底几何配置"""
    base_height: float = 0.0  # 基准平面高度 [nm]
    rectangular_defects: List[RectangularDefect] = None  # 矩形缺陷列表

    def __post_init__(self):
        if self.rectangular_defects is None:
            self.rectangular_defects = []


class SubstrateGeometryGenerator:
    """基底几何生成器"""

    def __init__(self, config: Dict):
        self.config = config
        self.geometry_config = config['geometry']
        self.substrate_config = config.get('substrate_geometry', {})

        # 解析配置
        self.substrate_geometry = self._parse_substrate_config()

    def _parse_substrate_config(self) -> SubstrateGeometry:
        """解析基底配置"""
        base_height = self.substrate_config.get('base_height', 0.0)
        defects_config = self.substrate_config.get('rectangular_defects', [])

        defects = []
        for i, defect_config in enumerate(defects_config):
            defect = RectangularDefect(
                x1=defect_config['x1'],
                y1=defect_config['y1'],
                x2=defect_config['x2'],
                y2=defect_config['y2'],
                height_offset=defect_config['height_offset'],
                name=defect_config.get('name', f'Defect_{i + 1}')
            )
            defects.append(defect)

        return SubstrateGeometry(
            base_height=base_height,
            rectangular_defects=defects
        )

    def generate_substrate_surface(self, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
        """
        生成基底表面高度分布

        Parameters:
        -----------
        x_grid : np.ndarray
            X方向网格坐标
        y_grid : np.ndarray
            Y方向网格坐标

        Returns:
        --------
        np.ndarray : 基底表面高度分布
        """
        # 创建网格
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)

        # 初始化为基准平面
        h_substrate = np.full_like(X_mesh, self.substrate_geometry.base_height, dtype=FLOAT_DTYPE)

        # 添加矩形特征
        for defect in self.substrate_geometry.rectangular_defects:
            mask = self._create_rectangular_mask(X_mesh, Y_mesh, defect)
            h_substrate[mask] += defect.height_offset

        return h_substrate

    def _create_rectangular_mask(self, X_mesh: np.ndarray, Y_mesh: np.ndarray,
                                 defect: RectangularDefect) -> np.ndarray:
        """创建矩形区域掩码"""
        # 确保坐标顺序正确
        x_min, x_max = min(defect.x1, defect.x2), max(defect.x1, defect.x2)
        y_min, y_max = min(defect.y1, defect.y2), max(defect.y1, defect.y2)

        mask = ((X_mesh >= x_min) & (X_mesh <= x_max) &
                (Y_mesh >= y_min) & (Y_mesh <= y_max))

        return mask

    def validate_substrate_config(self) -> bool:
        """验证基底配置"""
        try:
            print("🔍 验证基底几何配置...")

            # 检查基本参数
            if not isinstance(self.substrate_geometry.base_height, (int, float)):
                print("❌ base_height必须是数值")
                return False

            # 验证矩形缺陷
            geom = self.geometry_config
            simulation_bounds = (geom['X_min'], geom['X_max'], geom['Y_min'], geom['Y_max'])

            for i, defect in enumerate(self.substrate_geometry.rectangular_defects):
                if not self._validate_rectangular_defect(defect, simulation_bounds, i):
                    return False

            print(f"✓ 基底配置验证通过: 基准高度={self.substrate_geometry.base_height} nm, "
                  f"{len(self.substrate_geometry.rectangular_defects)} 个矩形特征")
            return True

        except Exception as e:
            print(f"❌ 基底配置验证失败: {e}")
            return False

    def _validate_rectangular_defect(self, defect: RectangularDefect,
                                     simulation_bounds: Tuple[float, float, float, float],
                                     index: int) -> bool:
        """验证单个矩形缺陷"""
        x_min_sim, x_max_sim, y_min_sim, y_max_sim = simulation_bounds

        # 检查坐标顺序
        if defect.x1 == defect.x2 or defect.y1 == defect.y2:
            print(f"❌ 矩形特征 {index + 1} ({defect.name}): 不能是零面积")
            return False

        # 检查是否在仿真区域内
        x_min, x_max = min(defect.x1, defect.x2), max(defect.x1, defect.x2)
        y_min, y_max = min(defect.y1, defect.y2), max(defect.y1, defect.y2)

        if (x_max < x_min_sim or x_min > x_max_sim or
                y_max < y_min_sim or y_min > y_max_sim):
            print(f"⚠️  矩形特征 {index + 1} ({defect.name}): 完全在仿真区域外")

        # 检查高度偏移合理性
        if abs(defect.height_offset) > 1000:  # 1微米限制
            print(f"⚠️  矩形特征 {index + 1} ({defect.name}): 高度偏移 {defect.height_offset} nm 可能过大")

        return True

    def get_substrate_statistics(self, h_substrate: np.ndarray,
                                 x_grid: np.ndarray, y_grid: np.ndarray) -> Dict:
        """获取基底统计信息"""
        stats = {
            'base_height': self.substrate_geometry.base_height,
            'min_height': float(np.min(h_substrate)),
            'max_height': float(np.max(h_substrate)),
            'mean_height': float(np.mean(h_substrate)),
            'height_range': float(np.max(h_substrate) - np.min(h_substrate)),
            'num_defects': len(self.substrate_geometry.rectangular_defects),
            'defect_details': []
        }

        # 计算每个缺陷的统计
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)

        for defect in self.substrate_geometry.rectangular_defects:
            mask = self._create_rectangular_mask(X_mesh, Y_mesh, defect)
            defect_area = np.sum(mask) * self.geometry_config['dx'] * self.geometry_config['dy']

            defect_stats = {
                'name': defect.name,
                'height_offset': defect.height_offset,
                'area': float(defect_area),
                'x_range': [min(defect.x1, defect.x2), max(defect.x1, defect.x2)],
                'y_range': [min(defect.y1, defect.y2), max(defect.y1, defect.y2)],
                'type': 'elevation' if defect.height_offset > 0 else 'depression'
            }
            stats['defect_details'].append(defect_stats)

        return stats

    def visualize_substrate(self, h_substrate: np.ndarray,
                            x_grid: np.ndarray, y_grid: np.ndarray,
                            save_path: str = None) -> None:
        """可视化基底形状"""
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)

        fig = plt.figure(figsize=(14, 10))

        # 3D表面图
        ax1 = fig.add_subplot(221, projection='3d')
        surf = ax1.plot_surface(X_mesh, Y_mesh, h_substrate,
                                cmap='terrain', alpha=0.9)
        ax1.set_xlabel('X Position [nm]')
        ax1.set_ylabel('Y Position [nm]')
        ax1.set_zlabel('Height [nm]')
        ax1.set_title('3D Substrate Topology')
        plt.colorbar(surf, ax=ax1, shrink=0.6)

        # 2D等高线图
        ax2 = fig.add_subplot(222)
        contour = ax2.contourf(X_mesh, Y_mesh, h_substrate, levels=20, cmap='terrain')
        ax2.set_xlabel('X Position [nm]')
        ax2.set_ylabel('Y Position [nm]')
        ax2.set_title('Substrate Height Contour')
        ax2.set_aspect('equal')
        plt.colorbar(contour, ax=ax2)

        # 标记矩形特征
        for defect in self.substrate_geometry.rectangular_defects:
            x_min, x_max = min(defect.x1, defect.x2), max(defect.x1, defect.x2)
            y_min, y_max = min(defect.y1, defect.y2), max(defect.y1, defect.y2)

            color = 'red' if defect.height_offset > 0 else 'blue'
            ax2.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                        fill=False, edgecolor=color, linewidth=2))

            # 添加标签
            center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
            ax2.text(center_x, center_y, defect.name,
                     ha='center', va='center', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # 横截面图
        ax3 = fig.add_subplot(223)
        mid_y_idx = len(y_grid) // 2
        ax3.plot(x_grid, h_substrate[mid_y_idx, :], 'b-', linewidth=2)
        ax3.set_xlabel('X Position [nm]')
        ax3.set_ylabel('Height [nm]')
        ax3.set_title(f'X-Direction Cross Section (Y={y_grid[mid_y_idx]:.1f} nm)')
        ax3.grid(True, alpha=0.3)

        # 纵截面图
        ax4 = fig.add_subplot(224)
        mid_x_idx = len(x_grid) // 2
        ax4.plot(y_grid, h_substrate[:, mid_x_idx], 'r-', linewidth=2)
        ax4.set_xlabel('Y Position [nm]')
        ax4.set_ylabel('Height [nm]')
        ax4.set_title(f'Y-Direction Cross Section (X={x_grid[mid_x_idx]:.1f} nm)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 基底可视化已保存: {save_path}")

        plt.show()

    def print_substrate_summary(self, stats: Dict) -> None:
        """打印基底配置摘要"""
        print("\n" + "=" * 60)
        print("🏗️  基底几何配置摘要")
        print("=" * 60)

        print(f"基准平面高度: {stats['base_height']:.2f} nm")
        print(f"最小高度: {stats['min_height']:.2f} nm")
        print(f"最大高度: {stats['max_height']:.2f} nm")
        print(f"高度范围: {stats['height_range']:.2f} nm")
        print(f"矩形特征数量: {stats['num_defects']}")

        if stats['num_defects'] > 0:
            print(f"\n📋 矩形特征详情:")
            for i, defect in enumerate(stats['defect_details']):
                defect_type = "凸起" if defect['type'] == 'elevation' else "凹陷"
                print(f"  {i + 1}. {defect['name']}: {defect_type} {abs(defect['height_offset']):.2f} nm")
                print(f"     区域: X=[{defect['x_range'][0]:.1f}, {defect['x_range'][1]:.1f}] nm, "
                      f"Y=[{defect['y_range'][0]:.1f}, {defect['y_range'][1]:.1f}] nm")
                print(f"     面积: {defect['area']:.0f} nm²")

        print("=" * 60)


def create_example_substrate_config() -> Dict:
    """创建示例基底配置"""
    return {
        'substrate_geometry': {
            'base_height': 0.0,  # 基准平面高度
            'rectangular_defects': [
                {
                    'name': 'Central_Elevation',
                    'x1': -15, 'y1': -15, 'x2': 15, 'y2': 15,
                    'height_offset': 8.0  # 8nm 凸起
                },
                {
                    'name': 'Left_Depression',
                    'x1': -35, 'y1': -10, 'x2': -25, 'y2': 10,
                    'height_offset': -5.0  # 5nm 凹陷
                },
                {
                    'name': 'Right_Step',
                    'x1': 25, 'y1': -8, 'x2': 35, 'y2': 8,
                    'height_offset': 3.0  # 3nm 台阶
                },
                {
                    'name': 'Top_Trench',
                    'x1': -20, 'y1': 20, 'x2': 20, 'y2': 30,
                    'height_offset': -2.0  # 2nm 沟槽
                }
            ]
        }
    }


def validate_substrate_geometry_config(config: Dict) -> bool:
    """验证基底几何配置（独立函数）"""
    try:
        if 'substrate_geometry' not in config:
            print("✓ 使用默认平面基底（无自定义几何）")
            return True

        generator = SubstrateGeometryGenerator(config)
        return generator.validate_substrate_config()

    except Exception as e:
        print(f"❌ 基底几何配置验证失败: {e}")
        return False


if __name__ == "__main__":
    """示例用法"""
    print("🔬 FEBID基底几何模块演示")

    # 创建示例配置
    example_config = {
        'geometry': {
            'X_min': -50, 'X_max': 50, 'Y_min': -50, 'Y_max': 50,
            'dx': 1, 'dy': 1
        }
    }
    example_config.update(create_example_substrate_config())

    # 创建基底生成器
    generator = SubstrateGeometryGenerator(example_config)

    # 验证配置
    if generator.validate_substrate_config():
        # 生成网格
        x_grid = np.arange(-50, 51, 1, dtype=FLOAT_DTYPE)
        y_grid = np.arange(-50, 51, 1, dtype=FLOAT_DTYPE)

        # 生成基底
        h_substrate = generator.generate_substrate_surface(x_grid, y_grid)

        # 获取统计信息
        stats = generator.get_substrate_statistics(h_substrate, x_grid, y_grid)
        generator.print_substrate_summary(stats)

        # 可视化
        generator.visualize_substrate(h_substrate, x_grid, y_grid,
                                      'substrate_geometry_example.png')

        print("✅ 基底几何模块演示完成")
    else:
        print("❌ 配置验证失败")
