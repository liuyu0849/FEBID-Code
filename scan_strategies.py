#!/usr/bin/env python3
"""
FEBID仿真扫描策略模块 - 优化版
使用策略模式和并行计算优化

Author: 刘宇
Date: 2025/7
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Set, Callable
from dataclasses import dataclass
from scipy.spatial import KDTree

from data_structures import ScanInfo, FLOAT_DTYPE
from base_classes import ScanStrategy, compute_edge_mask_parallel


@dataclass
class ScanGridInfo:
    """扫描网格信息"""
    x_coords: np.ndarray
    y_coords: np.ndarray
    nx_total: int
    ny_total: int
    edge_layers: int
    dx: float
    dy: float


class RasterScanStrategy(ScanStrategy):
    """光栅扫描策略"""

    def generate_path(self, x_positions: np.ndarray, y_positions: np.ndarray) -> np.ndarray:
        nx, ny = len(x_positions), len(y_positions)
        scan_path = np.zeros((nx * ny, 2), dtype=FLOAT_DTYPE)

        idx = 0
        for j in range(ny):
            for i in range(nx):
                scan_path[idx] = [x_positions[i], y_positions[j]]
                idx += 1

        return scan_path

    @property
    def description(self) -> str:
        return "光栅扫描：逐行从左到右扫描"


class SerpentineScanStrategy(ScanStrategy):
    """蛇形扫描策略"""

    def generate_path(self, x_positions: np.ndarray, y_positions: np.ndarray) -> np.ndarray:
        nx, ny = len(x_positions), len(y_positions)
        scan_path = np.zeros((nx * ny, 2), dtype=FLOAT_DTYPE)

        idx = 0
        for j in range(ny):
            if j % 2 == 0:  # 偶数行：从左到右
                for i in range(nx):
                    scan_path[idx] = [x_positions[i], y_positions[j]]
                    idx += 1
            else:  # 奇数行：从右到左
                for i in range(nx - 1, -1, -1):
                    scan_path[idx] = [x_positions[i], y_positions[j]]
                    idx += 1

        return scan_path

    @property
    def description(self) -> str:
        return "蛇形扫描：奇数行从左到右，偶数行从右到左"


class SquareSpiralStrategy(ScanStrategy):
    """方形螺旋扫描策略"""

    def __init__(self, direction: str = 'in2out'):
        self.direction = direction

    def generate_path(self, x_positions: np.ndarray, y_positions: np.ndarray) -> np.ndarray:
        nx, ny = len(x_positions), len(y_positions)

        # 创建坐标网格并计算层级
        layers = self._compute_layers(nx, ny, x_positions, y_positions)

        # 按层级排序
        sorted_layers = sorted(layers.keys())
        if self.direction == 'out2in':
            sorted_layers = sorted_layers[::-1]

        # 生成扫描路径
        scan_path = []
        for layer in sorted_layers:
            layer_points = layers[layer]
            if len(layer_points) == 1:
                scan_path.append([layer_points[0][2], layer_points[0][3]])
            else:
                sorted_points = self._sort_square_layer_points(layer_points, nx, ny)
                for _, _, x, y in sorted_points:
                    scan_path.append([x, y])

        return np.array(scan_path, dtype=FLOAT_DTYPE)

    def _compute_layers(self, nx: int, ny: int, x_positions: np.ndarray,
                        y_positions: np.ndarray) -> Dict:
        """计算每个点的层级"""
        layers = {}

        for j in range(ny):
            for i in range(nx):
                # 到四个边界的距离
                layer = min(i, nx - 1 - i, j, ny - 1 - j)

                if layer not in layers:
                    layers[layer] = []
                layers[layer].append((i, j, x_positions[i], y_positions[j]))

        return layers

    def _sort_square_layer_points(self, layer_points: List[Tuple], nx: int, ny: int) -> List[Tuple]:
        """对方形层内的点按顺时针方向排序"""
        if len(layer_points) <= 1:
            return layer_points

        # 使用字典按边分类，避免重复判断
        edges = {'top': [], 'right': [], 'bottom': [], 'left': []}

        for point in layer_points:
            i, j, x, y = point

            # 计算到各边的距离
            dists = {
                'left': i,
                'right': nx - 1 - i,
                'top': j,
                'bottom': ny - 1 - j
            }

            min_dist = min(dists.values())

            # 根据最近的边界分类（优先级：上→右→下→左）
            if dists['top'] == min_dist:
                edges['top'].append(point)
            elif dists['right'] == min_dist:
                edges['right'].append(point)
            elif dists['bottom'] == min_dist:
                edges['bottom'].append(point)
            else:
                edges['left'].append(point)

        # 对每条边内部排序
        edges['top'].sort(key=lambda p: p[0])  # 按i排序（从左到右）
        edges['right'].sort(key=lambda p: p[1])  # 按j排序（从上到下）
        edges['bottom'].sort(key=lambda p: -p[0])  # 按i倒序（从右到左）
        edges['left'].sort(key=lambda p: -p[1])  # 按j倒序（从下到上）

        # 顺时针连接
        return edges['top'] + edges['right'] + edges['bottom'] + edges['left']

    @property
    def description(self) -> str:
        direction_text = "从中心向外" if self.direction == 'in2out' else "从外围向中心"
        return f"方形螺旋扫描：{direction_text}"


class CircleSpiralStrategy(ScanStrategy):
    """圆形螺旋扫描策略"""

    def __init__(self, direction: str = 'in2out'):
        self.direction = direction

    def generate_path(self, x_positions: np.ndarray, y_positions: np.ndarray) -> np.ndarray:
        nx, ny = len(x_positions), len(y_positions)

        # 计算扫描区域中心
        center_x = (x_positions[0] + x_positions[-1]) / 2
        center_y = (y_positions[0] + y_positions[-1]) / 2

        # 创建点列表并计算极坐标
        points_with_polar = []
        for j in range(ny):
            for i in range(nx):
                x, y = x_positions[i], y_positions[j]
                r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                theta = np.arctan2(y - center_y, x - center_x)
                points_with_polar.append((r, theta, x, y))

        # 按距离分层
        layers = self._create_distance_layers(points_with_polar)

        # 按策略确定层的顺序
        sorted_indices = sorted(layers.keys())
        if self.direction == 'out2in':
            sorted_indices = sorted_indices[::-1]

        # 生成扫描路径
        scan_path = []
        for idx in sorted_indices:
            # 层内按角度排序
            layer_points = sorted(layers[idx], key=lambda p: p[1])
            for _, _, x, y in layer_points:
                scan_path.append([x, y])

        return np.array(scan_path, dtype=FLOAT_DTYPE)

    def _create_distance_layers(self, points_with_polar: List[Tuple]) -> Dict:
        """创建距离层"""
        if not points_with_polar:
            return {}

        # 获取距离范围
        distances = [p[0] for p in points_with_polar]
        min_r, max_r = min(distances), max(distances)

        # 自适应层数
        n_points = len(points_with_polar)
        num_layers = max(int(np.sqrt(n_points) / 2), 5)

        # 分配到层
        layers = {}
        if max_r > min_r:
            r_step = (max_r - min_r) / num_layers

            for r, theta, x, y in points_with_polar:
                layer_idx = min(int((r - min_r) / r_step), num_layers - 1)
                if layer_idx not in layers:
                    layers[layer_idx] = []
                layers[layer_idx].append((r, theta, x, y))
        else:
            # 所有点在同一位置
            layers[0] = points_with_polar

        return layers

    @property
    def description(self) -> str:
        direction_text = "从中心向外" if self.direction == 'in2out' else "从外围向中心"
        return f"圆形螺旋扫描：{direction_text}"


class ScanPathGenerator:
    """扫描路径生成器 - 优化版"""

    def __init__(self, config: Dict):
        self.config = config
        self.scan_grid_info = None
        self.edge_points_kdtree = None

        # 策略映射
        self.strategies = {
            'raster': RasterScanStrategy(),
            'serpentine': SerpentineScanStrategy(),
            'spiral_square_in2out': SquareSpiralStrategy('in2out'),
            'spiral_square_out2in': SquareSpiralStrategy('out2in'),
            'spiral_circle_in2out': CircleSpiralStrategy('in2out'),
            'spiral_circle_out2in': CircleSpiralStrategy('out2in')
        }

    def _precompute_edge_points_optimized(self, scan_config: Dict) -> np.ndarray:
        """优化的边缘点预计算 - 使用并行计算和KDTree"""
        x_start = scan_config['scan_x_start']
        x_end = scan_config['scan_x_end']
        y_start = scan_config['scan_y_start']
        y_end = scan_config['scan_y_end']
        edge_layers = scan_config['edge_layers']

        pixel_size_x = scan_config['pixel_size_x']
        pixel_size_y = scan_config['pixel_size_y']

        # 生成网格坐标
        x_coords = np.arange(x_start, x_end + pixel_size_x / 2, pixel_size_x)
        y_coords = np.arange(y_start, y_end + pixel_size_y / 2, pixel_size_y)

        nx_total = len(x_coords)
        ny_total = len(y_coords)

        # 检查边缘层数合理性
        max_edge_layers = min(nx_total // 2, ny_total // 2)
        if edge_layers > max_edge_layers:
            print(f"⚠️ 边缘层数 {edge_layers} 超过最大值 {max_edge_layers}，自动调整")
            edge_layers = max_edge_layers

        # 存储网格信息
        self.scan_grid_info = ScanGridInfo(
            x_coords=x_coords,
            y_coords=y_coords,
            nx_total=nx_total,
            ny_total=ny_total,
            edge_layers=edge_layers,
            dx=pixel_size_x,
            dy=pixel_size_y
        )

        if edge_layers <= 0:
            return np.array([])

        print(f"🔧 预计算全局边缘点 (优化版):")
        print(f"   扫描区域: X=[{x_start}, {x_end}], Y=[{y_start}, {y_end}]")
        print(f"   扫描步长: dx={pixel_size_x}, dy={pixel_size_y}")
        print(f"   基础网格尺寸: {nx_total} x {ny_total}")
        print(f"   边缘层数: {edge_layers}")

        # 使用并行计算边缘掩码
        edge_mask = compute_edge_mask_parallel(nx_total, ny_total, edge_layers)

        # 提取边缘点坐标
        edge_indices = np.where(edge_mask)
        edge_points = np.column_stack([
            x_coords[edge_indices[0]],
            y_coords[edge_indices[1]]
        ])

        # 创建KDTree以便快速查询
        if len(edge_points) > 0:
            self.edge_points_kdtree = KDTree(edge_points)

        print(f"✓ 预计算完成: 找到 {len(edge_points)} 个全局边缘点")

        # 显示边缘点分布信息
        if len(edge_points) > 0:
            print(f"   边缘X坐标范围: [{edge_points[:, 0].min():.1f}, {edge_points[:, 0].max():.1f}]")
            print(f"   边缘Y坐标范围: [{edge_points[:, 1].min():.1f}, {edge_points[:, 1].max():.1f}]")

        return edge_points

    def is_edge_point_fast(self, x: float, y: float, tolerance: float = 1e-6) -> bool:
        """快速判断是否为边缘点 - 使用KDTree"""
        if self.edge_points_kdtree is None:
            return False

        # 查询最近邻
        distance, _ = self.edge_points_kdtree.query([x, y])
        return distance < tolerance

    def generate_scan_positions(self, scan_params: Dict) -> Tuple[np.ndarray, ScanInfo]:
        """生成扫描位置 - 优化版"""
        # 预计算边缘点
        edge_points = self._precompute_edge_points_optimized(scan_params)

        # 验证循环参数
        total_subloops, scan_mode = self._validate_and_get_loop_params(scan_params)

        print(f"🔧 时间重复边缘加强模式: {scan_mode}")
        print(f"🎯 边缘重复策略: 停留时间延长 {scan_params['edge_repeat_times'] + 1}x")

        # 生成扫描位置
        scan_positions = self._generate_scan_positions_optimized(
            scan_params, total_subloops, edge_points
        )

        # 创建扫描信息
        scan_info = self._create_scan_info(scan_params, scan_positions, total_subloops)

        # 分析并显示统计
        self._display_scan_statistics(scan_positions, scan_params['scan_strategy'])

        return scan_positions, scan_info

    def _validate_and_get_loop_params(self, scan_params: Dict) -> Tuple[int, str]:
        """验证并获取循环参数"""
        loop = scan_params.get('loop')
        subloop = scan_params.get('subloop')
        pixel_size_x = scan_params['pixel_size_x']
        pixel_size_y = scan_params['pixel_size_y']

        if loop is not None and subloop is not None:
            raise ValueError("Cannot set both loop and subloop parameters")
        if loop is None and subloop is None:
            raise ValueError("Must set either loop or subloop parameter")

        if loop is not None:
            if loop < 1 or not isinstance(loop, int):
                raise ValueError("loop parameter must be a positive integer")
            total_subloops = loop * pixel_size_x * pixel_size_y
            scan_mode = f"Complete loop mode ({loop} full loops, {pixel_size_x}x{pixel_size_y} steps)"
        else:
            if subloop < 1 or not isinstance(subloop, int):
                raise ValueError("subloop parameter must be a positive integer")
            total_subloops = subloop
            scan_mode = f"Subloop mode ({subloop} subloops)"

        return total_subloops, scan_mode

    def _generate_scan_positions_optimized(self, scan_params: Dict, total_subloops: int,
                                           edge_points: np.ndarray) -> np.ndarray:
        """优化的扫描位置生成"""
        # 获取基础参数
        x_start = scan_params['scan_x_start']
        x_end = scan_params['scan_x_end']
        y_start = scan_params['scan_y_start']
        y_end = scan_params['scan_y_end']
        pixel_size_x = scan_params['pixel_size_x']
        pixel_size_y = scan_params['pixel_size_y']
        strategy_name = scan_params['scan_strategy']

        # 生成基础位置
        x_base = np.arange(x_start, x_end + pixel_size_x, pixel_size_x, dtype=FLOAT_DTYPE)
        y_base = np.arange(y_start, y_end + pixel_size_y, pixel_size_y, dtype=FLOAT_DTYPE)

        nx_base = len(x_base)
        ny_base = len(y_base)
        base_pixels_per_subloop = nx_base * ny_base

        # 预分配结果数组
        total_pixels = total_subloops * base_pixels_per_subloop
        scan_positions = np.zeros((total_pixels, 4), dtype=FLOAT_DTYPE)

        # 获取扫描策略
        if strategy_name not in self.strategies:
            raise ValueError(f"未知的扫描策略: {strategy_name}")

        strategy = self.strategies[strategy_name]

        # 生成每个子循环
        pixel_index = 0
        total_possible_steps = pixel_size_x * pixel_size_y

        for subloop_idx in range(1, total_subloops + 1):
            # 计算偏移
            step_idx = (subloop_idx - 1) % total_possible_steps
            x_offset = step_idx % pixel_size_x
            y_offset = step_idx // pixel_size_x

            # 生成当前子循环位置
            current_x = x_base + x_offset
            current_y = y_base + y_offset

            # 边界检查
            valid_x = (current_x >= x_start) & (current_x <= x_end)
            valid_y = (current_y >= y_start) & (current_y <= y_end)

            current_x = current_x[valid_x]
            current_y = current_y[valid_y]

            if len(current_x) == 0 or len(current_y) == 0:
                continue

            # 生成扫描路径
            path = strategy.generate_path(current_x, current_y)

            # 标记边缘点并添加到结果
            for x, y in path:
                is_edge = self.is_edge_point_fast(x, y)
                scan_positions[pixel_index] = [x, y, subloop_idx, int(is_edge)]
                pixel_index += 1

        # 裁剪到实际大小
        return scan_positions[:pixel_index]

    def _create_scan_info(self, scan_params: Dict, scan_positions: np.ndarray,
                          total_subloops: int) -> ScanInfo:
        """创建扫描信息对象"""
        nx_pixels = len(self.scan_grid_info.x_coords) if self.scan_grid_info else 0
        ny_pixels = len(self.scan_grid_info.y_coords) if self.scan_grid_info else 0
        base_pixels = nx_pixels * ny_pixels

        return ScanInfo(
            nx_pixels=nx_pixels,
            ny_pixels_per_subloop=ny_pixels,
            base_pixels_per_subloop=base_pixels,
            pixels_per_subloop=base_pixels,
            total_subloops=total_subloops,
            total_pixels=len(scan_positions),
            pixel_size_x=scan_params['pixel_size_x'],
            pixel_size_y=scan_params['pixel_size_y'],
            edge_layers=scan_params['edge_layers'],
            edge_repeat_times=scan_params['edge_repeat_times']
        )

    def _display_scan_statistics(self, scan_positions: np.ndarray, strategy_name: str):
        """显示扫描统计信息"""
        edge_mask = scan_positions[:, 3] == 1
        edge_points = np.sum(edge_mask)
        total_points = len(scan_positions)

        print(f"✅ 时间重复边缘加强扫描生成完成:")
        print(f"   总点数: {total_points:,}")
        print(f"   边缘点数: {edge_points:,} ({edge_points / total_points * 100:.1f}%)")
        print(f"   扫描策略: {self.strategies[strategy_name].description}")

    def analyze_edge_coverage(self, scan_positions: np.ndarray) -> Dict:
        """分析边缘覆盖情况 - 优化版"""
        edge_mask = scan_positions[:, 3] == 1

        edge_points = np.sum(edge_mask)
        normal_points = np.sum(~edge_mask)
        total_points = len(scan_positions)

        result = {
            'total_points': total_points,
            'edge_points': edge_points,
            'normal_points': normal_points,
            'edge_percentage': edge_points / total_points * 100 if total_points > 0 else 0
        }

        # 计算边缘点的空间分布
        if edge_points > 0:
            edge_positions = scan_positions[edge_mask, :2]
            unique_edge_positions = np.unique(
                np.round(edge_positions, 6), axis=0
            )

            result.update({
                'unique_edge_coords': len(unique_edge_positions),
                'edge_x_range': [edge_positions[:, 0].min(), edge_positions[:, 0].max()],
                'edge_y_range': [edge_positions[:, 1].min(), edge_positions[:, 1].max()]
            })
        else:
            result.update({
                'unique_edge_coords': 0,
                'edge_x_range': [0, 0],
                'edge_y_range': [0, 0]
            })

        return result

    def validate_scan_positions(self, scan_positions: np.ndarray, scan_params: Dict) -> bool:
        """验证扫描位置的有效性 - 向量化版本"""
        try:
            # 提取边界
            x_start = scan_params['scan_x_start']
            x_end = scan_params['scan_x_end']
            y_start = scan_params['scan_y_start']
            y_end = scan_params['scan_y_end']

            # 向量化检查
            x_coords = scan_positions[:, 0]
            y_coords = scan_positions[:, 1]
            subloop_indices = scan_positions[:, 2]
            edge_flags = scan_positions[:, 3]

            # 坐标范围检查
            x_valid = np.all((x_coords >= x_start - 1e-6) & (x_coords <= x_end + 1e-6))
            y_valid = np.all((y_coords >= y_start - 1e-6) & (y_coords <= y_end + 1e-6))

            if not x_valid:
                print(f"❌ X坐标超出范围: [{x_coords.min()}, {x_coords.max()}] vs [{x_start}, {x_end}]")
                return False

            if not y_valid:
                print(f"❌ Y坐标超出范围: [{y_coords.min()}, {y_coords.max()}] vs [{y_start}, {y_end}]")
                return False

            # 子循环索引检查
            if np.any(subloop_indices < 1):
                print("❌ 子循环索引包含无效值（小于1）")
                return False

            # 边缘标记检查
            if not np.all(np.isin(edge_flags, [0, 1])):
                print("❌ 边缘标记包含无效值（必须为0或1）")
                return False

            print("✅ 扫描位置验证通过")
            return True

        except Exception as e:
            print(f"❌ 扫描位置验证失败: {e}")
            return False
