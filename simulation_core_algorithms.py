#!/usr/bin/env python3
"""
FEBID仿真核心算法模块 - 精简版
只包含Numba优化的底层算法，删除所有备用方法

Author: 刘宇
Date: 2025/7
"""

import numpy as np
from numba import jit, prange

from base_classes import laplace_2d_parallel


@jit(nopython=True, fastmath=True)
def compute_surface_distance_numba(h_flat, start_y, start_x, end_y, end_x,
                                   dx, dy, grid_shape):
    """
    计算表面3D距离
    """
    ny, nx = grid_shape
    total_distance = 0.0

    # 计算路径采样点数
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    max_steps = max(abs(delta_x), abs(delta_y))

    if max_steps == 0:
        return 0.0

    # 获取起始高度
    start_idx = start_y * nx + start_x
    if start_idx < 0 or start_idx >= len(h_flat):
        return 0.0
    prev_height = h_flat[start_idx]

    # 沿路径计算3D距离
    for step in range(1, max_steps + 1):
        curr_x = start_x + int(step * delta_x / max_steps)
        curr_y = start_y + int(step * delta_y / max_steps)

        if curr_y < 0 or curr_y >= ny or curr_x < 0 or curr_x >= nx:
            break

        curr_idx = curr_y * nx + curr_x
        if curr_idx >= len(h_flat):
            break

        curr_height = h_flat[curr_idx]

        # 3D距离计算
        horizontal_dist = np.sqrt((dx * delta_x / max_steps) ** 2 + (dy * delta_y / max_steps) ** 2)
        vertical_dist = abs(curr_height - prev_height)
        total_distance += np.sqrt(horizontal_dist ** 2 + vertical_dist ** 2)

        prev_height = curr_height

    return total_distance


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def calculate_quad_gaussian_flux_numba(X_flat, Y_flat, h_flat, beam_pos_x, beam_pos_y,
                                       sub_params, dep_params, z_deposit,
                                       enable_surface_propagation, dx, dy, grid_shape):
    """
    并行四高斯通量计算
    """
    n_points = len(X_flat)
    f_surface = np.zeros(n_points, dtype=np.float32)

    beam_x = np.float32(beam_pos_x)
    beam_y = np.float32(beam_pos_y)
    z_dep = np.float32(z_deposit)
    dx_f = np.float32(dx)
    dy_f = np.float32(dy)

    ny, nx = grid_shape

    # 找到束流位置索引
    beam_idx_x = -1
    beam_idx_y = -1
    min_dist_sq = np.float32(1e10)

    for idx in range(n_points):
        dist_sq = (X_flat[idx] - beam_x) ** 2 + (Y_flat[idx] - beam_y) ** 2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            beam_idx_y = idx // nx
            beam_idx_x = idx % nx

    for idx in prange(n_points):
        x_pos = X_flat[idx]
        y_pos = Y_flat[idx]
        h_val = h_flat[idx]

        # 计算距离
        if enable_surface_propagation:
            curr_y = idx // nx
            curr_x = idx % nx

            if curr_x == beam_idx_x and curr_y == beam_idx_y:
                effective_distance = 0.0
            else:
                effective_distance = compute_surface_distance_numba(
                    h_flat, beam_idx_y, beam_idx_x, curr_y, curr_x,
                    dx_f, dy_f, grid_shape
                )
            r_squared = effective_distance * effective_distance
        else:
            dx_pos = x_pos - beam_x
            dy_pos = y_pos - beam_y
            r_squared = dx_pos * dx_pos + dy_pos * dy_pos

        # 材料权重计算
        if h_val <= 0:
            weight_substrate = 1.0
            weight_deposit = 0.0
        elif h_val < z_dep:
            inv_z_dep = 1.0 / z_dep
            weight_substrate = (z_dep - h_val) * inv_z_dep
            weight_deposit = h_val * inv_z_dep
        else:
            weight_substrate = 0.0
            weight_deposit = 1.0

        # 基底通量
        f_substrate = np.float32(0.0)
        for i in range(4):
            sigma = sub_params[i * 2]
            amplitude = sub_params[i * 2 + 1]
            sigma_sq = sigma * sigma
            f_substrate += amplitude * np.exp(-2.0 * r_squared / sigma_sq)

        # 沉积物通量
        f_deposit = np.float32(0.0)
        for i in range(4):
            sigma = dep_params[i * 2]
            amplitude = dep_params[i * 2 + 1]
            sigma_sq = sigma * sigma
            f_deposit += amplitude * np.exp(-2.0 * r_squared / sigma_sq)

        # 加权通量
        f_incident = weight_substrate * f_substrate + weight_deposit * f_deposit
        f_surface[idx] = max(f_incident, 0.0)

    return f_surface


@jit(nopython=True, parallel=True, fastmath=True)
def compute_area_factors(h_surface, dx, dy):
    """计算每个网格点的表面积增大因子"""
    ny, nx = h_surface.shape
    area_factors = np.ones_like(h_surface)

    for i in prange(1, ny - 1):
        for j in range(1, nx - 1):
            # 计算梯度
            grad_x = (h_surface[i, j + 1] - h_surface[i, j - 1]) / (2.0 * dx)
            grad_y = (h_surface[i + 1, j] - h_surface[i - 1, j]) / (2.0 * dy)

            # 面积因子 = sqrt(1 + |∇h|²)
            area_factors[i, j] = np.sqrt(1.0 + grad_x * grad_x + grad_y * grad_y)

    # 边界处理（保持为1）
    area_factors[0, :] = 1.0
    area_factors[-1, :] = 1.0
    area_factors[:, 0] = 1.0
    area_factors[:, -1] = 1.0

    return area_factors

@jit(nopython=True, parallel=True, fastmath=True)
def rk4_step_parallel(n_old, f_surface, dt, k_phi_n0, tau_inv, sigma, n0_inv,
                      D_surf_factor, dx, dy, h_surface=None):  # 添加h_surface参数
    """并行RK4步进 - 增强版（支持斜面吸附）"""

    def compute_rhs(n_current, f_surf):
        # 如果提供了高度场，计算面积因子
        if h_surface is not None:
            area_factors = compute_area_factors(h_surface, dx, dy)
            adsorption = k_phi_n0 * (1.0 - n_current * n0_inv) * area_factors
            desorption = n_current * tau_inv * area_factors  # 解吸也按面积增强
        else:
            # 兼容旧代码
            adsorption = k_phi_n0 * (1.0 - n_current * n0_inv)
            desorption = n_current * tau_inv

        dissociation = sigma * f_surf * n_current
        reaction = adsorption - desorption - dissociation
        diffusion = D_surf_factor * laplace_2d_parallel(n_current, dx, dy)
        return reaction + diffusion

    k1 = dt * compute_rhs(n_old, f_surface)
    k2 = dt * compute_rhs(n_old + 0.5 * k1, f_surface)
    k3 = dt * compute_rhs(n_old + 0.5 * k2, f_surface)
    k4 = dt * compute_rhs(n_old + k3, f_surface)

    return n_old + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


@jit(nopython=True, fastmath=True)
def exponential_slope_enhancement(slope, slope_min=0.1, slope_max=10.0):
    """
    斜率增强效应：基于物理的分段函数
    - 0.1 ~ 5.6: 使用 1/cos(ax+b) 形式
    - 5.6 ~ tan(85°): 线性增长到90%
    - tan(85°) ~ tan(89°): 线性下降到1%
    - > tan(89°): 返回0

    注：保留slope_max参数以保持接口兼容性，但实际使用固定的分段点
    """
    # 关键斜率点
    slope_transition1 = 5.6
    slope_85deg = 11.43  # np.tan(np.radians(85))
    slope_89deg = 57.29  # np.tan(np.radians(89))

    # 增强上限
    target_value_at_5_6 = 2.5

    if slope < slope_min:
        return 1.0

    # 第一段：0.1 到 5.6，使用 1/cos(ax+b) 形式
    elif slope_min <= slope <= slope_transition1:
        # 计算参数a和b
        # arccos(1/2.5) ≈ 1.159279
        a = 1.159279 / 5.5  # ≈ 0.2108
        b = -0.1 * a  # ≈ -0.02108

        # 计算增强因子
        cos_arg = a * slope + b
        cos_value = np.cos(cos_arg)

        # 避免除零错误
        if abs(cos_value) < 1e-10:
            return target_value_at_5_6
        return 1.0 / cos_value

    # 第二段：tan(80°) 到 tan(85°)，线性增长到前一段终值的90%
    elif slope_transition1 < slope <= slope_85deg:
        # 计算5.6处的值（应该接近2.5）
        a = 1.159279 / 5.5
        b = -0.1 * a
        cos_val = np.cos(a * slope_transition1 + b)
        if abs(cos_val) < 1e-10:
            value_at_5_6 = target_value_at_5_6
        else:
            value_at_5_6 = 1.0 / cos_val

        # 线性插值
        value_at_85deg = value_at_5_6 * 0.9
        t = (slope - slope_transition1) / (slope_85deg - slope_transition1)
        return value_at_5_6 + t * (value_at_85deg - value_at_5_6)

    # 第三段：tan(85°) 到 tan(89°)，线性下降到1%
    elif slope_85deg < slope <= slope_89deg:
        # 计算tan(85°)处的值
        a = 1.159279 / 5.5
        b = -0.1 * a
        cos_val = np.cos(a * slope_transition1 + b)
        if abs(cos_val) < 1e-10:
            value_at_5_6 = target_value_at_5_6
        else:
            value_at_5_6 = 1.0 / cos_val
        value_at_85deg = value_at_5_6 * 0.9

        # 线性插值下降到1%
        value_at_89deg = value_at_85deg * 0.01
        t = (slope - slope_85deg) / (slope_89deg - slope_85deg)
        return value_at_85deg + t * (value_at_89deg - value_at_85deg)

    # 第四段：大于tan(89°)
    else:
        return 0.0   #使用三段函数


@jit(nopython=True, parallel=True, fastmath=True)
def apply_surface_effects_numba(f_surface, h_surface, weight_substrate, weight_deposit,
                                sub_sigma1, dep_sigma1, depth_scale_factor, dx, dy,
                                slope_min=0.1, slope_max=10.0):
    """
    连续表面效应 - 删除gradient_factor，使用指数衰减

    Parameters:
    -----------
    f_surface : np.ndarray
        入射电子通量
    h_surface : np.ndarray
        表面高度
    weight_substrate, weight_deposit : np.ndarray
        材料权重
    sub_sigma1, dep_sigma1 : float
        基底和沉积物的第一个高斯参数sigma
    depth_scale_factor : float
        深度衰减尺度因子
    dx, dy : float
        网格间距
    slope_min, slope_max : float
        指数衰减的斜率范围
    flux_min_factor : float
        最小通量因子
    """
    ny, nx = f_surface.shape
    result = np.zeros_like(f_surface)

    for i in prange(ny):
        for j in range(nx):
            # 表面梯度计算
            grad_x = 0.0
            grad_y = 0.0

            if 0 < i < ny - 1:
                grad_y = (h_surface[i + 1, j] - h_surface[i - 1, j]) / (2.0 * dy)
            if 0 < j < nx - 1:
                grad_x = (h_surface[i, j + 1] - h_surface[i, j - 1]) / (2.0 * dx)

            surface_gradient = np.sqrt(grad_x * grad_x + grad_y * grad_y)
            #指数斜率衰减
            flux_decay_factor = exponential_slope_enhancement(surface_gradient,
                                                        slope_min, slope_max)

            # 深度因子
            reference_sigma = (weight_substrate[i, j] * sub_sigma1 +
                               weight_deposit[i, j] * dep_sigma1)
            if reference_sigma == 0:
                reference_sigma = sub_sigma1

            depth_for_attenuation = max(h_surface[i, j], 0.0)
            depth_factor = np.exp(-depth_for_attenuation /
                                  (depth_scale_factor * reference_sigma))

            # 最终通量：，使用flux_decay_factor和depth_factor
            result[i, j] = f_surface[i, j] * flux_decay_factor * depth_factor

    return result
