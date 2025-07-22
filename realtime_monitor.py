#!/usr/bin/env python3
"""
FEBID仿真实时监控模块
包含传统监控器和增强实时Web监控器

Author: 刘宇
Date: 2025/7
"""

import numpy as np
import json
import webbrowser
import os
import threading
import time
from collections import deque
import gc
from typing import Tuple, Dict, List, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver

from data_structures import FLOAT_DTYPE
from base_classes import BaseMonitor
from monitor_utils import ZAxisCalculator, MonitorHTMLGenerator, MonitorDataProcessor
import monitor_templates as templates


class FixedRangeRealTimeMonitor(BaseMonitor):
    """固定范围实时监控类 - 优化版 + Z轴自适应"""

    def __init__(self, x_grid: np.ndarray, y_grid: np.ndarray,
                 geometry_config: Dict, monitor_config: Dict,
                 save_interval: int = 50, max_memory_frames: int = 200,
                 simulation_config: Dict = None):
        """初始化固定范围实时监控器"""
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.geometry_config = geometry_config
        self.simulation_config = simulation_config or {}

        # 从 monitor_config 获取参数
        self.save_interval = save_interval or monitor_config.get('save_interval', 10)
        self.max_memory_frames = monitor_config.get('max_memory_frames', max_memory_frames)

        # 初始化 viz_config
        self.viz_config = {
            'height_range': monitor_config.get('height_range', [0, 8e-4]),
            'precursor_range': monitor_config.get('precursor_range', [0, 4e-4])
        }

        # Z轴自适应计算
        if simulation_config:
            adaptive_range, adjustment_info = ZAxisCalculator.calculate_adaptive_z_range(
                simulation_config, monitor_config
            )
            self.viz_config['height_range'] = adaptive_range
            print(f"📊 Z轴自适应: {adjustment_info}")
        else:
            print("📊 Z轴范围: 使用默认配置（未提供仿真配置）")

        # 网格信息
        self.X_mesh, self.Y_mesh = np.meshgrid(x_grid, y_grid)
        self.grid_shape = self.X_mesh.shape

        # 固定坐标轴范围
        self.x_axis_range = [self.geometry_config['X_min'], self.geometry_config['X_max']]
        self.y_axis_range = [self.geometry_config['Y_min'], self.geometry_config['Y_max']]

        # 时间序列数据存储
        self.time_data = self._initialize_time_data()

        print(f"✓ 传统监控器初始化完成（Z轴自适应）")
        self._print_monitor_info()

    def _initialize_time_data(self):
        """初始化时间序列数据结构"""
        return {
            'pixel_indices': [],
            'timestamps': [],
            'beam_positions': [],
            'max_heights': [],
            'mean_heights': [],
            'h_surface_snapshots': deque(maxlen=self.max_memory_frames),
            'n_surface_snapshots': deque(maxlen=self.max_memory_frames),
            'snapshot_pixel_indices': deque(maxlen=self.max_memory_frames)
        }

    def _print_monitor_info(self):
        """打印监控器信息"""
        print(f"📐 坐标轴范围: X={self.x_axis_range} nm, Y={self.y_axis_range} nm")
        print(f"🎨 颜色范围: Height={self.viz_config['height_range']} nm, "
              f"Precursor={self.viz_config['precursor_range']} molecules/nm²")
        print(f"💾 保存间隔: 每 {self.save_interval} 个像素保存一次快照")
        print(f"🔒 Z轴范围已锁定，监控过程中不再调整")

    def update_data(self, *args, **kwargs):
        """兼容BaseMonitor的抽象方法"""
        self.record_frame(*args)

    def record_frame(self, pixel_idx: int, h_surface: np.ndarray, n_surface: np.ndarray,
                     beam_pos: Tuple[float, float], timestamp: float):
        """记录当前帧数据"""
        # 记录基本信息
        self.time_data['pixel_indices'].append(pixel_idx)
        self.time_data['timestamps'].append(timestamp)
        self.time_data['beam_positions'].append(beam_pos)
        self.time_data['max_heights'].append(float(np.max(h_surface)))
        self.time_data['mean_heights'].append(float(np.mean(h_surface)))

        # 记录完整表面数据（按间隔保存）
        if pixel_idx % self.save_interval == 0:
            # 使用数据处理器优化内存
            self.time_data['h_surface_snapshots'].append(h_surface.astype(np.float32))
            self.time_data['n_surface_snapshots'].append(n_surface.astype(np.float32))
            self.time_data['snapshot_pixel_indices'].append(pixel_idx)

            if pixel_idx % (self.save_interval * 5) == 0:
                print(f"📸 保存快照: 像素 {pixel_idx}, 当前最大高度 {np.max(h_surface):.3e} nm")

    def launch_viewer(self):
        """实现BaseMonitor的抽象方法"""
        return self.launch_fixed_range_viewer()

    def launch_fixed_range_viewer(self):
        """启动固定范围的Web查看器"""
        if len(self.time_data['h_surface_snapshots']) == 0:
            print("⚠️ 没有快照数据可显示，请确保仿真已生成数据")
            return False

        print(f"🌐 启动传统Web查看器（Z轴自适应）...")
        print(f"📊 可用数据: {len(self.time_data['h_surface_snapshots'])} 个快照")

        try:
            # 转换数据为Web格式
            web_data = self._prepare_web_data()

            # 生成HTML
            html_content = self._create_fixed_range_html(json.dumps(web_data))

            # 保存文件
            html_file = 'febid_realtime_fixed_monitor.html'
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)

            print(f"✓ 传统Web界面已生成: {html_file}")
            webbrowser.open(f"file://{os.path.abspath(html_file)}")
            return True

        except Exception as e:
            print(f"❌ 传统界面创建失败: {e}")
            return False

    def _prepare_web_data(self):
        """准备Web数据"""
        web_data = []
        for i in range(len(self.time_data['h_surface_snapshots'])):
            frame_data = {
                'h_surface': self.time_data['h_surface_snapshots'][i].tolist(),
                'n_surface': self.time_data['n_surface_snapshots'][i].tolist(),
                'pixel_index': self.time_data['snapshot_pixel_indices'][i]
            }
            web_data.append(frame_data)
        return web_data

    def _create_fixed_range_html(self, web_data_json):
        """创建固定范围的Web界面HTML"""
        config_json = json.dumps({
            'x_range': self.x_axis_range,
            'y_range': self.y_axis_range,
            'height_range': self.viz_config['height_range'],
            'precursor_range': self.viz_config['precursor_range'],
            'dx': self.geometry_config['dx'],
            'dy': self.geometry_config['dy'],
            'plot_width': 650,
            'plot_height': 500
        })

        return f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>FEBID Monitor - Traditional Mode (Z-Adaptive)</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    {templates.get_common_styles()}
</head>
<body>
    <div class="header">
        <h1>🔬 FEBID Monitor - Traditional Mode</h1>
        <p>仿真完成后查看结果 | Z轴自适应基底几何</p>
    </div>

    {templates.get_traditional_control_panel()}

    <div class="container">
        <div id="surface-plot" class="plot-container" style="width: 650px; height: 500px;"></div>
        <div id="contour-plot" class="plot-container" style="width: 650px; height: 500px;"></div>
        <div id="precursor-plot" class="plot-container" style="width: 650px; height: 500px;"></div>
        <div id="timeseries-plot" class="plot-container" style="width: 650px; height: 500px;"></div>
    </div>

    <script>
        let frameData = {web_data_json};
        let config = {config_json};
        let currentFrame = 0;
        let isPlaying = false;
        let playInterval;

        {templates.get_plot_javascript()}

        function updateDisplay(frameIndex) {{
            if (!frameData || frameIndex >= frameData.length) return;

            currentFrame = frameIndex;
            const data = frameData[frameIndex];
            document.getElementById('frame-label').textContent = frameIndex + 1;
            document.getElementById('frame-slider').value = frameIndex;

            const coords = generateCoordinates([data.h_surface.length, data.h_surface[0].length]);

            Plotly.newPlot('surface-plot', [{{
                type: 'surface', x: coords.x, y: coords.y, z: data.h_surface,
                colorscale: 'Viridis', cmin: config.height_range[0], cmax: config.height_range[1],
                colorbar: getOptimizedColorbar('Height (nm)', config.height_range)
            }}], get3DLayout('3D Surface Height (Z-Adaptive)'));

            Plotly.newPlot('contour-plot', [{{
                type: 'contour', x: coords.x, y: coords.y, z: data.h_surface,
                colorscale: 'Viridis', zmin: config.height_range[0], zmax: config.height_range[1],
                colorbar: getOptimizedColorbar('Height (nm)', config.height_range)
            }}], getPlotLayout('Height Contour Map'));

            Plotly.newPlot('precursor-plot', [{{
                type: 'contour', x: coords.x, y: coords.y, z: data.n_surface,
                colorscale: 'Cool', zmin: config.precursor_range[0], zmax: config.precursor_range[1],
                colorbar: getOptimizedColorbar('Concentration (molecules/nm²)', config.precursor_range)
            }}], getPlotLayout('Precursor Concentration'));
        }}

        function nextFrame() {{ if (currentFrame < frameData.length - 1) updateDisplay(currentFrame + 1); }}
        function previousFrame() {{ if (currentFrame > 0) updateDisplay(currentFrame - 1); }}

        function playAnimation() {{
            if (isPlaying) return;
            isPlaying = true;
            playInterval = setInterval(() => {{
                if (currentFrame < frameData.length - 1) nextFrame();
                else pauseAnimation();
            }}, 1000);
        }}

        function pauseAnimation() {{
            isPlaying = false;
            if (playInterval) clearInterval(playInterval);
        }}

        document.getElementById('frame-slider').addEventListener('input', e => {{
            updateDisplay(parseInt(e.target.value));
        }});

        window.addEventListener('load', () => {{
            if (frameData.length > 0) {{
                document.getElementById('total-frames').textContent = frameData.length;
                document.getElementById('frame-slider').max = frameData.length - 1;
                updateDisplay(0);
            }}
        }});
    </script>
</body>
</html>
        '''

    def close(self):
        """关闭监控器"""
        gc.collect()
        print("✓ 传统监控器已关闭")


class RealTimeWebMonitor(BaseMonitor):
    """增强的实时Web监控类 - 优化版 + Z轴自适应"""

    def __init__(self, x_grid: np.ndarray, y_grid: np.ndarray,
                 geometry_config: Dict, monitor_config: Dict,
                 save_interval: int = 10, max_memory_frames: int = 8000,
                 web_port: int = 8888, simulation_config: Dict = None):
        """初始化增强实时Web监控器"""
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.geometry_config = geometry_config
        self.simulation_config = simulation_config or {}
        self.web_port = web_port
        self.data_processor = MonitorDataProcessor()  # 使用数据处理器

        # 从 monitor_config 获取参数
        self.save_interval = save_interval or monitor_config.get('save_interval', 10)
        self.max_memory_frames = monitor_config.get('max_memory_frames', max_memory_frames)

        # 初始化 viz_config
        self.viz_config = {
            'height_range': monitor_config.get('height_range', [0, 8e-4]),
            'precursor_range': monitor_config.get('precursor_range', [0, 4e-4])
        }
        self.z_adjustment_info = "使用默认Z轴范围"

        # Z轴自适应计算
        if simulation_config:
            adaptive_range, adjustment_info = ZAxisCalculator.calculate_adaptive_z_range(
                simulation_config, monitor_config
            )
            self.viz_config['height_range'] = adaptive_range
            self.z_adjustment_info = adjustment_info
            print(f"📊 Z轴自适应: {adjustment_info}")
        else:
            print("📊 Z轴范围: 使用默认配置（未提供仿真配置）")

        # 网格信息
        self.x_axis_range = [geometry_config['X_min'], geometry_config['X_max']]
        self.y_axis_range = [geometry_config['Y_min'], geometry_config['Y_max']]

        # 计算扫描区域比例
        self.scan_aspect_ratio = self._calculate_scan_aspect_ratio(geometry_config)

        # 实时数据存储
        self.current_data = self._initialize_current_data()

        # 历史快照存储
        self.snapshots = self._initialize_snapshots()

        self.data_lock = threading.Lock()
        self.server = None
        self.server_thread = None

        # 历史数据
        self.history_data = {
            'pixel_indices': deque(maxlen=1000),
            'max_heights': deque(maxlen=1000),
            'timestamps': deque(maxlen=1000)
        }

        self._print_monitor_info()

    def _initialize_current_data(self):
        """初始化当前数据结构"""
        return {
            'pixel_index': 0, 'timestamp': 0.0, 'beam_position': [0.0, 0.0],
            'max_height': 0.0, 'mean_height': 0.0, 'h_surface': [], 'n_surface': [],
            'simulation_progress': 0.0, 'total_pixels': 0, 'is_running': False
        }

    def _initialize_snapshots(self):
        """初始化快照存储"""
        return {
            'h_surface_snapshots': deque(maxlen=self.max_memory_frames),
            'n_surface_snapshots': deque(maxlen=self.max_memory_frames),
            'snapshot_metadata': deque(maxlen=self.max_memory_frames)
        }

    def _print_monitor_info(self):
        """打印监控器信息"""
        print(f"✓ 增强实时Web监控器初始化完成（Z轴自适应）")
        print(f"🌐 Web服务器端口: {self.web_port}")
        print(f"📊 更新间隔: 每 {self.save_interval} 个像素")
        print(f"📸 最大快照数: {self.max_memory_frames}")
        print(f"📐 扫描比例: {self.scan_aspect_ratio:.2f}")
        print(f"🎨 Z轴范围: {self.viz_config['height_range']} nm")
        print(f"🔒 Z轴范围已锁定，监控过程中不再调整")

    def _calculate_scan_aspect_ratio(self, config):
        """计算扫描区域的长宽比"""
        if 'scan_x_start' in config:
            x_range = config['scan_x_end'] - config['scan_x_start']
            y_range = config['scan_y_end'] - config['scan_y_start']
        else:
            x_range = config['X_max'] - config['X_min']
            y_range = config['Y_max'] - config['Y_min']

        return x_range / y_range if y_range != 0 else 1.0

    def start_web_server(self):
        """启动Web服务器"""
        monitor_instance = self

        class RealTimeHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    if self.path == '/':
                        self._serve_html(monitor_instance)
                    elif self.path == '/data':
                        self._serve_data(monitor_instance)
                    elif self.path == '/snapshots':
                        self._serve_snapshots(monitor_instance)
                    else:
                        self.send_response(404)
                        self.end_headers()
                except Exception as e:
                    print(f"❌ HTTP处理错误: {e}")
                    self.send_response(500)
                    self.end_headers()

            def _serve_html(self, monitor):
                """提供HTML页面"""
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.end_headers()
                html_content = monitor.create_realtime_html()
                self.wfile.write(html_content.encode('utf-8'))

            def _serve_data(self, monitor):
                """提供实时数据"""
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()

                with monitor.data_lock:
                    data = monitor.current_data.copy()
                    # 使用数据处理器压缩历史数据
                    data['history'] = monitor.data_processor.compress_history_data({
                        'pixel_indices': list(monitor.history_data['pixel_indices']),
                        'max_heights': list(monitor.history_data['max_heights']),
                        'timestamps': list(monitor.history_data['timestamps'])
                    })

                self.wfile.write(json.dumps(data).encode('utf-8'))

            def _serve_snapshots(self, monitor):
                """提供快照数据"""
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()

                with monitor.data_lock:
                    snapshots_data = []
                    for i in range(len(monitor.snapshots['h_surface_snapshots'])):
                        snapshot = {
                            'h_surface': monitor.snapshots['h_surface_snapshots'][i].tolist(),
                            'n_surface': monitor.snapshots['n_surface_snapshots'][i].tolist(),
                            'metadata': monitor.snapshots['snapshot_metadata'][i]
                        }
                        snapshots_data.append(snapshot)

                self.wfile.write(json.dumps(snapshots_data).encode('utf-8'))

            def log_message(self, format, *args):
                pass  # 禁用日志输出

        try:
            class ReusableTCPServer(socketserver.TCPServer):
                allow_reuse_address = True

            self.server = ReusableTCPServer(("localhost", self.web_port), RealTimeHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()

            print(f"🌐 增强实时Web服务器已启动: http://localhost:{self.web_port}")
            return True

        except Exception as e:
            print(f"❌ Web服务器启动失败: {e}")
            return False

    def update_data(self, pixel_idx: int, h_surface: np.ndarray, n_surface: np.ndarray,
                    beam_pos: Tuple[float, float], timestamp: float,
                    total_pixels: int, is_running: bool = True):
        """更新实时数据并存储历史快照"""
        should_update = (pixel_idx % self.save_interval == 0 or
                         pixel_idx == total_pixels or not is_running)

        if should_update:
            with self.data_lock:
                # 更新当前数据 - 使用float32优化内存
                self.current_data.update({
                    'pixel_index': int(pixel_idx),
                    'timestamp': float(timestamp),
                    'beam_position': [float(beam_pos[0]), float(beam_pos[1])],
                    'max_height': float(np.max(h_surface)),
                    'mean_height': float(np.mean(h_surface)),
                    'h_surface': h_surface.astype(np.float32).tolist(),
                    'n_surface': n_surface.astype(np.float32).tolist(),
                    'simulation_progress': float(pixel_idx / total_pixels * 100) if total_pixels > 0 else 0,
                    'total_pixels': int(total_pixels),
                    'is_running': is_running
                })

                # 存储快照
                self._store_snapshot(pixel_idx, timestamp, beam_pos, h_surface, n_surface, is_running)

                # 更新历史数据
                self.history_data['pixel_indices'].append(pixel_idx)
                self.history_data['max_heights'].append(float(np.mean(h_surface)))
                self.history_data['timestamps'].append(timestamp)

            if pixel_idx % (self.save_interval * 5) == 0:
                print(f"🔄 实时数据已更新: 像素 {pixel_idx}/{total_pixels}, "
                      f"最大高度 {np.max(h_surface):.3e} nm, "
                      f"快照数: {len(self.snapshots['h_surface_snapshots'])}")

    def _store_snapshot(self, pixel_idx, timestamp, beam_pos, h_surface, n_surface, is_running):
        """存储快照数据 - 优化内存使用"""
        self.snapshots['h_surface_snapshots'].append(h_surface.astype(np.float32))
        self.snapshots['n_surface_snapshots'].append(n_surface.astype(np.float32))
        self.snapshots['snapshot_metadata'].append({
            'pixel_index': int(pixel_idx),
            'timestamp': float(timestamp),
            'beam_position': [float(beam_pos[0]), float(beam_pos[1])],
            'max_height': float(np.max(h_surface)),
            'is_running': is_running
        })

    def record_frame(self, pixel_idx: int, h_surface: np.ndarray, n_surface: np.ndarray,
                     beam_pos: Tuple[float, float], timestamp: float,
                     total_pixels: int = 0, is_running: bool = True):
        """兼容原有接口"""
        self.update_data(pixel_idx, h_surface, n_surface, beam_pos, timestamp,
                         total_pixels, is_running)

    def create_realtime_html(self):
        """创建增强的实时监控HTML页面"""
        config_json = json.dumps({
            'x_range': self.x_axis_range,
            'y_range': self.y_axis_range,
            'height_range': self.viz_config['height_range'],
            'precursor_range': self.viz_config['precursor_range'],
            'dx': self.geometry_config['dx'],
            'dy': self.geometry_config['dy'],
            'aspect_ratio': self.scan_aspect_ratio,
            'plot_width': 650,
            'plot_height': 500
        })

        return f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>FEBID增强实时监控器 (Z-Adaptive)</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    {templates.get_common_styles()}
    <style>
        .plot-container {{ width: 650px; height: 500px; }}
        .history-controls {{ grid-column: 1 / -1; text-align: center; margin: 12px 0; 
                            padding: 12px; background-color: #d4edda; border-radius: 10px; display: none; }}
        .mode-tabs {{ display: flex; justify-content: center; margin-bottom: 10px; }}
        .mode-tab {{ padding: 8px 16px; margin: 0 4px; border: 2px solid #ddd; 
                    background: white; cursor: pointer; border-radius: 5px; font-size: 13px; }}
        .mode-tab.active {{ background: #4CAF50; color: white; border-color: #4CAF50; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 FEBID增强实时监控器</h1>
        <p>实时监控 + 历史回溯 + Z轴自适应基底几何</p>
    </div>

    <div class="status">
        <div><span id="status-indicator" class="status-indicator status-stopped"></span>
             <span id="status-text">等待连接...</span></div>
        <div style="margin-top: 8px;">
            <strong>仿真统计:</strong> 像素: <span id="pixel-count">-</span> |
            进度: <span id="progress">-</span>% | 平均高度: <span id="max-height">-</span> nm |
            束流位置: (<span id="beam-x">-</span>, <span id="beam-y">-</span>) nm |
            快照数: <span id="snapshot-count">-</span> | Z轴: <span id="z-range">{self.z_adjustment_info}</span>
        </div>
    </div>

    {templates.get_realtime_control_panel()}

    <div class="history-controls" id="history-controls">
        <h3>📸 历史回溯</h3>
        <label><strong>快照: <span id="history-frame-label">-</span> / <span id="total-snapshots">-</span></strong></label><br><br>
        <input type="range" id="history-slider" class="slider" min="0" max="0" value="0"><br><br>
        <button class="button" onclick="playHistory()">▶️ 播放历史</button>
        <button class="button" onclick="pauseHistory()">⏸️ 暂停播放</button>
        <button class="button" onclick="previousSnapshot()">⬅️ 上一个</button>
        <button class="button" onclick="nextSnapshot()">➡️ 下一个</button>
        <button class="button secondary" onclick="loadSnapshots()">📥 加载快照</button>
    </div>

    <div class="container">
        <div id="surface-plot" class="plot-container"></div>
        <div id="surface-plot-side" class="plot-container"></div>
        <div id="precursor-plot" class="plot-container"></div>
        <div id="timeseries-plot" class="plot-container"></div>
    </div>

    <script>
        let config = {config_json};
        let isAutoUpdate = true;
        let updateInterval = 2000;
        let updateTimer;
        let currentMode = 'realtime';

        // 历史数据相关
        let snapshotsData = [];
        let currentSnapshotIndex = 0;
        let isPlayingHistory = false;
        let historyPlayTimer;

        {templates.get_plot_javascript()}
        {templates.get_realtime_javascript()}
    </script>
</body>
</html>
        '''

    def launch_viewer(self):
        """实现BaseMonitor的抽象方法"""
        return self.launch_realtime_viewer()

    def launch_realtime_viewer(self):
        """启动实时查看器"""
        if self.start_web_server():
            url = f"http://localhost:{self.web_port}"
            webbrowser.open(url)
            print(f"🌐 增强实时监控界面已启动: {url}")
            print("🎯 新功能: 双3D视角 + 历史回溯 + Z轴自适应基底几何 + 固定Z轴范围")
            return True
        else:
            return False

    def stop_server(self):
        """停止Web服务器"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("🛑 增强实时Web服务器已停止")

    def close(self):
        """关闭监控器"""
        self.stop_server()
        print("✓ 增强实时监控器已关闭")
