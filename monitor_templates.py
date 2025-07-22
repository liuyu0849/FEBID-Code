#!/usr/bin/env python3
"""
FEBID监控器HTML/JavaScript模板
集中管理所有前端代码模板，减少主模块内存占用

Author: 刘宇
Date: 2025/7
"""


def get_common_styles():
    """获取通用CSS样式"""
    return """
    <style>
        body { font-family: Arial, sans-serif; margin: 12px; background-color: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  color: white; padding: 15px; border-radius: 10px; margin-bottom: 12px; text-align: center; }
        .status { background-color: #e8f5e8; border: 2px solid #4CAF50; border-radius: 8px; 
                  padding: 12px; margin: 12px 0; font-family: monospace; font-size: 13px; }
        .container { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; 
                     background-color: white; padding: 12px; border-radius: 10px; }
        .plot-container { border: 2px solid #ddd; border-radius: 8px; margin: auto; }
        .controls { grid-column: 1 / -1; text-align: center; margin: 12px 0; padding: 12px; 
                    background-color: #fff3cd; border-radius: 10px; }
        .button { background-color: #4CAF50; border: none; color: white; padding: 8px 16px; 
                  margin: 3px; cursor: pointer; border-radius: 5px; font-size: 13px; }
        .button.secondary { background-color: #17a2b8; }
        .slider { width: 80%; height: 18px; margin: 8px; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; 
                           border-radius: 50%; margin-right: 8px; }
        .status-running { background-color: #4CAF50; animation: pulse 2s infinite; }
        .status-stopped { background-color: #f44336; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    </style>
    """


def get_plot_javascript():
    """获取绘图相关的JavaScript函数"""
    return """
    function generateCoordinates(data_shape) {
        const rows = data_shape[0], cols = data_shape[1];
        const x_coords = [], y_coords = [];
        for (let i = 0; i < cols; i++) {
            x_coords.push(config.x_range[0] + (i / (cols - 1)) * (config.x_range[1] - config.x_range[0]));
        }
        for (let j = 0; j < rows; j++) {
            y_coords.push(config.y_range[0] + (j / (rows - 1)) * (config.y_range[1] - config.y_range[0]));
        }
        return {x: x_coords, y: y_coords};
    }

    function getOptimizedColorbar(title, range) {
        return {
            title: {text: title, font: {size: 12}},
            titleside: 'right',
            tickformat: '.2e',
            exponentformat: 'e',
            nticks: 8,
            thickness: 12,
            len: 0.8,
            x: 1.02,
            tickfont: {size: 10}
        };
    }

    function getPlotLayout(title) {
        return {
            title: {text: title, font: {size: 14}},
            width: config.plot_width,
            height: config.plot_height,
            margin: {l: 45, r: 80, t: 50, b: 45},
            font: {size: 11}
        };
    }

    function get3DLayout(title) {
        return {
            title: {text: title, font: {size: 14}},
            width: config.plot_width,
            height: config.plot_height,
            margin: {l: 45, r: 80, t: 50, b: 45},
            scene: {
                xaxis: {title: 'X Position (nm)', range: config.x_range, autorange: false},
                yaxis: {title: 'Y Position (nm)', range: config.y_range, autorange: false},
                zaxis: {
                    title: 'Height (nm)', range: config.height_range,
                    tickformat: '.2e', exponentformat: 'e',
                    autorange: false, fixedrange: true, type: 'linear'
                },
                aspectmode: 'manual',
                aspectratio: {x: 1, y: 1 / config.aspect_ratio, z: 0.6}
            }
        };
    }
    """


def get_realtime_control_panel():
    """获取实时模式控制面板HTML"""
    return """
    <div class="controls">
        <div class="mode-tabs">
            <div class="mode-tab active" id="realtime-tab" onclick="switchMode('realtime')">🔴 实时模式</div>
            <div class="mode-tab" id="history-tab" onclick="switchMode('history')">⏱️ 历史模式</div>
        </div>
        <div id="realtime-controls">
            <h3>📊 实时控制</h3>
            <button class="button" onclick="toggleAutoUpdate()" id="auto-update-btn">⏸️ 暂停自动更新</button>
            <button class="button" onclick="updateNow()">🔄 立即更新</button>
            <button class="button" onclick="resetZoom()">🔍 重置缩放</button>
            <br><br>
            <label>更新间隔: <select id="update-interval" onchange="setUpdateInterval()">
                <option value="1000">1秒</option>
                <option value="2000" selected>2秒</option>
                <option value="5000">5秒</option>
            </select></label>
        </div>
    </div>
    """


def get_traditional_control_panel():
    """获取传统模式控制面板HTML"""
    return """
    <div class="controls">
        <h3>📊 Frame Control</h3>
        <label><strong>Frame: <span id="frame-label">-</span> / <span id="total-frames">-</span></strong></label><br><br>
        <input type="range" id="frame-slider" class="slider" min="0" max="0" value="0"><br><br>
        <button class="button" onclick="playAnimation()">▶️ Play</button>
        <button class="button" onclick="pauseAnimation()">⏸️ Pause</button>
        <button class="button" onclick="previousFrame()">⬅️ Previous</button>
        <button class="button" onclick="nextFrame()">➡️ Next</button>
    </div>
    """


def get_realtime_javascript():
    """获取实时监控的完整JavaScript代码"""
    return """
    function initializePlots() {
        const emptyZ = [[config.height_range[0], config.height_range[0]],
                       [config.height_range[0], config.height_range[0]]];
        const emptyCoords = {x: [config.x_range[0], config.x_range[1]],
                            y: [config.y_range[0], config.y_range[1]]};

        // 初始化各个图表
        const surfaceData = [{
            type: 'surface',
            x: emptyCoords.x,
            y: emptyCoords.y,
            z: emptyZ,
            colorscale: 'Viridis',
            cmin: config.height_range[0],
            cmax: config.height_range[1],
            colorbar: getOptimizedColorbar('Height (nm)', config.height_range)
        }];

        Plotly.newPlot('surface-plot', surfaceData, get3DLayout('3D Surface Height (Z-LOCKED)'), {
            displayModeBar: true,
            scrollZoom: false,
            doubleClick: 'reset'
        });

        // 锁定Z轴范围 - 加强版
        document.getElementById('surface-plot').on('plotly_relayout', function(eventdata) {
            if (eventdata['scene.zaxis.autorange'] === true ||
                (eventdata['scene.zaxis.range'] &&
                 JSON.stringify(eventdata['scene.zaxis.range']) !== JSON.stringify(config.height_range))) {

                setTimeout(() => {
                    Plotly.relayout('surface-plot', {
                        'scene.zaxis.range': config.height_range,
                        'scene.zaxis.autorange': false,
                        'scene.zaxis.fixedrange': true
                    });
                }, 10);
            }
        });

        // 初始化第二个3D图（侧视角）
        Plotly.newPlot('surface-plot-side', surfaceData, {
            ...get3DLayout('3D Surface - Side View'),
            scene: {
                ...get3DLayout('3D Surface - Side View').scene,
                camera: {
                        eye: {x: 0.5, y: 0.2, z: 0.8},
                        up: {x: 0, y: 0, z: 1},
                        center: {x: 0, y: 0, z: 0}
                    }
            }
        }, {
            displayModeBar: true,
            scrollZoom: false,
            doubleClick: 'reset'
        });

        // 为第二个3D图也添加Z轴锁定
        document.getElementById('surface-plot-side').on('plotly_relayout', function(eventdata) {
            if (eventdata['scene.zaxis.autorange'] === true ||
                (eventdata['scene.zaxis.range'] &&
                 JSON.stringify(eventdata['scene.zaxis.range']) !== JSON.stringify(config.height_range))) {

                setTimeout(() => {
                    Plotly.relayout('surface-plot-side', {
                        'scene.zaxis.range': config.height_range,
                        'scene.zaxis.autorange': false,
                        'scene.zaxis.fixedrange': true
                    });
                }, 10);
            }
        });

        Plotly.newPlot('precursor-plot', [{
            type: 'contour',
            x: emptyCoords.x,
            y: emptyCoords.y,
            z: emptyZ,
            colorscale: 'Cool',
            zmin: config.precursor_range[0],
            zmax: config.precursor_range[1],
            colorbar: getOptimizedColorbar('Concentration (molecules/nm²)', config.precursor_range)
        }], {
            ...getPlotLayout('Precursor Distribution'),
            xaxis: {title: 'X Position (nm)'},
            yaxis: {title: 'Y Position (nm)'}
        });

        Plotly.newPlot('timeseries-plot', [{
            type: 'scatter',
            mode: 'lines',
            x: [],
            y: [],
            name: 'Mean Height',
            line: {color: '#1f77b4', width: 2}
        }], {
            ...getPlotLayout('Mean Height Evolution'),
            xaxis: {title: 'Pixel Index'},
            yaxis: {
                title: 'Mean Height (nm)',
                tickformat: '.2e',
                exponentformat: 'e'
            }
        });

        // 强化Z轴锁定检查
        setInterval(() => {
            const plotDiv = document.getElementById('surface-plot');
            if (plotDiv && plotDiv.layout && plotDiv.layout.scene && plotDiv.layout.scene.zaxis) {
                const currentRange = plotDiv.layout.scene.zaxis.range;
                if (!currentRange ||
                    Math.abs(currentRange[0] - config.height_range[0]) > 1e-12 ||
                    Math.abs(currentRange[1] - config.height_range[1]) > 1e-12) {

                    Plotly.relayout('surface-plot', {
                        'scene.zaxis.range': config.height_range,
                        'scene.zaxis.autorange': false,
                        'scene.zaxis.fixedrange': true
                    });
                }
            }

            // 同时检查侧视角的Z轴
            const plotDivSide = document.getElementById('surface-plot-side');
            if (plotDivSide && plotDivSide.layout && plotDivSide.layout.scene && plotDivSide.layout.scene.zaxis) {
                const currentRangeSide = plotDivSide.layout.scene.zaxis.range;
                if (!currentRangeSide ||
                    Math.abs(currentRangeSide[0] - config.height_range[0]) > 1e-12 ||
                    Math.abs(currentRangeSide[1] - config.height_range[1]) > 1e-12) {

                    Plotly.relayout('surface-plot-side', {
                        'scene.zaxis.range': config.height_range,
                        'scene.zaxis.autorange': false,
                        'scene.zaxis.fixedrange': true
                    });
                }
            }
        }, 1000);
    }

    function updateDisplay(data) {
        try {
            if (!data.h_surface || !data.n_surface) return;

            const coords = generateCoordinates([data.h_surface.length, data.h_surface[0].length]);

            // 更新3D图 - 强制锁定Z轴
            Plotly.update('surface-plot',
                {
                    x: [coords.x],
                    y: [coords.y],
                    z: [data.h_surface],
                    cmin: config.height_range[0],
                    cmax: config.height_range[1]
                },
                {
                    'scene.zaxis.range': config.height_range,
                    'scene.zaxis.autorange': false,
                    'scene.zaxis.fixedrange': true
                }
            );

            // 更新第二个3D图（侧视角）
            Plotly.update('surface-plot-side',
                {
                    x: [coords.x],
                    y: [coords.y],
                    z: [data.h_surface],
                    cmin: config.height_range[0],
                    cmax: config.height_range[1]
                },
                {
                    'scene.zaxis.range': config.height_range,
                    'scene.zaxis.autorange': false,
                    'scene.zaxis.fixedrange': true
                }
            );

            Plotly.restyle('precursor-plot', {
                'x': [coords.x], 'y': [coords.y], 'z': [data.n_surface],
                'zmin': config.precursor_range[0], 'zmax': config.precursor_range[1]
            });

            if (data.history && data.history.pixel_indices.length > 0) {
                Plotly.restyle('timeseries-plot', {
                    'x': [data.history.pixel_indices], 'y': [data.history.max_heights]
                });
            }

            updateStatusDisplay(data);
        } catch (error) {
            console.error('更新失败:', error);
        }
    }

    function updateStatusDisplay(data) {
        const indicator = document.getElementById('status-indicator');
        const text = document.getElementById('status-text');

        if (data.is_running) {
            indicator.className = 'status-indicator status-running';
            text.textContent = '仿真进行中...';
        } else {
            indicator.className = 'status-indicator status-stopped';
            text.textContent = '仿真已完成';
        }

        document.getElementById('pixel-count').textContent = `${data.pixel_index}/${data.total_pixels}`;
        document.getElementById('progress').textContent = data.simulation_progress.toFixed(1);
        document.getElementById('max-height').textContent = data.max_height.toExponential(3);
        document.getElementById('beam-x').textContent = data.beam_position[0].toFixed(1);
        document.getElementById('beam-y').textContent = data.beam_position[1].toFixed(1);
    }

    // 模式切换
    function switchMode(mode) {
        currentMode = mode;
        document.getElementById('realtime-tab').classList.toggle('active', mode === 'realtime');
        document.getElementById('history-tab').classList.toggle('active', mode === 'history');
        document.getElementById('realtime-controls').style.display = mode === 'realtime' ? 'block' : 'none';
        document.getElementById('history-controls').style.display = mode === 'history' ? 'block' : 'none';

        if (mode === 'history') {
            loadSnapshots();
        }
    }

    // 历史回溯功能
    function loadSnapshots() {
        fetch('/snapshots')
            .then(response => response.json())
            .then(data => {
                snapshotsData = data;
                document.getElementById('total-snapshots').textContent = snapshotsData.length;
                document.getElementById('snapshot-count').textContent = snapshotsData.length;
                if (snapshotsData.length > 0) {
                    document.getElementById('history-slider').max = snapshotsData.length - 1;
                    displaySnapshot(0);
                }
            })
            .catch(error => console.log('加载快照失败:', error));
    }

    function displaySnapshot(index) {
        if (!snapshotsData || index >= snapshotsData.length || index < 0) return;

        currentSnapshotIndex = index;
        const snapshot = snapshotsData[index];
        document.getElementById('history-frame-label').textContent = index + 1;
        document.getElementById('history-slider').value = index;

        updateDisplay({
            h_surface: snapshot.h_surface,
            n_surface: snapshot.n_surface,
            pixel_index: snapshot.metadata.pixel_index,
            timestamp: snapshot.metadata.timestamp,
            beam_position: snapshot.metadata.beam_position,
            max_height: snapshot.metadata.max_height,
            mean_height: 0,
            simulation_progress: 0,
            total_pixels: 0,
            is_running: snapshot.metadata.is_running
        });
    }

    function nextSnapshot() {
        if (currentSnapshotIndex < snapshotsData.length - 1) {
            displaySnapshot(currentSnapshotIndex + 1);
        }
    }

    function previousSnapshot() {
        if (currentSnapshotIndex > 0) {
            displaySnapshot(currentSnapshotIndex - 1);
        }
    }

    function playHistory() {
        if (isPlayingHistory) return;
        isPlayingHistory = true;
        historyPlayTimer = setInterval(() => {
            if (currentSnapshotIndex < snapshotsData.length - 1) {
                nextSnapshot();
            } else {
                pauseHistory();
            }
        }, 800);
    }

    function pauseHistory() {
        isPlayingHistory = false;
        if (historyPlayTimer) clearInterval(historyPlayTimer);
    }

    document.getElementById('history-slider').addEventListener('input', e => {
        displaySnapshot(parseInt(e.target.value));
    });

    // 实时更新功能
    function fetchData() {
        if (currentMode !== 'realtime') return;

        fetch('/data')
            .then(response => response.json())
            .then(data => {
                updateDisplay(data);
                if (data.pixel_index) {
                    const estimated_snapshots = Math.floor(data.pixel_index / 10);
                    document.getElementById('snapshot-count').textContent = estimated_snapshots;
                }
            })
            .catch(error => console.log('等待数据...'));
    }

    function toggleAutoUpdate() {
        isAutoUpdate = !isAutoUpdate;
        const btn = document.getElementById('auto-update-btn');
        if (isAutoUpdate) {
            btn.textContent = '⏸️ 暂停自动更新';
            startAutoUpdate();
        } else {
            btn.textContent = '▶️ 开始自动更新';
            stopAutoUpdate();
        }
    }

    function updateNow() { fetchData(); }

    function resetZoom() {
        // 主视角 - 鸟瞰稍微倾斜
        Plotly.relayout('surface-plot', {
            'scene.camera.eye': {x: 1.87, y: 0.88, z: 1.5},
            'scene.xaxis.range': config.x_range,
            'scene.yaxis.range': config.y_range,
            'scene.zaxis.range': config.height_range,
            'scene.zaxis.autorange': false,
            'scene.aspectmode': 'manual',
            'scene.aspectratio': {
                x: 1,
                y: 1 / config.aspect_ratio,
                z: 0.6
            }
        });

        // 侧视角 - 低角度观察
        Plotly.relayout('surface-plot-side', {
            'scene.camera.eye': {x: 0.5, y: -2.2, z: 0.8},
            'scene.xaxis.range': config.x_range,
            'scene.yaxis.range': config.y_range,
            'scene.zaxis.range': config.height_range,
            'scene.zaxis.autorange': false,
            'scene.aspectmode': 'manual',
            'scene.aspectratio': {
                x: 1,
                y: 1 / config.aspect_ratio,
                z: 0.6
            }
        });

        // 其他图表保持不变
        ['precursor-plot', 'timeseries-plot'].forEach(id => {
            Plotly.relayout(id, 'autosize');
        });
    }

    function setUpdateInterval() {
        const select = document.getElementById('update-interval');
        updateInterval = parseInt(select.value);
        if (isAutoUpdate) { stopAutoUpdate(); startAutoUpdate(); }
    }

    function startAutoUpdate() {
        if (updateTimer) clearInterval(updateTimer);
        updateTimer = setInterval(fetchData, updateInterval);
    }

    function stopAutoUpdate() {
        if (updateTimer) { clearInterval(updateTimer); updateTimer = null; }
    }

    window.addEventListener('load', () => {
        initializePlots();
        fetchData();
        startAutoUpdate();
    });

    window.addEventListener('beforeunload', () => {
        stopAutoUpdate();
        pauseHistory();
    });
    """
