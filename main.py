#!/usr/bin/env python3
"""
FEBID仿真主程序 - 优化版
减少代码重复，提高可维护性

Author: 刘宇
Date: 2025/7
"""

import sys
import argparse
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from simulation_core_main import MemoryOptimizedFEBID
from config import (
    SIMULATION_CONFIG,
    validate_config, validate_visualization_config, print_config_summary
)


class FEBIDSimulationRunner:
    """FEBID仿真运行器类"""

    def __init__(self):
        self.sim_config = None
        self.viz_config = None
        self.febid = None

    def parse_arguments(self):
        """解析命令行参数"""
        parser = argparse.ArgumentParser(
            description='FEBID仿真系统 - 优化版',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
            """
        )

        # 监控选项
        parser.add_argument('--no-monitor', action='store_true',
                            help='禁用实时监控功能')
        parser.add_argument('--monitor-interval', type=int, default=None,
                            help='监控数据保存间隔（像素数）')

        # 输出选项
        parser.add_argument('--no-plots', action='store_true',
                            help='禁用图表生成')
        parser.add_argument('--no-save', action='store_true',
                            help='禁用结果保存')

        # 调试选项
        parser.add_argument('--verbose', action='store_true',
                            help='显示详细信息')
        parser.add_argument('--validate-only', action='store_true',
                            help='仅验证配置参数，不运行仿真')

        return parser.parse_args()

    def apply_command_line_overrides(self, args):
        """应用命令行参数覆盖"""
        # 监控设置
        if args.no_monitor:
            self.sim_config['monitoring']['enable_realtime_monitor'] = False
            print("🚫 实时监控已通过命令行禁用")

        if args.monitor_interval is not None:
            self.sim_config['monitoring']['save_interval'] = args.monitor_interval
            self.viz_config['save_interval'] = args.monitor_interval
            print(f"📊 监控间隔设置为: {args.monitor_interval} 像素")

        # 输出设置
        if args.no_plots:
            self.sim_config['output']['create_plots'] = False
            print("📊 图表生成已禁用")

        if args.no_save:
            self.sim_config['output']['save_core_results'] = False
            print("💾 结果保存已禁用")

        if args.verbose:
            self.sim_config['output']['verbose'] = True
            print("🔍 详细信息模式已启用")

    def validate_configurations(self):
        """验证配置"""
        print("\n📋 验证配置参数...")

        if not validate_config(self.sim_config):
            raise ValueError("仿真配置验证失败")

        # 修改这部分逻辑
        enable_monitoring = self.sim_config['monitoring']['enable_realtime_monitor']
        if enable_monitoring:
            monitor_config = self.sim_config.get('monitoring', {})
            if not self._validate_monitor_visualization_params(monitor_config):
                raise ValueError("监控可视化参数验证失败")

    def _validate_monitor_visualization_params(self, monitor_config: dict) -> bool:
        """验证监控配置中的可视化参数"""
        try:
            # 检查必要的可视化参数
            if 'height_range' not in monitor_config or 'precursor_range' not in monitor_config:
                print("❌ 监控配置缺少可视化范围参数")
                return False

            # 如果不是 'auto'，验证范围值
            height_range = monitor_config.get('height_range')
            if height_range != 'auto':
                if not isinstance(height_range, list) or len(height_range) != 2:
                    print("❌ height_range 必须是 'auto' 或 [min, max]")
                    return False
                if height_range[1] <= height_range[0]:
                    print("❌ 高度显示范围设置错误")
                    return False

            precursor_range = monitor_config.get('precursor_range')
            if precursor_range != 'auto':
                if not isinstance(precursor_range, list) or len(precursor_range) != 2:
                    print("❌ precursor_range 必须是 'auto' 或 [min, max]")
                    return False
                if precursor_range[1] <= precursor_range[0]:
                    print("❌ 前驱体显示范围设置错误")
                    return False

            print("✓ 监控可视化参数验证通过")
            return True

        except Exception as e:
            print(f"❌ 监控可视化参数验证失败: {e}")
            return False

    def create_simulator(self):
        """创建仿真器实例"""
        print("\n🔧 初始化FEBID仿真器...")

        monitor_config = self.sim_config['monitoring']
        enable_monitoring = monitor_config['enable_realtime_monitor']

        self.febid = MemoryOptimizedFEBID(
            config=self.sim_config,
            enable_realtime_monitor=enable_monitoring,
            monitor_save_interval=monitor_config.get('save_interval', 50),
            use_realtime_mode=monitor_config.get('use_realtime_mode', True),  # 添加 get 和默认值
        )
        # *** 新增：检查Z轴自适应状态 ***
        if enable_monitoring and self.febid.monitor:
            if hasattr(self.febid.monitor, 'z_adjustment_info'):
                print(f"📊 监控器Z轴状态: {self.febid.monitor.z_adjustment_info}")


    def prepare_simulation(self):
        """准备仿真"""
        monitor_config = self.sim_config['monitoring']
        enable_monitoring = monitor_config['enable_realtime_monitor']

        if enable_monitoring:
            mode_type = "实时" if monitor_config['use_realtime_mode'] else "传统"
            print(f"\n🌐 {mode_type}监控已启用")

            if monitor_config['use_realtime_mode']:
                print("📊 Web界面将在仿真开始时自动打开")
                print("💡 您可以在仿真过程中实时查看进展")
                print("⏱️  3秒后开始仿真...")
                time.sleep(3)

    def run_simulation(self):
        """运行仿真"""
        print("\n🚀 开始仿真...")
        return self.febid.run_simulation()

    def process_results(self, results):
        """处理仿真结果"""
        # 监控提示
        monitor_config = self.sim_config['monitoring']
        enable_monitoring = monitor_config['enable_realtime_monitor']

        if enable_monitoring and self.febid.monitor is not None:
            print("\n📊 仿真完成！")
            if monitor_config['use_realtime_mode']:
                print("🌐 实时监控界面仍然可用")
                print("💡 关闭浏览器标签页或按Ctrl+C退出程序")

        # 可视化结果
        if self.sim_config['output']['create_plots']:
            print("\n📊 生成可视化图表...")
            self.febid.visualize_results(results)

        # 保存结果
        if self.sim_config['output']['save_core_results']:
            print("\n💾 保存仿真结果...")
            self.febid.save_results(results)

        # 显示摘要
        self._display_results_summary(results)

        # 保持监控服务器运行
        if enable_monitoring and self.febid.monitor is not None and monitor_config['use_realtime_mode']:
            self._keep_monitor_running()

    def _display_results_summary(self, results):
        """显示结果摘要"""
        print("\n" + "=" * 60)
        print("🎉 FEBID仿真完成！")
        print("=" * 60)

        print(f"📊 仿真统计:")
        print(f"   ⏱️  总时间: {results['simulation_time']:.2f} 秒")
        print(f"   📏 最大高度: {results['h_surface'].max():.3e} nm")
        print(f"   🎯 扫描点数: {results['scan_info'].total_pixels:,}")

        monitor_config = self.sim_config['monitoring']
        if (monitor_config['enable_realtime_monitor'] and
                self.febid.monitor is not None and
                monitor_config['use_realtime_mode']):
            print(f"   🌐 实时界面: http://localhost:{self.febid.monitor.web_port}")

        print("\n✨ 感谢使用FEBID仿真系统！")

    def _keep_monitor_running(self):
        """保持监控服务器运行"""
        print("\n🔄 实时监控服务器保持运行中...")
        print("   按 Ctrl+C 退出程序")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 用户中断，正在关闭...")
            self.febid.stop_realtime_monitor()

    def cleanup(self):
        """清理资源"""
        if self.febid is not None:
            if hasattr(self.febid, 'stop_realtime_monitor'):
                self.febid.stop_realtime_monitor()

            if hasattr(self.febid, 'monitor') and self.febid.monitor is not None:
                if hasattr(self.febid.monitor, 'close'):
                    self.febid.monitor.close()

    def run(self):
        """运行完整的仿真流程"""
        try:
            # 解析参数
            args = self.parse_arguments()

            # 加载配置
            self.sim_config = SIMULATION_CONFIG.copy()

            # 应用命令行覆盖
            self.apply_command_line_overrides(args)

            # 验证配置
            self.validate_configurations()

            # 显示配置摘要 - 不再传递 viz_config
            print_config_summary(self.sim_config)

            # 如果只是验证，则退出
            if args.validate_only:
                print("✅ 配置验证完成，程序退出")
                return None

            # 创建仿真器
            self.create_simulator()

            # 准备仿真
            self.prepare_simulation()

            # 运行仿真
            results = self.run_simulation()

            # 处理结果
            self.process_results(results)

            return results

        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断仿真")
            print("🧹 正在清理资源...")
            return None

        except Exception as e:
            print(f"\n❌ 仿真过程中发生错误: {e}")
            traceback.print_exc()
            return None

        finally:
            self.cleanup()


def main():
    """主函数"""
    print("🔬 FEBID仿真系统启动 (优化版)")
    print("=" * 60)

    runner = FEBIDSimulationRunner()
    results = runner.run()

    # 如果在交互式环境中，保留results变量供后续使用
    if results is not None:
        print(f"\n💡 仿真结果已保存在变量 'results' 中，可用于进一步分析")

    return results


def main_with_custom_config(sim_config: Dict, viz_config: Optional[Dict] = None, **kwargs):
    """
    使用自定义配置运行仿真的便捷函数
    """
    print("🔬 使用自定义配置运行FEBID仿真 (精简版)")
    # 如果提供了旧的 viz_config，合并到 monitoring 配置中
    if viz_config is not None:
        print("⚠️  viz_config 参数已废弃，将自动合并到 monitoring 配置中")
        if 'monitoring' not in sim_config:
            sim_config['monitoring'] = {}
        # 合并可视化配置到监控配置
        sim_config['monitoring']['height_range'] = viz_config.get('height_range', 'auto')
        sim_config['monitoring']['precursor_range'] = viz_config.get('precursor_range', 'auto')
        sim_config['monitoring']['save_interval'] = viz_config.get('save_interval', 10)
        sim_config['monitoring']['max_memory_frames'] = viz_config.get('max_memory_frames', 200)

    # 从kwargs中提取监控参数
    monitor_config = sim_config.get('monitoring', {})
    enable_monitor = kwargs.get('enable_monitor', monitor_config.get('enable_realtime_monitor', False))

    # 可以选择跳过验证（精简版特性）
    skip_validation = kwargs.get('skip_validation', False)

    if not skip_validation:
        # 验证配置
        if not validate_config(sim_config):
            raise ValueError("仿真配置验证失败")

        if enable_monitor and viz_config:
            if not validate_visualization_config(viz_config, sim_config):
                raise ValueError("可视化配置验证失败")

    # 创建并运行仿真（精简版 - 默认使用Numba）
    try:
        febid = MemoryOptimizedFEBID(
            config=sim_config,
            enable_realtime_monitor=enable_monitor,
            monitor_save_interval=monitor_config.get('save_interval', 50),
            use_realtime_mode=monitor_config.get('use_realtime_mode', True)
        )

        if enable_monitor:

            print("🌐 实时监控已启用，Web界面将自动启动")

        results = febid.run_simulation()

        # 保持实时监控服务器运行
        if enable_monitor and febid.monitor is not None and monitor_config.get('use_realtime_mode', True):
            print(f"🌐 实时监控界面: http://localhost:{febid.monitor.web_port}")
            print("💡 程序完成，但监控服务器继续运行")

        return results

    except Exception as e:
        print(f"❌ 自定义配置仿真失败: {e}")
        return None


if __name__ == "__main__":
    results = main()
