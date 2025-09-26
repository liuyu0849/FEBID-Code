#!/usr/bin/env python3
"""
FEBID仿真主程序 - 精简版（无监控）
删除所有监控相关功能

Author: 刘宇
Date: 2025/7
"""
import os

# 在导入其他模块之前设置环境变量
os.environ['NUMBA_NUM_THREADS'] = '14'
os.environ['OMP_NUM_THREADS'] = '14'
os.environ['MKL_NUM_THREADS'] = '14'
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
    validate_config, print_config_summary
)


class FEBIDSimulationRunner:
    """FEBID仿真运行器类 - 精简版"""

    def __init__(self):
        self.sim_config = None
        self.febid = None

    def parse_arguments(self):
        """解析命令行参数"""
        parser = argparse.ArgumentParser(
            description='FEBID仿真系统 - 精简版（无监控）',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )

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
        parser.add_argument('--fast', action='store_true',
                            help='快速模式：减少进度输出')

        return parser.parse_args()

    def apply_command_line_overrides(self, args):
        """应用命令行参数覆盖"""
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
        #print("\n📋 验证配置参数...")

        if not validate_config(self.sim_config):
            raise ValueError("仿真配置验证失败")

        print("✓ 配置验证通过")

    def create_simulator(self):
        """创建仿真器实例"""
        print("\n🔧 初始化FEBID仿真器...")
        self.febid = MemoryOptimizedFEBID(config=self.sim_config)

    def run_simulation(self):
        """运行仿真"""
        print("\n🚀 开始仿真...")
        return self.febid.run_simulation()

    def process_results(self, results):
        """处理仿真结果"""
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

    def _display_results_summary(self, results):
        """显示结果摘要"""
        print("\n" + "=" * 60)
        print("🎉 FEBID仿真完成！")
        print("=" * 60)

        print(f"📊 仿真统计:")
        print(f"   ⏱️  总时间: {results['simulation_time']:.2f} 秒")
        print(f"   📏 最大高度: {results['h_surface'].max():.3e} nm")
        print(f"   🎯 扫描点数: {results['scan_info'].total_pixels:,}")

        print("\n✨ 感谢使用FEBID仿真系统！")

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
            if not args.fast:
                self.validate_configurations()

            # 显示配置摘要
            print_config_summary(self.sim_config)

            # 如果只是验证，则退出
            if args.validate_only:
                print("✅ 配置验证完成，程序退出")
                return None

            # 创建仿真器
            self.create_simulator()

            # 运行仿真
            results = self.run_simulation()

            # 处理结果
            self.process_results(results)

            return results

        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断仿真")
            return None

        except Exception as e:
            print(f"\n❌ 仿真过程中发生错误: {e}")
            traceback.print_exc()
            return None


def main():
    """主函数"""
    #print("🔬 FEBID仿真系统启动 (精简版 - 无监控)")
    #print("=" * 60)

    runner = FEBIDSimulationRunner()
    results = runner.run()

    # 如果在交互式环境中，保留results变量供后续使用
    if results is not None:
        print(f"\n💡 仿真结果已保存在变量 'results' 中，可用于进一步分析")

    return results


def main_with_custom_config(sim_config: Dict, **kwargs):
    """
    使用自定义配置运行仿真的便捷函数 - 精简版
    """
    print("🔬 使用自定义配置运行FEBID仿真 (精简版)")

    # 可以选择跳过验证
    skip_validation = kwargs.get('skip_validation', False)

    if not skip_validation:
        # 验证配置
        if not validate_config(sim_config):
            raise ValueError("仿真配置验证失败")

    # 创建并运行仿真
    try:
        febid = MemoryOptimizedFEBID(config=sim_config)
        results = febid.run_simulation()

        # 可视化和保存
        if sim_config['output']['create_plots']:
            febid.visualize_results(results)
        if sim_config['output']['save_core_results']:
            febid.save_results(results)

        return results

    except Exception as e:
        print(f"❌ 自定义配置仿真失败: {e}")
        return None


if __name__ == "__main__":
    results = main()
