"""
分析训练曲线，判断模型是欠拟合还是过拟合

使用方法:
    python experiments/analyze_training_curves.py --logdir <训练日志目录>
    
或者直接指定实验目录:
    python experiments/analyze_training_curves.py --logdir runs/lane/spin/20240101T120000_12345
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("警告: 未安装tensorboard，将尝试使用其他方法读取日志")


def load_tensorboard_logs(logdir):
    """从TensorBoard日志目录加载训练指标"""
    if not HAS_TENSORBOARD:
        raise ImportError("需要安装tensorboard: pip install tensorboard")
    
    # 查找events文件
    event_files = list(Path(logdir).rglob('events.out.tfevents.*'))
    if not event_files:
        raise FileNotFoundError(f"在 {logdir} 中未找到TensorBoard事件文件")
    
    # 使用最新的events文件
    event_file = max(event_files, key=lambda x: x.stat().st_mtime)
    event_dir = str(event_file.parent)
    
    print(f"读取日志文件: {event_file}")
    
    # 加载事件
    ea = EventAccumulator(event_dir)
    ea.Reload()
    
    # 获取所有标量标签
    scalar_tags = ea.Tags()['scalars']
    print(f"找到的指标: {scalar_tags}")
    
    # 提取训练和验证指标
    metrics = {}
    for tag in scalar_tags:
        scalar_events = ea.Scalars(tag)
        steps = [s.step for s in scalar_events]
        values = [s.value for s in scalar_events]
        metrics[tag] = {'steps': steps, 'values': values}
    
    return metrics


def _get_metric_values(metrics, metric_name):
    """尝试多种格式获取指标值"""
    # 尝试多种可能的键名格式
    possible_keys = [
        f'{metric_name}/epoch',  # 带 /epoch 后缀
        metric_name,             # 不带后缀
        f'{metric_name}/step',   # 带 /step 后缀
    ]
    
    for key in possible_keys:
        if key in metrics:
            return metrics[key].get('values', [])
    
    return []


def analyze_fitting(metrics):
    """分析训练曲线，判断欠拟合/过拟合"""
    results = {
        'status': 'unknown',
        'train_mae': None,
        'val_mae': None,
        'train_loss': None,
        'val_loss': None,
        'gap': None,
        'convergence': None,
        'recommendations': []
    }
    
    # 提取关键指标（支持多种格式）
    train_mae = _get_metric_values(metrics, 'train_mae')
    val_mae = _get_metric_values(metrics, 'val_mae')
    train_loss = _get_metric_values(metrics, 'train_loss')
    val_loss = _get_metric_values(metrics, 'val_loss')
    
    if not train_mae or not val_mae:
        print("⚠️  警告: 未找到足够的训练指标数据")
        print(f"   找到的训练指标键: {list(metrics.keys())}")
        print(f"   train_mae 数据点数: {len(train_mae)}")
        print(f"   val_mae 数据点数: {len(val_mae)}")
        return results
    
    # 获取最后几个epoch的平均值（避免波动）
    n_epochs = min(len(train_mae), len(val_mae))
    if n_epochs < 5:
        print("⚠️  警告: 训练轮数太少，无法准确判断")
        return results
    
    # 计算最后5个epoch的平均值
    last_n = min(5, n_epochs)
    final_train_mae = np.mean(train_mae[-last_n:])
    final_val_mae = np.mean(val_mae[-last_n:])
    final_train_loss = np.mean(train_loss[-last_n:]) if train_loss else None
    final_val_loss = np.mean(val_loss[-last_n:]) if val_loss else None
    
    results['train_mae'] = final_train_mae
    results['val_mae'] = final_val_mae
    results['train_loss'] = final_train_loss
    results['val_loss'] = final_val_loss
    
    # 计算gap（验证集和训练集的差异）
    mae_gap = final_val_mae - final_train_mae
    loss_gap = (final_val_loss - final_train_loss) if (final_val_loss and final_train_loss) else None
    
    results['gap'] = mae_gap
    
    # 判断收敛情况
    if n_epochs >= 10:
        # 检查最后10个epoch是否还在下降
        recent_train = train_mae[-10:]
        recent_val = val_mae[-10:]
        train_trend = np.polyfit(range(len(recent_train)), recent_train, 1)[0]
        val_trend = np.polyfit(range(len(recent_val)), recent_val, 1)[0]
        
        if train_trend < -0.01 and val_trend < -0.01:
            results['convergence'] = 'still_improving'
        elif abs(train_trend) < 0.01 and abs(val_trend) < 0.01:
            results['convergence'] = 'converged'
        else:
            results['convergence'] = 'fluctuating'
    else:
        results['convergence'] = 'unknown'
    
    # 判断欠拟合/过拟合
    gap_ratio = mae_gap / final_train_mae if final_train_mae > 0 else 0
    
    if final_train_mae > final_val_mae * 1.1:
        # 训练集MAE明显高于验证集（异常情况，可能是数据问题）
        results['status'] = 'anomaly'
        results['recommendations'].append("训练集MAE高于验证集，可能是数据划分或mask设置有问题")
    elif gap_ratio > 0.3:
        # Gap > 30%，过拟合
        results['status'] = 'overfitting'
        results['recommendations'].extend([
            "增加正则化: l2_reg = 1e-4 或 1e-3",
            "增加dropout（如果模型支持）",
            "减少模型容量: hidden_size 或 n_layers",
            "增加数据增强或whiten_prob",
            "使用早停（patience）"
        ])
    elif gap_ratio < 0.05 and final_train_mae > 0.5:
        # Gap很小但MAE仍然很高，欠拟合
        results['status'] = 'underfitting'
        results['recommendations'].extend([
            "增加模型容量: hidden_size (16→32→64)",
            "增加层数: n_layers (3→4→5)",
            "增加训练轮数: epochs",
            "增加学习率: lr (0.0008→0.001→0.0015)",
            "检查数据预处理是否正确"
        ])
    elif gap_ratio < 0.05:
        # Gap小且MAE较低，可能是良好拟合
        results['status'] = 'good_fit'
        results['recommendations'].append("模型拟合良好，可以尝试微调超参数进一步提升")
    elif 0.05 <= gap_ratio <= 0.3:
        # 中等gap，可能是轻微过拟合或正常
        results['status'] = 'slight_overfitting'
        results['recommendations'].extend([
            "轻微过拟合，可以增加少量正则化",
            "或继续训练观察是否改善"
        ])
    else:
        results['status'] = 'unknown'
    
    return results


def _get_metric_data(metrics, metric_name):
    """获取指标的数据（steps和values）"""
    possible_keys = [
        f'{metric_name}/epoch',
        metric_name,
        f'{metric_name}/step',
    ]
    
    for key in possible_keys:
        if key in metrics:
            return metrics[key].get('steps', []), metrics[key].get('values', [])
    
    return None, None


def plot_training_curves(metrics, output_path=None):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves Analysis', fontsize=16, fontweight='bold')
    
    # MAE曲线
    ax1 = axes[0, 0]
    train_steps, train_values = _get_metric_data(metrics, 'train_mae')
    if train_steps and train_values:
        ax1.plot(train_steps, train_values, label='Train MAE', linewidth=2, alpha=0.8)
    
    val_steps, val_values = _get_metric_data(metrics, 'val_mae')
    if val_steps and val_values:
        ax1.plot(val_steps, val_values, label='Val MAE', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MAE')
    ax1.set_title('Mean Absolute Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss曲线
    ax2 = axes[0, 1]
    train_loss_steps, train_loss_values = _get_metric_data(metrics, 'train_loss')
    if train_loss_steps and train_loss_values:
        ax2.plot(train_loss_steps, train_loss_values, label='Train Loss', linewidth=2, alpha=0.8)
    
    val_loss_steps, val_loss_values = _get_metric_data(metrics, 'val_loss')
    if val_loss_steps and val_loss_values:
        ax2.plot(val_loss_steps, val_loss_values, label='Val Loss', linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # MSE曲线
    ax3 = axes[1, 0]
    train_mse_steps, train_mse_values = _get_metric_data(metrics, 'train_mse')
    if train_mse_steps and train_mse_values:
        ax3.plot(train_mse_steps, train_mse_values, label='Train MSE', linewidth=2, alpha=0.8)
    
    val_mse_steps, val_mse_values = _get_metric_data(metrics, 'val_mse')
    if val_mse_steps and val_mse_values:
        ax3.plot(val_mse_steps, val_mse_values, label='Val MSE', linewidth=2, alpha=0.8)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MSE')
    ax3.set_title('Mean Squared Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gap曲线（验证集 - 训练集）
    ax4 = axes[1, 1]
    train_mae_steps, train_mae_values = _get_metric_data(metrics, 'train_mae')
    val_mae_steps, val_mae_values = _get_metric_data(metrics, 'val_mae')
    
    if train_mae_steps and train_mae_values and val_mae_steps and val_mae_values:
        # 对齐steps
        common_steps = sorted(set(train_mae_steps) & set(val_mae_steps))
        if common_steps:
            train_aligned = [train_mae_values[train_mae_steps.index(s)] for s in common_steps]
            val_aligned = [val_mae_values[val_mae_steps.index(s)] for s in common_steps]
            gap = [v - t for t, v in zip(train_aligned, val_aligned)]
            
            ax4.plot(common_steps, gap, label='Val - Train Gap', linewidth=2, 
                    color='red', alpha=0.8)
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.fill_between(common_steps, 0, gap, alpha=0.3, color='red')
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Gap (Val - Train)')
    ax4.set_title('Overfitting Indicator (Gap)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {output_path}")
    else:
        plt.show()


def print_analysis_report(results, metrics):
    """打印分析报告"""
    print("\n" + "="*60)
    print("📊 训练曲线分析报告")
    print("="*60)
    
    # 基本信息
    print(f"\n🎯 模型状态: {results['status']}")
    print(f"\n📈 关键指标 (最后5个epoch的平均值):")
    if results['train_mae'] is not None:
        print(f"   训练集 MAE: {results['train_mae']:.4f}")
    if results['val_mae'] is not None:
        print(f"   验证集 MAE: {results['val_mae']:.4f}")
    if results['gap'] is not None:
        print(f"   Gap (Val - Train): {results['gap']:.4f} ({results['gap']/results['train_mae']*100:.1f}%)")
    
    if results['convergence']:
        conv_map = {
            'still_improving': '仍在改善',
            'converged': '已收敛',
            'fluctuating': '波动中',
            'unknown': '未知'
        }
        print(f"\n🔄 收敛状态: {conv_map.get(results['convergence'], results['convergence'])}")
    
    # 判断结果
    status_map = {
        'overfitting': '🔴 过拟合',
        'underfitting': '🟡 欠拟合',
        'good_fit': '🟢 良好拟合',
        'slight_overfitting': '🟠 轻微过拟合',
        'anomaly': '⚠️  异常',
        'unknown': '❓ 未知'
    }
    
    print(f"\n{status_map.get(results['status'], results['status'])}")
    
    # 建议
    if results['recommendations']:
        print(f"\n💡 调整建议:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='分析训练曲线判断欠拟合/过拟合')
    parser.add_argument('--logdir', type=str, required=True,
                       help='训练日志目录路径 (例如: runs/lane/spin/20240101T120000_12345)')
    parser.add_argument('--save-plot', type=str, default=None,
                       help='保存图表路径 (例如: training_curves.png)')
    parser.add_argument('--show-plot', action='store_true',
                       help='显示图表')
    
    args = parser.parse_args()
    
    # 检查目录
    if not os.path.exists(args.logdir):
        print(f"❌ 错误: 目录不存在: {args.logdir}")
        return
    
    # 加载指标
    try:
        metrics = load_tensorboard_logs(args.logdir)
    except Exception as e:
        print(f"❌ 错误: 无法加载日志: {e}")
        return
    
    # 分析
    results = analyze_fitting(metrics)
    
    # 打印报告
    print_analysis_report(results, metrics)
    
    # 绘制曲线
    if args.show_plot or args.save_plot:
        try:
            plot_training_curves(metrics, output_path=args.save_plot)
            if args.show_plot:
                plt.show()
        except Exception as e:
            print(f"⚠️  警告: 无法绘制图表: {e}")


if __name__ == '__main__':
    main()


