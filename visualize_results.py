#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练结果可视化脚本
分析损失曲线、学习率变化，并进行模型推理测试
"""

import os
import sys
import re
import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
from pathlib import Path

def find_latest_run():
    """查找最新的训练运行目录"""
    runs_dir = Path('runs')
    if not runs_dir.exists():
        print("错误: 找不到 runs 目录")
        return None
    
    # 获取所有运行目录并按时间排序
    run_dirs = sorted(runs_dir.glob('2026.*'), key=lambda x: x.name, reverse=True)
    
    if not run_dirs:
        print("错误: 找不到训练结果")
        return None
    
    latest_run = run_dirs[0]
    print(f"找到最新训练结果: {latest_run}")
    return latest_run


def parse_log_file(log_dir):
    """解析训练日志文件"""
    log_files = list(Path(log_dir).glob('*.log'))
    
    if not log_files:
        print(f"警告: 在 {log_dir} 中找不到日志文件")
        return None
    
    print(f"解析日志文件: {log_files[0]}")
    
    epochs = []
    steps = []
    losses = []
    lbox_values = []
    lcls_values = []
    dfl_values = []
    lr_values = []
    
    try:
        with open(log_files[0], 'r', encoding='utf-8') as f:
            for line in f:
                if 'imgsize (480, 480), loss:' in line:
                    # 解析训练信息
                    try:
                        # 示例: 2026-01-27 22:21:59,336 [INFO] Epoch 1/5, Step 50/4028, imgsize (480, 480), loss: 5.0810, lbox: 0.4255, lcls: 3.7792, dfl: 0.8763, cur_lr: 0.09962760657072067
                        
                        # 提取 Epoch 和 Step
                        epoch_match = re.search(r'Epoch (\d+)/', line)
                        step_match = re.search(r'Step (\d+)/', line)
                        epoch = int(epoch_match.group(1))
                        step = int(step_match.group(1))
                        
                        # 提取损失值 - 从 imgsize 后面开始
                        data_part = line.split('imgsize (480, 480),')[1]
                        
                        loss_match = re.search(r'loss:\s*([\d.]+)', data_part)
                        lbox_match = re.search(r'lbox:\s*([\d.]+)', data_part)
                        lcls_match = re.search(r'lcls:\s*([\d.]+)', data_part)
                        dfl_match = re.search(r'dfl:\s*([\d.]+)', data_part)
                        lr_match = re.search(r'cur_lr:\s*([\d.]+)', data_part)
                        
                        loss = float(loss_match.group(1))
                        lbox = float(lbox_match.group(1))
                        lcls = float(lcls_match.group(1))
                        dfl = float(dfl_match.group(1))
                        lr = float(lr_match.group(1)) if lr_match else 0.0
                        
                        epochs.append(epoch)
                        steps.append(step)
                        losses.append(loss)
                        lbox_values.append(lbox)
                        lcls_values.append(lcls)
                        dfl_values.append(dfl)
                        lr_values.append(lr)
                    except Exception as e:
                        print(f"解析行失败: {e}")
                        continue
    except Exception as e:
        print(f"解析日志时出错: {e}")
        return None
    
    if not losses:
        print("警告: 未能从日志中提取任何数据")
        return None
    
    return {
        'epochs': epochs,
        'steps': steps,
        'loss': losses,
        'lbox': lbox_values,
        'lcls': lcls_values,
        'dfl': dfl_values,
        'lr': lr_values
    }


def plot_training_curves(data, output_dir):
    """绘制训练曲线"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 创建全局步数（用于 x 轴）
    global_steps = []
    for i, (epoch, step) in enumerate(zip(data['epochs'], data['steps'])):
        global_steps.append((epoch - 1) * 4028 + step)
    
    # 1. 总损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(global_steps, data['loss'], label='Total Loss', linewidth=2)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curve.png', dpi=300)
    plt.close()
    print(f"✓ 保存损失曲线: {output_dir / 'loss_curve.png'}")
    
    # 2. 各组件损失曲线
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(global_steps, data['lbox'], label='Box Loss', color='blue', linewidth=1.5)
    plt.xlabel('Steps')
    plt.ylabel('lbox')
    plt.title('Bounding Box Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(global_steps, data['lcls'], label='Class Loss', color='green', linewidth=1.5)
    plt.xlabel('Steps')
    plt.ylabel('lcls')
    plt.title('Classification Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(global_steps, data['dfl'], label='DFL Loss', color='red', linewidth=1.5)
    plt.xlabel('Steps')
    plt.ylabel('dfl')
    plt.title('Distribution Focal Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(global_steps, data['lr'], label='Learning Rate', color='orange', linewidth=1.5)
    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detailed_metrics.png', dpi=300)
    plt.close()
    print(f"✓ 保存详细指标: {output_dir / 'detailed_metrics.png'}")
    
    # 3. 统计摘要
    summary = f"""
训练统计摘要
{'='*50}

总体指标:
  最终 loss: {data['loss'][-1]:.4f}
  最低 loss: {min(data['loss']):.4f}
  
边界框损失 (lbox):
  最终值: {data['lbox'][-1]:.4f}
  最低值: {min(data['lbox']):.4f}
  
分类损失 (lcls):
  最终值: {data['lcls'][-1]:.4f}
  最低值: {min(data['lcls']):.4f}
  
分布损失 (dfl):
  最终值: {data['dfl'][-1]:.4f}
  最低值: {min(data['dfl']):.4f}
  
学习率:
  初始: {data['lr'][0]:.6f}
  最终: {data['lr'][-1]:.6f}

训练完成度:
  总 Epochs: {max(data['epochs'])}
  总 Steps: {max(global_steps)}
"""
    
    with open(output_dir / 'training_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(summary)
    print(f"✓ 保存训练摘要: {output_dir / 'training_summary.txt'}")


def main():
    print("=" * 60)
    print("FreshFruit_YOLO 训练结果可视化")
    print("=" * 60)
    
    # 1. 找到最新的训练结果
    latest_run = find_latest_run()
    if not latest_run:
        return
    
    # 2. 检查权重文件
    weights_dir = latest_run / 'weights'
    if weights_dir.exists():
        weight_files = list(weights_dir.glob('*.ckpt'))
        print(f"\n找到 {len(weight_files)} 个模型权重文件:")
        for wf in sorted(weight_files):
            size_mb = wf.stat().st_size / (1024 * 1024)
            print(f"  - {wf.name} ({size_mb:.2f} MB)")
    
    # 3. 解析日志
    log_dir = latest_run / 'logs'
    if not log_dir.exists():
        print(f"\n错误: 找不到日志目录 {log_dir}")
        return
    
    print(f"\n正在解析训练日志...")
    data = parse_log_file(log_dir)
    
    if data is None:
        print("无法解析日志数据")
        return
    
    print(f"成功解析 {len(data['loss'])} 个训练步骤")
    
    # 4. 绘制曲线
    print(f"\n正在生成可视化图表...")
    visualization_dir = latest_run / 'visualizations'
    plot_training_curves(data, visualization_dir)
    
    print(f"\n" + "=" * 60)
    print("✓ 可视化完成！")
    print(f"结果保存在: {visualization_dir}")
    print("=" * 60)
    
    # 5. 推理建议
    print("\n下一步建议:")
    print("1. 查看可视化结果:")
    print(f"   {visualization_dir}\\loss_curve.png")
    print(f"   {visualization_dir}\\detailed_metrics.png")
    print(f"   {visualization_dir}\\training_summary.txt")
    print("\n2. 使用最佳模型进行推理测试:")
    ema_weights = list(weights_dir.glob('EMA_*.ckpt'))
    if ema_weights:
        print(f"   推荐使用: {ema_weights[-1].name}")


if __name__ == '__main__':
    main()
