#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FreshFruit_YOLO 训练脚本
基于 MindYOLO 的水果新鲜度检测训练入口

使用方法:
    python train.py                              # 使用默认配置
    python train.py --epochs 100                 # 指定训练轮数
    python train.py --config configs/xxx.yaml   # 指定配置文件
"""

import os
import sys
import argparse


def find_mindyolo_path():
    """自动查找 MindYOLO 安装路径"""
    # 常见的 MindYOLO 路径
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'mindyolo'),
        os.path.join(os.path.dirname(__file__), '..', 'mindyolo_new'),
        os.path.expanduser('~/mindyolo'),
        'D:/code/mindyolo',
        'D:/code/mindyolo_new',
    ]
    
    for path in possible_paths:
        if os.path.exists(os.path.join(path, 'mindyolo', 'models')):
            return os.path.abspath(path)
    
    return None


def main():
    parser = argparse.ArgumentParser(description='FreshFruit_YOLO 训练脚本')
    parser.add_argument('--config', type=str, default='configs/fruit_yolov8n.yaml',
                        help='配置文件路径 (默认: configs/fruit_yolov8n.yaml)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数 (覆盖配置文件中的值)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='批大小 (覆盖配置文件中的值)')
    parser.add_argument('--mindyolo-path', type=str, default=None,
                        help='MindYOLO 安装路径')
    args = parser.parse_args()
    
    # 切换到项目目录
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    # 查找 MindYOLO 路径
    mindyolo_path = args.mindyolo_path or find_mindyolo_path()
    
    if mindyolo_path is None:
        print("错误: 找不到 MindYOLO 安装路径!")
        print("请使用 --mindyolo-path 参数指定路径，或确保 MindYOLO 已正确安装。")
        print("\n安装 MindYOLO:")
        print("  git clone https://github.com/mindspore-lab/mindyolo.git")
        print("  cd mindyolo && pip install -e .")
        sys.exit(1)
    
    print(f"使用 MindYOLO 路径: {mindyolo_path}")
    
    # 添加到 Python 路径
    if mindyolo_path not in sys.path:
        sys.path.insert(0, mindyolo_path)
    
    # 构建训练命令
    train_script = os.path.join(mindyolo_path, 'train.py')
    if not os.path.exists(train_script):
        print(f"错误: 找不到训练脚本 {train_script}")
        sys.exit(1)
    
    # 构建命令行参数
    cmd_args = ['python', train_script, '--config', args.config]
    
    if args.epochs is not None:
        cmd_args.extend(['--epochs', str(args.epochs)])
    
    if args.batch_size is not None:
        cmd_args.extend(['--per_batch_size', str(args.batch_size)])
    
    # 打印执行的命令
    print(f"\n执行命令: {' '.join(cmd_args)}\n")
    
    # 执行训练
    import subprocess
    subprocess.run(cmd_args)


if __name__ == '__main__':
    main()
