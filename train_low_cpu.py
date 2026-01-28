#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
低负载训练脚本 - 限制 CPU 使用，防止过热重启
"""
import os
import sys

# ============ 关键：限制 CPU 核心数 ============
# 只使用一半的 CPU 核心，大幅降低发热
CPU_CORES = os.cpu_count() or 4
USE_CORES = max(2, CPU_CORES // 2)  # 使用一半核心，最少2个

os.environ['OMP_NUM_THREADS'] = str(USE_CORES)
os.environ['MKL_NUM_THREADS'] = str(USE_CORES)
os.environ['OPENBLAS_NUM_THREADS'] = str(USE_CORES)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(USE_CORES)
os.environ['NUMEXPR_NUM_THREADS'] = str(USE_CORES)

print(f"[INFO] CPU 核心数: {CPU_CORES}, 限制使用: {USE_CORES} 核心")
print(f"[INFO] 这将降低 CPU 负载约 50%")

# 添加 MindYOLO 路径
MINDYOLO_PATH = r"D:\code\mindyolo_new"
if MINDYOLO_PATH not in sys.path:
    sys.path.insert(0, MINDYOLO_PATH)

# 切换到项目目录
os.chdir(r"D:\code\FreshFruit_YOLO")

# 导入并运行训练
if __name__ == '__main__':
    from train import main as train_main
    
    # 设置命令行参数
    sys.argv = [
        'train.py',
        '--config', 'configs/fruit_yolov8n.yaml',
        '--weight', 'weights/yolov8n_coco.ckpt',
        '--strict_load', 'False',
        '--epochs', '3',
    ]
    
    # 运行 MindYOLO 训练
    exec(open(os.path.join(MINDYOLO_PATH, 'train.py')).read())
