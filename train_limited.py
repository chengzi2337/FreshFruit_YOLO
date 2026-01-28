#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
强制限制 CPU 使用的训练脚本
使用 Windows 进程亲和性限制只能使用部分 CPU 核心
"""
import os
import sys
import subprocess
import psutil

def limit_cpu_affinity(num_cores=4):
    """限制当前进程只使用指定数量的 CPU 核心"""
    p = psutil.Process()
    # 获取可用 CPU 核心数
    total_cores = psutil.cpu_count()
    # 设置只使用前 num_cores 个核心
    cores_to_use = list(range(min(num_cores, total_cores)))
    p.cpu_affinity(cores_to_use)
    print(f"[INFO] CPU 亲和性已设置: 只使用核心 {cores_to_use}")
    return cores_to_use

if __name__ == '__main__':
    # ========== 1. 限制 CPU 亲和性 ==========
    try:
        limit_cpu_affinity(4)  # 只使用 4 个核心
    except Exception as e:
        print(f"[WARNING] 无法设置 CPU 亲和性: {e}")
    
    # ========== 2. 设置环境变量 ==========
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    os.environ['NUMEXPR_NUM_THREADS'] = '2'
    
    # MindSpore 特定设置
    os.environ['MS_DEV_SIDE_EFFECT_LOAD_ELIM'] = '3'
    
    print("[INFO] 环境变量已设置")
    print(f"[INFO] 当前工作目录: {os.getcwd()}")
    
    # ========== 3. 切换目录并运行训练 ==========
    os.chdir(r"D:\code\FreshFruit_YOLO")
    
    # 构建命令
    cmd = [
        r"D:\software\anaconda\envs\ms_win_cpu\python.exe",
        r"D:\code\mindyolo_new\train.py",
        "--config", r"D:\code\FreshFruit_YOLO\configs\fruit_yolov8n.yaml",
        "--weight", r"D:\code\FreshFruit_YOLO\weights\yolov8n_coco.ckpt",
        "--strict_load", "False",
        "--epochs", "3",
    ]
    
    print(f"[INFO] 执行命令: {' '.join(cmd)}")
    
    # 使用 subprocess 运行，继承 CPU 亲和性
    process = subprocess.Popen(cmd)
    
    # 设置子进程的 CPU 亲和性
    try:
        child = psutil.Process(process.pid)
        child.cpu_affinity([0, 1, 2, 3])  # 只使用 4 个核心
        print(f"[INFO] 子进程 CPU 亲和性已设置")
    except:
        pass
    
    # 等待完成
    process.wait()
