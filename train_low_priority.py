#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
低优先级训练脚本
设置进程优先级为"低于正常"，减少对系统的影响
"""

import os
import sys
import subprocess

# 设置环境变量限制线程数
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

# Windows 设置进程优先级
if sys.platform == 'win32':
    import psutil
    p = psutil.Process(os.getpid())
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)  # 低于正常优先级
    print(f"进程优先级已设置为: 低于正常")
    
    # 限制 CPU 亲和性到一半核心
    cpu_count = psutil.cpu_count()
    half_cpus = list(range(cpu_count // 2))  # 使用一半的核心
    p.cpu_affinity(half_cpus)
    print(f"CPU 亲和性限制为核心: {half_cpus}")

# MindSpore 设置
os.environ['MS_DEV_RUNTIME_CONF'] = '{"thread_num": 4}'

# 切换到项目目录
project_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_dir)

print(f"\n当前目录: {project_dir}")
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")

# 启动训练 - 使用 subprocess 继承优先级设置
# 从第5轮权重继续训练
resume_weight = 'runs/2026.01.27-22.21.07/weights/EMA_fruit_yolov8n-5_4028.ckpt'

cmd = [
    sys.executable,
    'D:/code/mindyolo_new/train.py',
    '--config', 'configs/fruit_yolov8n.yaml',
    '--weight', resume_weight,  # 使用之前训练的权重
    '--epochs', '50',  # 再训练50轮
    '--strict_load', 'False'
]

print(f"\n执行命令: {' '.join(cmd)}\n")

# 创建低优先级的子进程
if sys.platform == 'win32':
    # Windows: 使用低优先级，但不使用 CREATE_NEW_PROCESS_GROUP，以便响应 Ctrl+C
    import subprocess
    import signal
    
    startupinfo = subprocess.STARTUPINFO()
    process = subprocess.Popen(
        cmd,
        creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS
    )
    print(f"训练进程已启动 (PID: {process.pid})，优先级: 低于正常")
    print(f"按 Ctrl+C 可以中断训练\n")
    
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，正在终止训练进程...")
        process.terminate()
        try:
            process.wait(timeout=10)
            print("训练进程已终止")
        except subprocess.TimeoutExpired:
            print("进程未响应，强制结束...")
            process.kill()
            process.wait()
else:
    subprocess.run(cmd)
