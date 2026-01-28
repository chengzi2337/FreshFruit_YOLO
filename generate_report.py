#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成HTML报告展示推理结果
"""

import os
from pathlib import Path
import base64

def generate_html_report(result_dir="runs/2026.01.27-22.21.07/inference_results"):
    """生成HTML报告展示所有检测结果"""
    
    result_path = Path(result_dir)
    images = sorted(result_path.glob("result_*.jpg"))
    
    if not images:
        print("未找到结果图像")
        return
    
    html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FreshFruit_YOLO 推理结果</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        header {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
            text-align: center;
        }
        h1 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #666;
            font-size: 1.2em;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            text-align: center;
        }
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 25px;
        }
        .card {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.25);
        }
        .card img {
            width: 100%;
            height: auto;
            display: block;
        }
        .card-title {
            padding: 15px;
            font-size: 1.1em;
            color: #333;
            background: #f8f9fa;
            border-top: 3px solid #667eea;
        }
        footer {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-top: 30px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🍎 FreshFruit_YOLO 推理结果</h1>
            <p class="subtitle">基于MindSpore YOLOv8n的水果检测模型</p>
        </header>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">""" + str(len(images)) + """</div>
                <div class="stat-label">测试图像</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">17</div>
                <div class="stat-label">类别数量</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">5</div>
                <div class="stat-label">训练轮数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">480×480</div>
                <div class="stat-label">输入尺寸</div>
            </div>
        </div>
        
        <div class="gallery">
"""
    
    # 添加每张图像
    for idx, img_path in enumerate(images, 1):
        # 读取图像并转换为base64
        with open(img_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
        
        img_name = img_path.stem.replace('result_', '').replace('_', ' ')
        
        html_content += f"""
            <div class="card">
                <img src="data:image/jpeg;base64,{img_data}" alt="{img_name}">
                <div class="card-title">
                    #{idx} {img_name}
                </div>
            </div>
"""
    
    html_content += """
        </div>
        
        <footer>
            <p>模型权重: EMA_fruit_yolov8n-5_4028.ckpt</p>
            <p>训练配置: batch_size=1, img_size=480, accumulate=4</p>
            <p>生成时间: """ + str(Path(result_dir).stat().st_mtime) + """</p>
        </footer>
    </div>
</body>
</html>
"""
    
    # 保存HTML文件
    html_path = result_path / "inference_report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ HTML报告已生成: {html_path}")
    print(f"  共包含 {len(images)} 张检测结果图像")
    return html_path


if __name__ == "__main__":
    print("="*60)
    print("生成推理结果HTML报告")
    print("="*60)
    
    html_file = generate_html_report()
    
    print("\n" + "="*60)
    print("请在浏览器中打开以下文件查看结果:")
    print(f"  {html_file.absolute()}")
    print("="*60)
