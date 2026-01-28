#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型推理测试脚本
对验证集图像进行检测并可视化结果
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import mindspore as ms
from mindspore import Tensor
import random

# 添加MindYOLO路径
MINDYOLO_PATH = r"D:\code\mindyolo"
if MINDYOLO_PATH not in sys.path:
    sys.path.insert(0, MINDYOLO_PATH)

from mindyolo.models import create_model
from mindyolo.data import COCO80_TO_COCO91_CLASS
from mindyolo.utils.config import parse_args, Config
import yaml


def load_class_names(classes_file='dataset/classes.txt'):
    """加载类别名称"""
    with open(classes_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names


def load_model(config_path, weight_path):
    """加载训练好的模型"""
    print(f"加载配置: {config_path}")
    print(f"加载权重: {weight_path}")
    
    # 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)
    
    # 转换为Config对象
    cfg = Config(cfg_dict)
    
    # 创建模型
    network = create_model(
        model_name=cfg.network.model_name,
        model_cfg=cfg.network,
        num_classes=cfg.data.nc,
        sync_bn=False
    )
    
    # 加载权重
    param_dict = ms.load_checkpoint(weight_path)
    ms.load_param_into_net(network, param_dict, strict_load=False)
    network.set_train(False)
    
    print("✓ 模型加载成功")
    return network, cfg


def preprocess_image(image_path, img_size=480):
    """预处理图像"""
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    img_h, img_w = img.shape[:2]
    
    # 保持宽高比缩放
    scale = min(img_size / img_h, img_size / img_w)
    new_h, new_w = int(img_h * scale), int(img_w * scale)
    
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # 填充到目标尺寸
    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    top = (img_size - new_h) // 2
    left = (img_size - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = img_resized
    
    # 转换为CHW格式并归一化
    img_normalized = canvas.astype(np.float32) / 255.0
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    img_batch = np.expand_dims(img_chw, axis=0)
    
    return Tensor(img_batch, ms.float32), img, (scale, top, left)


def postprocess(prediction, conf_thres=0.25, iou_thres=0.45):
    """后处理预测结果 - NMS
    YOLOv8格式: (num_boxes, 4+nc)
    [x, y, w, h, cls0, cls1, ..., cls16]
    """
    # prediction shape: (1, num_boxes, 4+nc)
    pred = prediction[0]  # 移除batch维度 -> (4725, 21)
    
    # 分离box和类别
    boxes = pred[:, :4]  # (4725, 4) - x,y,w,h
    class_scores = pred[:, 4:]  # (4725, 17) - 类别分数
    
    # 获取最大类别分数和类别ID
    class_confs = np.max(class_scores, axis=1)  # (4725,)
    class_ids = np.argmax(class_scores, axis=1)  # (4725,)
    
    # 过滤低置信度
    mask = class_confs > conf_thres
    boxes = boxes[mask]
    class_confs = class_confs[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0:
        return np.array([])
    
    # 转换box格式 (cx, cy, w, h) -> (x1, y1, x2, y2)
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    
    detections = np.column_stack([
        x1, y1, x2, y2,
        class_confs,
        class_ids
    ])
    
    # NMS
    detections = nms(detections, iou_thres)
    
    return detections


def nms(boxes, iou_threshold):
    """非极大值抑制"""
    if len(boxes) == 0:
        return np.array([])
    
    # 按置信度排序
    indices = np.argsort(boxes[:, 4])[::-1]
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # 计算IoU
        current_box = boxes[current, :4]
        other_boxes = boxes[indices[1:], :4]
        
        ious = compute_iou(current_box, other_boxes)
        
        # 保留IoU小于阈值的框
        indices = indices[1:][ious < iou_threshold]
    
    return boxes[keep]


def compute_iou(box, boxes):
    """计算IoU"""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    union = box_area + boxes_area - intersection
    
    return intersection / (union + 1e-6)


def draw_detections(image, detections, class_names, scale_info):
    """在图像上绘制检测框"""
    scale, top, left = scale_info
    img_h, img_w = image.shape[:2]
    
    # 随机颜色
    np.random.seed(42)
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255), 
               np.random.randint(0, 255)) for _ in range(len(class_names))]
    
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cls_id = int(cls_id)
        
        # 将坐标从480x480映射回原图
        x1 = (x1 - left) / scale
        y1 = (y1 - top) / scale
        x2 = (x2 - left) / scale
        y2 = (y2 - top) / scale
        
        # 限制在图像范围内
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))
        
        # 绘制边界框
        color = colors[cls_id]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # 绘制标签
        label = f"{class_names[cls_id]}: {conf:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        y1_label = max(y1, label_size[1])
        cv2.rectangle(image, 
                     (int(x1), int(y1_label - label_size[1])),
                     (int(x1 + label_size[0]), int(y1_label + baseline)),
                     color, -1)
        cv2.putText(image, label, (int(x1), int(y1_label)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image


def run_inference(model, image_path, class_names, img_size=480, conf_thres=0.25):
    """对单张图像进行推理"""
    # 预处理
    input_tensor, orig_img, scale_info = preprocess_image(image_path, img_size)
    
    # 推理
    prediction = model(input_tensor)
    
    # 如果返回是tuple，取第一个元素
    if isinstance(prediction, tuple):
        prediction = prediction[0]
    
    # 后处理
    detections = postprocess(prediction.asnumpy(), conf_thres=conf_thres)
    
    # 绘制结果
    result_img = draw_detections(orig_img.copy(), detections, class_names, scale_info)
    
    return result_img, len(detections)


def main():
    """主函数"""
    print("="*60)
    print("FreshFruit_YOLO 推理测试")
    print("="*60)
    
    # 设置MindSpore上下文
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    
    # 配置路径
    config_path = "configs/fruit_yolov8n.yaml"
    weight_path = "runs/2026.01.27-22.21.07/weights/EMA_fruit_yolov8n-5_4028.ckpt"
    
    # 加载类别名称
    class_names = load_class_names()
    print(f"类别数量: {len(class_names)}")
    print(f"类别: {', '.join(class_names)}\n")
    
    # 加载模型
    model, cfg = load_model(config_path, weight_path)
    img_size = cfg.img_size
    
    # 获取验证集图像
    val_image_dir = "dataset/images/val"
    val_images = list(Path(val_image_dir).glob("*.jpg")) + list(Path(val_image_dir).glob("*.png"))
    val_images = [str(img) for img in val_images]
    
    print(f"验证集图像数量: {len(val_images)}")
    
    # 随机选择10张图像进行测试
    test_images = random.sample(val_images, min(10, len(val_images)))
    
    # 创建输出目录
    output_dir = Path("runs/2026.01.27-22.21.07/inference_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n开始推理测试...")
    print(f"输出目录: {output_dir}\n")
    
    # 对每张图像进行推理
    total_detections = 0
    for idx, img_path in enumerate(test_images, 1):
        if not os.path.exists(img_path):
            print(f"[{idx}/{len(test_images)}] 图像不存在: {img_path}")
            continue
        
        try:
            result_img, num_dets = run_inference(model, img_path, class_names, 
                                                 img_size=img_size, conf_thres=0.25)
            
            # 保存结果
            img_name = Path(img_path).name
            output_path = output_dir / f"result_{idx}_{img_name}"
            cv2.imwrite(str(output_path), result_img)
            
            total_detections += num_dets
            print(f"[{idx}/{len(test_images)}] {img_name}: 检测到 {num_dets} 个对象")
            
        except Exception as e:
            print(f"[{idx}/{len(test_images)}] 推理失败: {img_path}")
            print(f"  错误: {e}")
    
    print("\n" + "="*60)
    print(f"✓ 推理完成！")
    print(f"  测试图像: {len(test_images)}")
    print(f"  总检测数: {total_detections}")
    print(f"  平均每图: {total_detections/len(test_images):.1f}")
    print(f"  结果保存在: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
