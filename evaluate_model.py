#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型评估脚本 - 计算mAP等指标
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import mindspore as ms
from mindspore import Tensor
import yaml

# 添加MindYOLO路径
MINDYOLO_PATH = r"D:\code\mindyolo"
if MINDYOLO_PATH not in sys.path:
    sys.path.insert(0, MINDYOLO_PATH)

from mindyolo.models import create_model
from mindyolo.utils.config import Config


def load_class_names(classes_file='dataset/classes.txt'):
    """加载类别名称"""
    with open(classes_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names


def load_model(config_path, weight_path):
    """加载训练好的模型"""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = Config(cfg_dict)
    
    network = create_model(
        model_name=cfg.network.model_name,
        model_cfg=cfg.network,
        num_classes=cfg.data.nc,
        sync_bn=False
    )
    
    param_dict = ms.load_checkpoint(weight_path)
    ms.load_param_into_net(network, param_dict, strict_load=False)
    network.set_train(False)
    
    return network, cfg


def preprocess_image(image_path, img_size=480):
    """预处理图像"""
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None
    
    img_h, img_w = img.shape[:2]
    scale = min(img_size / img_h, img_size / img_w)
    new_h, new_w = int(img_h * scale), int(img_w * scale)
    
    img_resized = cv2.resize(img, (new_w, new_h))
    
    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    top = (img_size - new_h) // 2
    left = (img_size - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = img_resized
    
    img_normalized = canvas.astype(np.float32) / 255.0
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    img_batch = np.expand_dims(img_chw, axis=0)
    
    return Tensor(img_batch, ms.float32), (img_h, img_w), (scale, top, left)


def postprocess(prediction, conf_thres=0.001, iou_thres=0.6):
    """后处理 - 使用较低阈值以获取完整的PR曲线"""
    pred = prediction[0]
    boxes = pred[:, :4]
    class_scores = pred[:, 4:]
    
    class_confs = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)
    
    mask = class_confs > conf_thres
    boxes = boxes[mask]
    class_confs = class_confs[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0:
        return np.array([])
    
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    
    detections = np.column_stack([x1, y1, x2, y2, class_confs, class_ids])
    detections = nms_per_class(detections, iou_thres)
    
    return detections


def nms_per_class(boxes, iou_threshold):
    """按类别进行NMS"""
    if len(boxes) == 0:
        return np.array([])
    
    unique_classes = np.unique(boxes[:, 5])
    keep_boxes = []
    
    for cls in unique_classes:
        cls_mask = boxes[:, 5] == cls
        cls_boxes = boxes[cls_mask]
        
        indices = np.argsort(cls_boxes[:, 4])[::-1]
        keep = []
        
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            current_box = cls_boxes[current, :4]
            other_boxes = cls_boxes[indices[1:], :4]
            ious = compute_iou(current_box, other_boxes)
            indices = indices[1:][ious < iou_threshold]
        
        keep_boxes.extend(cls_boxes[keep])
    
    return np.array(keep_boxes) if keep_boxes else np.array([])


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


def load_ground_truth(label_path, img_size, orig_size, scale_info):
    """加载真实标签"""
    if not os.path.exists(label_path):
        return np.array([])
    
    scale, top, left = scale_info
    img_h, img_w = orig_size
    
    gt_boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            
            # 转换为原图坐标
            cx_orig = cx * img_w
            cy_orig = cy * img_h
            w_orig = w * img_w
            h_orig = h * img_h
            
            # 转换为缩放后的坐标（与预测对齐）
            cx_scaled = cx_orig * scale + left
            cy_scaled = cy_orig * scale + top
            w_scaled = w_orig * scale
            h_scaled = h_orig * scale
            
            x1 = cx_scaled - w_scaled / 2
            y1 = cy_scaled - h_scaled / 2
            x2 = cx_scaled + w_scaled / 2
            y2 = cy_scaled + h_scaled / 2
            
            gt_boxes.append([x1, y1, x2, y2, cls_id])
    
    return np.array(gt_boxes) if gt_boxes else np.array([])


def compute_ap(recalls, precisions):
    """计算AP (11点插值法)"""
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])
    
    # 确保precision单调递减
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # 11点插值
    ap = 0
    for t in np.linspace(0, 1, 11):
        p = precisions[recalls >= t]
        ap += p.max() if len(p) > 0 else 0
    
    return ap / 11


def evaluate_class(all_detections, all_gt, iou_threshold=0.5):
    """评估单个类别"""
    if len(all_gt) == 0:
        return 0.0, 0.0, 0.0
    
    if len(all_detections) == 0:
        return 0.0, 0.0, 0.0
    
    # 按置信度排序
    sorted_indices = np.argsort(all_detections[:, 4])[::-1]
    all_detections = all_detections[sorted_indices]
    
    tp = np.zeros(len(all_detections))
    fp = np.zeros(len(all_detections))
    
    # 标记每个gt是否已匹配 (按图像ID)
    gt_matched = {}
    
    for det_idx, det in enumerate(all_detections):
        img_id = int(det[6])  # 图像ID在第7列
        det_box = det[:4]
        
        # 获取该图像的gt (GT的图像ID在第5列，索引4)
        img_gt_mask = all_gt[:, 4] == img_id
        img_gt = all_gt[img_gt_mask]
        
        if len(img_gt) == 0:
            fp[det_idx] = 1
            continue
        
        # 初始化该图像的gt匹配状态
        if img_id not in gt_matched:
            gt_matched[img_id] = np.zeros(len(img_gt))
        
        # 计算与所有gt的IoU
        ious = []
        for gt in img_gt:
            iou = compute_iou(det_box, gt[:4].reshape(1, -1))[0]
            ious.append(iou)
        ious = np.array(ious)
        
        # 找最大IoU
        max_iou_idx = np.argmax(ious)
        max_iou = ious[max_iou_idx]
        
        if max_iou >= iou_threshold:
            # 检查该gt是否已被匹配
            if gt_matched[img_id][max_iou_idx] == 0:
                tp[det_idx] = 1
                gt_matched[img_id][max_iou_idx] = 1
            else:
                fp[det_idx] = 1
        else:
            fp[det_idx] = 1
    
    # 计算累积TP和FP
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    
    # 计算precision和recall
    precision = cum_tp / (cum_tp + cum_fp + 1e-6)
    recall = cum_tp / len(all_gt)
    
    # 计算AP
    ap = compute_ap(recall, precision)
    
    # 最终的precision和recall（使用conf=0.5阈值）
    conf_mask = all_detections[:, 4] >= 0.5
    final_tp = tp[conf_mask].sum() if conf_mask.any() else 0
    final_fp = fp[conf_mask].sum() if conf_mask.any() else 0
    
    final_precision = final_tp / (final_tp + final_fp + 1e-6)
    final_recall = final_tp / len(all_gt)
    
    return ap, final_precision, final_recall


def main():
    """主函数"""
    print("="*60)
    print("FreshFruit_YOLO 模型评估 (mAP计算)")
    print("="*60)
    
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    
    # 配置
    config_path = "configs/fruit_yolov8n.yaml"
    weight_path = "runs/2026.01.27-22.21.07/weights/EMA_fruit_yolov8n-5_4028.ckpt"
    val_image_dir = "dataset/images/val"
    val_label_dir = "dataset/labels/val"
    img_size = 480
    
    # 加载类别
    class_names = load_class_names()
    num_classes = len(class_names)
    print(f"类别数量: {num_classes}")
    
    # 加载模型
    print("加载模型...")
    model, cfg = load_model(config_path, weight_path)
    print("✓ 模型加载成功\n")
    
    # 获取验证集图像
    val_images = list(Path(val_image_dir).glob("*.jpg")) + list(Path(val_image_dir).glob("*.png"))
    print(f"验证集图像: {len(val_images)}")
    
    # 收集所有检测结果和GT
    all_detections = defaultdict(list)  # {class_id: [(x1,y1,x2,y2,conf,cls,img_id), ...]}
    all_gt = defaultdict(list)  # {class_id: [(x1,y1,x2,y2,img_id), ...]}
    
    print("\n开始评估...")
    for img_idx, img_path in enumerate(val_images):
        if img_idx % 50 == 0:
            print(f"  处理: {img_idx+1}/{len(val_images)}")
        
        # 获取标签路径
        label_name = img_path.stem + ".txt"
        label_path = Path(val_label_dir) / label_name
        
        # 预处理
        input_tensor, orig_size, scale_info = preprocess_image(str(img_path), img_size)
        if input_tensor is None:
            continue
        
        # 推理
        prediction = model(input_tensor)
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        
        # 后处理
        detections = postprocess(prediction.asnumpy())
        
        # 加载GT
        gt_boxes = load_ground_truth(str(label_path), img_size, orig_size, scale_info)
        
        # 收集检测结果
        for det in detections:
            cls_id = int(det[5])
            all_detections[cls_id].append(np.append(det, img_idx))
        
        # 收集GT
        for gt in gt_boxes:
            cls_id = int(gt[4])
            all_gt[cls_id].append(np.append(gt[:4], img_idx))
    
    print("\n" + "="*60)
    print("评估结果 (IoU=0.5)")
    print("="*60)
    
    # 计算每个类别的AP
    aps = []
    precisions = []
    recalls = []
    
    print(f"\n{'类别':<25} {'AP':>8} {'Precision':>10} {'Recall':>10} {'GT数量':>8}")
    print("-"*65)
    
    for cls_id in range(num_classes):
        cls_dets = np.array(all_detections[cls_id]) if all_detections[cls_id] else np.array([])
        cls_gt = np.array(all_gt[cls_id]) if all_gt[cls_id] else np.array([])
        
        # GT格式已经是 [x1, y1, x2, y2, img_id]
        
        ap, prec, rec = evaluate_class(cls_dets, cls_gt, iou_threshold=0.5)
        
        aps.append(ap)
        precisions.append(prec)
        recalls.append(rec)
        
        gt_count = len(cls_gt)
        if gt_count > 0:
            print(f"{class_names[cls_id]:<25} {ap*100:>7.2f}% {prec*100:>9.2f}% {rec*100:>9.2f}% {gt_count:>8}")
    
    print("-"*65)
    
    # 计算mAP
    valid_aps = [ap for ap, gt in zip(aps, [len(all_gt[i]) for i in range(num_classes)]) if gt > 0]
    mAP = np.mean(valid_aps) if valid_aps else 0
    mean_precision = np.mean([p for p, gt in zip(precisions, [len(all_gt[i]) for i in range(num_classes)]) if gt > 0])
    mean_recall = np.mean([r for r, gt in zip(recalls, [len(all_gt[i]) for i in range(num_classes)]) if gt > 0])
    
    print(f"\n{'总体指标':=^60}")
    print(f"\n  mAP@0.5:     {mAP*100:.2f}%")
    print(f"  Precision:   {mean_precision*100:.2f}%")
    print(f"  Recall:      {mean_recall*100:.2f}%")
    print(f"  F1-Score:    {2*mean_precision*mean_recall/(mean_precision+mean_recall+1e-6)*100:.2f}%")
    
    # 保存结果
    result_path = Path("runs/2026.01.27-22.21.07/evaluation_results.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("FreshFruit_YOLO 模型评估结果\n")
        f.write("="*60 + "\n\n")
        f.write(f"模型: {weight_path}\n")
        f.write(f"验证集: {len(val_images)} 张图像\n")
        f.write(f"IoU阈值: 0.5\n\n")
        f.write(f"{'类别':<25} {'AP':>8} {'Precision':>10} {'Recall':>10}\n")
        f.write("-"*55 + "\n")
        for cls_id in range(num_classes):
            gt_count = len(all_gt[cls_id])
            if gt_count > 0:
                f.write(f"{class_names[cls_id]:<25} {aps[cls_id]*100:>7.2f}% {precisions[cls_id]*100:>9.2f}% {recalls[cls_id]*100:>9.2f}%\n")
        f.write("-"*55 + "\n")
        f.write(f"\nmAP@0.5:     {mAP*100:.2f}%\n")
        f.write(f"Precision:   {mean_precision*100:.2f}%\n")
        f.write(f"Recall:      {mean_recall*100:.2f}%\n")
        f.write(f"F1-Score:    {2*mean_precision*mean_recall/(mean_precision+mean_recall+1e-6)*100:.2f}%\n")
    
    print(f"\n✓ 评估结果已保存: {result_path}")
    print("="*60)


if __name__ == "__main__":
    main()
