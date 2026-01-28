# FreshFruit_YOLO - 水果新鲜度检测

基于 MindSpore + MindYOLO 的水果新鲜度检测项目，使用 YOLOv8 模型进行 17 类水果（健康/腐烂）分类检测。

## 🍎 项目介绍

本项目用于检测水果的新鲜程度，支持以下 17 个类别：
- Apple (healthy/rotten)
- Banana (healthy/rotten)  
- Beans (healthy/rotten)
- Grape (healthy/rotten)
- Mango (healthy/rotten)
- Orange (healthy/rotten)
- Potato (healthy/rotten)
- Tomato (healthy/rotten)
- non_produce (非农产品)

## 📁 项目结构

```
FreshFruit_YOLO/
├── configs/
│   ├── fruit_yolov8n.yaml      # YOLOv8n 训练配置 (推荐使用)
│   └── fruit_detect.yaml       # YOLOv5 配置 (备用)
├── dataset/
│   ├── classes.txt             # 类别列表
│   ├── data.yaml               # 数据集配置
│   ├── images/                 # 图片目录 (需自行准备)
│   │   ├── train/
│   │   └── val/
│   └── labels/                 # 标签目录 (需自行准备)
│       ├── train/
│       └── val/
├── .gitignore
├── README.md
└── requirements.txt
```

## 🔧 环境配置

### 1. 安装 MindSpore

```bash
# Windows CPU 版本
pip install mindspore==2.7.2

```

### 2. 安装 MindYOLO

```bash
git clone https://github.com/mindspore-lab/mindyolo.git
cd mindyolo
pip install -r requirements.txt
pip install -e .
```

### 3. 安装其他依赖

```bash
pip install -r requirements.txt
```

## 📊 数据集准备

数据集采用 YOLO 格式：

1. **图片目录结构**:
   ```
   dataset/images/train/xxx.jpg
   dataset/images/val/xxx.jpg
   ```

2. **标签格式** (每行一个目标):
   ```
   class_id x_center y_center width height
   ```
   - 所有值都是归一化的 (0-1)
   - 例如: `0 0.5 0.5 0.3 0.4`

3. **标签文件位置**:
   ```
   dataset/labels/train/xxx.txt
   dataset/labels/val/xxx.txt
   ```

## 🚀 训练

```bash
# 使用 YOLOv8n 训练 (推荐)
python path/to/mindyolo/train.py --config configs/fruit_yolov8n.yaml

# 指定 epochs
python path/to/mindyolo/train.py --config configs/fruit_yolov8n.yaml --epochs 100
```

### 配置说明

主要参数 (`configs/fruit_yolov8n.yaml`):
- `epochs`: 训练轮数 (默认 5，正式训练建议 100+)
- `per_batch_size`: 批大小 (CPU 建议 4，GPU 可增大)
- `lr_init`: 初始学习率 (默认 0.01)
- `img_size`: 输入图片尺寸 (默认 640)

## 📈 推理

```bash
python path/to/mindyolo/infer.py \
    --config configs/fruit_yolov8n.yaml \
    --weight runs/xxx/weights/best.ckpt \
    --image_path test.jpg
```

## 📝 注意事项

1. **Windows 用户**: 建议使用 Anaconda 创建虚拟环境
2. **CPU 训练**: 第一个 epoch 会进行图编译，约需 10-15 分钟
3. **数据路径**: 配置文件中的路径需要修改为你的实际路径

## 🔗 相关链接

- [MindSpore 官网](https://www.mindspore.cn/)
- [MindYOLO GitHub](https://github.com/mindspore-lab/mindyolo)

## 📄 License

MIT License
