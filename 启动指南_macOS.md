# CoralSCOP macOS 启动指南

## 第一步：创建 conda 环境

```bash
conda create -n coralscop python=3.10 -y
conda activate coralscop
```

## 第二步：安装依赖（macOS CPU版）

```bash
pip install torch torchvision
pip install opencv-python pycocotools pillow numpy==1.23.5
pip install segment-anything tqdm
```

## 第三步：准备输入数据

把图片放到 `./input_images/` 目录下。

## 第四步：运行图像检测

```bash
python test.py \
  --model_type vit_b \
  --checkpoint_path ./checkpoints/vit_b_coralscop.pth \
  --test_img_path ./input_images/ \
  --output_path ./output_jsons/ \
  --iou_threshold 0.72 \
  --sta_threshold 0.62 \
  --point_number 32
```

> macOS 无 NVIDIA GPU，不加 `--gpu 0`，自动使用 CPU 运行（速度较慢）。
