# CoralSCOP_video 架构设计文档

## 总体定位

这是一个**珊瑚礁自动检测与分割**系统，基于 Meta 的 **Segment Anything Model (SAM)** 进行领域微调，支持静态图像和视频的分析，输出 COCO 标准 JSON（含 RLE 编码掩码）。

---

## 目录结构

```
CoralSCOP_video/
├── checkpoints/                  # 模型权重 (vit_b_coralscop.pth, ~382 MB)
├── segment_anything/             # 自定义 SAM 实现（核心算法包）
│   ├── modeling/                 # 神经网络组件
│   │   ├── sam.py                # 顶层 Sam 模型类
│   │   ├── image_encoder.py      # Vision Transformer 图像编码器
│   │   ├── prompt_encoder.py     # 点/框提示编码器
│   │   ├── mask_decoder.py       # 掩码解码器
│   │   └── transformer.py        # TwoWayTransformer
│   ├── automatic_mask_generator.py  # 全自动掩码生成（核心算法）
│   ├── predictor.py              # SamPredictor（带提示的推理）
│   └── utils/                    # 图像变换、NMS、RLE 编码等
├── test.py                       # 入口：静态图像检测
├── coralscop_video.py            # 入口：视频逐帧检测
├── coralscop_video_temporal.py   # 入口：视频时序追踪
└── coralscop_visualization.py    # 可视化工具
```

---

## 核心数据流

```
输入图像/视频帧
    ↓
Letterbox 缩放 → 1024×1024（保持纵横比）
    ↓
ImageEncoderViT → 提取图像特征嵌入（256维/patch）
    ↓
Point Grid 采样（32×32 均匀网格点）
    ↓
MaskDecoder → 生成候选掩码（每个点3个候选）
    ↓
过滤（IoU阈值0.72 / 稳定性0.62 / NMS / 面积过滤）
    ↓
逆变换 → 坐标还原到原始图像尺寸
    ↓
RLE 压缩 → 输出 COCO JSON
```

---

## 两种处理模式

| 模式 | 文件 | 特点 |
|------|------|------|
| **全自动模式** | `test.py` / `coralscop_video.py` | 图像级均匀网格点采样，全覆盖，无监督 |
| **时序追踪模式** | `coralscop_video_temporal.py` | 复用前一帧的掩码中心点作为下一帧的提示，速度更快，时序一致性更强 |

---

## 架构设计亮点

1. **三层分离**：`modeling/`（神经网络）→ `automatic_mask_generator.py`（算法）→ `test.py / coralscop_video.py`（应用）
2. **可插拔编码器**：支持 ViT-B / ViT-L / ViT-H 三种规模
3. **两种提示机制**：自动（均匀点网格）和手动（点、框、掩码提示）可独立使用
4. **高效存储**：掩码用 RLE 编码，比二进制存储节省 70-80%

---

## 关键参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `iou_threshold` | 0.72 | 低于此置信度的掩码被丢弃 |
| `sta_threshold` | 0.62 | 不稳定掩码（阈值偏移下变化大）被丢弃 |
| `point_number` | 32 | 每轴网格点数（32×32=1024个点） |
| `min_area` | 4096 | 最小掩码面积（像素），过滤噪声 |
| `skip` | 5 | 视频跳帧数 |


## 特别说明

研究专用 