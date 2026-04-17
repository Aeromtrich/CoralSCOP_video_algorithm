# CoralSCOP 珊瑚礁检测系统 - 迁移与使用文档

## 一、环境打包与迁移（当前机 → 内网机）

### 方法 A：使用 conda-pack（推荐，最可靠）

在当前机执行：

```bash
# 1. 安装打包工具
conda install -c conda-forge conda-pack -y

# 2. 打包当前环境（coralscop）
conda pack -n coralscop -o coralscop_env.tar.gz

# 3. 将以下文件传输到内网机：
# - coralscop_env.tar.gz（环境包，约500MB-2GB）
# - 本代码仓库（CoralSCOP/目录，含test.py, coralscop_video.py等）
# - 模型权重文件（checkpoints/vit_b_coralscop.pth，约几百MB）
```

在内网机恢复：

```bash
# 1. 解压环境到目标目录
mkdir -p /path/to/coralscop_env
tar -xzf coralscop_env.tar.gz -C /path/to/coralscop_env

# 2. 激活环境（无需conda，直接用python）
# Windows:
/path/to/coralscop_env/python.exe test.py ...

# Linux:
/path/to/coralscop_env/bin/python test.py ...
```

### 方法 B：pip freeze + 离线安装（备选）

如果 conda-pack 失败，使用 pip 方式：

```bash
# 在当前机导出依赖列表
pip list --format=freeze > requirements.txt

# 下载所有 whl 文件到本地目录（用于离线安装）
pip download -r requirements.txt -d ./offline_packages/ --platform win_amd64

# 传输到内网机后安装
pip install --no-index --find-links=./offline_packages/ -r requirements.txt
```

### 方法 C：Docker 镜像（如果内网机支持）

```dockerfile
FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04
COPY coralscop_env.tar.gz /tmp/
RUN tar -xzf /tmp/coralscop_env.tar.gz -C /opt/conda/envs/
WORKDIR /app
COPY . /app
CMD ["/opt/conda/envs/coralscop/bin/python", "test.py"]
```

---

## 二、环境恢复检查清单

在内网机首次运行前，确认以下组件：

```bash
# 1. Python 版本（应为 3.10）
python --version

# 2. CUDA 可用性（必须显示 True）
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"

# 3. 关键依赖版本检查
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import pycocotools; print('pycocotools: OK')"
python -c "from segment_anything import sam_model_registry; print('SAM: OK')"

# 4. 模型文件存在性检查
ls checkpoints/vit_b_coralscop.pth
```

**常见问题修复**：
- **NumPy 版本冲突**：`pip install numpy==1.23.5`
- **OpenCV 读取失败**：检查视频文件路径，确保无中文/空格
- **CUDA 不可用**：检查 NVIDIA 驱动，或改用 CPU 模式（修改 device="cpu"，速度极慢）

---

## 三、珊瑚礁图像检测算法使用办法

### 3.1 单张/批量图像检测（test.py）

**功能**：对静态图像进行珊瑚分割，输出 COCO 格式 JSON

**基本命令**：

```bash
python test.py   --model_type vit_b   --checkpoint_path ./checkpoints/vit_b_coralscop.pth   --test_img_path ./input_images/   --output_path ./output_jsons/   --iou_threshold 0.72   --sta_threshold 0.62   --point_number 32   --gpu 0
```

**参数说明**：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--model_type` | 模型类型 | `vit_b`（平衡速度/精度）或 `vit_h`（高精度慢速） |
| `--checkpoint_path` | 训练好的权重文件路径 | 必须存在 |
| `--test_img_path` | 输入图像目录 | 支持任意分辨率，自动预处理为1024x1024 |
| `--output_path` | JSON输出目录 | 每图生成一个同名.json文件 |
| `--iou_threshold` | IoU阈值，过滤重叠mask | 0.72（珊瑚密集场景）或 0.85（稀疏场景） |
| `--sta_threshold` | 稳定性阈值，过滤噪声 | 0.62 |
| `--point_number` | 每边采样点数，控制mask密度 | 32（默认），64（更精细但慢4倍） |
| `--gpu` | GPU设备ID | 0（单卡）或 0/1/2/3（多卡选择） |

**输出文件结构**：

```
output_jsons/
├── IMG_001.json          # 与输入图像同名，仅扩展名改为.json
├── IMG_002.json
└── ...

# JSON内容格式（COCO标准+RLE编码）：
{
  "image": {
    "width": 1920,           # 原始图像宽度
    "height": 1080,          # 原始图像高度
    "file_name": "IMG_001.jpg"
  },
  "annotations": [
    {
      "id": 0,
      "segmentation": {"size": [1080, 1920], "counts": "...RLE字符串..."},
      "bbox": [x, y, w, h],  # 边界框，与原始图像坐标一致
      "area": 12345,         # mask面积（像素）
      "predicted_iou": 0.95, # SAM置信度
      "stability_score": 0.88
    },
    ...
  ]
}
```

**重要特性**：
- **自动分辨率适配**：任意输入分辨率 → 内部1024x1024处理 → 输出映射回原图坐标
- **跳过已处理文件**：如果输出目录已有同名.json，自动跳过（断点续传）
- **RLE压缩**：分割结果使用COCO标准RLE编码，空间效率高

### 3.2 结果可视化（coralscop_visualization.py）

将JSON结果叠加到原图生成可视化图像：

```bash
python coralscop_visualization.py   --img_path ./input_images/   --json_path ./output_jsons/   --output_path ./visualizations/   --alpha 0.4   --min_area 4096
```

**可视化效果**：
- 紫色半透明 overlay：珊瑚mask区域
- 红色轮廓线：mask边界
- 绿色数字标注：mask ID

---

## 四、视频检测算法使用办法

### 4.1 视频逐帧处理（coralscop_video.py）

**功能**：对视频逐帧（或跳帧）进行珊瑚分割，每帧输出独立JSON，可选生成带mask的可视化视频

**基本命令**：

```bash
python coralscop_video.py   --video ./underwater_video.mp4   --checkpoint ./checkpoints/vit_b_coralscop.pth   --output_dir ./video_results/jsons/   --vis_video ./video_results/visualization.mp4   --skip 5   --gpu 0   --min_area 4096
```

**关键参数**：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--video` | 输入视频路径 | MP4/AVI/MOV等OpenCV支持格式 |
| `--output_dir` | JSON输出目录 | 每帧生成 frame_000001.json |
| `--vis_video` | 可视化视频路径（可选） | 带mask叠加的输出视频 |
| `--skip` | 跳帧间隔 | 5（30fps视频→6fps处理，省算力） |
| `--min_area` | 最小mask面积过滤 | 4096（过滤噪声点） |

**输出结构**：

```
video_results/
├── jsons/
│   ├── frame_000001.json     # 第1帧结果
│   ├── frame_000006.json     # 第6帧结果（skip=5）
│   ├── frame_000011.json
│   └── ...
└── visualization.mp4         # 带mask的可视化视频（如果指定--vis_video）

# JSON内容（含帧号）：
{
  "frame_id": 6,              # 视频帧号（从1开始）
  "image": {"width": 1920, "height": 1080, "file_name": "frame_000006.jpg"},
  "annotations": [...]        # 同图像检测格式
}
```

**性能优化建议**：

1. **跳帧处理（--skip）**：
   - 视频30fps，珊瑚移动缓慢时，每5帧处理1帧即可（--skip 5）
   - 处理速度提升5倍，精度损失极小

2. **分辨率预处理**：
   - 如果视频是4K（3840x2160），内部自动下采样到1024x576处理
   - 无需手动缩放，算法自动Letterbox处理


---

## 五、目录结构规范（建议）

```
CoralSCOP/
├── checkpoints/
│   └── vit_b_coralscop.pth          # 模型权重（必需）
├── input/                             # 输入数据
│   ├── images/                        # 静态图像
│   └── videos/                        # 视频文件
├── output/                            # 输出结果
│   ├── image_jsons/                   # 图像检测结果
│   ├── video_jsons/                   # 视频检测结果（按视频分子目录）
│   └── visualizations/                # 可视化图像/视频
├── test.py                            # 图像检测主程序
├── coralscop_video.py                 # 视频检测主程序
├── coralscop_visualization.py         # 可视化工具
└── README.md                          # 本说明文档
```

---



