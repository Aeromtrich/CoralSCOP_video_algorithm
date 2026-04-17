"""
CoralSCOP 可视化脚本（零依赖 detectron2 最终修复版）
修复内容：修正 Alpha 混合逻辑（仅 mask 区域混合，背景 100% 保留，避免全图变暗）
"""
import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端，避免 filter 警告
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
import cv2
import argparse

# 硬编码颜色（RGB 0-255 范围）
MASK_COLOR = np.array([138, 43, 226], dtype=np.float32)  # 珊瑚紫色
EDGE_COLOR_BGR = (0, 0, 255)  # OpenCV BGR 格式红色

class SimpleMaskVisualizer:
    """
    零依赖 detectron2 的掩膜可视化器
    关键修复：仅在 mask 区域进行 alpha 混合，背景像素完全保留原值
    """
    def __init__(self, img_rgb):
        """
        Args:
            img_rgb: numpy array (H, W, 3), RGB格式, uint8 (0-255)
        """
        if img_rgb.dtype != np.uint8:
            img_rgb = img_rgb.astype(np.uint8)
        self.base_img = img_rgb.copy()  # 保留原始图像作为基准
        self.result = img_rgb.astype(np.float32).copy()
        self.height, self.width = img_rgb.shape[:2]

    def draw_binary_mask_with_number(self, mask, color=None, edge_color=None, 
                                     text="", label_mode='1', alpha=0.4, anno_mode=None, **kwargs):
        """
        绘制二值掩膜（修复版：正确的 Alpha 混合）

        Args:
            mask: numpy array (H, W), bool 或 uint8 (0/1)
            color: array-like, RGB颜色 0-255 或 0-1
            edge_color: array-like, 边缘颜色（在OpenCV BGR中处理）
            alpha: float, 透明度 0-1（仅在mask区域生效）
        Returns:
            self（支持链式调用）
        """
        if color is None:
            color = MASK_COLOR
        else:
            # 统一转换为 0-255 float32
            color = np.array(color, dtype=np.float32)
            if color.max() <= 1.0:
                color = color * 255.0

        # 确保 mask 是 bool 数组
        mask_bool = mask.astype(bool)

        # 关键修复：仅在 mask 区域进行 alpha 混合
        # background: 100% 保留原像素
        # mask区域: base * (1-alpha) + color * alpha
        mask_y, mask_x = np.where(mask_bool)
        if len(mask_y) > 0:
            base_pixels = self.base_img[mask_y, mask_x].astype(np.float32)
            color_pixels = np.tile(color, (len(mask_y), 1))
            self.result[mask_y, mask_x] = base_pixels * (1.0 - alpha) + color_pixels * alpha

        # 绘制边缘（在 BGR 空间操作 OpenCV）
        mask_uint8 = mask_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 将当前结果转为 BGR 供 OpenCV 绘制
            result_bgr = cv2.cvtColor(self.result.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.drawContours(result_bgr, contours, -1, EDGE_COLOR_BGR, thickness=2)
            # 转回 RGB 继续后续处理
            self.result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        return self

    def get_image(self):
        """返回绘制后的图像，uint8格式，RGB"""
        return np.clip(self.result, 0, 255).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description='CoralSCOP 可视化工具（无 detectron2 依赖）')
    parser.add_argument("--img_path", type=str, required=True, help="原始图像目录路径")
    parser.add_argument("--json_path", type=str, required=True, help="SAM 输出的 JSON 目录路径")
    parser.add_argument("--output_path", type=str, required=True, help="可视化输出目录路径")
    parser.add_argument("--alpha", type=float, default=0.4, help="掩膜透明度 (0-1)，默认 0.4")
    parser.add_argument("--min_area", type=float, default=4096, help="最小掩膜面积，小于此值的掩膜将被过滤")
    args = parser.parse_args()

    alpha = args.alpha
    min_area = args.min_area

    # 确保输出目录存在
    os.makedirs(args.output_path, exist_ok=True)

    # 查找所有 JSON 文件
    json_files = sorted(glob.glob(os.path.join(args.json_path, "*.json")))
    if not json_files:
        print(f"警告：在 {args.json_path} 未找到 JSON 文件")
        return

    print(f"找到 {len(json_files)} 个 JSON 文件，开始处理...")

    for json_file in json_files:
        with open(json_file, "r", encoding='utf-8') as f:
            data = json.load(f)

        image_info = data['image']
        annotations = data['annotations']
        img_name = image_info['file_name']

        _, file_name = os.path.split(img_name)
        output_file = os.path.join(args.output_path, file_name)

        # 如果输出已存在则跳过
        if os.path.exists(output_file):
            print(f"  跳过（已存在）: {file_name}")
            continue

        # 查找对应的图像文件（支持 .jpg 和 .png）
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        img_candidates = [
            os.path.join(args.img_path, base_name + ".jpg"),
            os.path.join(args.img_path, base_name + ".png"),
            os.path.join(args.img_path, base_name + ".jpeg"),
        ]

        img_path = None
        for candidate in img_candidates:
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f"  跳过（未找到图像）: {base_name}.[jpg/png]")
            continue

        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"  错误：无法读取图像 {img_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 创建可视化器（使用修复后的类）
        visualizer = SimpleMaskVisualizer(image)

        # 统计绘制的掩膜数量
        drawn_count = 0
        total_area = 0

        for i, ann in enumerate(annotations):
            # 跳过空分割
            if not ann.get('segmentation'):
                continue

            # 解码 RLE
            try:
                mask = mask_util.decode(ann['segmentation'])
            except Exception as e:
                print(f"  警告：解码掩膜 {i} 失败: {e}")
                continue

            # 面积过滤
            mask_area = np.sum(mask)
            if mask_area < min_area:
                continue

            # 绘制掩膜（使用统一颜色，关键修复：仅 mask 区域混合）
            visualizer.draw_binary_mask_with_number(
                mask,
                color=MASK_COLOR,
                edge_color=EDGE_COLOR_BGR,
                text="",
                alpha=alpha
            )
            drawn_count += 1
            total_area += mask_area

        # 获取结果并保存
        result_img = visualizer.get_image()

        # 使用 matplotlib 保存（保持与原代码一致的输出质量）
        plt.figure(figsize=(20, 20), dpi=100)
        plt.imshow(result_img)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=150, 
                   facecolor='none', edgecolor='none')
        plt.close()

        print(f"  完成：{file_name} （{drawn_count} 个掩膜，总面积 {total_area} 像素）")

    print("全部处理完成！")

if __name__ == "__main__":
    main()