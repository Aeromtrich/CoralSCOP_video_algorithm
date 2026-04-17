import os.path
import pycocotools.mask as mask_util
import numpy as np
import cv2
import json
import glob
import os
import argparse
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def preprocess_for_sam(image, target_size=1024):
    """
    强制预处理图像为 1024x1024（SAM 模型输入要求）
    策略：等比例缩放 + 填充（letterbox）

    Returns:
        processed_image: 1024x1024 的 RGB 图像
        scale_ratio: 缩放比例（用于后续 mask 反变换）
        pad_info: (pad_top, pad_bottom, pad_left, pad_right) 填充信息
        original_size: (orig_h, orig_w)
    """
    orig_h, orig_w = image.shape[:2]

    # 计算等比例缩放，使长边等于 target_size
    scale = target_size / max(orig_h, orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # 等比例缩放
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    # 创建 1024x1024 画布并居中放置（letterbox）
    processed = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    processed[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized

    pad_info = (pad_top, target_size - new_h - pad_top, pad_left, target_size - new_w - pad_left)

    return processed, scale, pad_info, (orig_h, orig_w)

def postprocess_mask(mask_1024, pad_info, original_size):
    """
    将 1024x1024 的 mask 反变换回原始图像尺寸
    """
    pad_top, pad_bottom, pad_left, pad_right = pad_info
    orig_h, orig_w = original_size

    # 去除 padding
    h_valid = 1024 - pad_top - pad_bottom
    w_valid = 1024 - pad_left - pad_right
    mask_cropped = mask_1024[pad_top:pad_top+h_valid, pad_left:pad_left+w_valid]

    # 缩放回原始尺寸
    mask_orig = cv2.resize(mask_cropped.astype(np.uint8), (orig_w, orig_h), 
                           interpolation=cv2.INTER_NEAREST)
    return mask_orig.astype(bool)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=['vit_h', 'vit_l', 'vit_b'])
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--iou_threshold", type=float, required=True)
    parser.add_argument("--sta_threshold", type=float, required=True)
    parser.add_argument("--point_number", type=int, required=True)
    parser.add_argument("--test_img_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=1024, 
                       help="模型输入尺寸，固定为1024，非1024图像将被letterbox处理")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"[INFO] 强制输入尺寸: {args.image_size}x{args.image_size}")
    print(f"[INFO] 任意尺寸图像将被等比缩放+填充至 {args.image_size}x{args.image_size} 处理")
    print(f"[INFO] 加载模型: {args.model_type}")

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint_path)
    sam.to(device="cuda")

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.point_number,
        pred_iou_thresh=args.iou_threshold,
        stability_score_thresh=args.sta_threshold,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    all_files = glob.glob(os.path.join(args.test_img_path, "*.*"))
    img_files = [f for f in all_files if os.path.splitext(f)[1].lower() in valid_exts]
    img_files = sorted(img_files)

    print(f"[INFO] 找到 {len(img_files)} 个图像")

    if not img_files:
        print("[ERROR] 没有找到图像文件")
        return

    processed = 0
    skipped = 0
    errors = 0

    for img_path in img_files:
        basename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(basename)[0]
        json_path = os.path.join(args.output_path, name_no_ext + ".json")

        if os.path.exists(json_path):
            print(f"[SKIP] {basename} (已存在)")
            skipped += 1
            continue

        # 读取原图
        image = cv2.imread(img_path)
        if image is None:
            print(f"[ERROR] 无法读取: {basename}")
            errors += 1
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # 关键修复：强制预处理为 1024x1024（letterbox 方式）
        if orig_h == args.image_size and orig_w == args.image_size:
            # 已经是 1024x1024，直接处理
            image_sam = image
            scale = 1.0
            pad_info = (0, 0, 0, 0)
            print(f"[PROCESS] {basename} ({orig_w}x{orig_h}) -> 无需预处理")
        else:
            # 需要 letterbox 处理
            image_sam, scale, pad_info, orig_size = preprocess_for_sam(image, args.image_size)
            print(f"[PROCESS] {basename} ({orig_w}x{orig_h}) -> 预处理为 {args.image_size}x{args.image_size} (scale={scale:.3f})")

        # SAM 处理（现在输入严格为 1024x1024）
        try:
            masks = mask_generator.generate(image_sam)
        except Exception as e:
            print(f"  [ERROR] SAM 失败: {e}")
            errors += 1
            continue

        print(f"  -> 生成 {len(masks)} 个 masks")

        # 构建 JSON（使用原始尺寸）
        output_json = {
            'image': {
                'image_id': 0,
                'width': orig_w,
                'height': orig_h,
                'file_name': basename
            },
            'annotations': []
        }

        # 处理每个 mask：从 1024x1024 映射回原始尺寸
        for i, m in enumerate(masks):
            mask_1024 = m['segmentation']

            # 如果原图不是 1024x1024，需要将 mask 反变换
            if orig_h != args.image_size or orig_w != args.image_size:
                mask_orig = postprocess_mask(mask_1024, pad_info, (orig_h, orig_w))
            else:
                mask_orig = mask_1024

            # 编码为 RLE
            fortran_mask = np.asfortranarray(mask_orig)
            compressed_rle = mask_util.encode(fortran_mask)
            compressed_rle['counts'] = str(compressed_rle['counts'], encoding="utf-8")

            # bbox 也需要映射回原始坐标
            bbox = m['bbox']  # [x, y, w, h] 在 1024x1024 坐标系中
            if orig_h != args.image_size or orig_w != args.image_size:
                # 去除 pad 并除以 scale
                pad_top, pad_bottom, pad_left, pad_right = pad_info
                x, y, w, h = bbox
                # 减去 pad
                x = max(0, x - pad_left)
                y = max(0, y - pad_top)
                # 除以 scale 回到原图
                x = x / scale
                y = y / scale
                w = w / scale
                h = h / scale
                bbox = [x, y, w, h]

            output_json['annotations'].append({
                'segmentation': compressed_rle,
                'bbox': bbox,
                'area': float(np.sum(mask_orig)),
                'predicted_iou': m['predicted_iou'],
                'crop_box': m['crop_box'],
                'stability_score': m['stability_score'],
                'point_coords': m['point_coords'],
                'id': i
            })

        # 保存
        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(output_json, f)

        print(f"  -> 已保存 {os.path.basename(json_path)} ({len(masks)} masks)")
        processed += 1

    print(f"\n[DONE] 完成: 成功 {processed}, 跳过 {skipped}, 错误 {errors}")

if __name__ == "__main__":
    main()