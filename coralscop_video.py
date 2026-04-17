import os
import cv2
import json
import numpy as np
import argparse
from tqdm import tqdm
import pycocotools.mask as mask_util
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch

def preprocess_for_sam(image, target_size=1024):
    """强制预处理为target_sizextarget_size（Letterbox）"""
    orig_h, orig_w = image.shape[:2]
    scale = target_size / max(orig_h, orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = cv2.resize(image, (new_w, new_h), 
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    processed = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    processed[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized

    pad_bottom = target_size - new_h - pad_top
    pad_right = target_size - new_w - pad_left

    return processed, scale, (pad_top, pad_bottom, pad_left, pad_right), (orig_h, orig_w)

def postprocess_mask(mask_input, pad_info, original_size, target_size=1024):
    """
    将target_sizextarget_size的mask映射回原始尺寸
    修复：处理SAM可能返回的一维张量情况
    """
    # 转换为numpy并处理torch张量
    if isinstance(mask_input, torch.Tensor):
        mask_input = mask_input.cpu().numpy()

    # 处理一维异常情况（如果SAM错误地返回了扁平化数组）
    if mask_input.ndim == 1:
        expected_size = target_size * target_size
        if mask_input.size == expected_size:
            print(f"    [DEBUG] Reshaping 1D mask ({mask_input.size}) to {target_size}x{target_size}")
            mask_input = mask_input.reshape(target_size, target_size)
        else:
            print(f"    [警告] Mask 形状异常: {mask_input.shape}，期望 {expected_size} 或 ({target_size},{target_size})，跳过")
            return None
    elif mask_input.ndim != 2:
        print(f"    [警告] Mask 维度异常: {mask_input.ndim}D，期望 2D，跳过")
        return None

    pad_top, pad_bottom, pad_left, pad_right = pad_info
    orig_h, orig_w = original_size

    h_valid = target_size - pad_top - pad_bottom
    w_valid = target_size - pad_left - pad_right

    if h_valid <= 0 or w_valid <= 0:
        print(f"    [警告] 有效尺寸非法: {h_valid}x{w_valid}，跳过")
        return None

    # 裁剪有效区域（去除letterbox填充）
    mask_cropped = mask_input[pad_top:pad_top+h_valid, pad_left:pad_left+w_valid]

    # 缩放回原始尺寸
    mask_orig = cv2.resize(mask_cropped.astype(np.uint8), (orig_w, orig_h), 
                           interpolation=cv2.INTER_NEAREST)
    return mask_orig.astype(bool)

class VideoCoralSCOP:
    def __init__(self, model_type, checkpoint, gpu=0, 
                 iou_thresh=0.72, sta_thresh=0.62, point_number=32):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.to(device="cuda")
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=point_number,
            pred_iou_thresh=iou_thresh,
            stability_score_thresh=sta_thresh,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        self.target_size = 1024
        print(f"[初始化完成] 模型: {model_type}, GPU: {gpu}, 强制输入尺寸: {self.target_size}x{self.target_size}")

    def process_video(self, video_path, output_dir, vis_output=None, skip_frames=1, min_area=4096):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[视频信息] {width}x{height}, {fps}fps, 共{total_frames}帧")
        print(f"[处理策略] 每{skip_frames}帧处理1帧，预计处理{total_frames//skip_frames}帧")

        os.makedirs(output_dir, exist_ok=True)

        writer = None
        if vis_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(vis_output, fourcc, fps//skip_frames, (width, height))

        frame_idx = 0
        processed_count = 0
        error_count = 0

        pbar = tqdm(total=total_frames, desc="处理视频")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            pbar.update(1)

            if (frame_idx - 1) % skip_frames != 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = frame_rgb.shape[:2]

            # 预处理为1024x1024（或target_size）
            frame_sam, scale, pad_info, (orig_h, orig_w) = preprocess_for_sam(frame_rgb, self.target_size)

            # SAM生成masks
            try:
                masks = self.mask_generator.generate(frame_sam)
            except Exception as e:
                print(f"\n[警告] 第{frame_idx}帧 SAM 失败: {e}")
                error_count += 1
                continue

            # 构建结果
            annotations = []
            valid_mask_count = 0

            for i, m in enumerate(masks):
                try:
                    mask_1024 = m['segmentation']

                    # 后处理：映射回原始尺寸
                    if orig_h != self.target_size or orig_w != self.target_size:
                        mask_orig = postprocess_mask(mask_1024, pad_info, (orig_h, orig_w), self.target_size)
                        if mask_orig is None:
                            continue
                    else:
                        mask_orig = mask_1024 if isinstance(mask_1024, np.ndarray) else mask_1024.cpu().numpy()

                    area = np.sum(mask_orig)
                    if area < min_area:
                        continue

                    # 编码RLE
                    fortran_mask = np.asfortranarray(mask_orig)
                    compressed_rle = mask_util.encode(fortran_mask)
                    compressed_rle['counts'] = str(compressed_rle['counts'], encoding="utf-8")

                    # bbox映射（SAM输出的是1024坐标系）
                    bbox = m['bbox']  # [x, y, w, h]
                    if orig_h != self.target_size or orig_w != self.target_size:
                        pad_top, _, pad_left, _ = pad_info
                        # 去除pad并除以scale
                        x = (max(0, bbox[0] - pad_left)) / scale
                        y = (max(0, bbox[1] - pad_top)) / scale
                        w = bbox[2] / scale
                        h = bbox[3] / scale
                        bbox = [x, y, w, h]

                    annotations.append({
                        'id': len(annotations),
                        'segmentation': compressed_rle,
                        'bbox': bbox,
                        'area': float(area),
                        'predicted_iou': m['predicted_iou'],
                        'crop_box': m['crop_box'],
                        'stability_score': m['stability_score'],
                        'point_coords': m['point_coords']
                    })
                    valid_mask_count += 1

                except Exception as e:
                    print(f"    [警告] 处理第{i}个mask时出错: {e}")
                    continue

            # 保存JSON
            output_data = {
                'frame_id': frame_idx,
                'image': {
                    'width': orig_w,
                    'height': orig_h,
                    'file_name': f"frame_{frame_idx:06d}.jpg"
                },
                'annotations': annotations
            }

            json_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f)

            # 可视化
            if writer and annotations:
                vis_frame = self._visualize_frame(frame, annotations)
                writer.write(vis_frame)

            processed_count += 1
            if processed_count % 10 == 0:
                pbar.set_postfix({'帧': frame_idx, 'masks': valid_mask_count, '错误': error_count})

        pbar.close()
        cap.release()
        if writer:
            writer.release()

        print(f"\n[完成] 成功处理 {processed_count} 帧，跳过 {total_frames - processed_count * skip_frames} 帧，错误 {error_count} 帧")
        print(f"结果保存至: {output_dir}")

    def _visualize_frame(self, frame_bgr, annotations):
        vis = frame_bgr.copy()
        mask_color = np.array([138, 43, 226], dtype=np.uint8)

        for ann in annotations:
            try:
                mask = mask_util.decode(ann['segmentation'])
                overlay = vis.copy()
                overlay[mask == 1] = mask_color
                vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)

                x, y, w, h = map(int, ann['bbox'])
                cv2.putText(vis, str(ann['id']), (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except:
                continue

        return vis

def main():
    parser = argparse.ArgumentParser(description='CoralSCOP 视频处理（修复版）')
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--model_type", type=str, default='vit_b', choices=['vit_h', 'vit_l', 'vit_b'])
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--output_dir", type=str, required=True, help="JSON输出目录")
    parser.add_argument("--vis_video", type=str, default=None, help="可视化视频输出路径（可选）")
    parser.add_argument("--skip", type=int, default=1, help="跳帧数，每N帧处理1帧（默认1=逐帧）")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--iou", type=float, default=0.72, help="IoU阈值")
    parser.add_argument("--sta", type=float, default=0.62, help="稳定性阈值")
    parser.add_argument("--points", type=int, default=32, help="每边点数")
    parser.add_argument("--min_area", type=int, default=4096, help="最小mask面积")
    args = parser.parse_args()

    processor = VideoCoralSCOP(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        gpu=args.gpu,
        iou_thresh=args.iou,
        sta_thresh=args.sta,
        point_number=args.points
    )

    processor.process_video(
        video_path=args.video,
        output_dir=args.output_dir,
        vis_output=args.vis_video,
        skip_frames=args.skip,
        min_area=args.min_area
    )

if __name__ == "__main__":
    main()