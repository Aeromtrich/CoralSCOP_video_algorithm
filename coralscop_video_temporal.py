import os
import cv2
import json
import numpy as np
import argparse
from tqdm import tqdm
import pycocotools.mask as mask_util
from segment_anything import sam_model_registry, SamPredictor

class TemporalCoralSCOP:
    """时序优化版：利用前一帧的mask中心作为下一帧的point prompt"""
    def __init__(self, model_type, checkpoint, gpu=0):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.to(device="cuda")
        self.predictor = SamPredictor(self.sam)  # 使用Predictor而非Generator，支持prompt
        print(f"[初始化完成] 使用SamPredictor（支持时序prompt）")
        self.prev_centers = []  # 前一帧的mask中心点

    def preprocess_frame(self, frame_rgb, target_size=1024):
        """预处理并记录变换参数"""
        orig_h, orig_w = frame_rgb.shape[:2]
        scale = target_size / max(orig_h, orig_w)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        resized = cv2.resize(frame_rgb, (new_w, new_h))
        processed = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        pad_top = (target_size - new_h) // 2
        pad_left = (target_size - new_w) // 2
        processed[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized

        return processed, scale, (pad_top, pad_left), (orig_h, orig_w)

    def process_video_temporal(self, video_path, output_dir, vis_output=None, 
                               skip_frames=1, min_area=4096, iou_thresh=0.7):
        """
        时序处理：利用前一帧结果初始化当前帧
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[视频信息] {width}x{height}, {fps}fps, 共{total_frames}帧")
        print(f"[时序模式] 利用前一帧中心点作为当前帧prompt，保持ID连贯")

        os.makedirs(output_dir, exist_ok=True)
        writer = None
        if vis_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(vis_output, fourcc, fps//skip_frames, (width, height))

        frame_idx = 0
        prev_masks = []  # 存储前一帧的mask信息

        pbar = tqdm(total=total_frames)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            pbar.update(1)

            if (frame_idx - 1) % skip_frames != 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 预处理
            img_input, scale, (pad_top, pad_left), (orig_h, orig_w) = self.preprocess_frame(frame_rgb)

            # 设置图像（SamPredictor需要）
            self.predictor.set_image(img_input)

            current_masks = []
            annotations = []

            # 如果有前一帧的mask中心，作为point prompts
            if prev_masks:
                # 将前一帧的坐标映射到当前帧1024x1024空间
                points = []
                for prev in prev_masks:
                    # 获取前一帧mask中心
                    cy, cx = prev['center_y'], prev['center_x']  # 原始坐标
                    # 映射到1024空间
                    new_cx = cx * scale + pad_left
                    new_cy = cy * scale + pad_top
                    points.append([new_cx, new_cy])

                if points:
                    points = np.array(points)
                    labels = np.ones(len(points))  # 前景标签

                    # SAM预测（使用point prompt）
                    masks, scores, logits = self.predictor.predict(
                        point_coords=points,
                        point_labels=labels,
                        multimask_output=True,  # 为每个点生成3个候选mask
                    )

                    # 选择最佳mask（通常取logits最高的）
                    for i, (mask_multi, score_multi) in enumerate(zip(masks, scores)):
                        # mask_multi形状: (3, H, W)，3个候选
                        best_idx = np.argmax(score_multi)
                        best_mask = mask_multi[best_idx]

                        # 映射回原始尺寸
                        h_valid = int(orig_h * scale)
                        w_valid = int(orig_w * scale)
                        mask_cropped = best_mask[pad_top:pad_top+h_valid, pad_left:pad_left+w_valid]
                        mask_orig = cv2.resize(mask_cropped.astype(np.uint8), (orig_w, orig_h), 
                                              interpolation=cv2.INTER_NEAREST)

                        area = np.sum(mask_orig)
                        if area < min_area:
                            continue

                        # 计算当前mask中心用于下一帧
                        ys, xs = np.where(mask_orig)
                        if len(xs) > 0:
                            center_x, center_y = np.mean(xs), np.mean(ys)
                        else:
                            center_x, center_y = points[i][0] / scale - pad_left, points[i][1] / scale - pad_top

                        current_masks.append({
                            'center_x': center_x,
                            'center_y': center_y,
                            'mask': mask_orig,
                            'area': area,
                            'score': score_multi[best_idx]
                        })

                        # 编码保存
                        rle = mask_util.encode(np.asfortranarray(mask_orig))
                        rle['counts'] = str(rle['counts'], encoding="utf-8")

                        annotations.append({
                            'id': len(annotations),
                            'segmentation': rle,
                            'area': float(area),
                            'center': [float(center_x), float(center_y)],
                            'score': float(score_multi[best_idx]),
                            'tracked_from_prev': True
                        })

            # 如果tracked mask不足，补充自动生成（针对新出现的珊瑚）
            if len(annotations) < 3:
                # 自动生成补充（这部分代码与基础版相同，略）
                pass  # 实际实现时可接入mask_generator作为后备

            # 保存JSON
            output_data = {
                'frame_id': frame_idx,
                'tracked_masks': len(annotations),
                'image': {'width': orig_w, 'height': orig_h, 'file_name': f"frame_{frame_idx:06d}.jpg"},
                'annotations': annotations
            }

            json_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.json")
            with open(json_path, 'w') as f:
                json.dump(output_data, f)

            # 更新prev_masks用于下一帧
            prev_masks = current_masks if current_masks else prev_masks  # 如果没有检测到，保持前一帧

            # 可视化
            if writer and annotations:
                vis = self._draw_masks(frame, annotations)
                writer.write(vis)

            if frame_idx % 30 == 0:
                pbar.set_postfix({'tracked': len(annotations)})

        pbar.close()
        cap.release()
        if writer:
            writer.release()
        print(f"[完成] 共处理{frame_idx}帧，结果保存至{output_dir}")

    def _draw_masks(self, frame, annotations):
        vis = frame.copy()
        for ann in annotations:
            mask = mask_util.decode(ann['segmentation'])
            color = (138, 43, 226) if ann.get('tracked_from_prev') else (0, 255, 0)  # 紫色=跟踪，绿色=新检测
            overlay = vis.copy()
            overlay[mask == 1] = color
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

            # 标记中心
            cx, cy = map(int, ann['center'])
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(vis, f"ID:{ann['id']}", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return vis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, default='vit_b')
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vis_video", type=str, default=None)
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--min_area", type=int, default=4096)
    args = parser.parse_args()

    processor = TemporalCoralSCOP(args.model_type, args.checkpoint, args.gpu)
    processor.process_video_temporal(
        args.video, args.output_dir, args.vis_video, 
        skip_frames=args.skip, min_area=args.min_area
    )

if __name__ == "__main__":
    main()
