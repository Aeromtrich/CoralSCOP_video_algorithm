# 珊瑚礁算法 

## 架构设计亮点

1. **三层分离**：`modeling/`（神经网络）→ `automatic_mask_generator.py`（算法）→ `test.py / coralscop_video.py`（应用）
2. **可插拔编码器**：支持 ViT-B / ViT-L / ViT-H 三种规模
3. **两种提示机制**：自动（均匀点网格）和手动（点、框、掩码提示）可独立使用
4. **高效存储**：掩码用 RLE 编码，比二进制存储节省 70-80%
