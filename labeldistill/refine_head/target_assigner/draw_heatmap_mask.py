# 用于lidar分支的特征级蒸馏的掩码生成
# LabelDistill/labeldistill/refine_head/target_assigner/draw_heatmap_mask.py
import numpy as np
import torch
import torch.nn as nn
from mmdet3d.core import draw_heatmap_gaussian, gaussian_radius


class BEVDistillationMaskGenerator(nn.Module):
    """
    基于ROI和GT匹配结果生成BEV特征蒸馏权重掩码。
    新版本将配置硬编码在类中。
    """
    def __init__(self):
        super().__init__()
        
        # 硬编码的 pts 配置
        # 1. 定义配置常量
        grid_size = [1024, 1024, 40]
        voxel_size = [0.1, 0.1, 0.2]
        out_size_factor = 8
        point_cloud_range_list = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        gaussian_overlap = 0.1
        min_radius = 2

        # 2. 将必要的配置转换为张量并存储为实例属性
        self.grid_size = torch.tensor(grid_size)
        self.pc_range = torch.tensor(point_cloud_range_list)
        self.voxel_size = torch.tensor(voxel_size)
        self.out_factor = out_size_factor
        self.gaussian_overlap = gaussian_overlap
        self.min_radius = min_radius
        
        # 预计算特征图尺寸，维度为 [W, H]，转换为 numpy 以匹配原始逻辑
        self.feature_map_size = (self.grid_size[:2] // self.out_factor).cpu().numpy()
        
        # 定义小类别的类别ID（用于中等质量匹配的特殊处理）
        self.small_classes = [5, 6, 8, 9]  # barrier, motorcycle, pedestrian, traffic_cone
        
    def generate_mask(self, matched_results, batch_size):
   
        # 直接使用硬编码的配置属性
        pc_range = self.pc_range
        voxel_size = self.voxel_size
        out_factor = self.out_factor
        feature_map_size = self.feature_map_size
        gaussian_overlap = self.gaussian_overlap
        min_radius = self.min_radius

        # 提取匹配结果（使用新的键名）
        refined_high_rois = matched_results['refined_high_quality_rois']
        refined_high_scores = matched_results['refined_high_quality_roi_scores']

        # 中等质量匹配
        medium_rois = matched_results['medium_quality_rois']
        medium_scores = matched_results['medium_quality_roi_scores']
        medium_gts = matched_results['medium_quality_gt']

        # 未匹配GT的细粒度距离划分
        near_unmatched_gts = matched_results['unmatched_gt_near']      # <30m
        medium_unmatched_gts = matched_results['unmatched_gt_medium']  # 30-50m
        far_unmatched_gts = matched_results['unmatched_gt_far']        # >50m

        # ===== 关键修复：验证列表长度 =====
        assert len(refined_high_rois) == batch_size, \
            f"refined_high_rois length {len(refined_high_rois)} != batch_size {batch_size}"
        assert len(medium_rois) == batch_size, \
            f"medium_rois length {len(medium_rois)} != batch_size {batch_size}"
        assert len(medium_scores) == batch_size, \
            f"medium_scores length {len(medium_scores)} != batch_size {batch_size}"

        # 存储每个batch的掩码
        mask_list = []

        for idx in range(batch_size):
            # 安全地获取当前batch的数据
            cur_refined_rois = refined_high_rois[idx]
            cur_refined_scores = refined_high_scores[idx]

            cur_medium_rois = medium_rois[idx]
            cur_medium_scores = medium_scores[idx]  # 现在可以安全访问
            cur_medium_gts = medium_gts[idx]

            cur_near_unmatched = near_unmatched_gts[idx]
            cur_medium_unmatched = medium_unmatched_gts[idx]

            # 确保配置张量在正确的设备上
            device = cur_refined_rois.device
            pc_range_dev = pc_range.to(device)
            voxel_size_dev = voxel_size.to(device)

            # 初始化掩码 (1, H, W)
            mask = torch.zeros((1, feature_map_size[1], feature_map_size[0]), 
                             device=device, dtype=torch.float32)

            # ===== 1. 绘制高质量匹配ROI =====
            if len(cur_refined_rois) > 0:
                mask = self._draw_boxes_to_mask(
                    boxes=cur_refined_rois,
                    mask=mask,
                    center_values=cur_refined_scores,
                    pc_range=pc_range_dev,
                    voxel_size=voxel_size_dev,
                    out_factor=out_factor,
                    feature_map_size=feature_map_size,
                    gaussian_overlap=gaussian_overlap,
                    min_radius=min_radius
                )

            # ===== 2. 绘制中等质量匹配GT =====
            if len(cur_medium_gts) > 0:
                gt_classes = cur_medium_gts[:, -1].long()

                small_class_mask = torch.zeros(len(gt_classes), dtype=torch.bool, device=device)
                for small_class_id in self.small_classes:
                    small_class_mask |= (gt_classes == small_class_id)

                # 处理小类别GT
                if small_class_mask.any():
                    small_gts = cur_medium_gts[small_class_mask]
                    small_roi_scores = cur_medium_scores[small_class_mask]
                    small_center_values = (small_roi_scores + 1.0) / 2.0

                    mask = self._draw_boxes_to_mask(
                        boxes=small_gts[:, :7],
                        mask=mask,
                        center_values=small_center_values,
                        pc_range=pc_range_dev,
                        voxel_size=voxel_size_dev,
                        out_factor=out_factor,
                        feature_map_size=feature_map_size,
                        gaussian_overlap=gaussian_overlap,
                        min_radius=min_radius
                    )

                # 处理其他类别GT
                other_class_mask = ~small_class_mask
                if other_class_mask.any():
                    other_gts = cur_medium_gts[other_class_mask]
                    other_center_values = torch.full((len(other_gts),), 0.5, 
                                                    device=device, dtype=torch.float32)

                    mask = self._draw_boxes_to_mask(
                        boxes=other_gts[:, :7],
                        mask=mask,
                        center_values=other_center_values,
                        pc_range=pc_range_dev,
                        voxel_size=voxel_size_dev,
                        out_factor=out_factor,
                        feature_map_size=feature_map_size,
                        gaussian_overlap=gaussian_overlap,
                        min_radius=min_radius
                    )

            # ===== 3. 绘制近距离未匹配GT =====
            if len(cur_near_unmatched) > 0:
                near_scores = torch.full((len(cur_near_unmatched),), 0.4, 
                                        device=device, dtype=torch.float32)
                mask = self._draw_boxes_to_mask(
                    boxes=cur_near_unmatched[:, :7],
                    mask=mask,
                    center_values=near_scores,
                    pc_range=pc_range_dev,
                    voxel_size=voxel_size_dev,
                    out_factor=out_factor,
                    feature_map_size=feature_map_size,
                    gaussian_overlap=gaussian_overlap,
                    min_radius=min_radius
                )

            # ===== 4. 绘制中距离未匹配GT =====
            if len(cur_medium_unmatched) > 0:
                medium_scores_unmatched = torch.full((len(cur_medium_unmatched),), 0.2, 
                                          device=device, dtype=torch.float32)
                mask = self._draw_boxes_to_mask(
                    boxes=cur_medium_unmatched[:, :7],
                    mask=mask,
                    center_values=medium_scores_unmatched,
                    pc_range=pc_range_dev,
                    voxel_size=voxel_size_dev,
                    out_factor=out_factor,
                    feature_map_size=feature_map_size,
                    gaussian_overlap=gaussian_overlap,
                    min_radius=min_radius
                )

            mask_list.append(mask)

        # 合并为 (B, 1, H, W)
        weight_mask = torch.stack(mask_list, dim=0)

        return weight_mask
    
    def _draw_boxes_to_mask(self, boxes, mask, center_values, pc_range, voxel_size, 
                           out_factor, feature_map_size, gaussian_overlap, min_radius):
        """
        在掩码上绘制框的高斯热图
        
        Args:
            boxes: (N, 7) 框坐标 [x, y, z, w, l, h, yaw]
            mask: (1, H, W) 当前掩码
            center_values: (N,) 每个框的中心高斯值
            其他参数: 配置参数
            
        Returns:
            mask: 更新后的掩码
        """
        num_boxes = len(boxes)
        feat_width, feat_height = feature_map_size

        # 裁剪出 HxW 的二维热图供绘制
        heatmap = mask[0]

        for k in range(num_boxes):
            # 提取尺寸 [W, L]
            width = boxes[k][3]  
            length = boxes[k][4]

            width_fm = width / voxel_size[0] / out_factor
            length_fm = length / voxel_size[1] / out_factor

            if width_fm.item() <= 0 or length_fm.item() <= 0:
                continue

            # ===== 添加这个检查 =====
            # 跳过异常大的框（特征图空间超过1000像素明显不合理）
            if width_fm.item() > 1000 or length_fm.item() > 1000:
                continue
            # =====================

            # 2. 计算高斯半径
            radius_tensor = gaussian_radius(
                (length_fm, width_fm),
                min_overlap=gaussian_overlap
            )
            # 限制最小半径并转换为整数
            radius = max(min_radius, int(radius_tensor.item()))

            # 3. 中心点坐标转换（到特征图像素坐标）
            x, y = boxes[k][0], boxes[k][1]
            coor_x = (x - pc_range[0]) / voxel_size[0] / out_factor
            coor_y = (y - pc_range[1]) / voxel_size[1] / out_factor

            # 转换为整数坐标
            center_int = torch.stack([coor_x, coor_y], dim=0).to(torch.int32)

            # 4. 边界检查
            if not (0 <= center_int[0] < feat_width and 
                    0 <= center_int[1] < feat_height):
                continue

            # 5. 绘制高斯热图，使用 center_values[k] 作为峰值系数
            heatmap = draw_heatmap_gaussian(
                heatmap, 
                center_int, 
                radius, 
                k=center_values[k].item()
            )

        return mask
    
    
    def forward(self, matched_results, batch_size):
        """前向传播"""
        return self.generate_mask(matched_results, batch_size)