#labeldistill/refine_head/target_assigner/draw_heatmap_mask_fp.py
import numpy as np
import torch
import torch.nn as nn
from mmdet3d.core import draw_heatmap_gaussian, gaussian_radius


class StudentFPMaskGenerator(nn.Module):
    """
    为学生网络特有的误检ROI生成BEV特征蒸馏权重掩码。
    用于抑制学生网络在这些误检位置的特征响应。
    """
    def __init__(self):
        super().__init__()
        
        # 硬编码的配置常量
        grid_size = [1024, 1024, 40]
        voxel_size = [0.1, 0.1, 0.2]
        out_size_factor = 8
        point_cloud_range_list = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        gaussian_overlap = 0.2  # 修改：从0.1改为0.3，缩小高斯半径
        min_radius = 2          # 修改：从2改为1，减小最小半径

        # 将配置转换为张量并存储为实例属性
        self.grid_size = torch.tensor(grid_size)
        self.pc_range = torch.tensor(point_cloud_range_list)
        self.voxel_size = torch.tensor(voxel_size)
        self.out_factor = out_size_factor
        self.gaussian_overlap = gaussian_overlap
        self.min_radius = min_radius
        
        # 预计算特征图尺寸，维度为 [W, H]
        self.feature_map_size = (self.grid_size[:2] // self.out_factor).cpu().numpy()
        
    def generate_mask(self, filtered_results, batch_size):
        """
        根据学生特有误检ROI生成权重掩码
        
        Args:
            filtered_results: StudentSpecificFPFilter的输出字典
                'student_specific_fp_rois': List[Tensor]
                'student_specific_fp_scores': List[Tensor]
                'student_specific_fp_labels': List[Tensor]
            batch_size: int
            
        Returns:
            weight_mask: (B, 1, H, W) 权重掩码张量
                在学生特有误检位置有高斯分布的权重（峰值=对应ROI的分数）
                其他位置权重为0
        """
        # 使用硬编码的配置属性
        pc_range = self.pc_range
        voxel_size = self.voxel_size
        out_factor = self.out_factor
        feature_map_size = self.feature_map_size
        gaussian_overlap = self.gaussian_overlap
        min_radius = self.min_radius
        
        # 提取学生特有误检ROI
        student_specific_rois = filtered_results['student_specific_fp_rois']
        student_specific_scores = filtered_results['student_specific_fp_scores']
        
        # 存储每个batch的掩码
        mask_list = []
        
        for idx in range(batch_size):
            # 获取当前batch的学生特有误检ROI
            cur_rois = student_specific_rois[idx]  # (N, 7+C)
            cur_scores = student_specific_scores[idx]  # (N,)
            
            # 确定设备
            device = cur_rois.device
            pc_range_dev = pc_range.to(device)
            voxel_size_dev = voxel_size.to(device)
            
            # 初始化掩码 (1, H, W)
            mask = torch.zeros((1, feature_map_size[1], feature_map_size[0]), 
                               device=device, dtype=torch.float32)
            
            # 绘制学生特有误检ROI（中心值=对应分数）
            if len(cur_rois) > 0:
                mask = self._draw_boxes_to_mask(
                    boxes=cur_rois,
                    mask=mask,
                    center_values=cur_scores,  # 使用实际分数作为高斯峰值
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
        在掩码上绘制高斯热图
        
        Args:
            boxes: (N, 7+C) ROI boxes
            mask: (1, H, W) 当前掩码
            center_values: (N,) 每个ROI的中心峰值
            其他参数: 配置参数
            
        Returns:
            mask: (1, H, W) 更新后的掩码
        """
        num_boxes = len(boxes)
        if num_boxes == 0:
            return mask
        
        # 提取特征图尺寸
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
            center_x = int(coor_x.item())
            center_y = int(coor_y.item())
            
            # 4. 边界检查
            if not (0 <= center_x < feat_width and 
                    0 <= center_y < feat_height):
                continue
            
            # 5. 绘制高斯热图
            center_tuple = (center_x, center_y)
            k_value = center_values[k].item()
            
            try:
                heatmap = draw_heatmap_gaussian(
                    heatmap, 
                    center_tuple,
                    radius, 
                    k=k_value
                )
            except Exception as e:
                # 即使清理了打印，保留异常捕获和引发以防万一，但移除了额外的调试信息
                raise
        
        return mask
    
    def forward(self, filtered_results, batch_size):
        """
        前向传播接口
        
        Args:
            filtered_results: StudentSpecificFPFilter的输出
            batch_size: int
            
        Returns:
            weight_mask: (B, 1, H, W)
        """
        return self.generate_mask(filtered_results, batch_size)