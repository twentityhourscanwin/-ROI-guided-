# 用于label分支的特征级蒸馏的掩码生成
# LabelDistill/labeldistill/refine_head/target_assigner/draw_heatmap_mask_label.py
import numpy as np
import torch
import torch.nn as nn
from mmdet3d.core import draw_heatmap_gaussian, gaussian_radius

class BEVDistillationMaskGenerator_label(nn.Module):
    """
    基于ROI和GT匹配结果生成BEV特征蒸馏权重掩码。
    支持灵活配置各类别的高斯中心值。
    """
    def __init__(self):
        super().__init__()
        
        # ========== 硬编码的 pts 配置 ==========
        grid_size = [1024, 1024, 40]
        voxel_size = [0.1, 0.1, 0.2]
        out_size_factor = 8
        point_cloud_range_list = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        gaussian_overlap = 0.1
        min_radius = 2

        self.grid_size = torch.tensor(grid_size)
        self.pc_range = torch.tensor(point_cloud_range_list)
        self.voxel_size = torch.tensor(voxel_size)
        self.out_factor = out_size_factor
        self.gaussian_overlap = gaussian_overlap
        self.min_radius = min_radius
        
        # 预计算特征图尺寸，维度为 [W, H]
        self.feature_map_size = (self.grid_size[:2] // self.out_factor).cpu().numpy()
        
        # ========== 高斯中心值配置（方便修改！）==========
        self.gaussian_center_values = {
            'refined_high_quality_rois': 0.5,      # 精准匹配ROI
            'medium_quality_gt': 1.0,              # 中等质量匹配GT
            'unmatched_gt_near': 1.0,              # 近距离未匹配GT (<30m)
            'unmatched_gt_medium': 1.0,            # 中距离未匹配GT (30-50m)
            'unmatched_gt_far': 1.0,               # 远距离未匹配GT (>50m)
        }
        
        # ========== 参与绘制的类别配置 ==========
        # 如果某个键不在这个列表中，就不会被绘制
        self.categories_to_draw = [
            'refined_high_quality_rois',
            'medium_quality_gt',
            'unmatched_gt_near',
            'unmatched_gt_medium',
            'unmatched_gt_far',
        ]

    def generate_mask(self, matched_results, batch_size):
        """
        根据匹配结果生成权重掩码

        Args:
            matched_results: ProposalTargetLayer的输出字典
            batch_size: int

        Returns:
            weight_mask: (B, 1, H, W) 权重掩码张量
        """
        pc_range = self.pc_range
        voxel_size = self.voxel_size
        out_factor = self.out_factor
        feature_map_size = self.feature_map_size
        gaussian_overlap = self.gaussian_overlap
        min_radius = self.min_radius

        # 存储每个batch的掩码
        mask_list = []

        # ==========================================================
        # 🚀 关键修复步骤 1: 确定模型实际运行的设备 (target_device)
        # 我们应该获取模型实际运行的设备，而不是依赖于数据是否为空。

        # 尝试从输入数据中获取设备
        target_device = None
        for key in self.categories_to_draw:
            # 假设 matched_results[key] 是一个张量列表
            if key in matched_results and len(matched_results[key]) > 0 and isinstance(matched_results[key][0], torch.Tensor):
                target_device = matched_results[key][0].device
                break

        # 如果数据为空，通过一个已知的类成员张量（比如 self.pc_range）获取模型设备
        # 即使 self.pc_range 仍是 CPU 张量，我们也可以先用它的设备信息，
        # 但更安全的回退是直接假设是 cuda:0，因为错误提示是 cuda:0 vs cpu。
        if target_device is None:
            # 确保至少能获取一个设备信息，如果代码运行到这里，通常是在 GPU 模式下
            try:
                target_device = self.pc_range.device
            except:
                # 最终回退，根据错误提示，你的程序运行在 cuda:0 上
                target_device = torch.device('cuda:0')

        # 🚀 关键修复步骤 2: 将 CPU 常量张量移动到 target_device (策略二的实现)
        pc_range_dev = pc_range.to(target_device)
        voxel_size_dev = voxel_size.to(target_device)
        # ==========================================================

        for idx in range(batch_size):
            # ⚠️ 原代码中的设备获取逻辑可以简化，因为我们已经有了 target_device

            # 初始化掩码 (1, H, W) 并确保它位于 target_device
            mask = torch.zeros((1, feature_map_size[1], feature_map_size[0]), 
                               device=target_device, dtype=torch.float32)

            # ========== 遍历所有需要绘制的类别 ==========
            for category_key in self.categories_to_draw:
                if category_key not in matched_results:
                    continue

                # 获取当前batch的数据
                data = matched_results[category_key][idx]

                if len(data) == 0:
                    continue

                # 获取该类别的高斯中心值
                center_value = self.gaussian_center_values.get(category_key, 1.0)

                # 判断数据格式：ROI是7维，GT是8维（包含类别）
                if data.shape[-1] >= 8:
                    # GT格式：只取前7维 (x,y,z,w,l,h,yaw)
                    boxes = data[:, :7]
                else:
                    # ROI格式：直接使用
                    boxes = data

                # 确保 center_values 也位于 target_device
                center_values = torch.full((len(boxes),), center_value, 
                                           device=target_device, dtype=torch.float32)

                # 绘制到掩码上
                # 使用已移动到 target_device 的 pc_range_dev 和 voxel_size_dev
                mask = self._draw_boxes_to_mask(
                    boxes=boxes,
                    mask=mask,
                    center_values=center_values,
                    pc_range=pc_range_dev,
                    voxel_size=voxel_size_dev,
                    out_factor=out_factor,
                    feature_map_size=feature_map_size,
                    gaussian_overlap=gaussian_overlap,
                    min_radius=min_radius
                )

            mask_list.append(mask)

        # 这一步现在是安全的，因为 mask_list 中所有张量都在 target_device 上
        weight_mask = torch.stack(mask_list, dim=0)

        return weight_mask
    
    def _draw_boxes_to_mask(self, boxes, mask, center_values, pc_range, voxel_size, 
                           out_factor, feature_map_size, gaussian_overlap, min_radius):
        """
        将boxes绘制到mask上，使用高斯分布
        
        Args:
            boxes: (N, 7) 包含 [x, y, z, w, l, h, yaw]
            mask: (1, H, W) 当前掩码
            center_values: (N,) 每个box的高斯中心峰值
            其他参数为配置项
        """
        num_boxes = len(boxes)
        if num_boxes == 0:
            return mask
        
        # 提取 H, W 供边界检查
        feat_width, feat_height = feature_map_size
        
        # 裁剪出 HxW 的二维热图供绘制
        heatmap = mask[0]
        
        for k in range(num_boxes):
            # 提取尺寸 [W, L]
            width = boxes[k][3]
            length = boxes[k][4]
            
            # 1. 转换到特征图空间
            width_fm = width / voxel_size[0] / out_factor
            length_fm = length / voxel_size[1] / out_factor
            
            if width_fm.item() <= 0 or length_fm.item() <= 0:
                continue
            
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
            
            # mmdet3d 的 draw_heatmap_gaussian 期望 center 是整数坐标
            center_int = torch.stack([coor_x, coor_y], dim=0).to(torch.int32)
            
            # 4. 边界检查
            if not (0 <= center_int[0] < feat_width and 
                    0 <= center_int[1] < feat_height):
                continue
            
            # 5. 绘制高斯热图，使用 center_values[k] 作为峰值系数 k
            heatmap = draw_heatmap_gaussian(
                heatmap, 
                center_int, 
                radius, 
                k=center_values[k].item()
            )
        
        return mask
    
    def forward(self, matched_results, batch_size):
        """
        前向传播
        
        Args:
            matched_results: ProposalTargetLayer的输出字典
            batch_size: int
            
        Returns:
            weight_mask: (B, 1, H, W) 权重掩码张量
        """
        return self.generate_mask(matched_results, batch_size)