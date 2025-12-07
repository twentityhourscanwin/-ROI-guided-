#labeldistill/refine_head/target_assigner/roi_distill_scale.py
import torch
import numpy as np
from copy import deepcopy


class AdaptiveGTScaler:
    
    def __init__(self, scale_cfg=None):
        self.scale_cfg = scale_cfg
        
        # 高质量匹配的缩放配置
        self.high_quality_cfg = {
            'roi_direction_scale': 0.3,    # 朝向ROI方向的缩放系数
            'max_roi_offset': 2.0,          # ROI偏移对缩放的最大影响
            'velocity_scale': 0.2,          # 速度影响系数
            'max_scale': 1.0,               # 最大缩放幅度
            'min_scale': 0.2,               # 最小缩放幅度
        }
        
        # 中等质量匹配的缩放配置
        self.medium_quality_cfg = {
            'roi_direction_scale': 0.4,    # 更大的朝向ROI方向的缩放
            'max_roi_offset': 3.0,
            'velocity_scale': 0.25,
            'max_scale': 1.5,
            'min_scale': 0.2,
        }
        
        # 未匹配GT的缩放配置（只考虑距离和速度）
        self.unmatched_cfg = {
            'near_range': 30.0,
            'far_range': 50.0,
            'near_scale': {'x': 0.20, 'y': 0.20},      # <30m
            'medium_scale': {'x': 0.25, 'y': 0.25},      # 30-50m
            'far_scale': {'x': 0.2, 'y': 0.2},         # >50m
            'velocity_scale': 0.3,
            'max_scale': 1.5,
            'min_scale': 0.1,
        }
    
    def compute_direction_vector(self, gt_center, roi_center):
        """
        计算GT中心指向ROI中心的方向向量
        
        Args:
            gt_center: (3,) GT中心坐标
            roi_center: (3,) ROI中心坐标
            
        Returns:
            direction: (2,) BEV平面归一化方向向量 (x, y)
            offset_distance: float ROI相对GT的偏移距离
        """
        # 只在BEV平面(x-y)计算
        diff = roi_center[:2] - gt_center[:2]
        offset_distance = torch.sqrt((diff ** 2).sum())
        
        if offset_distance < 0.2:  # 几乎重合
            return torch.zeros(2, device=gt_center.device), 0.0
        
        direction = diff / offset_distance
        return direction, offset_distance.item()
    
    def compute_velocity_scale(self, velocity, velocity_scale_factor, max_scale):
        """
        根据速度计算缩放增量
        
        Args:
            velocity: (2,) 速度向量 (vx, vy)
            velocity_scale_factor: 速度影响系数
            max_scale: 最大缩放
            
        Returns:
            scale_x, scale_y: x和y方向的缩放增量
        """
        vx = abs(velocity[0].item()) if len(velocity) > 0 else 0
        vy = abs(velocity[1].item()) if len(velocity) > 1 else 0
        
        scale_x = 0.0
        scale_y = 0.0
        
        # 速度分级缩放
        if vx > 0.3 and vx < 0.8:
            scale_x = velocity_scale_factor * 0.5
        elif vx >= 0.8:
            scale_x = velocity_scale_factor * 1.0
        
        if vy > 0.3 and vy < 0.8:
            scale_y = velocity_scale_factor * 0.5
        elif vy >= 0.8:
            scale_y = velocity_scale_factor * 1.0
        
        # 限制最大值
        scale_x = min(scale_x, max_scale)
        scale_y = min(scale_y, max_scale)
        
        return scale_x, scale_y
    
    def scale_high_quality(self, gt_boxes, roi_centers, roi_scores, gt_original_sizes):
        """
        高质量匹配GT的单侧扩展
        根据ROI相对GT的方位，只向ROI所在方向扩展边界
        
        Args:
            gt_boxes: (N, 7+) GT框 [x, y, z, l, w, h, heading, ...]
            roi_centers: (N, 3) 对应的ROI中心
            roi_scores: (N,) ROI分数
            gt_original_sizes: (N, 2) GT原始尺寸 [l, w]
            
        Returns:
            scaled_gt_boxes: (N, 7+) 单侧扩展后的GT框
        """
        if len(gt_boxes) == 0:
            return gt_boxes
        
        cfg = self.high_quality_cfg
        scaled_boxes = gt_boxes.clone()
        
        for i in range(len(gt_boxes)):
            gt_box = gt_boxes[i]
            roi_center = roi_centers[i]
            roi_score = roi_scores[i].item()
            original_l, original_w = gt_original_sizes[i]
            
            # 1. 计算ROI相对GT的偏移向量（在BEV平面）
            offset_x = roi_center[0] - gt_box[0]  # ROI在GT的x方向偏移
            offset_y = roi_center[1] - gt_box[1]  # ROI在GT的y方向偏移
            offset_dist = torch.sqrt(offset_x**2 + offset_y**2).item()
            
            if offset_dist < 0.1:  # 几乎重合，不扩展
                continue
            
            # 2. 获取GT的朝向角（heading）
            heading = gt_box[6].item()
            cos_h = np.cos(heading)
            sin_h = np.sin(heading)
            
            # 3. 将偏移向量转换到GT的局部坐标系
            # 局部坐标系：x轴沿车头方向（length方向），y轴沿车宽方向
            local_offset_x = offset_x.item() * cos_h + offset_y.item() * sin_h
            local_offset_y = -offset_x.item() * sin_h + offset_y.item() * cos_h
            
            # 4. 基于ROI分数计算扩展系数
            # 分数越高，说明检测越可信，扩展幅度可以更大
            score_factor = roi_score * cfg['roi_direction_scale']
            
            # 5. 基于速度计算额外扩展
            velocity = gt_box[7:9] if gt_box.shape[0] > 8 else torch.zeros(2, device=gt_box.device)
            vx, vy = abs(velocity[0].item()), abs(velocity[1].item())
            
            # 速度分级扩展量
            vel_extend_x = 0.0
            vel_extend_y = 0.0
            if vx > 0.3 and vx < 0.8:
                vel_extend_x = original_l.item() * cfg['velocity_scale'] * 0.5
            elif vx >= 0.8:
                vel_extend_x = original_l.item() * cfg['velocity_scale']
            
            if vy > 0.3 and vy < 0.8:
                vel_extend_y = original_w.item() * cfg['velocity_scale'] * 0.5
            elif vy >= 0.8:
                vel_extend_y = original_w.item() * cfg['velocity_scale']
            
            # 6. 计算单侧扩展量
            # 在局部坐标系中，判断ROI在哪一侧
            extend_l = 0.0  # length方向扩展
            extend_w = 0.0  # width方向扩展
            
            # length方向（前后）：如果ROI在前方(local_offset_x > 0)，向前扩展
            if abs(local_offset_x) > 0.1:
                # 基于ROI分数和偏移距离的扩展
                roi_extend_l = score_factor * min(abs(local_offset_x), cfg['max_roi_offset'])
                extend_l = roi_extend_l + vel_extend_x
                extend_l = min(extend_l, original_l.item() * cfg['max_scale'])
            
            # width方向（左右）：如果ROI在左侧(local_offset_y > 0)，向左扩展
            if abs(local_offset_y) > 0.1:
                roi_extend_w = score_factor * min(abs(local_offset_y), cfg['max_roi_offset'])
                extend_w = roi_extend_w + vel_extend_y
                extend_w = min(extend_w, original_w.item() * cfg['max_scale'])
            
            # 7. 应用单侧扩展
            # 扩展尺寸
            new_l = original_l.item() + extend_l
            new_w = original_w.item() + extend_w
            
            # 8. 调整中心位置
            # 因为是单侧扩展，中心需要向扩展方向偏移一半的扩展量
            # 在局部坐标系中计算新中心偏移
            local_center_shift_x = extend_l / 2.0 if local_offset_x > 0 else -extend_l / 2.0 if local_offset_x < -0.1 else 0
            local_center_shift_y = extend_w / 2.0 if local_offset_y > 0 else -extend_w / 2.0 if local_offset_y < -0.1 else 0
            
            # 转换回全局坐标系
            global_shift_x = local_center_shift_x * cos_h - local_center_shift_y * sin_h
            global_shift_y = local_center_shift_x * sin_h + local_center_shift_y * cos_h
            
            # 更新框参数
            scaled_boxes[i, 0] = gt_box[0] + global_shift_x  # 新的x中心
            scaled_boxes[i, 1] = gt_box[1] + global_shift_y  # 新的y中心
            scaled_boxes[i, 3] = new_l  # 新的length
            scaled_boxes[i, 4] = new_w  # 新的width
        
        return scaled_boxes
    
    def scale_medium_quality(self, gt_boxes, roi_centers, gt_original_sizes):
        """
        缩放中等质量匹配的GT框（策略与高质量类似但幅度更大）
        
        Args:
            gt_boxes: (N, 7+) GT框
            roi_centers: (N, 3) 对应的ROI中心
            gt_original_sizes: (N, 2) GT原始尺寸
            
        Returns:
            scaled_gt_boxes: (N, 7+) 缩放后的GT框
        """
        if len(gt_boxes) == 0:
            return gt_boxes
        
        cfg = self.medium_quality_cfg
        scaled_boxes = gt_boxes.clone()
        
        for i in range(len(gt_boxes)):
            gt_box = gt_boxes[i]
            roi_center = roi_centers[i]
            original_l, original_w = gt_original_sizes[i]
            
            # 1. 计算ROI偏移方向和距离
            direction, offset_dist = self.compute_direction_vector(
                gt_box[:3], roi_center
            )
            
            # 2. 根据偏移距离计算朝向ROI的缩放
            offset_factor = min(offset_dist / cfg['max_roi_offset'], 1.0)
            roi_direction_scale = cfg['roi_direction_scale'] * offset_factor
            
            # 3. 计算速度影响
            velocity = gt_box[7:9] if gt_box.shape[0] > 8 else torch.zeros(2, device=gt_box.device)
            vel_scale_x, vel_scale_y = self.compute_velocity_scale(
                velocity, cfg['velocity_scale'], cfg['max_scale']
            )
            
            # 4. 计算总缩放
            scale_x = roi_direction_scale * abs(direction[0].item()) + vel_scale_x
            scale_y = roi_direction_scale * abs(direction[1].item()) + vel_scale_y
            
            # 5. 限制缩放范围
            scale_x = min(max(scale_x, cfg['min_scale']), cfg['max_scale'])
            scale_y = min(max(scale_y, cfg['min_scale']), cfg['max_scale'])
            
            # 6. 应用缩放
            scaled_boxes[i, 3] = original_l * (1 + scale_x)
            scaled_boxes[i, 4] = original_w * (1 + scale_y)
        
        return scaled_boxes
    
    def scale_unmatched_gt(self, gt_boxes):
        """
        缩放未匹配的GT框（只考虑距离和速度）
        
        Args:
            gt_boxes: (N, 7+) GT框
            
        Returns:
            scaled_gt_boxes: (N, 7+) 缩放后的GT框
        """
        if len(gt_boxes) == 0:
            return gt_boxes
        
        cfg = self.unmatched_cfg
        scaled_boxes = gt_boxes.clone()
        
        for i in range(len(gt_boxes)):
            gt_box = gt_boxes[i]
            original_l = gt_box[3].item()
            original_w = gt_box[4].item()
            
            # 1. 计算距离
            x, y = gt_box[0].item(), gt_box[1].item()
            bbox_range = np.sqrt(x**2 + y**2)
            
            # 2. 基于距离的基础缩放
            if bbox_range < cfg['near_range']:
                base_scale_x = cfg['near_scale']['x']
                base_scale_y = cfg['near_scale']['y']
            elif bbox_range < cfg['far_range']:
                base_scale_x = cfg['medium_scale']['x']
                base_scale_y = cfg['medium_scale']['y']
            else:
                base_scale_x = cfg['far_scale']['x']
                base_scale_y = cfg['far_scale']['y']
            
            # 3. 计算速度影响
            velocity = gt_box[7:9] if gt_box.shape[0] > 8 else torch.zeros(2, device=gt_box.device)
            vel_scale_x, vel_scale_y = self.compute_velocity_scale(
                velocity, cfg['velocity_scale'], cfg['max_scale']
            )
            
            # 4. 总缩放 = 基础缩放 + 速度缩放
            total_scale_x = base_scale_x + vel_scale_x
            total_scale_y = base_scale_y + vel_scale_y
            
            # 5. 限制缩放范围
            total_scale_x = min(max(total_scale_x, cfg['min_scale']), cfg['max_scale'])
            total_scale_y = min(max(total_scale_y, cfg['min_scale']), cfg['max_scale'])
            
            # 6. 应用缩放
            scaled_boxes[i, 3] = original_l * (1 + total_scale_x)
            scaled_boxes[i, 4] = original_w * (1 + total_scale_y)
        
        return scaled_boxes
    
    def forward(self, matched_results):
        """
        对所有类型的GT框进行自适应缩放
        
        Args:
            matched_results: ProposalTargetLayer的输出结果
            
        Returns:
            scaled_results: 包含缩放后GT框的结果字典
        """
        batch_size = len(matched_results['refined_high_quality_gt'])
        
        scaled_high_gt_list = []
        scaled_medium_gt_list = []
        scaled_unmatched_near_list = []
        scaled_unmatched_medium_list = []
        scaled_unmatched_far_list = []
        
        for idx in range(batch_size):
            # 1. 高质量GT单侧扩展
            high_gt = matched_results['refined_high_quality_gt'][idx]
            high_rois = matched_results['refined_high_quality_rois'][idx]
            high_scores = matched_results['refined_high_quality_roi_scores'][idx]
            
            if len(high_gt) > 0:
                # 保存原始尺寸
                original_sizes = high_gt[:, 3:5].clone()
                scaled_high_gt = self.scale_high_quality(
                    high_gt, high_rois[:, :3], high_scores, original_sizes
                )
            else:
                scaled_high_gt = high_gt
            scaled_high_gt_list.append(scaled_high_gt)
            
            # 2. 中等质量GT缩放
            medium_gt = matched_results['medium_quality_gt'][idx]
            medium_rois = matched_results['medium_quality_rois'][idx]
            
            if len(medium_gt) > 0:
                original_sizes = medium_gt[:, 3:5].clone()
                scaled_medium_gt = self.scale_medium_quality(
                    medium_gt, medium_rois[:, :3], original_sizes
                )
            else:
                scaled_medium_gt = medium_gt
            scaled_medium_gt_list.append(scaled_medium_gt)
            
            # 3. 未匹配GT缩放
            unmatched_near = matched_results['unmatched_gt_near'][idx]
            unmatched_medium = matched_results['unmatched_gt_medium'][idx]
            unmatched_far = matched_results['unmatched_gt_far'][idx]
            
            scaled_unmatched_near = self.scale_unmatched_gt(unmatched_near)
            scaled_unmatched_medium = self.scale_unmatched_gt(unmatched_medium)
            scaled_unmatched_far = self.scale_unmatched_gt(unmatched_far)
            
            scaled_unmatched_near_list.append(scaled_unmatched_near)
            scaled_unmatched_medium_list.append(scaled_unmatched_medium)
            scaled_unmatched_far_list.append(scaled_unmatched_far)
        
        # 创建新的结果字典
        scaled_results = deepcopy(matched_results)
        scaled_results['refined_high_quality_gt'] = scaled_high_gt_list
        scaled_results['medium_quality_gt'] = scaled_medium_gt_list
        scaled_results['unmatched_gt_near'] = scaled_unmatched_near_list
        scaled_results['unmatched_gt_medium'] = scaled_unmatched_medium_list
        scaled_results['unmatched_gt_far'] = scaled_unmatched_far_list
        
        return scaled_results


