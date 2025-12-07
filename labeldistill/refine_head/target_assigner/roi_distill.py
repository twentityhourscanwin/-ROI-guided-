#labeldistill/refine_head/target_assigner/roi_distill.py
import torch
import torch.nn as nn


class ProposalTargetLayer(nn.Module):
    """
    ROI筛选器：以GT为中心，基于中心距离（BEV平面）分类GT并生成高质量ROI
    支持类别相关的距离阈值和任务组内匹配
    """
    def __init__(self, roi_sampler_cfg):
        super().__init__()
        self.roi_sampler_cfg = roi_sampler_cfg
        
        # 类别任务组配置（用于判断是否属于同一任务）
        self.class_groups = [
            {'num_class': 1, 'class_names': ['car'], 'class_ids': [0]},
            {'num_class': 2, 'class_names': ['truck', 'construction_vehicle'], 'class_ids': [1, 2]},
            {'num_class': 2, 'class_names': ['bus', 'trailer'], 'class_ids': [3, 4]},
            {'num_class': 1, 'class_names': ['barrier'], 'class_ids': [5]},
            {'num_class': 2, 'class_names': ['motorcycle', 'bicycle'], 'class_ids': [6, 7]},
            {'num_class': 2, 'class_names': ['pedestrian', 'traffic_cone'], 'class_ids': [8, 9]},
        ]
        
        # 构建类别到任务组的映射
        self.class_to_group = {}
        for group_idx, group in enumerate(self.class_groups):
            for class_id in group['class_ids']:
                self.class_to_group[class_id] = group_idx
        
        # 类别相关的中心距离阈值配置
        self.distance_thresholds = {
            0: {'high': 2.0, 'medium': 4.0},  # car
            1: {'high': 2.5, 'medium': 4.0},  # truck
            2: {'high': 2.5, 'medium': 4.0},  # construction_vehicle
            3: {'high': 3.5, 'medium': 4.0},  # bus
            4: {'high': 2.5, 'medium': 4.0},  # trailer
            5: {'high': 0.5, 'medium': 2.5},  # barrier
            6: {'high': 1.0, 'medium': 2.5},  # motorcycle
            7: {'high': 1.0, 'medium': 2.5},  # bicycle
            8: {'high': 0.5, 'medium': 2.0},  # pedestrian
            9: {'high': 0.5, 'medium': 2.0},  # traffic_cone
        }
        
    def get_distance_thresholds_for_class(self, class_id):
        """
        根据类别ID获取中心距离阈值
        
        Args:
            class_id: 类别ID (0-9)
            
        Returns:
            high_thresh: 高质量匹配阈值
            medium_thresh: 中等质量匹配阈值
        """
        class_id = int(class_id)
        thresholds = self.distance_thresholds.get(class_id, {'high': 2.0, 'medium': 4.0})
        return thresholds['high'], thresholds['medium']
    
    def same_task_group(self, class_id1, class_id2):
        """
        判断两个类别是否属于同一任务组
        
        Args:
            class_id1: 第一个类别ID
            class_id2: 第二个类别ID
            
        Returns:
            bool: 是否属于同一任务组
        """
        class_id1 = int(class_id1)
        class_id2 = int(class_id2)
        
        group1 = self.class_to_group.get(class_id1, -1)
        group2 = self.class_to_group.get(class_id2, -1)
        
        return group1 == group2 and group1 != -1
    
    def compute_bev_center_distance(self, roi_centers, gt_centers):
        """
        计算BEV平面（x-y平面）上的中心距离
        
        Args:
            roi_centers: (N, 2 or 3) ROI中心坐标
            gt_centers: (M, 2 or 3) GT中心坐标
            
        Returns:
            distances: (N, M) 距离矩阵
        """
        # 只使用x和y坐标
        roi_xy = roi_centers[:, :2]  # (N, 2)
        gt_xy = gt_centers[:, :2]    # (M, 2)
        
        # 计算欧氏距离
        # (N, 1, 2) - (1, M, 2) -> (N, M, 2)
        diff = roi_xy.unsqueeze(1) - gt_xy.unsqueeze(0)
        distances = torch.sqrt((diff ** 2).sum(dim=-1))  # (N, M)
        
        return distances
        
    def forward(self, batch_dict):
        """
        Args:
            batch_dict = {
                'batch_size': batch_size,
                'rois': rois_padded,  # (B, max_rois, 9)
                'roi_scores': roi_scores_padded,  # (B, max_rois)
                'roi_labels': roi_labels_padded,  # (B, max_rois)
                'gt_boxes_and_cls': gt_boxes_padded,  # (B, max_gts, 10)
            }
                
        Returns:
            matched_results: dict
        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        roi_scores = batch_dict['roi_scores']
        roi_labels = batch_dict['roi_labels']
        gt_boxes = batch_dict['gt_boxes_and_cls']
        
        # 距离阈值配置（用于未匹配GT的分类）
        near_distance_thresh = self.roi_sampler_cfg.get('NEAR_DISTANCE_THRESH', 30.0)
        far_distance_thresh = self.roi_sampler_cfg.get('FAR_DISTANCE_THRESH', 50.0)
        
        # 存储结果
        refined_high_rois_list = []
        refined_high_scores_list = []
        refined_high_labels_list = []
        refined_high_gt_list = []
        
        medium_rois_list = []
        medium_scores_list = []
        medium_labels_list = []
        medium_gt_list = []
        
        unmatched_gt_near_list = []    # <30m
        unmatched_gt_medium_list = []  # 30-50m
        unmatched_gt_far_list = []     # >50m
        
        false_positive_rois_list = []
        false_positive_scores_list = []
        false_positive_labels_list = []
        
        for idx in range(batch_size):
            cur_roi = rois[idx]
            cur_gt = gt_boxes[idx]
            cur_scores = roi_scores[idx]
            cur_labels = roi_labels[idx]
            
            # 过滤padding
            valid_gt_mask = cur_gt.sum(dim=-1) != 0
            cur_gt = cur_gt[valid_gt_mask]
            
            valid_roi_mask = cur_roi.sum(dim=-1) != 0
            cur_roi = cur_roi[valid_roi_mask]
            cur_scores = cur_scores[valid_roi_mask]
            cur_labels = cur_labels[valid_roi_mask]
            
            roi_dim = cur_roi.shape[-1] if len(cur_roi) > 0 else 7
            gt_dim = cur_gt.shape[-1] if len(cur_gt) > 0 else 8
            
            # 处理空GT情况
            if len(cur_gt) == 0:
                refined_high_rois_list.append(torch.zeros((0, roi_dim), device=cur_roi.device))
                refined_high_scores_list.append(torch.zeros(0, device=cur_roi.device))
                refined_high_labels_list.append(torch.zeros(0, dtype=torch.long, device=cur_roi.device))
                refined_high_gt_list.append(torch.zeros((0, gt_dim), device=cur_roi.device))
                
                medium_rois_list.append(torch.zeros((0, roi_dim), device=cur_roi.device))
                medium_scores_list.append(torch.zeros(0, device=cur_roi.device))
                medium_labels_list.append(torch.zeros(0, dtype=torch.long, device=cur_roi.device))
                medium_gt_list.append(torch.zeros((0, gt_dim), device=cur_roi.device))
                
                unmatched_gt_near_list.append(torch.zeros((0, gt_dim), device=cur_roi.device))
                unmatched_gt_medium_list.append(torch.zeros((0, gt_dim), device=cur_roi.device))
                unmatched_gt_far_list.append(torch.zeros((0, gt_dim), device=cur_roi.device))
                
                # 所有ROI都是误检
                false_positive_rois_list.append(cur_roi)
                false_positive_scores_list.append(cur_scores)
                false_positive_labels_list.append(cur_labels)
                continue
            
            # 处理空ROI情况
            if len(cur_roi) == 0:
                refined_high_rois_list.append(torch.zeros((0, roi_dim), device=cur_gt.device))
                refined_high_scores_list.append(torch.zeros(0, device=cur_gt.device))
                refined_high_labels_list.append(torch.zeros(0, dtype=torch.long, device=cur_gt.device))
                refined_high_gt_list.append(torch.zeros((0, gt_dim), device=cur_gt.device))
                
                medium_rois_list.append(torch.zeros((0, roi_dim), device=cur_gt.device))
                medium_scores_list.append(torch.zeros(0, device=cur_gt.device))
                medium_labels_list.append(torch.zeros(0, dtype=torch.long, device=cur_gt.device))
                medium_gt_list.append(torch.zeros((0, gt_dim), device=cur_gt.device))
                
                # 对所有未匹配GT进行距离细分
                gt_distances = torch.sqrt(cur_gt[:, 0]**2 + cur_gt[:, 1]**2)
                near_mask = gt_distances < near_distance_thresh
                far_mask = gt_distances > far_distance_thresh
                medium_mask = ~(near_mask | far_mask)
                
                unmatched_gt_near_list.append(cur_gt[near_mask])
                unmatched_gt_medium_list.append(cur_gt[medium_mask])
                unmatched_gt_far_list.append(cur_gt[far_mask])
                
                false_positive_rois_list.append(torch.zeros((0, roi_dim), device=cur_gt.device))
                false_positive_scores_list.append(torch.zeros(0, device=cur_gt.device))
                false_positive_labels_list.append(torch.zeros(0, dtype=torch.long, device=cur_gt.device))
                continue
            
            # 计算BEV平面中心距离
            center_distances = self.compute_bev_center_distance(
                cur_roi[:, :3], cur_gt[:, :3]
            )  # (num_roi, num_gt)
            
            # 标记误检ROI：与所有GT的最小距离都大于4米
            min_distance_to_any_gt = center_distances.min(dim=1)[0]  # (num_roi,)
            false_positive_mask = min_distance_to_any_gt > 4.0
            
            # 以GT为中心进行分类
            num_gt = len(cur_gt)
            
            refined_high_rois = []
            refined_high_scores = []
            refined_high_labels = []
            refined_high_gts = []
            
            medium_rois = []
            medium_scores = []
            medium_labels = []
            medium_gts = []
            
            unmatched_gt_indices = []
            
            for gt_idx in range(num_gt):
                # 获取当前GT的类别和对应阈值
                gt_class_id = cur_gt[gt_idx, -1].item()  # 最后一维是类别
                high_thresh, medium_thresh = self.get_distance_thresholds_for_class(gt_class_id)
                
                distances_to_gt = center_distances[:, gt_idx]
                
                # 筛选同任务组的ROI
                same_group_mask = torch.tensor([
                    self.same_task_group(roi_label.item(), gt_class_id) 
                    for roi_label in cur_labels
                ], device=cur_roi.device, dtype=torch.bool)
                
                # 高质量ROI：距离 <= high_thresh 且同任务组
                high_quality_mask = (distances_to_gt <= high_thresh) & same_group_mask
                high_quality_indices = high_quality_mask.nonzero().view(-1)
                
                # 中等质量ROI：high_thresh < 距离 <= medium_thresh 且同任务组
                medium_quality_mask = (distances_to_gt > high_thresh) & \
                                     (distances_to_gt <= medium_thresh) & \
                                     same_group_mask
                medium_quality_indices = medium_quality_mask.nonzero().view(-1)
                
                has_high = len(high_quality_indices) > 0
                has_medium = len(medium_quality_indices) > 0
                
                if has_high:
                    # 存在高质量ROI（无论是否有中等质量）
                    high_scores = cur_scores[high_quality_indices]
                    best_high_idx = high_scores.argmax()
                    best_high_roi_idx = high_quality_indices[best_high_idx]
                    best_high_roi = cur_roi[best_high_roi_idx]
                    best_high_score = cur_scores[best_high_roi_idx]

                    # 直接使用原始ROI，保留其相对GT的真实偏移信息
                    # 用于后续单侧扩展的方向和幅度判断
                    new_roi = best_high_roi.clone()

                    # 计算新的score: min((1.2 + score) / 2, 1)
                    new_score = torch.min(
                        torch.tensor((1.2 + best_high_score.item()) / 2, device=best_high_score.device),
                        torch.tensor(1.0, device=best_high_score.device)
                    )

                    refined_high_rois.append(new_roi)
                    refined_high_scores.append(new_score)
                    refined_high_labels.append(cur_labels[best_high_roi_idx])
                    refined_high_gts.append(cur_gt[gt_idx])
                    
                elif has_medium:
                    # 只存在中等质量ROI
                    medium_quality_scores = cur_scores[medium_quality_indices]
                    best_medium_idx = medium_quality_scores.argmax()
                    best_medium_roi_idx = medium_quality_indices[best_medium_idx]
                    
                    medium_rois.append(cur_roi[best_medium_roi_idx])
                    medium_scores.append(cur_scores[best_medium_roi_idx])
                    medium_labels.append(cur_labels[best_medium_roi_idx])
                    medium_gts.append(cur_gt[gt_idx])
                    
                else:
                    # 未被检测到
                    unmatched_gt_indices.append(gt_idx)
            
            # 整理精准匹配结果
            if len(refined_high_rois) > 0:
                refined_high_rois_list.append(torch.stack(refined_high_rois))
                refined_high_scores_list.append(torch.stack(refined_high_scores))
                refined_high_labels_list.append(torch.stack(refined_high_labels))
                refined_high_gt_list.append(torch.stack(refined_high_gts))
            else:
                refined_high_rois_list.append(torch.zeros((0, roi_dim), device=cur_gt.device))
                refined_high_scores_list.append(torch.zeros(0, device=cur_gt.device))
                refined_high_labels_list.append(torch.zeros(0, dtype=torch.long, device=cur_gt.device))
                refined_high_gt_list.append(torch.zeros((0, gt_dim), device=cur_gt.device))
            
            # 整理中等质量匹配结果
            if len(medium_rois) > 0:
                medium_rois_list.append(torch.stack(medium_rois))
                medium_scores_list.append(torch.stack(medium_scores))
                medium_labels_list.append(torch.stack(medium_labels))
                medium_gt_list.append(torch.stack(medium_gts))
            else:
                medium_rois_list.append(torch.zeros((0, roi_dim), device=cur_gt.device))
                medium_scores_list.append(torch.zeros(0, device=cur_gt.device))
                medium_labels_list.append(torch.zeros(0, dtype=torch.long, device=cur_gt.device))
                medium_gt_list.append(torch.zeros((0, gt_dim), device=cur_gt.device))
            
            # 整理未匹配结果并进行距离细分
            if len(unmatched_gt_indices) > 0:
                unmatched_gts = cur_gt[unmatched_gt_indices]
                
                # 计算距离并细分
                gt_distances = torch.sqrt(unmatched_gts[:, 0]**2 + unmatched_gts[:, 1]**2)
                near_mask = gt_distances < near_distance_thresh
                far_mask = gt_distances > far_distance_thresh
                medium_mask = ~(near_mask | far_mask)
                
                unmatched_gt_near_list.append(unmatched_gts[near_mask])
                unmatched_gt_medium_list.append(unmatched_gts[medium_mask])
                unmatched_gt_far_list.append(unmatched_gts[far_mask])
            else:
                unmatched_gt_near_list.append(torch.zeros((0, gt_dim), device=cur_gt.device))
                unmatched_gt_medium_list.append(torch.zeros((0, gt_dim), device=cur_gt.device))
                unmatched_gt_far_list.append(torch.zeros((0, gt_dim), device=cur_gt.device))
            
            # 整理误检ROI结果
            false_positive_indices = false_positive_mask.nonzero().view(-1)
            if len(false_positive_indices) > 0:
                false_positive_rois_list.append(cur_roi[false_positive_indices])
                false_positive_scores_list.append(cur_scores[false_positive_indices])
                false_positive_labels_list.append(cur_labels[false_positive_indices])
            else:
                false_positive_rois_list.append(torch.zeros((0, roi_dim), device=cur_gt.device))
                false_positive_scores_list.append(torch.zeros(0, device=cur_gt.device))
                false_positive_labels_list.append(torch.zeros(0, dtype=torch.long, device=cur_gt.device))
       
       
                
                
        matched_results = {
            # 高质量匹配GT生成的融合ROI
            'refined_high_quality_rois': refined_high_rois_list,
            'refined_high_quality_roi_scores': refined_high_scores_list,
            'refined_high_quality_roi_labels': refined_high_labels_list,
            'refined_high_quality_gt': refined_high_gt_list,
            
            # 中等质量匹配GT的最高分ROI
            'medium_quality_rois': medium_rois_list,
            'medium_quality_roi_scores': medium_scores_list,
            'medium_quality_roi_labels': medium_labels_list,
            'medium_quality_gt': medium_gt_list,
            
            # 未匹配GT的细粒度距离划分
            'unmatched_gt_near': unmatched_gt_near_list,      # <30m
            'unmatched_gt_medium': unmatched_gt_medium_list,  # 30-50m
            'unmatched_gt_far': unmatched_gt_far_list,        # >50m
            
            # 误检ROI（与所有GT距离都>4m）
            'false_positive_rois': false_positive_rois_list,
            'false_positive_roi_scores': false_positive_scores_list,
            'false_positive_roi_labels': false_positive_labels_list,
        }
        
        
        return matched_results