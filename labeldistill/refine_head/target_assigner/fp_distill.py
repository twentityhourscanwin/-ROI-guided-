#LabelDistill/labeldistill/refine_head/target_assigner/fp_distill.py
import torch
import torch.nn as nn
from mmdet3d.core.bbox.iou_calculators import BboxOverlapsNearest3D


class StudentSpecificFPFilter(nn.Module):
    """
    学生网络特有误检筛选器：
    1. 先通过IoU筛选出学生特有的误检ROI（与教师误检ROI的IoU都低于阈值）
    2. 在学生特有的误检ROI中，按类别筛选出每个类别得分最高的ROI
    """
    def __init__(self, filter_cfg):
        super().__init__()
        self.filter_cfg = filter_cfg
        # 初始化BEV IoU计算器
        self.iou_calculator = BboxOverlapsNearest3D()
        
        # 类别分组信息
        self.class_groups = [
            {'num_class': 1, 'class_names': ['car'], 'class_ids': [0]},
            {'num_class': 2, 'class_names': ['truck', 'construction_vehicle'], 'class_ids': [1, 2]},
            {'num_class': 2, 'class_names': ['bus', 'trailer'], 'class_ids': [3, 4]},
            {'num_class': 1, 'class_names': ['barrier'], 'class_ids': [5]},
            {'num_class': 2, 'class_names': ['motorcycle', 'bicycle'], 'class_ids': [6, 7]},
            {'num_class': 2, 'class_names': ['pedestrian', 'traffic_cone'], 'class_ids': [8, 9]},
        ]
        
        # 提取所有类别ID
        self.all_class_ids = []
        for group in self.class_groups:
            self.all_class_ids.extend(group['class_ids'])
        
    def forward(self, student_filter_results, teacher_matched_results):
        """
        Args:
            student_filter_results: dict - 学生网络FalsePositiveRoiFilter的输出
                'high_score_fp_rois': List[Tensor] - 学生网络的高分误检ROI
                    每个元素: (HFP_i, 7+C)
                'high_score_fp_roi_scores': List[Tensor]
                    每个元素: (HFP_i,)
                'high_score_fp_roi_labels': List[Tensor]
                    每个元素: (HFP_i,)
                'low_score_fp_rois': List[Tensor] - 学生网络的低分误检ROI
                    每个元素: (LFP_i, 7+C)
                'low_score_fp_roi_scores': List[Tensor]
                    每个元素: (LFP_i,)
                'low_score_fp_roi_labels': List[Tensor]
                    每个元素: (LFP_i,)
                    
            teacher_matched_results: dict - 教师网络ProposalTargetLayer的输出
                'false_positive_rois': List[Tensor] - 教师网络的误检ROI
                    每个元素: (FP_teacher_i, 7+C)
                'false_positive_roi_scores': List[Tensor]
                    每个元素: (FP_teacher_i,)
                'false_positive_roi_labels': List[Tensor]
                    每个元素: (FP_teacher_i,)
                
        Returns:
            filtered_results: dict
                'student_specific_fp_rois': List[Tensor]
                    每个元素: (SFP_i, 7+C) - 每个类别最多1个ROI
                'student_specific_fp_scores': List[Tensor]
                    每个元素: (SFP_i,)
                'student_specific_fp_labels': List[Tensor]
                    每个元素: (SFP_i,)
                
                # 统计信息
                'num_total_student_fp_per_batch': List[int] - 每个batch的学生总误检数量
                'num_teacher_fp_per_batch': List[int] - 每个batch的教师误检数量
                'num_student_specific_before_class_filter': List[int] - 类别筛选前的学生特有误检数量
                'num_student_specific_fp_per_batch': List[int] - 最终筛选出的数量（按类别）
                'selected_classes_per_batch': List[List[int]] - 每个batch中被选中的类别ID列表
        """
        # 获取筛选参数
        iou_thresh = self.filter_cfg.get('TEACHER_FP_IOU_THRESH', 0.3)
        
        # 获取学生的高分和低分误检ROI（合并处理）
        student_high_fp_rois = student_filter_results['high_score_fp_rois']
        student_high_fp_scores = student_filter_results['high_score_fp_roi_scores']
        student_high_fp_labels = student_filter_results['high_score_fp_roi_labels']
        
        student_low_fp_rois = student_filter_results['low_score_fp_rois']
        student_low_fp_scores = student_filter_results['low_score_fp_roi_scores']
        student_low_fp_labels = student_filter_results['low_score_fp_roi_labels']
        
        # 获取教师的误检ROI
        teacher_fp_rois = teacher_matched_results['false_positive_rois']
        
        batch_size = len(student_high_fp_rois)
        
        # 存储结果
        student_specific_fp_rois_list = []
        student_specific_fp_scores_list = []
        student_specific_fp_labels_list = []
        
        # 统计信息
        num_total_student_fp_list = []
        num_teacher_fp_list = []
        num_before_class_filter_list = []
        num_student_specific_fp_list = []
        selected_classes_list = []
        
        for idx in range(batch_size):
            # 合并高分和低分误检ROI
            cur_high_fp_rois = student_high_fp_rois[idx]
            cur_high_fp_scores = student_high_fp_scores[idx]
            cur_high_fp_labels = student_high_fp_labels[idx]
            
            cur_low_fp_rois = student_low_fp_rois[idx]
            cur_low_fp_scores = student_low_fp_scores[idx]
            cur_low_fp_labels = student_low_fp_labels[idx]
            
            # 合并所有学生误检ROI
            if len(cur_high_fp_rois) > 0 and len(cur_low_fp_rois) > 0:
                all_student_fp_rois = torch.cat([cur_high_fp_rois, cur_low_fp_rois], dim=0)
                all_student_fp_scores = torch.cat([cur_high_fp_scores, cur_low_fp_scores], dim=0)
                all_student_fp_labels = torch.cat([cur_high_fp_labels, cur_low_fp_labels], dim=0)
            elif len(cur_high_fp_rois) > 0:
                all_student_fp_rois = cur_high_fp_rois
                all_student_fp_scores = cur_high_fp_scores
                all_student_fp_labels = cur_high_fp_labels
            elif len(cur_low_fp_rois) > 0:
                all_student_fp_rois = cur_low_fp_rois
                all_student_fp_scores = cur_low_fp_scores
                all_student_fp_labels = cur_low_fp_labels
            else:
                all_student_fp_rois = cur_high_fp_rois  # 空tensor
                all_student_fp_scores = cur_high_fp_scores
                all_student_fp_labels = cur_high_fp_labels
            
            cur_teacher_fp_rois = teacher_fp_rois[idx]
            
            # 获取维度
            roi_dim = 7
            if len(all_student_fp_rois) > 0:
                roi_dim = all_student_fp_rois.shape[-1]
            
            # 统计原始数量
            num_total_student_fp = len(all_student_fp_rois)
            num_teacher_fp = len(cur_teacher_fp_rois)
            
            num_total_student_fp_list.append(num_total_student_fp)
            num_teacher_fp_list.append(num_teacher_fp)
            
            # 处理空情况：学生没有误检
            if num_total_student_fp == 0:
                device = cur_teacher_fp_rois.device if num_teacher_fp > 0 else torch.device('cuda')
                student_specific_fp_rois_list.append(torch.zeros((0, roi_dim), device=device))
                student_specific_fp_scores_list.append(torch.zeros(0, device=device))
                student_specific_fp_labels_list.append(torch.zeros(0, dtype=torch.long, device=device))
                
                num_before_class_filter_list.append(0)
                num_student_specific_fp_list.append(0)
                selected_classes_list.append([])
                continue
            
            # 步骤1：通过IoU筛选出学生特有的误检ROI
            if num_teacher_fp == 0:
                # 教师没有误检，所有学生误检都是"特有"的
                student_specific_mask = torch.ones(num_total_student_fp, dtype=torch.bool, device=all_student_fp_rois.device)
            else:
                # 计算学生误检ROI与教师误检ROI之间的BEV IoU
                bev_ious = self.iou_calculator(
                    all_student_fp_rois[:, :7], 
                    cur_teacher_fp_rois[:, :7]
                )  # (N_student, N_teacher)
                
                # 对每个学生误检ROI，计算其与所有教师误检ROI的最大IoU
                max_ious_with_teacher = bev_ious.max(dim=1)[0]  # (N_student,)
                
                # 筛选与所有教师误检ROI的IoU都低于阈值的学生误检ROI
                student_specific_mask = max_ious_with_teacher <= iou_thresh
            
            # 获取学生特有的误检ROI
            student_specific_rois = all_student_fp_rois[student_specific_mask]
            student_specific_scores = all_student_fp_scores[student_specific_mask]
            student_specific_labels = all_student_fp_labels[student_specific_mask]
            
            num_before_class_filter = len(student_specific_rois)
            num_before_class_filter_list.append(num_before_class_filter)
            
            # 如果没有学生特有的误检ROI
            if num_before_class_filter == 0:
                student_specific_fp_rois_list.append(torch.zeros((0, roi_dim), device=all_student_fp_rois.device))
                student_specific_fp_scores_list.append(torch.zeros(0, device=all_student_fp_rois.device))
                student_specific_fp_labels_list.append(torch.zeros(0, dtype=torch.long, device=all_student_fp_rois.device))
                
                num_student_specific_fp_list.append(0)
                selected_classes_list.append([])
                continue
            
            # 步骤2：在学生特有的误检ROI中，按类别筛选出每个类别得分最高的ROI
            selected_rois = []
            selected_scores = []
            selected_labels = []
            selected_classes = []
            
            for class_id in self.all_class_ids:
                # 找到该类别的所有ROI
                class_mask = student_specific_labels == class_id
                
                if class_mask.any():
                    # 该类别存在误检ROI，选择得分最高的
                    class_rois = student_specific_rois[class_mask]
                    class_scores = student_specific_scores[class_mask]
                    class_labels = student_specific_labels[class_mask]
                    
                    best_idx = class_scores.argmax()
                    
                    selected_rois.append(class_rois[best_idx:best_idx+1])
                    selected_scores.append(class_scores[best_idx:best_idx+1])
                    selected_labels.append(class_labels[best_idx:best_idx+1])
                    selected_classes.append(class_id)
            
            # 合并所有选中的ROI
            if len(selected_rois) > 0:
                final_rois = torch.cat(selected_rois, dim=0)
                final_scores = torch.cat(selected_scores, dim=0)
                final_labels = torch.cat(selected_labels, dim=0)
            else:
                final_rois = torch.zeros((0, roi_dim), device=all_student_fp_rois.device)
                final_scores = torch.zeros(0, device=all_student_fp_rois.device)
                final_labels = torch.zeros(0, dtype=torch.long, device=all_student_fp_rois.device)
            
            student_specific_fp_rois_list.append(final_rois)
            student_specific_fp_scores_list.append(final_scores)
            student_specific_fp_labels_list.append(final_labels)
            
            num_student_specific_fp_list.append(len(final_rois))
            selected_classes_list.append(selected_classes)
        
        filtered_results = {
            # 学生网络特有的误检ROI（按类别筛选后）
            'student_specific_fp_rois': student_specific_fp_rois_list,
            'student_specific_fp_scores': student_specific_fp_scores_list,
            'student_specific_fp_labels': student_specific_fp_labels_list,
            
            # 统计信息
            'num_total_student_fp_per_batch': num_total_student_fp_list,
            'num_teacher_fp_per_batch': num_teacher_fp_list,
            'num_student_specific_before_class_filter': num_before_class_filter_list,
            'num_student_specific_fp_per_batch': num_student_specific_fp_list,
            'selected_classes_per_batch': selected_classes_list,
        }
        
        return filtered_results
    
    def get_statistics(self, filtered_results):
        """
        获取筛选统计信息
        
        Returns:
            stats: dict
                'total_student_fp': int - 总的学生误检数量
                'total_teacher_fp': int - 总的教师误检数量
                'total_student_specific_before_filter': int - 类别筛选前的学生特有误检数量
                'total_student_specific_fp': int - 最终筛选出的学生特有误检数量
                'class_filter_ratio': float - 类别筛选率（筛选后/筛选前）
                'class_coverage': dict - 每个类别被选中的次数
        """
        total_student_fp = sum(filtered_results['num_total_student_fp_per_batch'])
        total_teacher_fp = sum(filtered_results['num_teacher_fp_per_batch'])
        total_before_filter = sum(filtered_results['num_student_specific_before_class_filter'])
        total_student_specific_fp = sum(filtered_results['num_student_specific_fp_per_batch'])
        
        class_filter_ratio = (
            total_student_specific_fp / total_before_filter 
            if total_before_filter > 0 else 0.0
        )
        
        # 统计每个类别被选中的次数
        class_coverage = {class_id: 0 for class_id in self.all_class_ids}
        for selected_classes in filtered_results['selected_classes_per_batch']:
            for class_id in selected_classes:
                class_coverage[class_id] += 1
        
        stats = {
            'total_student_fp': total_student_fp,
            'total_teacher_fp': total_teacher_fp,
            'total_student_specific_before_filter': total_before_filter,
            'total_student_specific_fp': total_student_specific_fp,
            'class_filter_ratio': class_filter_ratio,
            'class_coverage': class_coverage,
        }
        
        return stats