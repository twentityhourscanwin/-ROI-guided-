#labeldistill/refine_head/target_assigner/roi_distill_fp.py
import torch
import torch.nn as nn

class FalsePositiveRoiFilter(nn.Module):
    """
    误检ROI筛选器：筛选出与所有GT的BEV中心距离都超过阈值的ROI（误检）
    """
    def __init__(self, filter_cfg):
        super().__init__()
        self.filter_cfg = filter_cfg
        
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)
                gt_boxes_and_cls: (B, N, 7 + C + 1)
                
        Returns:
            filter_results: dict
                # 误检ROI（与所有GT的中心距离都超过阈值）
                false_positive_rois: List[Tensor] - 误检ROI
                    每个元素: (FP_i, 7+C)
                false_positive_roi_scores: List[Tensor] - 误检ROI的分数
                    每个元素: (FP_i,)
                false_positive_roi_labels: List[Tensor] - 误检ROI的标签
                    每个元素: (FP_i,)
                
                # 按分数细分的误检ROI
                high_score_fp_rois: List[Tensor] - 高分误检ROI
                    每个元素: (HFP_i, 7+C)
                high_score_fp_roi_scores: List[Tensor] - 高分误检ROI的分数
                    每个元素: (HFP_i,)
                high_score_fp_roi_labels: List[Tensor] - 高分误检ROI的标签
                    每个元素: (HFP_i,)
                    
                low_score_fp_rois: List[Tensor] - 低分误检ROI
                    每个元素: (LFP_i, 7+C)
                low_score_fp_roi_scores: List[Tensor] - 低分误检ROI的分数
                    每个元素: (LFP_i,)
                low_score_fp_roi_labels: List[Tensor] - 低分误检ROI的标签
                    每个元素: (LFP_i,)
        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        roi_scores = batch_dict['roi_scores']
        roi_labels = batch_dict['roi_labels']
        gt_boxes = batch_dict['gt_boxes_and_cls']
        
        # 筛选参数
        distance_thresh = self.filter_cfg.get('FP_DISTANCE_THRESH', 4.0)  # 中心距离阈值（米）
        high_score_thresh = self.filter_cfg.get('HIGH_SCORE_THRESH', 0.5)  # 高分阈值
        
        # 存储结果
        fp_rois_list = []
        fp_scores_list = []
        fp_labels_list = []
        
        high_score_fp_rois_list = []
        high_score_fp_scores_list = []
        high_score_fp_labels_list = []
        
        low_score_fp_rois_list = []
        low_score_fp_scores_list = []
        low_score_fp_labels_list = []
        
        for idx in range(batch_size):
            cur_roi = rois[idx]  # (num_rois, 7+C)
            cur_gt = gt_boxes[idx]  # (N, 7+C+1)
            cur_scores = roi_scores[idx]  # (num_rois,)
            cur_labels = roi_labels[idx]  # (num_rois,)
            
            # 过滤padding的GT
            valid_gt_mask = cur_gt.sum(dim=-1) != 0
            cur_gt = cur_gt[valid_gt_mask]
            
            # 过滤padding的ROI
            valid_roi_mask = cur_roi.sum(dim=-1) != 0
            cur_roi = cur_roi[valid_roi_mask]
            cur_scores = cur_scores[valid_roi_mask]
            cur_labels = cur_labels[valid_roi_mask]
            
            # 获取维度
            roi_dim = cur_roi.shape[-1] if len(cur_roi) > 0 else 7
            
            # 处理空情况
            if len(cur_roi) == 0:
                # 没有ROI，所有列表都为空
                fp_rois_list.append(torch.zeros((0, roi_dim), device=rois.device))
                fp_scores_list.append(torch.zeros(0, device=rois.device))
                fp_labels_list.append(torch.zeros(0, dtype=torch.long, device=rois.device))
                
                high_score_fp_rois_list.append(torch.zeros((0, roi_dim), device=rois.device))
                high_score_fp_scores_list.append(torch.zeros(0, device=rois.device))
                high_score_fp_labels_list.append(torch.zeros(0, dtype=torch.long, device=rois.device))
                
                low_score_fp_rois_list.append(torch.zeros((0, roi_dim), device=rois.device))
                low_score_fp_scores_list.append(torch.zeros(0, device=rois.device))
                low_score_fp_labels_list.append(torch.zeros(0, dtype=torch.long, device=rois.device))
                continue
            
            if len(cur_gt) == 0:
                # 没有GT，所有ROI都是误检
                fp_rois_list.append(cur_roi)
                fp_scores_list.append(cur_scores)
                fp_labels_list.append(cur_labels)
                
                # 按分数细分
                high_score_mask = cur_scores > high_score_thresh
                high_score_fp_rois_list.append(cur_roi[high_score_mask])
                high_score_fp_scores_list.append(cur_scores[high_score_mask])
                high_score_fp_labels_list.append(cur_labels[high_score_mask])
                
                low_score_fp_rois_list.append(cur_roi[~high_score_mask])
                low_score_fp_scores_list.append(cur_scores[~high_score_mask])
                low_score_fp_labels_list.append(cur_labels[~high_score_mask])
                continue
            
            # 计算BEV中心距离
            # ROI中心: (num_rois, 2) - 取x,y坐标
            roi_centers = cur_roi[:, :2]  # (num_rois, 2)
            # GT中心: (num_gts, 2)
            gt_centers = cur_gt[:, :2]  # (num_gts, 2)
            
            # 计算所有ROI与所有GT之间的欧式距离: (num_rois, num_gts)
            # 使用广播机制
            distances = torch.sqrt(
                (roi_centers[:, None, 0] - gt_centers[None, :, 0])**2 + 
                (roi_centers[:, None, 1] - gt_centers[None, :, 1])**2
            )  # (num_rois, num_gts)
            
            # === 以ROI为中心进行分类 ===
            # 对每个ROI，计算其与所有GT的最小距离
            min_distances_per_roi = distances.min(dim=1)[0]  # (num_rois,)
            
            # 误检ROI：与所有GT的距离都超过阈值
            fp_mask = min_distances_per_roi > distance_thresh
            
            # 提取误检ROI
            fp_rois = cur_roi[fp_mask]
            fp_scores = cur_scores[fp_mask]
            fp_labels = cur_labels[fp_mask]
            
            # 添加到列表
            fp_rois_list.append(fp_rois)
            fp_scores_list.append(fp_scores)
            fp_labels_list.append(fp_labels)
            
            # === 按分数细分误检ROI ===
            if len(fp_rois) > 0:
                high_score_mask = fp_scores > high_score_thresh
                
                # 高分误检ROI
                high_score_fp_rois_list.append(fp_rois[high_score_mask])
                high_score_fp_scores_list.append(fp_scores[high_score_mask])
                high_score_fp_labels_list.append(fp_labels[high_score_mask])
                
                # 低分误检ROI
                low_score_fp_rois_list.append(fp_rois[~high_score_mask])
                low_score_fp_scores_list.append(fp_scores[~high_score_mask])
                low_score_fp_labels_list.append(fp_labels[~high_score_mask])
            else:
                # 没有误检ROI
                high_score_fp_rois_list.append(torch.zeros((0, roi_dim), device=cur_roi.device))
                high_score_fp_scores_list.append(torch.zeros(0, device=cur_roi.device))
                high_score_fp_labels_list.append(torch.zeros(0, dtype=torch.long, device=cur_roi.device))
                
                low_score_fp_rois_list.append(torch.zeros((0, roi_dim), device=cur_roi.device))
                low_score_fp_scores_list.append(torch.zeros(0, device=cur_roi.device))
                low_score_fp_labels_list.append(torch.zeros(0, dtype=torch.long, device=cur_roi.device))
        
        filter_results = {
            # 所有误检ROI
            'false_positive_rois': fp_rois_list,
            'false_positive_roi_scores': fp_scores_list,
            'false_positive_roi_labels': fp_labels_list,
            
            # 高分误检ROI
            'high_score_fp_rois': high_score_fp_rois_list,
            'high_score_fp_roi_scores': high_score_fp_scores_list,
            'high_score_fp_roi_labels': high_score_fp_labels_list,
            
            # 低分误检ROI
            'low_score_fp_rois': low_score_fp_rois_list,
            'low_score_fp_roi_scores': low_score_fp_scores_list,
            'low_score_fp_roi_labels': low_score_fp_labels_list,
        }
        
        return filter_results