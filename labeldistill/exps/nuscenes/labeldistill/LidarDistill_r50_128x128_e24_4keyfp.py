#LabelDistill/labeldistill/exps/nuscenes/labeldistill/LidarDistill_r50_128x128_e24_4keyfp.py
from labeldistill.exps.base_cli import run_cli
from labeldistill.exps.nuscenes.base_exp import \
    LabelDistillModel as BaseLabelDistillModel
from labeldistill.models.lidardistill import LabelDistill
from torch.optim.lr_scheduler import MultiStepLR
from mmcv.runner import build_optimizer
from labeldistill.datasets.nusc_det_dataset_lidar import NuscDetDataset, collate_fn
from functools import partial
from mmdet3d.models import build_detector
import torch
import torch.nn.functional as F
import os
from labeldistill.layers.relate_loss.channel_inner_loss import TiGBEVDistillationLoss
from labeldistill.refine_head.target_assigner.roi_distill import ProposalTargetLayer
from labeldistill.refine_head.target_assigner.draw_heatmap_mask import BEVDistillationMaskGenerator
from labeldistill.refine_head.target_assigner.draw_heatmap_mask_label import BEVDistillationMaskGenerator_label
from labeldistill.layers.relate_loss.weight_channel_loss import RefinedTiGBEVDistillationLoss
from labeldistill.refine_head.target_assigner.roi_distill_fp import FalsePositiveRoiFilter
from labeldistill.refine_head.target_assigner.fp_distill import StudentSpecificFPFilter
from labeldistill.refine_head.target_assigner.draw_heatmap_mask_fp import StudentFPMaskGenerator
from labeldistill.refine_head.target_assigner.roi_distill_scale import AdaptiveGTScaler


class LabelDistillModel(BaseLabelDistillModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.key_idxes = [-2, -4, -6, -8]
       
        roi_sampler_cfg = {
    'NEAR_DISTANCE_THRESH': 30.0,  # 未匹配 GT 的近距离阈值(米)
    'FAR_DISTANCE_THRESH': 50.0    # 未匹配 GT 的远距离阈值(米)
}
                        
        fp_filter_cfg = {
    'FP_DISTANCE_THRESH': 4.0,  # 误检距离阈值(米)
    'HIGH_SCORE_THRESH': 0.15    # 高分误检分数阈值
}
        filter_cfg = {
  
    'TEACHER_FP_IOU_THRESH': 0.1,
    # 'MAX_FP_PER_BATCH': 100,  # 限制每个batch学生特有误检的最大数量 (如果代码中用到)
    # 'SCORE_THRESH': 0.1,      # 误检的最低分数阈值 (假设已在外部FalsePositiveRoiFilter中处理)
}

# 初始化
        self.student_specific_filter = StudentSpecificFPFilter(filter_cfg)
        self.generate_bev_mask = BEVDistillationMaskGenerator()
        self.generate_bev_mask_label = BEVDistillationMaskGenerator_label()
        self.proposal_target_layer = ProposalTargetLayer(roi_sampler_cfg=roi_sampler_cfg) 
        self.proposal_target_layer_fp = FalsePositiveRoiFilter(filter_cfg=fp_filter_cfg)
        self.change_gt = AdaptiveGTScaler()
        self.generate_bev_mask_fp = StudentFPMaskGenerator()
        self.fp_distill_start_epoch = 1  # 开始计算的epoch
        self.fp_distill_warmup_epochs = 2  # 权重从0增加到目标值的epoch数
        self.fp_distill_max_weight = 0.3  # 最终权重
        self.backbone_conf['output_channels'] = 150
        self.head_conf['bev_backbone_conf']['in_channels'] = 150 * (
            len(self.key_idxes) + 1)
        self.head_conf['bev_backbone_conf']['base_channels'] = 150 * 2
        self.head_conf['bev_neck_conf']['in_channels'] = [
            150 * (len(self.key_idxes) + 1), 150*2, 150*4, 150*8
        ]
        self.head_conf['train_cfg']['code_weights'] = [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]

        self.data_return_lidar = True
        
        #self.relate_loss = TiGBEVDistillationLoss()
        self.relate_loss = RefinedTiGBEVDistillationLoss()
        self.optimizer_config = dict(
            type='AdamW',
            lr=4e-4,
            paramwise_cfg=dict(
                custom_keys={
                    'backbone': dict(lr_mult=0.5),
                }),
            weight_decay=1e-2)

        point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        voxel_size = [0.1, 0.1, 0.2]

        bbox_coder = dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
            code_size=9)

        train_cfg = dict(
            pts=dict(
                grid_size=[1024, 1024, 40],
                voxel_size=voxel_size,
                out_size_factor=8,
                dense_reg=1,
                gaussian_overlap=0.1,
                max_objs=500,
                min_radius=2,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]))
        test_cfg = dict(
            pts=dict(
                post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_per_img=500,
                max_pool_nms=False,
                min_radius=[4, 12, 10, 1, 0.85, 0.175],
                score_threshold=0.15,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                nms_type='circle',
                pre_max_size=1000,
                post_max_size=83,
                nms_thr=0.2))

        self.lidar_conf = dict(type='CenterPoint',
            pts_voxel_layer=dict(
                point_cloud_range=point_cloud_range, max_num_points=10, voxel_size=voxel_size,
                max_voxels=(90000, 120000)),
            pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
            pts_middle_encoder=dict(
                type='SparseEncoder',
                in_channels=5,
                sparse_shape=[41, 1024, 1024],
                output_channels=128,
                order=('conv', 'norm', 'act'),
                encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
                encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
                block_type='basicblock'),
            pts_backbone=dict(
                type='SECOND',
                in_channels=256,
                out_channels=[128, 256],
                layer_nums=[5, 5],
                layer_strides=[1, 2],
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                conv_cfg=dict(type='Conv2d', bias=False)),
            pts_neck=dict(
                type='SECONDFPN',
                in_channels=[128, 256],
                out_channels=[256, 256],
                upsample_strides=[1, 2],
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                upsample_cfg=dict(type='deconv', bias=False),
                use_conv_for_no_stride=True),
             pts_bbox_head=dict(
                 type='CenterHead',
                 in_channels=sum([256, 256]),
                 tasks=[
                     dict(num_class=1, class_names=['car']),
                     dict(num_class=2, class_names=['truck', 'construction_vehicle']),
                     dict(num_class=2, class_names=['bus', 'trailer']),
                     dict(num_class=1, class_names=['barrier']),
                     dict(num_class=2, class_names=['motorcycle', 'bicycle']),
                     dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
                 ],
                 common_heads=dict(
                     reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
                 share_conv_channel=64,
                 bbox_coder=bbox_coder,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 separate_head=dict(
                     type='SeparateHead', init_bias=-2.19, final_kernel=3),
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
                 norm_bbox=True),
        )
        #############################################################################################
        "reproduced centerpoint"
        lidar_ckpt_path = './ckpts/centerpoint_vox01_128x128_20e_10sweeps.pth'

        "original centerpoint"
        # lidar_ckpt_path = './pretrained/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth'
        #############################################################################################
       

        self.model = LabelDistill(self.backbone_conf,
                                    self.head_conf,
                                 
                                    self.lidar_conf,
                                    
                                    lidar_ckpt_path,
                                    is_train_depth=True)
                                     # build lidar detection model
        self.centerpoint = build_detector(self.lidar_conf)

        # load pretrained parameters for lidar detection model
        lidar_params = torch.load(lidar_ckpt_path, map_location='cpu')

        prefix = 'model.centerpoint.'
        load_keys = [k for k in lidar_params['state_dict'] if k.startswith(prefix)]
        self.centerpoint.load_state_dict({k[len(prefix):]: lidar_params['state_dict'][k] for k in load_keys})
        self.centerpoint.eval()
        for param in self.centerpoint.parameters():
            param.requires_grad = False

    def training_step(self, batch):
        (sweep_imgs, mats, _, _, gt_boxes, gt_labels, lidar_pts, depth_labels) = batch
       
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
            gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
            gt_labels = [gt_label.cuda() for gt_label in gt_labels]
            self.model = self.model.cuda()

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            bev_mask, bev_box, bev_label, targets = self.model.module.get_targets(gt_boxes, gt_labels)
        else:
            bev_mask, bev_box, bev_label, targets = self.model.get_targets(gt_boxes, gt_labels)

        preds, lidar_preds, depth_preds, distill_feats_lidar, lidar_feats, neck_feats,neck_output,lidar_pred_box,image_pred_box= self.model(bev_mask,
                                                                                                                         bev_box,
                                                                                                                         bev_label,
                                                                                                                         sweep_imgs,
                                                                                                                         mats,
                                                                                                                         lidar_pts)   
        
      
       

        batch_dict_pre = self.prepare_batch_dict_from_pred_box(lidar_pred_box, gt_boxes,gt_labels)
        batch_dict_pre_image = self.prepare_batch_dict_from_pred_box(image_pred_box, gt_boxes,gt_labels)
        batch_result = self.proposal_target_layer(batch_dict_pre)
        batch_result_fp = self.proposal_target_layer_fp(batch_dict_pre_image)
        filtered_results = self.student_specific_filter(
     batch_result_fp,  # 学生的误检roi
    batch_result  # 教师的误检roi
)       
        B = len(gt_boxes)
        batch_result = self.change_gt.forward(batch_result)
        bev_roi_mask = self.generate_bev_mask(batch_result,B)
        student_feat_tensor = neck_output[0]
        teacher_feat_tensor = neck_feats[0] 
        
        
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            detection_loss, response_loss = self.model.module.response_loss(targets, preds, lidar_preds)
        else:
            detection_loss, response_loss = self.model.response_loss(targets, preds, lidar_preds)
       
        if len(depth_labels.shape) == 5:
            # only key-frame will calculate depth loss
            depth_labels = depth_labels[:, 0, ...]
        depth_loss = self.get_depth_loss(depth_labels.cuda(), depth_preds)
        lidar_distill_loss = self.get_feature_distill_loss1(lidar_feats, distill_feats_lidar, bev_roi_mask, binary_mask=False)*0.6
        if self.current_epoch < self.fp_distill_start_epoch:
            fp_weight = 0.0
        else:
            fp_weight = self.fp_distill_max_weight

        # 计算fp_distill_loss
        if fp_weight > 0:
            bev_roi_mask_fp =self.generate_bev_mask_fp(filtered_results,4)
            fp_distill_loss = self.get_feature_distill_loss1(
                lidar_feats, distill_feats_lidar, bev_roi_mask_fp, binary_mask=False
            ) * fp_weight
        else:
            fp_distill_loss = torch.tensor(0.0, device=lidar_feats[0].device)
        self.log('fp_distill_loss',fp_distill_loss)
        
        self.log('detection_loss', detection_loss)
        self.log('response_loss', response_loss)
        self.log('depth_loss', depth_loss)
        self.log('lidar_distill_loss', lidar_distill_loss)
      

        return detection_loss + depth_loss + lidar_distill_loss  + response_loss+fp_distill_loss

    def get_feature_distill_loss(self, lidar_feat, distill_feats, bev_mask=None, binary_mask=False):

        label_losses = 0

        if bev_mask is not None:
            bev_mask = torch.cat(bev_mask, dim=1).sum(1).unsqueeze(1)
            B, _, W, H = bev_mask.shape

            bev_mask = [bev_mask,
                        F.interpolate(bev_mask.type(torch.float32), size=(W//2, H//2), mode='bilinear', align_corners=True)]

            if binary_mask:
                bev_mask[0][bev_mask[0] > 0] = 1.0
                bev_mask[1][bev_mask[1] > 0] = 1.0

        for i in range(len(lidar_feat)):
            label_loss = F.mse_loss(
                lidar_feat[i],
                distill_feats[i],
                reduction='none',
            )

            if bev_mask is not None:
                label_loss = ((label_loss.sum(1) * bev_mask[i].squeeze()).sum()) / max(1.0, bev_mask[i].sum())
            else:
                B, C, W, H = label_loss.shape
                label_loss = label_loss.sum() / (B*W*H)
            label_losses += label_loss

        return label_losses
    
    def get_feature_distill_loss1(self, lidar_feat, distill_feats, bev_mask=None, binary_mask=False):

        label_losses = 0

        if bev_mask is not None:
            B, _, W, H = bev_mask.shape

            bev_mask = [bev_mask,
                        F.interpolate(bev_mask.type(torch.float32), size=(W//2, H//2), mode='bilinear', align_corners=True)]

            if binary_mask:
                bev_mask[0][bev_mask[0] > 0] = 1.0
                bev_mask[1][bev_mask[1] > 0] = 1.0

        for i in range(len(lidar_feat)):
            label_loss = F.mse_loss(
                lidar_feat[i],
                distill_feats[i],
                reduction='none',
            )

            if bev_mask is not None:
                label_loss = ((label_loss.sum(1) * bev_mask[i].squeeze()).sum()) / max(1.0, bev_mask[i].sum())
            else:
                B, C, W, H = label_loss.shape
                label_loss = label_loss.sum() / (B*W*H)
            label_losses += label_loss

        return label_losses
    
   

    def prepare_batch_dict_from_pred_box(self, pred_box, gt_boxes, gt_labels, max_rois=256, max_gts=128):
        """
        将预测框转换为 ProposalTargetLayer 需要的格式
        Args:
            pred_box: List[List[Tensor, Tensor, Tensor]] - 预测结果
                每个batch: [boxes (N, 9), scores (N,), labels (N,)]
            gt_boxes: List[Tensor] - GT框列表,每个元素 shape 为 (num_gts, 9)
            gt_labels: List[Tensor] - GT标签列表,每个元素 shape 为 (num_gts,)
            max_rois: int - 最大ROI数量
            max_gts: int - 最大GT数量
        Returns:
            batch_dict: dict - 包含以下键值:
                - batch_size: int
                - rois: Tensor (B, max_rois, 9)
                - roi_scores: Tensor (B, max_rois)
                - roi_labels: Tensor (B, max_rois)
                - gt_boxes_and_cls: Tensor (B, max_gts, 10) - 前9列为boxes,第10列为类别
        """

        batch_size = len(pred_box)

        # 获取device (从第一个有效的tensor中获取)
        device = None
        for batch_item in pred_box:
            if isinstance(batch_item, list) and len(batch_item) > 0:
                if isinstance(batch_item[0], torch.Tensor):
                    device = batch_item[0].device
                    break
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化输出张量 (用0填充,作为padding)
        rois_padded = torch.zeros((batch_size, max_rois, 9), device=device, dtype=torch.float32)
        roi_scores_padded = torch.zeros((batch_size, max_rois), device=device, dtype=torch.float32)
        roi_labels_padded = torch.zeros((batch_size, max_rois), device=device, dtype=torch.long)
        gt_boxes_padded = torch.zeros((batch_size, max_gts, 10), device=device, dtype=torch.float32)

        # 填充每个batch的数据
        for i in range(batch_size):
            # === 处理 pred_box (ROI boxes, scores, labels) ===
            if not isinstance(pred_box[i], list) or len(pred_box[i]) != 3:
                print(f"[WARNING] pred_box[{i}] format unexpected: {type(pred_box[i])}")
                continue

            cur_rois, cur_scores, cur_labels = pred_box[i]

            # 转换数据类型
            cur_rois = cur_rois.float()
            cur_scores = cur_scores.float()
            cur_labels = cur_labels.long()

            num_rois = min(cur_rois.shape[0], max_rois)

            # 填充ROI数据
            if num_rois > 0:
                rois_padded[i, :num_rois] = cur_rois[:num_rois]
                roi_scores_padded[i, :num_rois] = cur_scores[:num_rois]
                roi_labels_padded[i, :num_rois] = cur_labels[:num_rois] 

            # === 处理 GT boxes 和 labels ===
            cur_gts = gt_boxes[i].float()  # (num_gts, 9)
            cur_gt_labels = gt_labels[i].long()  # (num_gts,)

            num_gts = min(cur_gts.shape[0], max_gts)

            # 填充GT数据
            if num_gts > 0:
                # 前9列为box坐标
                gt_boxes_padded[i, :num_gts, :9] = cur_gts[:num_gts]
                gt_boxes_padded[i, :num_gts, 9] = cur_gt_labels[:num_gts] 

        # 构建 batch_dict
        batch_dict = {
            'batch_size': batch_size,
            'rois': rois_padded,  # (B, max_rois, 9)
            'roi_scores': roi_scores_padded,  # (B, max_rois)
            'roi_labels': roi_labels_padded,  # (B, max_rois)
            'gt_boxes_and_cls': gt_boxes_padded,  # (B, max_gts, 10)
        }

        return batch_dict
    

    def eval_step(self, batch, batch_idx, prefix: str):
        (sweep_imgs, mats, _, img_metas, _, _) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
        preds = self.model(x=sweep_imgs, mats_dict=mats)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            results = self.model.module.get_bboxes(preds, img_metas)
        else:
            results = self.model.get_bboxes(preds, img_metas)
        for i in range(len(results)):
            results[i][0] = results[i][0].detach().cpu().numpy()
            results[i][1] = results[i][1].detach().cpu().numpy()
            results[i][2] = results[i][2].detach().cpu().numpy()
            results[i].append(img_metas[i])
        return results

    def train_dataloader(self):
        # 根据use_train_val参数决定使用哪些数据
        if self.use_train_val:
            # 使用train+val数据集
            info_paths = [self.train_info_paths, self.val_info_paths]
        else:
            # 仅使用train数据集
            info_paths = self.train_info_paths
            
        train_dataset = NuscDetDataset(ida_aug_conf=self.ida_aug_conf,
                                       bda_aug_conf=self.bda_aug_conf,
                                       classes=self.class_names,
                                       data_root=self.data_root,
                                       info_paths=info_paths,
                                       is_train=True,
                                       use_cbgs=self.data_use_cbgs,
                                       img_conf=self.img_conf,
                                       num_sweeps=self.num_sweeps,
                                       sweep_idxes=self.sweep_idxes,
                                       key_idxes=self.key_idxes,
                                       return_depth=self.data_return_depth,
                                       return_lidar=self.data_return_lidar,
                                       use_fusion=self.use_fusion)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=4,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(collate_fn,
                               is_return_depth=self.data_return_depth
                               or self.use_fusion,
                               is_return_lidar=self.data_return_lidar),
            sampler=None,
        )
        return train_loader

    def configure_optimizers(self):
        optimizer = build_optimizer(self.model, self.optimizer_config)
        scheduler = MultiStepLR(optimizer, [19, 21])
        return [[optimizer], [scheduler]]


if __name__ == '__main__':
    run_cli(LabelDistillModel,
            'LidarDistill_r50_128x128_e24_4keyfp',
            extra_trainer_config_args={'epochs': 24},
            use_ema=True)