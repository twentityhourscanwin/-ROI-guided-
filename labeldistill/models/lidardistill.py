# labeldistill/models/lidardistill.py
from torch import nn
import matplotlib.pyplot as plt
from labeldistill.layers.backbones.base_lss_fpn import BaseLSSFPN
from labeldistill.layers.backbones.adaptor import DistillAdaptor
from labeldistill.layers.backbones.label_backbone import LabelBackbone
from labeldistill.models.labelencoder import LabelEncoder
from labeldistill.layers.heads.kd_head import KDHead
from mmdet3d.models import build_detector
import torch

__all__ = ['LabelDistill']

def print_memory_usage(stage_name, device_id=None):
    if not torch.cuda.is_available():
        print(f"[{stage_name} Memory] CUDA not available.")
        return

    if device_id is None:
        device_id = torch.cuda.current_device()
        
    # 获取当前已分配内存 (allocated)
    allocated = torch.cuda.memory_allocated(device_id) / 1024**3
    # 获取当前缓存/预留内存 (reserved)
    reserved = torch.cuda.memory_reserved(device_id) / 1024**3
    print(f"[{stage_name} Memory] Allocated: {allocated:.2f} GiB, Reserved: {reserved:.2f} GiB")
# --- 内存打印辅助函数结束 ---

"""
LiDAR Distillation (backbone to backbone)
"""

class LabelDistill(nn.Module):
    
    def __init__(self,
                 backbone_conf,
                 head_conf,
                 lidar_conf=None,
                 lidar_ckpt_path=None,
                 is_train_depth=False,
                 ):
        super(LabelDistill, self).__init__()
        self.backbone = BaseLSSFPN(**backbone_conf)
        self.head = KDHead(**head_conf)
        self.is_train_depth = is_train_depth

        distill_in_feature = head_conf['bev_neck_conf']['in_channels'][:2]
        self.distill_encoder_lidar = DistillAdaptor([x // 2 for x in distill_in_feature],
                                                    out_features=[128, 256],
                                                    stride=[1, 1]
                                                    )
        
        # build lidar detection model
        self.centerpoint = build_detector(lidar_conf)

        # load pretrained parameters for lidar detection model
        lidar_params = torch.load(lidar_ckpt_path, map_location='cpu')

        prefix = 'model.centerpoint.'
        load_keys = [k for k in lidar_params['state_dict'] if k.startswith(prefix)]
        self.centerpoint.load_state_dict({k[len(prefix):]: lidar_params['state_dict'][k] for k in load_keys})
        self.centerpoint.eval()


    def forward(
        self,
        bev_mask=None,
        bev_box=None,
        bev_label=None,
        x=None,
        mats_dict=None,
        lidar_pts=None,
        timestamps=None,
    ):
        
        if self.is_train_depth and self.training:
            lidar_preds = None
            lidar_pred_box = None
            lidar_feats_out = None
            neck_feats = None
            
            
            with torch.no_grad():
                voxels, num_points, coors = self.centerpoint.voxelize(lidar_pts.squeeze(1))
                
                voxel_features = self.centerpoint.pts_voxel_encoder(voxels, num_points, coors)
                batch_size = coors[-1, 0] + 1
                lidar_feats = self.centerpoint.pts_middle_encoder(voxel_features, coors, batch_size)

                lidar_feats = self.centerpoint.pts_backbone(lidar_feats)
                lidar_feats_out = lidar_feats
                # '''
                # lidar_feats_out 的类型：<class 'tuple'>
                # lidar_feats_out 的长度：2
                # 元素 0 的类型：<class 'torch.Tensor'>
                # 元素 0 的形状：torch.Size([4, 128, 128, 128])
                # 元素 1 的类型：<class 'torch.Tensor'>
                # 元素 1 的形状：torch.Size([4, 256, 64, 64])
                # '''
                neck_feats = self.centerpoint.pts_neck(lidar_feats)
                # '''
                # lidar_feats 的类型：<class 'tuple'>
                # lidar_feats 的长度：2
                # 元素 0 的类型：<class 'torch.Tensor'>
                # 元素 0 的形状：torch.Size([4, 128, 128, 128])
                # 元素 1 的类型：<class 'torch.Tensor'>
                # 元素 1 的形状：torch.Size([4, 256, 64, 64])
                # '''
                
                lidar_preds = self.centerpoint.pts_bbox_head(neck_feats)
                lidar_pred_box = self.get_bboxes(lidar_preds)
          
         
            x, depth_pred = self.backbone(x,
                                          mats_dict,
                                          timestamps,
                                          is_return_depth=True)
            
           
            preds, backbone_out, neck_output = self.head(x) #backbone output list of [B, C, W, H]
            image_pred_box = self.get_bboxes(preds)
            
            # '''
            # backone_outs
            # [0] Shape: torch.Size([4, 450, 128, 128])
            # [1] Shape: torch.Size([4, 300, 64, 64])
            # [2] Shape: torch.Size([4, 600, 32, 32])
            # '''
            c1 = backbone_out[0].shape[1] // 2
            c2 = backbone_out[1].shape[1] // 2

            #adaptor
            distill_feats_lidar = self.distill_encoder_lidar([backbone_out[0][:, :c1],
                                                              backbone_out[1][:, :c2]])
                
            # '''
            # 经过适配器之后的用于作用lidar的特征：
            # 特征 0: torch.Size([4, 128, 128, 128])
            # 特征 1: torch.Size([4, 256, 64, 64])
            # '''
            
            return preds, lidar_preds, depth_pred, distill_feats_lidar, lidar_feats_out, neck_feats, neck_output, lidar_pred_box, image_pred_box
        else:
            x = self.backbone(x, mats_dict, timestamps)
            preds, _, x = self.head(x)
            return preds
        

    def get_targets(self, gt_boxes, gt_labels):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        return self.head.get_targets(gt_boxes, gt_labels)

    def loss(self, targets, preds_dicts):
        """Loss function for BEVDepth.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        return self.head.loss(targets, preds_dicts)

    def response_loss(self, targets, preds_dicts, teacher_dicts):
        """Loss function for BEVDepth.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        return self.head.response_loss(targets, preds_dicts, teacher_dicts)
    
    def distill_loss(self, targets, preds_dicts, teacher_dicts):

        return self.head.distill_loss(targets, preds_dicts, teacher_dicts)


    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)