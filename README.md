看到这篇文章的朋友你好，如果你是HR，请看以下内容：

本项目基于pytorch-lightning 主要三个脚本：
LabelDistill/labeldistill/exps/nuscenes/labeldistill/LidarDistill_r50_128x128_e24_4keyfp.py（核心）

LabelDistill/labeldistill/exps/nuscenes/base_exp.py

LabelDistill/labeldistill/exps/base_cli.py

使用ROI改进蒸馏策略的核心代码在 如下路径LabelDistill/labeldistill/refine_head/target_assigner

因为idea是从kitti数据集上二阶段微调框的策略中获取的灵感，所以文件夹用来这个名字来作为纪念，整个项目独立完成，参考项目：

https://github.com/sanmin0312/LabelDistill/tree/master/labeldistill

https://github.com/Megvii-BaseDetection/BEVDepth/tree/main/bevdepth/exps/nuscenes







如果你是在写论文或者找寻跨模态蒸馏的灵感的朋友，请看以下内容：

本项目基于pytorch-lightning 主要三个脚本

LabelDistill/labeldistill/exps/nuscenes/labeldistill/LidarDistill_r50_128x128_e24_4keyfp.py（核心）

LabelDistill/labeldistill/exps/nuscenes/base_exp.py

LabelDistill/labeldistill/exps/base_cli.py

使用ROI改进蒸馏策略的核心代码在 如下路径LabelDistill/labeldistill/refine_head/target_assigner

设置backbone为res101的版本我也写好了，我想你可能最大的原因是不知道怎么在res101的情况维持稳定的训练

目前主流文章很少公开res101的backbone的情况，因为debug比较麻烦，我贡献的代码部分是可以直接使用的，你可以参考我可以分享下一些稳定训练的关键设置，首先：

backbone_conf = {
    'x_bound': [-51.2, 51.2, 0.8],
    'y_bound': [-51.2, 51.2, 0.8],
    'z_bound': [-5, 3, 8],
    'd_bound': [2.0, 58.0, 0.5],
    'final_dim': final_dim,
    'output_channels': 80,
    'downsample_factor': 16,
    'img_backbone_conf': dict(
        type='ResNet',
        depth=101,  # 修改：从 50 改为 101
        frozen_stages=0,
        out_indices=[0, 1, 2, 3],
        norm_eval=False,
        norm_cfg=dict(type='BN', requires_grad=True),  
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101') , 
        with_cp=True,
    ),
    
  with_cp=True 和 norm_eval=False,norm_cfg=dict(type='BN', requires_grad=True), 的设置是必须的

  然后就是训练的时候需要额外降低backbone的学习率来维持稳定，训练预热是必要的
，具体的参数设置可以参考LabelDistill/labeldistill/exps/nuscenes/labeldistill/LidarDistill_r101_128x128_e24_4keyfp.py

  最后就是学生网络bevdepth的损失，官方的bevdepth项目其实有一个问题，就是loss其实是有问题的，这个问题涉及到原理部分，
  并且实际训练中预测的depth分布会越来越尖锐，训练很不稳定，我采用了一个简要的做法，就是深度标签换成了软标签。
  
  最后还有一个我没有采用的改动，LabelDistill/labeldistill/layers/backbones/adaptor.py 这个适配器可能不是太好，只不过我
  没有修改适配器的情况下就完成了训练，达到了想要的结果，所以这个部分只留作考虑吧

  
