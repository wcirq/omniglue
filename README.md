# 图像配置 

- 源于 [omniglue](https://github.com/google-research/omniglue)

# 安装

```
mkdir models
cd models

# SuperPoint.
git clone https://github.com/rpautrat/SuperPoint.git
mv SuperPoint/pretrained_models/sp_v6.tgz . && rm -rf SuperPoint
tar zxvf sp_v6.tgz && rm sp_v6.tgz

# DINOv2 - vit-b14.
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth

# OmniGlue.
wget https://storage.googleapis.com/omniglue/og_export.zip
unzip og_export.zip && rm og_export.zip
```

# models 目录结构

-- [og_export](models%2Fog_export)

-- [sp_v6](models%2Fsp_v6)

-- [dinov2_vitb14_pretrain.pth](models%2Fdinov2_vitb14_pretrain.pth)