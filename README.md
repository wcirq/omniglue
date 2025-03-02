# 图像配准

## 源于 
- [omniglue](https://github.com/google-research/omniglue) 
- [SuperPointPretrainedNetwork](https://github.com/magicleap/SuperPointPretrainedNetwork)

## 安装

```
mkdir models
cd models

# 1.1.SuperPoint. (本仓库已包括，可不用重复下载，请跳到下一步)
git clone https://github.com/rpautrat/SuperPoint.git
mv SuperPoint/pretrained_models/sp_v6.tgz . && rm -rf SuperPoint
tar zxvf sp_v6.tgz && rm sp_v6.tgz

# or 1.2 SuperPoint torch版
from https://github.com/magicleap/SuperPointPretrainedNetwork.git

# 2.DINOv2 - vit-b14.
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth

# 3.OmniGlue.
wget https://storage.googleapis.com/omniglue/og_export.zip
unzip og_export.zip && rm og_export.zip
```

## models 目录结构

    |-- models
        |-- og_export
            |-- assets
            |-- variables
                |-- variables.data-00000-of-00001
                |-- variables.index
            | -- fingerprint.pb
            | -- saved_model.pb
        |-- sp_torch
            |-- superpoint_v1.pth
        |-- sp_v6
            |-- variables
                |-- variables.data-00000-of-00001
                |-- variables.index
            | -- saved_model.pb
        |-- dinov2_vitb14_pretrain.pth

## Demo
```shell
# 使用usb摄像头实时特征点跟踪，测试torch版 SuperPoint
python demo_superpoint_extract_torch.py camera --camid=0 --show_extra  

# 使用图片测试图像配准，测试omniglue，注意修改代码中的图片路径为自己的
python demo.py
```