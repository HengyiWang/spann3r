# Datasets



## Training datasets

### ScanNet++

1. Download the [dataset](https://kaldir.vc.in.tum.de/scannetpp/)
2. Pre-process the dataset using [pre-process code](https://github.com/Nik-V9/scannetpp) in SplaTAM to generate undistorted DSLR depth.
3. Place it in `./data/scannetpp`

NOTE: Scannetpp is a great dataset for debugging and test purposes.



### ScanNet

1. Download the [dataset](http://www.scan-net.org/)
2. Extract and organize the dataset using [pre-process script](https://github.com/nianticlabs/simplerecon/tree/main/data_scripts/scannet_wrangling_scripts) in SimpleRecon
3. Place it in `./data/scannet`



### Habitat

1. Download the [dataset](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md)
2. Render 5-frame video as in [Croco](https://github.com/naver/croco/blob/master/datasets/habitat_sim/generate_multiview_images.py). You may want to read the [instructions](./croco/datasets/habitat_sim)
3. Place it in `./data/habitat_5frame`

NOTE: We render the 5-frame using aminimum covisiblity of [0.1](https://github.com/HengyiWang/spann3r/blob/8e0b8455484f8e3d480f60caed97f6fcf5a8d07d/croco/datasets/habitat_sim/generate_multiview_images.py#L155). This can improve the rendering speed, but the generated data may not be optimal for training Spann3R.



### ArkitScenes

1. Download the [dataset](https://github.com/apple/ARKitScenes/blob/9ec0b99c3cd55e29fc0724e1229e2e6c2909ab45/DATA.md)
2. Place it in `./data/arkit_lowres`

NOTE: Due to the limit of storage, we use low-resolution input to supervise Spann3R. Ideally, you can use a higher resolution i.e. `vga_wide`, as in DUSt3R, for training.



### Co3D

1. Download the [dataset](https://github.com/facebookresearch/co3d)
2. Pre-process dataset as in [DUSt3R](https://github.com/naver/dust3r/blob/main/datasets_preprocess/preprocess_co3d.py)
3. Place it in `./data/co3d_preprocessed_50`

NOTE: For Co3D, we use two sampling strategies to train our model, one is the same as in DUSt3R, another is our own sampling strategy as in other datasets that contain videos.



### BlendedMVS

1. Download the [dataset](https://github.com/YoYo000/BlendedMVS)
2. Place it in `./data/blendmvg`



## Evaluation datasets



### 7 Scenes

1. Download the [dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/). You may want to use [code](https://github.com/nianticlabs/simplerecon/blob/477aa5b32aa1b93f53abc72828f86023b6e46ce7/data_scripts/7scenes_preprocessing.py#L43) in SimpleRecon to download the data
2. Use [pre-process code](https://github.com/nianticlabs/simplerecon/blob/main/data_scripts/7scenes_preprocessing.py) in SimpleRecon to generate pseudo gt depth
3. Place it in `./data/7scenes`



### Neural RGBD

1. Download the [dataset](http://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip)
2. Place it in `./data/neural_rgbd`



### DTU

1. Download the [dataset](https://github.com/YoYo000/MVSNet?tab=readme-ov-file). Note that we [render the depth](./spann3r/tools/render_dtu.py) as in MVSNet and use our own mask annotations for evaluation. You can download our pre-processed DTU that contains the rendered depth map for evaluation [here](https://drive.google.com/drive/folders/1bqtcVf8lK4VC8LgG-SIGRBECcrFqM7Wy).
2. Place it in `./data/dtu_test`

