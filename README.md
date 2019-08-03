# Pixel2Mesh

This is an implementation of Pixel2Mesh in PyTorch. Besides, we also:

- Provide retrained Pixel2Mesh checkpoints. Besides, the pretrained tensorflow pretrained model provided in [official implementation](https://github.com/nywang16/Pixel2Mesh) is also converted into a PyTorch checkpoint file for convenience.
- Provide a modified version of Pixel2Mesh whose backbone is ResNet instead of VGG.
- Clarify some details in previous implementation and provide a flexible training framework.

## Get Started

### Environment

Current version only supports training and inference on GPU. It works well under dependencies as follows:

- Ubuntu 16.04 / 18.04
- Python 3.7
- PyTorch 1.1
- CUDA 9.0 (10.0 should also work)
- OpenCV 4.1
- Scipy 1.3
- Scikit-Image 0.15

Some minor dependencies are also needed, for which the latest version provided by conda/pip works well:

> easydict, pyyaml, tensorboardx, trimesh, shapely

Two another steps to prepare the codebase:

1. `git submodule update --init` to get [Neural Renderer](https://github.com/daniilidis-group/neural_renderer) ready.
2. `python setup.py install` in directory [external/chamfer](external/chamfer) and `external/neural_renderer` to compile the modules.

### Configuration

You should specify your configuration in a `yml` file, which can override default settings in [options.py](options.py). We provide some examples in the [experiments](experiments) directory. If you just want to look around, you don't have to change everything. Options provided in [experiments/default](experiments/default) are everything you need.

### Datasets

We use [ShapeNet](https://www.shapenet.org/) for model training and evaluation. The official tensorflow implementation provides a subset of ShapeNet for it, you can download it [here](https://drive.google.com/drive/folders/131dH36qXCabym1JjSmEpSQZg4dmZVQid). Extract it and link it to `data_tf` directory as follows. Before that, some meta files [here](https://drive.google.com/file/d/16d9druvCpsjKWsxHmsTD5HSOWiCWtDzo/view?usp=sharing) will help you establish the folder tree, demonstrated as follows.

**P.S.** In case more data is needed, another larger data package of ShapeNet is also [available](https://drive.google.com/file/d/1Z8gt4HdPujBNFABYrthhau9VZW10WWYe/view). You can extract it and place it in the `data` directory. But this would take much time and needs about 300GB storage.

```
datasets/data
├── ellipsoid
│   ├── face1.obj
│   ├── face2.obj
│   ├── face3.obj
│   └── info_ellipsoid.dat
├── pretrained
│   ... (.pth files)
└── shapenet
    ├── data (larger data package, optional)
    │   ├── 02691156
    │   │   └── 3a123ae34379ea6871a70be9f12ce8b0_02.dat
    │   ├── 02828884
    │   └── ...
    ├── data_tf (standard data used in official implementation)
    │   ├── 02691156 (put the folders directly in data_tf)
    │   │   └── 10115655850468db78d106ce0a280f87
    │   ├── 02828884
    │   └── ...
    └── meta
        ...
```

Difference between the two versions of dataset is worth some explanation:

- `data_tf` has images of 137x137 resolution and four channels (RGB + alpha), 175,132 samples for training and 43,783 for evaluation.
- `data` has RGB images of 224x224 resolution with background set all white. It contains altogether 1,050,240 for training and evaluation.

We trained model with both datasets and evaluated on both benchmarks. To save time and align our results with the official paper/implementation, we use `data_tf` by default.

### Train your own model

```
python entrypoint_train.py --name xxx --options path_to_yaml
```

**P.S.** To train on slurm clusters, we also provide settings reference. Refer to [slurm](slurm) folder for details.

### Evaluation

Take evaluation on our checkpoint for example:

```
python entrypoint_eval.py --name resnet_eval --options experiments/default/resnet.yml --checkpoint checkpoints/resnet.pth.tar --gpus 1 --shuffle
```

## Results

We provide results from the implementation tested by us here.

First, the [official tensorflow implementation](https://github.com/nywang16/Pixel2Mesh) reports much higher performance than claimed in the [original paper](https://arxiv.org/abs/1804.01654). The results are listed as follows, which is close to that reported in [MeshRCNN](https://arxiv.org/abs/1906.02739).

| Category        | # of samples | F1$^{\tau}$ | F1$^{2\tau}$ | CD        | EMD       |
| --------------- | ------------ | ----------- | ------------ | --------- | --------- |
| firearm         | 2372         | 77.24       | 85.85        | 0.382     | 2.671     |
| cellphone       | 1052         | 74.63       | 86.15        | 0.342     | 1.500     |
| speaker         | 1618         | 54.11       | 70.77        | 0.633     | 2.318     |
| cabinet         | 1572         | 66.50       | 81.85        | 0.331     | 1.615     |
| lamp            | 2318         | 56.93       | 69.27        | 1.033     | 3.765     |
| bench           | 1816         | 65.57       | 78.76        | 0.474     | 2.395     |
| couch           | 3173         | 56.49       | 74.44        | 0.441     | 2.073     |
| chair           | 6778         | 59.57       | 74.80        | 0.507     | 2.808     |
| plane           | 4045         | 76.35       | 85.02        | 0.372     | 2.243     |
| table           | 8509         | 71.44       | 83.38        | 0.385     | 2.021     |
| monitor         | 1095         | 58.02       | 73.08        | 0.569     | 2.127     |
| car             | 7496         | 70.59       | 86.43        | 0.242     | 3.335     |
| watercraft      | 1939         | 60.39       | 74.56        | 0.558     | 2.558     |
| *Mean*          |              | **65.22**   | **78.80**    | **0.482** | **2.418** |
| *Weighted-mean* |              | **66.56**   | **80.17**    | **0.439** | **2.545** |

The original paper evaluates based on simple mean, without considerations of different categories containing different number of samples, while some later papers use weighted-mean to calculate final performance. We report results under both two metrics for caution.

### Pretrained checkpoints

- **Migrated:** We provide scripts to migrate tensorflow checkpoints into PyTorch `.pth` files in [utils/migrations](utils/migrations). The checkpoint converted from official pretrained model can be downloaded [here](https://drive.google.com/file/d/1Gk3M4KQekEenG9qQm60OFsxNar0sG8bN/view?usp=sharing). We find that there is a performance drop (79.51 vs. 80.17), although we tried to align the ops of these two as close as possible.
- **VGG backbone:** We also trained a model with almost identical settings, using VGG as backbone, with subtle different choices of camera intrinsics among other settings, but the training is still running (will be added once completed).
- **ResNet backbone:** As we provide another backbone choice of resenet, we also provide a corresponding checkpoint [here](https://drive.google.com/file/d/1pZm_IIWDUDje6gRZHW-GDhx5FCDM2Qg_/view?usp=sharing). The training takes about 5 days on eight 1080 Ti GPUs. Refer to [yml](experiments/default/resnet.yml) for the settings of this training.

The performances of all these checkpoints, as compared to the official result, are listed below:

<table>
  <thead>
    <tr>
      <th>Checkpoint</th>
      <th>Eval Protocol
      <th>CD</th>
      <th>F1$^{\tau}$</th>
      <th>F1$^{2\tau}$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan=2>Official Result</td>
      <td>Mean</td>
      <td>0.482</td>
      <td>65.22</td>
      <td>78.80</td>
    </tr>
    <tr>
      <td>Weighted-mean</td>
      <td>0.439</td>
      <td><b>66.56</b></td>
      <td><b>80.17</b></td>
    </tr>
    <tr>
      <td rowspan=2>Migrated Checkpoint</td>
      <td>Mean</td>
      <td>0.498</td>
      <td>64.21</td>
      <td>78.03</td>
    </tr>
    <tr>
      <td>Weighted-mean</td>
      <td>0.451</td>
      <td>65.67</td>
      <td>79.51</td>
    </tr>
    <tr>
      <td rowspan=2>ResNet</td>
      <td>Mean</td>
      <td><b>0.443</b></td>
      <td><b>65.36</b></td>
      <td><b>79.24</b></td>
    </tr>
    <tr>
      <td>Weighted-mean</td>
      <td><b>0.411</b></td>
      <td>66.13</td>
      <td>80.13</td>
    </tr>
  </tbody>
</table>

## Details of Improvement

We explain some improvement of this version of implementation compared with the official version here.

- **Larger batch size:** We support larger batch size on multiple GPUs for training. Since Chamfer distances cannot be calculated if samples in a batch with different ground-truth pointcloud, "resizing" the pointcloud is necessary. Instead of resampling points, we simply upsample/downsample from the dataset.
- **Better backbone:** We enable replacing VGG by ResNet50 for model backbone. The training progress is more stable and final performance is higher.
- **More stable training:** We do normalization on the deformed sphere, so that it's deformed at location $(0,0,0)$; we use a threshold activation on $z$-axis during projection, so that $z$ will always be positive or negative and never be $0$. These seem not to result in better performance but more stable training loss.

You may notice another differences on choices of hyper-parameters if you look into the configuration. Most of these changes are unintentional and definitely not a result of careful tuning.

## Demo

We provide demos generated with images in [datasets/examples](datasets/examples) on our ResNet checkpoint. Here are some samples:

![](datasets/examples/airplane.gif)

![](datasets/examples/lamp.gif)

![](datasets/examples/table.gif)

![](datasets/examples/display.gif)

You can do inference on your own image folder by running

```
python --name predict --options /path/to/yml --checkpoint /path/to/checkpoint --folder /path/to/your/image/folder
```

## Known Issues

- Currently, CPU inference is not supported. CUDA is required for training, evaluation and prediction.
- We tried to pretrain the original mini-VGG (fewer channels than standard VGG) on ImageNet, and we release our pretrained results [here](https://drive.google.com/file/d/1kODNfwPBPQIYPQTki4ev5FyXK_UGfL-w/view?usp=sharing). However, using VGG with pretrained weights would backfire, resulting in loss turning **NaN**, for reasons we are not sure so far.

## Acknowledgements

Our work is based on the official version of [Pixel2Mesh](https://github.com/nywang16/Pixel2Mesh); Some part of code are borrowed from [a previous PyTorch implementation of Pixel2Mesh](https://github.com/Tong-ZHAO/Pixel2Mesh-Pytorch), even though this version seems incomplete. The packed files for two version of datasets are also provided by them two. Most codework is done by [Yuge Zhang](https://github.com/ultmaster).
