# Pixel2Mesh

This is an implementation of Pixel2Mesh in PyTorch. Besides, we also:

* Provide retrained Pixel2Mesh checkpoints. Besides, the pretrained tensorflow pretrained model provided in [official implementation](https://github.com/nywang16/Pixel2Mesh) is also converted into a PyTorch checkpoint file for convenience. 
* Provide a modified version of Pixel2Mesh whose backbone is ResNet instead of VGG.
* Clarify some details in previous implementation and provide a flexible training framework.

## Get Started

### Environment

Current version only supports training and inference on GPU. It works well under dependencies as follows:

-   PyTorch 1.1
-   CUDA 9.0 (10.0 should also work)
-   OpenCV 4.1
-   Scipy 1.3
-   Scikit-Image 0.15

Some minor dependencies are also needed, for which the latest version provided by conda/pip works well:

> easydict, pyyaml, tensorboardx, trimesh, shapely

Two another steps to prepare the codebase:

1.  `git submodule update --init`  to get  [Neural Renderer](https://github.com/daniilidis-group/neural_renderer)  ready.
2.  `python setup.py install`  in directory  `external/chamfer`  and  `external/neural_renderer`  to compile the modules.

### Configuration

You should specify your configuration in a `yml` file, which can override default settings in `options.py`. We provide some examples in the `experiment` directory. If you just want to look around, you don't have to change everything. Options provided in `experiments/default` are everything you need.

### Datasets

We use [ShapeNet](https://www.shapenet.org/) for model training and evaluation. The official tensorflow implementation provides a subset of ShapeNet for it, you can download it [here](https://drive.google.com/drive/folders/131dH36qXCabym1JjSmEpSQZg4dmZVQid). Extract it and link it to `data_tf` directory as follows. Before that, some meta files [here](xxx) will help you establish the folder tree, demonstrated as follows.

**P.S. ** In case more data is needed, another larger data package of ShapeNet is also [available](https://drive.google.com/file/d/1Z8gt4HdPujBNFABYrthhau9VZW10WWYe/view). You can extract it and place it in the `data` directory. But this would take much time and needs about 300GB storage.

```text
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

* `data_tf` has images of 137x137 resolution and four channels (RGB + alpha), 175,132 samples for training and 43,783 for evaluation. 
* `data` has RGB images of 224x224 resolution with background set all white. It divides xxx for training and xxx for evaluation.

We trained model with both datasets and evaluated on both benchmarks. To save time and align our results with the official paper/implementation, we use `data_tf` by default.

### Train your own model

```shell
python entrypoint_train.py --name xxx --options path_to_yaml
```

**P.S.** To train on slurm clusters, we also provide settings reference. Refer to `slurm` folder for details.

### Evaluation

```shell
python entrypoint_eval.py --options path_to_yaml --checkpoint path_to_pth
```

## Results

We provide results from the implementation tested by us here.

First, the [official tensorflow implementation](https://github.com/nywang16/Pixel2Mesh) reports much higher performance than claimed in the [original paper](https://arxiv.org/abs/1804.01654). The results are listed as follows, which is close to that reported in [MeshRCNN](https://arxiv.org/abs/1906.02739).

| Category      | # of samples | F1$^{\tau}$ | F1$^{2\tau}$ | CD    | EMD   |
|---------------|--------------|---------|---------|-------|-------|
| firearm       | 2372         | 77.24   | 85.85   | 0.382 | 2.671 |
| cellphone     | 1052         | 74.63   | 86.15   | 0.342 | 1.500 |
| speaker       | 1618         | 54.11   | 70.77   | 0.633 | 2.318 |
| cabinet       | 1572         | 66.50   | 81.85   | 0.331 | 1.615 |
| lamp          | 2318         | 56.93   | 69.27   | 1.033 | 3.765 |
| bench         | 1816         | 65.57   | 78.76   | 0.474 | 2.395 |
| couch         | 3173         | 56.49   | 74.44   | 0.441 | 2.073 |
| chair         | 6778         | 59.57   | 74.80   | 0.507 | 2.808 |
| plane         | 4045         | 76.35   | 85.02   | 0.372 | 2.243 |
| table         | 8509         | 71.44   | 83.38   | 0.385 | 2.021 |
| monitor       | 1095         | 58.02   | 73.08   | 0.569 | 2.127 |
| car           | 7496         | 70.59   | 86.43   | 0.242 | 3.335 |
| watercraft    | 1939         | 60.39   | 74.56   | 0.558 | 2.558 |
| *Mean*        |         | **65.22** | **78.80** | **0.482** | **2.418** |
| *Weighted-mean* |              | **66.56** | **80.17** | **0.439** | **2.545** |

The original paper evaluates based on simple mean, without considerations of different categories containing different number of samples, while some later papers use weighted-mean to calculate final performance. We report results under both two metrics for caution.

### Pretrained checkpoints

* **Migrated:** We provide scripts to migrate tensorflow checkpoints into PyTorch `.pth` files in `utils/migration`. The checkpoint converted from official pretrained model can be downloaded [here](...).
* **VGG backbone:** We also trained a model with almost identical settings, using VGG as backbone, with subtle different choices of camera intrinsics among [other settings](...), but the training is still running (will be added once completed).
* **ResNet backbone:** As we provide another backbone choice of resenet, we also provide a corresponding checkpoint [here](). 

The performances of all these checkpoints are listed in the following table:

to be added

## Details of improvement

We explain some improvement of this version of implementation compared with the official version here.

* **Larger batch size:** We support larger batch size on multiple GPUs for training. Since Chamfer distances cannot be calculated if samples in a batch with different ground-truth pointcloud, "resizing" the pointcloud is necessary. Instead of resampling points, we simply upsample/downsample from the dataset.
* **Better backbone:** We enable replacing VGG by ResNet50 for model backbone. The training progress is more stable and final performance is higher. 
* **More stable training:** We do normalization on the deformed sphere, so that it's deformed at location $(0,0,0)$; we use a threshold activation on $z$-axis during projection, so that $z$ will always be positive or negative and never be $0$. These seem not to result in better performance but more stable training loss.

## Demo

For a quick look around, follow the following steps:

1. `git clone https://github.com/noahcao/Pixel2Mesh`
2. Do the installation as instructed in the last step
3. Download our best checkpoint [here](to be added)
4. Run `python --name predict --options experiments/to_be_added.yml --checkpoint /path/to/your/checkpoint --folder datasets/examples`

We provide demos generated by our implementation in `datasets/examples`. Here are some samples:

[add examples]

## Some known issues

We tried to pretrain the original mini-VGG (fewer channels than standard VGG) on ImageNet, and we release our pretrained results [here](to be added). However, using VGG with pretrained weights would backfire, resulting in loss turning **NaN**, for reasons we are not sure so far.

## Acknowledgements

Our work is based on the official version of [Pixel2Mesh](https://github.com/nywang16/Pixel2Mesh); Some part of code are borrowed from [a previous PyTorch implementation of Pixel2Mesh](https://github.com/Tong-ZHAO/Pixel2Mesh-Pytorch), even though this version seems incomplete. The packed files for two version of datasets are also provided by them two. Most codework is done by [Yuge Zhang](https://github.com/ultmaster).
