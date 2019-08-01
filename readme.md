# Pixel2Mesh

In this repo are:
* A reimplementation of Pixel2Mesh ([paper](https://arxiv.org/abs/1804.01654), [code](https://github.com/nywang16/Pixel2Mesh)) in PyTorch
* Retrained Pixel2Mesh checkpoints
* Backbone replacement and several minor improvements to Pixel2Mesh
* A flexible training framework, which supports more than Pixel2Mesh
* A clarification of all the details you might miss in Pixel2Mesh

## Installation

**Note: CPU training and inference has NOT yet been supported.**

The implementation works well under following environment settings:

-   PyTorch 1.1
-   CUDA 9.0
-   OpenCV 4.1
-   Scipy 1.3
-   Scikit-Image 0.15

The following tools also needs to be installed:

- easydict
- pyyaml
- tensorboardx
- trimesh
- shapely

Up till now, you can follow the latest version of all above. We've tested our code on CUDA 9.0, but 10.0 should also work. Post an issue if anything breaks.

Besides, you should also do two steps to complete environment preparation:

1.  Do  `git submodule update --init`  to get  [Neural Renderer](https://github.com/daniilidis-group/neural_renderer)  ready.
2.  Do  `python setup.py install`  in directory  `external/chamfer`  and  `external/neural_renderer`  to compile the modules.

## Demo

For a quick look around, follow the following steps:

1. `git clone https://github.com/noahcao/Pixel2Mesh`
2. Do the installation as instructed in the last step
3. Download our best checkpoint [here](to be added), and extract it under your Pixel2Mesh directory
4. Run `python --name predict --options experiments/to_be_added.yml --checkpoint to_be_added --folder datasets/examples`

Look into `datasets/examples`, there will be a few more obj files and GIFs generated. The GIF shows how the mesh is deformed and refined. Here are some examples.

[add examples]

## Use/write an experiment

The first thing you should know before you intent to run this project, is that, you need to specify everything you need in a `yml` file, which will override the default settings in `options.py`. You can see from the `experiments` directory that there are plenty of examples on how to write these files, and of course you can read `options.py` to see how it works.

If you just want to look around, you don't have to change everything. Options provided in `experiments/default` are everything you need.

## Datasets

As used by default in original implementation, we use ShapeNet to do the model training and evaluation.

We've packed everything needed for training and evaluation. So other than `data` folder and `data_tf` folder in `shapenet`, it can be downloaded [here](to be added). `data_tf` can be downloaded in the [official repo](https://github.com/nywang16/Pixel2Mesh), and `data` can be downloaded from [Tong Zhao's PyTorch repo](https://github.com/Tong-ZHAO/Pixel2Mesh-Pytorch). Download them and organize the extracted files as listed in the tree below.

If you only need `train_tf` and `test_tf`, **it's recommended to download `data_tf` only**, as downloading the full `data` takes much longer and the extracted files are about 300GB.

```
datasets/data
├── ellipsoid
│   ├── face1.obj
│   ├── face2.obj
│   ├── face3.obj
│   └── info_ellipsoid.dat
├── pretrained
│   ├── resnet50-19c8e357.pth
│   ├── vgg16-p2m.pth
│   └── vgg16-397923af.pth
└── shapenet
    ├── data
    │   ├── 02691156
    │   │   └── 3a123ae34379ea6871a70be9f12ce8b0_02.dat
    │   └── 02828884
    ├── data_tf
    │   ├── 02691156
    │   │   └── 10115655850468db78d106ce0a280f87
    │   └── 02828884
    └── meta
        ├── shapenet.json
        ├── shapenet_labels.pkl
        ├── test_all.txt
        ├── test_plane.txt
        ├── test_small.txt
        ├── test_tf.txt
        ├── train_all.txt
        ├── train_plane.txt
        ├── train_small.txt
        └── train_tf.txt
```

It might be worth some efforts to explain the difference between the two datasets.

The dataset provided by the original Tensorflow repository has images with resolution 137x137, four channels (RGB + alpha), and it contains 175132 samples for training and 43783 samples for evaluation; while the dataset provided by Zhao has RGB images with resolution 224x224 (the background has already been painted white, which, in the former version of dataset, is done in the data processing step), and the number of samples is much larger (??? for training and ??? for evaluation).

We did experiments on both datasets, alternatively. However, due to the limitations on time and resources, there is no fair comparison on which dataset trains better, or performs better. All our checkpoints provided here are trained on the Tensorflow version, that is, the files under `data_tf`.

## Evaluation

### Metrics of Original Paper

According to [MeshRCNN](https://arxiv.org/abs/1906.02739), the performance of the pretrained checkpoint is much higher than the performance of the original paper (68.94 vs 59.72 on F1$^{\tau}$, 80.75 vs. 74.19 on F1$^{2\tau}$). We rerun the evaluation provided in the original tensorflow repository for verification. The results are listed below:

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
| Mean          | 43783        | 65.22   | 78.80   | 0.482 | 2.418 |
| Weighted-mean |              | 66.56   | 80.17   | 0.439 | 2.545 |

Note that when the paper reports its performance, metrics of each category are calculated separately and the overall mean is, simply, their mean, without considerations on some categories containing more samples than others, which, when taken into account, gives a higher performance (over 80% $F1_{2\tau}$).

Considering the gap of about 2% (between 78.80 and 80.75), we suspect that MeshRCNN takes the weighted-mean instead of mean in their paper, which might be a more natural practice. And we, thus, provide both weighted-mean and mean for all our checkpoints shown below.

### Our Checkpoints

We have converted the pretrained checkpoint from two repositories mentioned above to the format that is readable for our code. The conversion scripts can be found in `utils/migration`, and we also provided our converted version to download for your convenience.

The evaluation results and corresponding experiment files of all converted checkpoints and our retrained checkpoints are listed below.

[to be added]

### Do Your Own Evaluation

```
python entrypoint_eval.py --name xxx --options path_to_yaml --checkpoint weights_to_evaluate
```

## Training

### Improvement of Training Recipes

Here we want to stress some training recipes that might differ from the official version.

<<<<<<< HEAD
To evaluate the original trained model, you can download the model from the [author's repo](https://github.com/nywang16/Pixel2Mesh),
and convert the model into our checkpoint using the tools provided in `utils/migrations`; or you can directly downloaded 
our converted model [here](to be added).
=======
* **Adopt a larger batch size.** For VGG, the batch size can be 24 per GPU. We leverage PyTorch DataParallel to support multi-GPU training. Since Chamfer distances cannot have ground-truths with different number of points in the same batch, instead of resampling from the original surface, we simply to downsampling or upsampling from the provided dataset.
* **Replace backbone.** We tried to replace the original VGG with something deeper, like ResNet50. We adopt a similar strategy, like VGG, by extracting the last four convolution layers and use them in Graph Projection. Loss curve trained with ResNet seems to be more stable and smooth, which is likely due to the pretraining on ImageNet. We also tried to pretrain the original mini-VGG (fewer channels than standard VGG) on ImageNet, and we release our pretrained results [here](to be added). However, using VGG with pretrained weights would backfire, resulting in loss turning NaN, for reasons we are not sure.
* **Stabilize loss curve.** There are also a few more things we have done in order to prevent the loss curve from oscillation. For example, we tried to do normalization on the deformed sphere, so that it's deformed at location $(0,0,0)$; we tried to use a threshold activation on $z$-axis when projection, so that $z$ will always be positive or negative and never be $0$ during projection. None of these cast a significant improvement on evaluation performance, and other effects have not been thoroughly investigated.
* **Camera Intrinsics.** Original authors used to set their camera focal length to 248, camera center at [112, 112], but later changed that to 250 with [111.5, 111.5]. We didn't investigate in the further reason of this modification and retrieved similar results on both settings.
>>>>>>> 66d241e6fdfb018df03a34a63f527aaa41def28b

### Do Your Own Training

```
python entrypoint_train.py --name xxx --options path_to_yaml
```

For training on a cluster with, for example, slurm, we have also provided examples of scripts for job launching. Refer to `slurm` folder for details.

## Future work

To be added

## Acknowledgements

Our work is based on the official version of [Pixel2Mesh](https://github.com/nywang16/Pixel2Mesh); Some part of code are taken and modified from [a previous PyTorch implementation of Pixel2Mesh](https://github.com/Tong-ZHAO/Pixel2Mesh-Pytorch), even though this version seems buggy and incomplete. We thank them for their great efforts.

## License

To be added
