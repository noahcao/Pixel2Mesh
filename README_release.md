# Pixel2Mesh-Pytorch

This is a Pixel2Mesh implementation in Pytorch while [the official released version](https://github.com/nywang16/Pixel2Mesh) is in Tensorflow.

## Get Started

### Environment

The implementation works well under following environment settings: **Add libs version**

* pytorch 
* cudatoolkit =9.0
* opencv
* scipy
* scikit-image

Besides, you should also do two steps to complete environment preparation:

1. Do `git submodule update --init` to get [neural renderer](https://github.com/daniilidis-group/neural_renderer) ready.
2. Do `python setup.py install` in directory `external/chamfer` and `external/neural_renderer` to set up requirements. 

### Dataset

As used by default in original implementation, we use ShapeNet to do the model training and evaluation. **Add data preparation details**

### Train

To train a new model, an one-step script is provided:

``` shell
python entrypoint_train.py --name xxx --options path_to_yaml
```

### Evaluate

To evaluate the model performance, you can simply do as:

```shell
python entrypoint_eval.py --name xxx --options path_to_yaml --checkpoint weights_to_evaluate
```

To reproduce the model performance provided by paper author, you can download the tensorflow model weights in [their repo](https://github.com/nywang16/Pixel2Mesh), and use the provided tools in `utils/migrations` to convert into model weights in our pytorch implementation. For convenience, it can be also downloaded [here](to be added). 

### Performance

We list the Pixel2Mesh model performance on full [ShapeNet](https://www.shapenet.org/) reported in different literatures here: **Add performance**

| Source                                               | Chamfer | $F_1^{\tau}$ | $F_1^{2\tau}$ |
| ---------------------------------------------------- | ------- | ------------ | ------------- |
| [Pixel2Mesh paper](https://arxiv.org/abs/1804.01654) | 0.591   | 59.72        | 74.19         |
| [Mesh RCNN](https://arxiv.org/pdf/1906.02739.pdf)    | 0.444   | 68.94        | 80.75         |
| Our implementation with VGG                          |         |              |               |
| Our implementation with ResNet                       |         |              |               |

