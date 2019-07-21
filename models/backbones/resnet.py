import torch
from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck

import config


class P2MResNet(ResNet):

    def __init__(self, *args, **kwargs):
        self.output_dim = 0
        super().__init__(*args, **kwargs)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        res = super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)
        self.output_dim += self.inplanes
        return res

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)

        return features

    @property
    def features_dim(self):
        return self.output_dim


def resnet50():
    model = P2MResNet(Bottleneck, [3, 4, 6, 3])
    state_dict = torch.load(config.PRETRAINED_WEIGHTS_PATH["resnet50"])
    model.load_state_dict(state_dict)
    return model
