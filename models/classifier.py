import torch.nn as nn

from models.backbones import get_backbone


class Classifier(nn.Module):

    def __init__(self, options, num_classes):
        super(Classifier, self).__init__()

        self.nn_encoder, self.nn_decoder = get_backbone(options)

        if "vgg" in options.backbone:
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Linear(list(self.nn_encoder.children())[-1].out_channels * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        elif "resnet" in options.backbone:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(self.nn_encoder.inplanes, num_classes)
        else:
            raise NotImplementedError

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        x = self.nn_encoder(img)[-1]  # last layer
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
