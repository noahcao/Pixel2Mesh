import torch.nn as nn

from models.backbones import get_backbone


class Classifier(nn.Module):

    def __init__(self, options, num_classes):
        super(Classifier, self).__init__()

        self.nn_encoder, self.nn_decoder = get_backbone(options)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(self.nn_encoder.children()[-1].out_channels, num_classes)

    def forward(self, img):
        features = self.nn_encoder(img)
        pooling = self.avgpool(features)
        output = self.fc(pooling.reshape(pooling.size(0), -1))
        return output
