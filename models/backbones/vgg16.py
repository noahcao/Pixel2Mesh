import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class VGG16TensorflowAlign(nn.Module):

    def __init__(self, n_classes_input=3):
        super(VGG16TensorflowAlign, self).__init__()

        self.features_dim = 960
        # this is to align with tensorflow padding (with stride)
        # https://bugxch.github.io/tf%E4%B8%AD%E7%9A%84padding%E6%96%B9%E5%BC%8FSAME%E5%92%8CVALID%E6%9C%89%E4%BB%80%E4%B9%88%E5%8C%BA%E5%88%AB/
        self.same_padding = nn.ZeroPad2d(1)
        self.tf_padding = nn.ZeroPad2d((0, 1, 0, 1))
        self.tf_padding_2 = nn.ZeroPad2d((1, 2, 1, 2))

        self.conv0_1 = nn.Conv2d(n_classes_input, 16, 3, stride=1, padding=0)
        self.conv0_2 = nn.Conv2d(16, 16, 3, stride=1, padding=0)

        self.conv1_1 = nn.Conv2d(16, 32, 3, stride=2, padding=0)  # 224 -> 112
        self.conv1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
        self.conv1_3 = nn.Conv2d(32, 32, 3, stride=1, padding=0)

        self.conv2_1 = nn.Conv2d(32, 64, 3, stride=2, padding=0)  # 112 -> 56
        self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.conv2_3 = nn.Conv2d(64, 64, 3, stride=1, padding=0)

        self.conv3_1 = nn.Conv2d(64, 128, 3, stride=2, padding=0)  # 56 -> 28
        self.conv3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=0)
        self.conv3_3 = nn.Conv2d(128, 128, 3, stride=1, padding=0)

        self.conv4_1 = nn.Conv2d(128, 256, 5, stride=2, padding=0)  # 28 -> 14
        self.conv4_2 = nn.Conv2d(256, 256, 3, stride=1, padding=0)
        self.conv4_3 = nn.Conv2d(256, 256, 3, stride=1, padding=0)

        self.conv5_1 = nn.Conv2d(256, 512, 5, stride=2, padding=0)  # 14 -> 7
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=0)
        self.conv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=0)
        self.conv5_4 = nn.Conv2d(512, 512, 3, stride=1, padding=0)

    def forward(self, img):
        img = F.relu(self.conv0_1(self.same_padding(img)))
        img = F.relu(self.conv0_2(self.same_padding(img)))

        img = F.relu(self.conv1_1(self.tf_padding(img)))
        img = F.relu(self.conv1_2(self.same_padding(img)))
        img = F.relu(self.conv1_3(self.same_padding(img)))

        img = F.relu(self.conv2_1(self.tf_padding(img)))
        img = F.relu(self.conv2_2(self.same_padding(img)))
        img = F.relu(self.conv2_3(self.same_padding(img)))
        img2 = img

        img = F.relu(self.conv3_1(self.tf_padding(img)))
        img = F.relu(self.conv3_2(self.same_padding(img)))
        img = F.relu(self.conv3_3(self.same_padding(img)))
        img3 = img

        img = F.relu(self.conv4_1(self.tf_padding_2(img)))
        img = F.relu(self.conv4_2(self.same_padding(img)))
        img = F.relu(self.conv4_3(self.same_padding(img)))
        img4 = img

        img = F.relu(self.conv5_1(self.tf_padding_2(img)))
        img = F.relu(self.conv5_2(self.same_padding(img)))
        img = F.relu(self.conv5_3(self.same_padding(img)))
        img = F.relu(self.conv5_4(self.same_padding(img)))
        img5 = img

        return [img2, img3, img4, img5]


class VGG16P2M(nn.Module):

    def __init__(self, n_classes_input=3, pretrained=False):
        super(VGG16P2M, self).__init__()

        self.features_dim = 960

        self.conv0_1 = nn.Conv2d(n_classes_input, 16, 3, stride=1, padding=1)
        self.conv0_2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.conv1_1 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 224 -> 112
        self.conv1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 112 -> 56
        self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 56 -> 28
        self.conv3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(128, 256, 5, stride=2, padding=2)  # 28 -> 14
        self.conv4_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(256, 512, 5, stride=2, padding=2)  # 14 -> 7
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, 3, stride=1, padding=1)

        if "vgg16p2m" in config.PRETRAINED_WEIGHTS_PATH and pretrained:
            state_dict = torch.load(config.PRETRAINED_WEIGHTS_PATH["vgg16p2m"])
            self.load_state_dict(state_dict)
        else:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        img = F.relu(self.conv0_1(img))
        img = F.relu(self.conv0_2(img))
        # img0 = torch.squeeze(img) # 224

        img = F.relu(self.conv1_1(img))
        img = F.relu(self.conv1_2(img))
        img = F.relu(self.conv1_3(img))
        # img1 = torch.squeeze(img) # 112

        img = F.relu(self.conv2_1(img))
        img = F.relu(self.conv2_2(img))
        img = F.relu(self.conv2_3(img))
        img2 = img

        img = F.relu(self.conv3_1(img))
        img = F.relu(self.conv3_2(img))
        img = F.relu(self.conv3_3(img))
        img3 = img

        img = F.relu(self.conv4_1(img))
        img = F.relu(self.conv4_2(img))
        img = F.relu(self.conv4_3(img))
        img4 = img

        img = F.relu(self.conv5_1(img))
        img = F.relu(self.conv5_2(img))
        img = F.relu(self.conv5_3(img))
        img = F.relu(self.conv5_4(img))
        img5 = img

        return [img2, img3, img4, img5]


class VGG16Recons(nn.Module):

    def __init__(self, input_dim=512, image_channel=3):
        super(VGG16Recons, self).__init__()

        self.conv_1 = nn.ConvTranspose2d(input_dim, 256, kernel_size=2, stride=2, padding=0)  # 7 -> 14
        self.conv_2 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)  # 14 -> 28
        self.conv_3 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)  # 28 -> 56
        self.conv_4 = nn.ConvTranspose2d(128, 32, kernel_size=6, stride=2, padding=2)  # 56 -> 112
        self.conv_5 = nn.ConvTranspose2d(32, image_channel, kernel_size=6, stride=2, padding=2)  # 112 -> 224

    def forward(self, img_feats):
        x = F.relu(self.conv_1(img_feats[-1]))
        x = torch.cat((x, img_feats[-2]), dim=1)
        x = F.relu(self.conv_2(x))
        x = torch.cat((x, img_feats[-3]), dim=1)
        x = F.relu(self.conv_3(x))
        x = torch.cat((x, img_feats[-4]), dim=1)
        x = F.relu(self.conv_4(x))
        x = F.relu(self.conv_5(x))

        return torch.sigmoid(x)
