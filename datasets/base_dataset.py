from torch.utils.data.dataset import Dataset
from torchvision.transforms import Normalize

import config


class BaseDataset(Dataset):

    def __init__(self):
        self.normalize_img = Normalize(mean=config.IMG_NORM_MEAN, std=config.IMG_NORM_STD)
