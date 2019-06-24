import tensorflow as tf

import random
import numpy as np
import pickle
import threading
from skimage import io, transform


class DataFetcher(threading.Thread):
    def __init__(self, file_list):
        super(DataFetcher, self).__init__()
        self.stopped = False
        self.queue = Queue(64)

        self.pkl_list = []
        with open(file_list, 'r') as f:
            while True:
                line = f.readline().strip()
                if not line:
                    break
                self.pkl_list.append(line)
        self.index = 0
        self.number = len(self.pkl_list)
        random.shuffle(self.pkl_list)

    def work(self, idx):
        pkl_path = self.pkl_list[idx]
        with open(pkl_path, 'rb') as pkl_file:
            label = pickle.load(pkl_file)

        img_path = pkl_path.replace('.dat', '.png')
        '''
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img[np.where(img[:,:,3]==0)] = 255
        img = cv2.resize(img, (224,224))
        img = img[:,:,:3]/255.
        '''
        img = io.imread(img_path)
        img[np.where(img[:, :, 3] == 0)] = 255
        img = transform.resize(img, (224, 224))
        img = img[:, :, :3].astype('float32')

        return img, label, pkl_path.split('/')[-1]

    def run(self):
        while self.index < 90000000 and not self.stopped:
            self.queue.put(self.work(self.index % self.number))
            self.index += 1
            if self.index % self.number == 0:
                random.shuffle(self.pkl_list)

    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        while not self.queue.empty():
            self.queue.get()


def create_dataset(filename):
    with open(filename, "r") as f:
        file_list = [s.strip() for s in f.readlines()]
    return tf.data.Dataset.from_tensor_slices(file_list)


def shapenet_p2m_process(filename):
    pass

