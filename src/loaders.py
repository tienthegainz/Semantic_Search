import os

import numpy as np
import keras
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from tqdm import tqdm


class TextSequenceGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, fpath, mode="train",
                 split_ratio = 0.0, batch_size=64,
                 img_size=(224, 224), no_channels=3,
                 n_classes = 7, shuffle=True):
        """
        fpath: the absolute path to the images folder
        mode: two mode train and val
        split_ratio: % of dataset will be validation. 0 for no validation
        """
        # train 95, test 5
        self.imgs, self.labels, self.unique_labels = [], [], []
        if split_ratio > 1 or split_ratio < 0:
            print("Invalid split_ration parameter\n")
            exit()
        if mode == "train":
            base_train = fpath
            id_labels = sorted(os.listdir(base_train))
            for folder in tqdm(id_labels):
                label_path = os.path.join(base_train, folder)
                fn_paths = sorted(os.listdir(label_path))
                fn_len = len(fn_paths)
                # split
                fn_paths = fn_paths[ :int((1-split_ratio)*fn_len)]
                for fn_path in fn_paths:
                    # imgs : path to images
                    # labels: coressponding label for that image
                    # unique_labels: for classes' names
                    self.imgs.append(os.path.join(label_path, fn_path))
                    self.labels.append(folder)
                    if folder not in self.unique_labels:
                        self.unique_labels.append(folder)
        elif mode == "val":
            base_train = fpath
            id_labels = sorted(os.listdir(base_train))
            for folder in tqdm(id_labels):
                label_path = os.path.join(base_train, folder)
                fn_paths = sorted(os.listdir(label_path))
                fn_len = len(fn_paths)
                # split
                fn_paths = fn_paths[int((1-split_ratio)*fn_len): ]
                for fn_path in fn_paths:
                    # imgs : path to images
                    # labels: coressponding label for that image
                    # unique_labels: for classes' names
                    self.imgs.append(os.path.join(label_path, fn_path))
                    self.labels.append(folder)
                    if folder not in self.unique_labels:
                        self.unique_labels.append(folder)
        #ids: data amount
        self.ids = range(len(self.imgs))
        self.indexes = np.arange(len(self.ids))
        # img_size, img_w, img_h: image size for later use
        self.img_size = img_size
        self.img_w, self.img_h = self.img_size

        self.batch_size = batch_size
        self.no_channels = no_channels

        self.n_classes = n_classes

        self.unique_labels = sorted(self.unique_labels)

        self.shuffle = shuffle
        if mode == "train":
            self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        ids = [self.ids[k] for k in indexes]

        X, Y = self.__data_generation(ids)

        return X, Y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        #self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids):
        size = len(ids)

        X = np.empty(
            (size, self.img_w, self.img_h, self.no_channels),
            dtype=np.float32
        )
        Y = np.empty((size, self.n_classes), dtype=np.float32)

        for i, id_ in enumerate(ids):
            img = image.load_img(self.imgs[id_], target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            X[i] = img
            Y[i] = self.wv_label_mapping(self.labels[id_])
        return X, Y

    def wv_label_mapping(self, y_label):
        """Making one hot code from labels"""
        one_hot_code = []
        for label in self.unique_labels:
            if label == y_label:
                one_hot_code.append(1)
            else:
                one_hot_code.append(0)
        return np.array(one_hot_code)
