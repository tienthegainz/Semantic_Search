from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image

from annoy import AnnoyIndex
import numpy as np
from numpy import linalg
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import logging
import time
import argparse
parser = argparse.ArgumentParser(description='Building Annoy tree to serach image')
parser.add_argument('--img_path', '-imp', type=str,
                    help='Image path to find similar vector')
parser.add_argument('--tree_path', '-dtp', type=str, required = True,
                    help='Tree ANN path')

parser.add_argument('--build', '-B', action='store_true',
                    help='Build the .ann file')
parser.add_argument('--search', '-S', action='store_true',
                    help='Search the image .ann file')
args = parser.parse_args()

class VetorSearch:
    def __init__(self, model, data_path='../Vietnam_Food/Training/',
                image_size = (224, 224), dims=4096):
        self.model = model
        self.data_path = data_path
        self.image_size = image_size
        self.dims = dims
        self.id_labels = sorted(os.listdir(self.data_path))


    def preprocess_img(self, fn_path):
        img = image.load_img(fn_path, target_size=self.image_size)
        img = img.convert('RGB')
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        input_ = preprocess_input(img)
        feature = self.model.predict(img)[0]
        feature = feature / np.linalg.norm(feature)
        return feature


    def generate_features(self):
        """
        Yield out the vector feature and path to that images
        """
        # self.id_labels = sorted(os.listdir(self.data_path))
        for folder in tqdm(self.id_labels):
            label_path = os.path.join(self.data_path, folder)
            fn_paths = sorted(os.listdir(label_path))
            fn_paths = [os.path.join(label_path, fn_path) for fn_path in fn_paths]
            for fn_path in fn_paths:
                feature = self.preprocess_img(fn_path)

                yield feature, fn_path


    def _map_path_item(self):
        self.item_mapping = []
        for folder in tqdm(self.id_labels):
            label_path = os.path.join(self.data_path, folder)
            fn_paths = sorted(os.listdir(label_path))
            fn_paths = [os.path.join(label_path, fn_path) for fn_path in fn_paths]
            for fn_path in fn_paths:
                self.item_mapping.append(fn_path)


    def make_index_features(self, mode="image", n_trees=1000):
        self.feature_index = AnnoyIndex(self.dims, metric='angular')
        #self.item_mapping = []
        for i, row in enumerate(self.generate_features()):
            vec = row
            self.feature_index.add_item(i, vec[0])
            #self.item_mapping.append(vec[1])
        self.feature_index.build(n_trees)
        self._map_path_item()


    def save_tree(self, path):
        self.feature_index.save(path)


    def load_tree(self, path):
        self.feature_index = AnnoyIndex(self.dims)
        self.feature_index.load(path)
        self._map_path_item()


    def search_index_by_vector(self, vector, top_n=10):
        distances = self.feature_index.get_nns_by_vector(
            vector, top_n, include_distances=True
        )
        return [{'index': a, 'path': self.item_mapping[a], 'distance': distances[1][i]}
                for i, a in enumerate(distances[0])]


if __name__ == '__main__':
    start_time = time.time()
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    vs = VetorSearch(model)
    if args.build:
        vs.make_index_features()
        vs.save_tree(args.tree_path)
    elif args.search:
        vs.load_tree(args.tree_path)
    print("Setup: %s seconds" % (time.time() - start_time))
    if args.search:
        start_time = time.time()
        print(vs.search_index_by_vector(vs.preprocess_img(fn_path=args.img_path),
                top_n=4))
        print("Search: %s seconds" % (time.time() - start_time))
