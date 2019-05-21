import random
import os
import logging

from annoy import AnnoyIndex
import numpy as np
from numpy import linalg
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from tqdm import tqdm
from feature_extractor import FeatureExtractor

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def generate_features(id_labels, fe):
    """
    Yield out the vector feature and path to that images
    """
    TRAIN_BASE = '../Vietnam_Food/Training/'
    id_labels = os.listdir(TRAIN_BASE)
    for folder in tqdm(id_labels):
        label_path = os.path.join(TRAIN_BASE, folder)
        fn_paths = sorted(os.listdir(label_path))
        fn_paths = [os.path.join(label_path, fn_path) for fn_path in fn_paths]
        for fn_path in fn_paths:
            img = image.load_img(fn_path, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            input_ = preprocess_input(img)
            feature = fe.extract(input_)

            yield feature, fn_path


def build_image_mapping(id_labels):
    """
    
    """
    TRAIN_BASE = '../Vietnam_Food/Training/'
    i = 0
    #id_labels = os.listdir(TRAIN_BASE)
    images_mapping = {}
    for folder in tqdm(id_labels):
        label_path = os.path.join(TRAIN_BASE, folder)
        fn_paths = sorted(os.listdir(label_path))
        fn_paths = [os.path.join(label_path, fn_path) for fn_path in fn_paths]
        for fn_path in fn_paths:
            images_mapping[i] = fn_path
            i += 1
    return images_mapping


def index_features(features, mode="image", n_trees=1000, dims=4096):
    feature_index = AnnoyIndex(dims, metric='angular')
    for i, row in enumerate(features):
        vec = row
        if mode == "image":
            feature_index.add_item(i, vec[0][0])
        elif mode == "word":
            feature_index.add_item(i, vec)
    feature_index.build(n_trees)
    return feature_index


def search_index_by_key(key, feature_index, item_mapping, top_n=10):
    distances = feature_index.get_nns_by_item(
        key, top_n, include_distances=True
    )
    return [[a, item_mapping[a], distances[1][i]]
            for i, a in enumerate(distances[0])]


def show_sim_imgs(search_key, feature_index, feature_mapping):

    results = search_index_by_key(
        search_key, feature_index, feature_mapping, 10
    )

    main_img = mpimg.imread(results[0][1])
    plt.imshow(main_img)
    fig = plt.figure(figsize=(8, 8))
    columns = 3
    rows = 3
    for i in range(1, columns * rows + 1):
        img = mpimg.imread(results[i][1])
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


def show_imgs(id_folder):
    fmt = 'path-to-train-folder/train/{}/images/{}_{}.JPEG'
    random_imgs = [fmt.format(id_folder, id_folder, num)
                   for num in random.sample(range(0, 500), 9)]
    fig = plt.figure(figsize=(16, 16))
    columns = 3
    rows = 3
    for i in range(1, columns * rows + 1):
        img = mpimg.imread(random_imgs[i - 1])
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()
