from keras import optimizers
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.applications import ResNet50
from keras.layers import Dense, BatchNormalization, Activation, Dropout, Flatten
from keras.layers import Conv2D, AveragePooling2D
from keras.losses import cosine_proximity
from keras.models import Model
from keras.optimizers import Adam, SGD
import keras.backend as K


def setup_custom_model_feature(n_classes):
    # Get vgg16 frozen the weight
    # Clear memory for new model
    K.clear_session()
    # Put the Inception V3 (cut out the classifer part) and our custom classifier on top
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Sequential()
    model.add(base_model)
    model.add(AveragePooling2D(pool_size=(7, 7)))
    model.add(Dropout(0.4))
    model.add(Dense(n_classes))
    model.add(Flatten())
    for layer in base_model.layers:
        layer.trainable = False
    sgd = optimizers.Adam()
    model.compile(optimizer=sgd, loss=cosine_proximity)
    return model


if __name__ == '__main__':
    VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)).summary()
