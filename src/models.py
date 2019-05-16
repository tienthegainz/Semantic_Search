from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, BatchNormalization, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.losses import cosine_proximity
from keras.models import Model


def load_headless_pretrained_model():
    pretrained_vgg16 = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=pretrained_vgg16.input,
                  outputs=pretrained_vgg16.get_layer('fc2').output)
    return model

def setup_custom_model_feature():
    # Get vgg16 frozen the weight
    headless_pretrained_vgg16 = VGG16(
        weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in headless_pretrained_vgg16.layers:
        layer.trainable = False

    # Make custom top layer
    image_dense = Conv2D()
    image_dense = Flatten()(headless_pretrained_vgg16)
    image_dense = Conv2D
    complete_model = Model(
        inputs=[headless_pretrained_vgg16.input], outputs=image_dense)

    return complete_model

def setup_custom_model(intermediate_dim=2000, word_embedding_dim=100):
    headless_pretrained_vgg16 = VGG16(
        weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    x = headless_pretrained_vgg16.get_layer('fc2').output

    for layer in headless_pretrained_vgg16.layers:
        layer.trainable = False

    image_dense1 = Dense(intermediate_dim, name="image_dense1")(x)
    image_dense1 = BatchNormalization()(image_dense1)
    image_dense1 = Activation("relu")(image_dense1)
    image_dense1 = Dropout(0.5)(image_dense1)

    image_dense2 = Dense(word_embedding_dim, name="image_dense2")(image_dense1)
    image_dense2 = BatchNormalization()(image_dense2)

    complete_model = Model(
        inputs=[headless_pretrained_vgg16.input], outputs=image_dense2)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    complete_model.compile(optimizer=sgd, loss=cosine_proximity)
    return complete_model

if __name__ == '__main__':
    VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)).summary()
