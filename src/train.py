from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications import ResNet50
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
import keras.backend as K
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam, SGD
from keras.models import load_model

import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import os
import argparse

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Add argument
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='new',
                    help='Choose model type: \'new\' = New model, \'old\' = Continue to train')
parser.add_argument('--model_type', type=str, default='resnet50',
                    help='Choose model type')
parser.add_argument('--model_path', type=str,
                    help='If mode = \'old\' then add model path')
parser.add_argument('--batch_size', type=int, default=200,
                    help='Input batch size for training, validation, evaluation')
parser.add_argument('--n_classes', type=int, default=7,
                    help='Number of classes')
parser.add_argument('--n_epochs', type=int, default=10,
                    help='Number of epochs')
parser.add_argument('--data_path', type=str,
                    help='Path to training data folder')
args = parser.parse_args()


def build_resnet50(n_classes):
    # Clear memory for new model
    K.clear_session()
    # Put the ResNet50 (cut out the classifer part) and our custom classifier on top
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Sequential()
    model.add(base_model)
    model.add(AveragePooling2D(pool_size=(7, 7)))
    model.add(Dropout(0.4))
    model.add(Flatten(name='vector_flatten'))
    model.add(Dense(n_classes, activation='softmax'))
    for layer in base_model.layers:
        layer.trainable = True
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                 metrics=['accuracy', 'top_k_categorical_accuracy'])
    return model


def main():
    # argument
    mode = args.mode
    n_classes = args.n_classes
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    TRAIN_PATH = args.data_path

    # Image processing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range = 20,
        width_shift_range = 10,
        height_shift_range = 10,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
        )

    train_gen = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
        )
    val_gen = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
        )
    # Write class convert
    try:
        index_file = open('index_file.txt', 'r')
    except FileNotFoundError:
        labels = (train_gen.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        print('Making index_file.txt\n')
        index_file = open('index_file.txt', 'w')
        for k,v in labels.items():
            index_file.write('{}.{}\n'.format(k, v))
        index_file.close()
    STEP_SIZE_TRAIN=(train_gen.n//train_gen.batch_size)+1
    STEP_SIZE_VALID=(val_gen.n//val_gen.batch_size)+1

    ### TRAIN ###
    if args.mode == 'new':
        model_type = args.model_type
        print('New model\n')
        if model_type == 'resnet50':
            model = build_resnet50(n_classes)
    elif args.mode == 'old':
        model_path = args.model_path
        model = load_model(model_path)
        model_type = args.model_type

    checkpointer = ModelCheckpoint(filepath='../train_data/vietnam_food_resnet50.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
    csv_logger = CSVLogger('../history/vietnam_food_vgg16.log')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=0.001)

    """Fit models by generator"""
    history = model.fit_generator(generator=train_gen,
                        validation_data=val_gen,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_steps=STEP_SIZE_VALID,
                        callbacks=[csv_logger, checkpointer, reduce_lr],
                        epochs=n_epochs)
    """Plot training history"""
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('history/vietnam_food_{}_acc.png'.format(model_type))

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('history/vietnam_food_{}.png'.format(model_type))



if __name__ == '__main__':
    main()
