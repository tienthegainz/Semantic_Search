from keras.callbacks import ModelCheckpoint

from src.models import setup_custom_model
from src.loaders import TextSequenceGenerator


if __name__ == '__main__':
    num_epochs = 10
    batch_size = 16
    ratio = 0.2
    custom_model = setup_custom_model()
    train_generator = TextSequenceGenerator(fpath='Vietnam_Food/Training/',
                                            mode="train",
                                            batch_size=batch_size,
                                            split_ratio = ratio,
                                            shuffle=True)
    val_generator = TextSequenceGenerator(fpath='Vietnam_Food/Training/',
                                            mode="val",
                                            batch_size=batch_size,
                                            split_ratio = ratio,
                                            shuffle=True)

    checkpointer = ModelCheckpoint(
        filepath='../models/checkpoint.{epoch:02d}-{loss:.2f}.hdf5',
        verbose=1, save_best_only=True
    )
    custom_model.fit_generator(
        data_generator, steps_per_epoch=2799 // batch_size,
        epochs=num_epochs,
        # validation_data=val_data_generator, validation_steps=10000 // batch_size,
        callbacks=[checkpointer]
    )
    model_json = custom_model.to_json()
    with open('../models/config.json', 'w') as f:
        f.write(model_json)

    custom_model.save_weights(
        'path-to-model-folder/models/best_model.hdf5')
