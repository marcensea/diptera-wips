import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

from pathlib import Path
import numpy as np
import cv2
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import pickle  # to save sklearn scaler
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../..')

from diptera import config
from diptera.datasets import wip
from diptera.utils.callbacks import HistoryPlot
from diptera.utils.scaling import scaler_transform_util

from diptera.models.mobilenet9 import build_model


lr = config.LEARNING_RATE
def scheduler(epoch):
    # Reduce lr by half every 25 epochs.
    n_reductions = epoch // 25
    return lr / (2**n_reductions)


if __name__ == '__main__':

    show_samples = False
    show_augmented_samples = False

    # input image dimensions
    input_shape = (config.image_height, config.image_width, config.channels)

    MODEL_DATA_PATH = Path("model_data").resolve()
    save_path = MODEL_DATA_PATH / config.model_name
    print("Model will be saved to", save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Training models '{config.model_name}' with {config.FOLDS}-fold crossvalidation")

    n_classes = len(wip.classes)
    print("Number of classes:", n_classes)

    x_train, _ = wip.load_images(dataset=config.dataset)
    y_train, _ = wip.load_database_split(dataset=config.dataset, col='Label')

    n_train = len(x_train)
    print("Train images format: %s, Train labels format: %s" % (x_train.shape, y_train.shape))

    Y_train = to_categorical(y_train, n_classes)

    if show_samples:
        # Show sample images & labels
        names_train, _ = wip.load_database_split(dataset=config.dataset, col='Picture')
        indexes = np.arange(6)
        for index in indexes:
            image = x_train[index]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = y_train[index]
            print(f"Image n°{index} is named {names_train[index]}, has shape {x_train[index].shape} and label {label}. Full name: {wip.fullname(label)}")
            plt.figure()
            plt.imshow(image)
            plt.show()
        del names_train  # free memory

    # Use StratifiedKFold to generate (K-1) sets for training and 1 set for validation, returning all combinations.
    kfold = StratifiedKFold(n_splits=config.FOLDS, shuffle=False)#, random_state=config.SEED)

    start_epoch = config.START_EPOCH
    ind_fold = 1
    for fold_train_index, fold_val_index in kfold.split(x_train, y_train):
        if ind_fold < config.START_FOLD:
            ind_fold += 1
            continue

        print('\n' + '_'*50)
        print(f"Starting K-Fold cross-validation {ind_fold} / {config.FOLDS}")
        print(f"Training: {len(fold_train_index)}, validation: {len(fold_val_index)}")
        print('\n' + '_'*50)

        # Create directory for current fold
        fold_path = save_path / f"{ind_fold}k{config.FOLDS}"
        fold_path.mkdir(exist_ok=True)

        X_train = x_train[fold_train_index]
        X_val = x_train[fold_val_index]
        print("Center, std:", X_train.mean(), X_val.std())

        if rescale:
            scaler = StandardScaler()
            X_train = scaler_transform_util(X_train, scaler, fit=True)
            X_val = scaler_transform_util(X_val, scaler)
            with open(str(fold_path / "scaler.pkl"), 'wb') as f:
                pickle.dump(scaler, f)

        # Reduce learning rate over training
        # reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=10, verbose=1)
        reduce_lr = LearningRateScheduler(scheduler, verbose=1)

        # Backup the models weights which give best accuracy
        checkpoint = ModelCheckpoint(str(fold_path / (config.model_name + "-weights-{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5")),
                                     monitor='val_categorical_accuracy',
                                     verbose=1, save_best_only=False, save_weights_only=False, mode='max',
                                     save_freq='epoch', period=20)
        checkpoint_last = ModelCheckpoint(str(fold_path / (config.model_name + "-weights-last.hdf5")),
                                     monitor='val_categorical_accuracy',
                                     verbose=1, save_best_only=False, save_weights_only=False, mode='max')
        checkpoint_best = ModelCheckpoint(str(fold_path / (config.model_name + "-weights-best.hdf5")),
                                     monitor='val_categorical_accuracy',
                                     verbose=1, save_best_only=True, save_weights_only=False, mode='max')

        logger = CSVLogger(str(fold_path / "training.log"), append=True)
        history = HistoryPlot(fold_path / 'training.log', "categorical_accuracy", "loss")
        # After each epoch, history will generate a plot image from CSV logger.

        # callbacks_list = [reduce_lr, checkpoint, checkpoint_last, checkpoint_best, logger, history]
        callbacks_list = [checkpoint, checkpoint_last, checkpoint_best, logger, history, reduce_lr]

        print('Using real-time data augmentation.')

        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(**config.augmentations)

        train_generator = datagen.flow(X_train, Y_train[fold_train_index],
                                       batch_size=config.BATCH_SIZE, shuffle=True)

        if show_augmented_samples:
            augmented_batch = train_generator.next()[0][:12]
            plt.figure()
            n_cols = 4
            n_rows = 3
            for i in range(n_cols*n_rows):
                plt.subplot(n_rows, n_cols, i+1)
                augmented_img = augmented_batch[i, :,:,::-1]  # bgr to rgb
                plt.imshow(augmented_img)
                plt.axis('off')
            plt.tight_layout()
            plt.show()
            # Reset train generator
            train_generator = datagen.flow(X_train, Y_train[fold_train_index],
                                           batch_size=config.BATCH_SIZE, shuffle=True)

        validation_generator = datagen.flow(X_val, Y_train[fold_val_index],
                                            batch_size=config.BATCH_SIZE, shuffle=True)

        # Build a CNN models and print its summary
        if start_epoch > 0:
            model = load_model(fold_path / (config.model_name + "-weights-last.hdf5"))
        else:
            model = build_model(input_shape, n_classes)

        model.summary()

        # fit the models on the batches generated by datagen.flow()
        steps_per_epoch = int(np.ceil(X_train.shape[0] / config.BATCH_SIZE))
        print(f"Fitting model with {steps_per_epoch} steps per epoch.")
        model.fit_generator(train_generator,
                            # steps_per_epoch=steps_per_epoch,
                            epochs=config.EPOCHS,
                            initial_epoch=start_epoch,
                            validation_data=validation_generator,
                            callbacks=callbacks_list)

        del model
        K.clear_session()
        ind_fold += 1
        start_epoch = 0

