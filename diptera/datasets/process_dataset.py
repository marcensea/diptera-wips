# coding: utf-8

"""
    Author: Marc Souchaud
    This script generates files representing a subset of the initial dataset.
    The following files are generated:
        \database.csv ->
            Contains picture names and labels.
            We filter the initial excel dataset, by selecting a subset of classes and removing errors.
            Our subset contains classes with an image count of 10 or more, for a total of 68 classes.
        \train_indices.csv & test_indices.csv ->
            Contains the indices of data to use in training & testing,
            with balanced classes partition and random shuffling.
        \image_data.hdf5 ->
            Contains preprocessed images from training and testing sets.
            We read the dataset image directory in batches, resize and process images, then save on disk.
"""

from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
import cv2
from math import ceil
import time

import sys
sys.path.insert(0, '../..')

from diptera import config
from diptera.datasets import wip

# =============================================================================
# Parameters
# =============================================================================

subset = config.dataset
data_path = wip.DATA_PATH
save_path = data_path / subset
images_path = Path(data_path, 'images')

ratio_test = 0.2
seed = 2023
df = None

print(data_path)
save_path.mkdir(parents=False, exist_ok=True)

# =============================================================================
# Dataset filtering
# =============================================================================
if True:
    print("> Loading dataframe...")
    df = wip.load_database()
    print("Number of unique classes:", df['Classes'].nunique())
    print("")

    # Select classes
    classes = wip.classes
    n_classes = len(classes)
    print("> Number of classes for training:", n_classes)
    print("Removing other classes...")

    df = df[df['Classes'].isin(classes['Classes'])]
    print(f"Dataframe has now {len(df)} entries and {df['Classes'].nunique()} unique classes.")

    # Remove duplicate entries
    df.drop_duplicates(subset=['Picture'], keep='first', inplace=True)
    print("\n> Removing duplicates...")
    print("Dataframe has now %d entries." % len(df))

    # Check for missing image files
    image_files = np.array([f.stem for f in (images_path.glob('*/*.jpg'))], dtype=str)
    print(f"\n> Found {len(image_files)} image files in data folder.")
    invalids = np.flatnonzero(~df['Picture'].isin(image_files).to_numpy())
    invalids = df.index[invalids]
    print("Number of invalid pictures:", len(invalids), "(missing files)")
    print("Sample of invalid pictures:", df.loc[invalids, 'Picture'][:10].to_numpy())
    df.drop(invalids, inplace=True)
    print("Removing invalid pictures...")
    print("Dataframe has now %d entries." % len(df))
    # print(np.count_nonzero(df['Picture'].isin(['1766', '3766', '3911', '4062'])))

    # Get the number of pictures for each class (can be used to drop classes with low population)
    print("\n> Information on dataset:")
    populations = df['Classes'].value_counts()
    ind = populations.argmax()
    c = populations.index[ind]
    print("Most populated class:", wip.fullname(spp=c), '->', populations.iat[ind])
    ind = populations.argmin()
    c = populations.index[ind]
    print("Least populated class:", wip.fullname(spp=c), '->', populations.iat[ind])
    print("")

    # Convert the initial database classes to our model classes
    # These will be our training & testing labels.
    mapping = classes.set_index('Classes').to_dict()['Model output']
    df['Label'] = df['Classes'].map(mapping)

    # Save picture names
    df[['Picture', 'Label']].to_csv(save_path / 'database.csv', index=False)

# =============================================================================
# Train-test split
# =============================================================================
if True:
    print("> Splitting data...")
    if df is None:
        df = wip.load_database(subset=subset)
    indices = np.arange(len(df))
    labels = df['Label'].to_numpy()
    train_ind, test_ind = train_test_split(indices, test_size=ratio_test,
                                           random_state=seed, shuffle=True, stratify=labels)
    print(len(train_ind), "in train ;", len(test_ind), "in test")

    # # Check the effects of class stratification in train-test splits
    # train_samples = df.iloc[train_ind]
    # test_samples = df.iloc[test_ind]
    # print("Train samples")
    # print(train_samples[:10])
    # n1 = len(train_samples[train_samples['Label']==51])
    # n2 = len(test_samples[test_samples['Label']==51])
    # print("Distribution of class #51:")
    # print(f"{n1} in train, {n2} in test. ratio={n2/(n1+n2):.3f}")

    np.savetxt(save_path / 'train_indices.csv', train_ind, fmt='%d')
    np.savetxt(save_path / 'test_indices.csv', test_ind, fmt='%d')

# =============================================================================
# Image preprocessing
# =============================================================================
if True:

    def preprocess_images(dset, pictures, _images_path, _config):
        N = len(pictures)
        batch_size = _config.PREPROCESS_BATCH_SIZE

        # Using batches for faster data writes
        n_batch = ceil(N / batch_size)
        offset = 0
        total_time = 0
        for batch in range(1, n_batch + 1):
            start_time = time.time()
            img_arr = None
            for i in range(min(batch_size, N - offset)):
                img_name = pictures[offset + i]
                img_path = str(_images_path.glob('*/' + img_name + '.jpg').__next__())
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (_config.image_width, _config.image_height), interpolation=cv2.INTER_AREA)
                if img_arr is None:
                    img_arr = img[None, ...]
                else:
                    img_arr = np.concatenate((img_arr, img[None, ...]))
            img_arr = img_arr.astype('float32') / 255.
            dset[offset:offset + len(img_arr), ...] = img_arr
            offset += batch_size
            end_time = time.time()
            batch_time = end_time - start_time
            total_time += batch_time
            print(
                f"Batch: {batch:03d}/{n_batch:03d} -- time elapsed: {batch_time:.1f}s -- remaining estimate: {total_time * (n_batch - batch) / batch:.1f}s")


    # input image dimensions
    model_input_shape = (config.image_height, config.image_width, config.channels)
    h5py_path = save_path / 'image_data.hdf5'

    if df is None:
        df = wip.load_database(subset=subset)
    train_ind, test_ind = wip.load_indices(subset)
    # train_ind = train_ind[:600]
    # test_ind = test_ind[:300]
    n_train = len(train_ind)
    n_test = len(test_ind)
    train_pictures = df['Picture'].iloc[train_ind].to_numpy()
    test_pictures = df['Picture'].iloc[test_ind].to_numpy()

    print("\n> Preprocessing image dataset and compressing to .hdf5 file...")
    with h5py.File(h5py_path, 'w') as f:
        print("--- TRAINING DATA ---")
        train_dset = f.create_dataset("train", (n_train, *model_input_shape), dtype='float32', compression="gzip",
                                      compression_opts=6)
        preprocess_images(train_dset, train_pictures, images_path, config)
        print("--- TESTING DATA ---")
        test_dset = f.create_dataset("test", (n_test, *model_input_shape), dtype='float32', compression="gzip",
                                     compression_opts=6)
        preprocess_images(test_dset, test_pictures, images_path, config)

# Test hdf5 data
if True:
    print("\n> Testing preprocessed data:")
    with h5py.File(save_path / 'image_data.hdf5', 'r') as f:
        X_train = f['train'][:50]  # read 50 first images in train dataset
        assert (X_train.ndim == 4)
        assert (X_train.dtype == 'float32')
        assert (X_train.min() >= 0 and X_train.max() <= 1.0)
    print("OK \n")
    ## Uncomment this to view sample images
    # for img in X_train:
    #     cv2.imshow('sample', img)
    #      cv2.waitKey(0)

print("Done!")
