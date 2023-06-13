# coding: utf-8

"""
Import functions for the Wing Interference Pattern dataset.
"""
import os

import numpy as np
from pathlib import Path
import pandas as pd
import h5py


class MetaWIP(type):
    """ Metaclass for the WIP class.
    Contains the definition of properties which can be accessed directly from class.
    This is useful to protect class attributes and allows lazy loading of data.
    Warning: panda's dataframes are NOT immutable, so these attributes cannot be protected without an explicit copy.
    """

    def __init__(cls, *args, **kwargs):
        _ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
        cls._DATA_PATH = _ROOT_DIR / 'data'
        cls._abbreviations = None
        cls._classes = None

    @property
    def DATA_PATH(cls):
        return cls._DATA_PATH

    @property
    def abbreviations(cls):
        if cls._abbreviations is None:
            # Import abbreviations
            cls._abbreviations = pd.read_csv(cls._DATA_PATH / 'abbreviations.csv',
                                             header=None, index_col=0).squeeze("columns").to_dict()
        return cls._abbreviations

    @property
    def classes(cls):
        if cls._classes is None:
            cls._classes = cls.load_classes()
        return cls._classes


class WIP(metaclass=MetaWIP):

    @classmethod
    def load_classes(cls, dataset='spp68'):
        # Import classes labels
        cls._classes = pd.read_csv(cls.DATA_PATH / f"classes_{dataset}.csv")
        return cls._classes

    @classmethod
    def get_abbreviation(cls, x):
        """
        Return the abbreviated form of x's genus.
        If no particular abbreviation is known, take the first letter in genus name.
        :param x:
        :return:
        """
        if type(x) == int:  # x is a label indice
            genus_label = cls.classes.at['Genera', x]
            return cls.abbreviations.get(genus_label, genus_label[0] + '.')
        elif type(x) == str:  # x is a str
            return cls.abbreviations.get(x, x[0] + '.')
        else:
            raise TypeError('Value must be either int or str.')

    @classmethod
    def fullname(cls, model_class=None, spp=None):
        """ Get the fullname from an integer, either model class or spp class.
            names[0] is the genera, names[1] is the species/subspecies class number. """
        col = 'Model output' if spp is None else 'Classes'
        x = model_class if spp is None else spp
        try:
            names = cls.classes.loc[cls.classes[col] == x, ['Genera', 'Classes']].iloc[0]
            return f"[{names[0]} spp:{names[1]}]"
        except IndexError:
            print(f"Error: {col} {x} does not exist in database.")


# =====================================
# References to class methods
DATA_PATH = WIP.DATA_PATH
classes = WIP.classes
load_classes = WIP.load_classes
abbreviations = WIP.abbreviations
get_abbreviation = WIP.get_abbreviation
fullname = WIP.fullname
# =====================================


def load_database(subset=""):
    if not subset:
        filepath = DATA_PATH / "IdentifiantDiptera.xlsx"  # filepath must be an absolute path
        df = pd.read_excel(filepath, dtype={'Picture': str, 'Classes': 'int64'}, sheet_name=0, engine='openpyxl')
    else:
        filepath = DATA_PATH / subset / "database.csv"
        df = pd.read_csv(filepath, dtype={'Picture': str, 'Label': 'int64'})

    print("Loaded dataframe with", len(df), "entries.")
    return df


def load_indices(dataset='spp68'):
    # Import indices
    return (np.loadtxt(DATA_PATH / dataset / 'train_indices.csv', dtype=int),
            np.loadtxt(DATA_PATH / dataset / 'test_indices.csv', dtype=int))


# def load_valid_pictures():
#     # Import a list of valid picture names (no missing image/label).
#     return pd.read_csv(DATA_PATH / 'valid_pictures.csv', header=None).squeeze("columns")


def load_database_split(dataset='spp68', col='Label'):
    # Import picture names or labels from train-test split
    assert col in ['Picture', 'Label']
    column_array = load_database(dataset)[col].to_numpy()
    train_indices, test_indices = load_indices(dataset)
    return column_array[train_indices], column_array[test_indices]


def load_images(dataset='spp68'):
    # Import image data
    # dset must be closed with close() when done
    dset = h5py.File(DATA_PATH / dataset / 'image_data.hdf5', 'r')
    x_train = dset["train"][:]
    x_test = dset["test"][:]
    dset.close()
    return x_train, x_test


if __name__ == '__main__':

    print(DATA_PATH)

    # Testing 'get_abbreviation'
    print(get_abbreviation('culex'))
    print(get_abbreviation('anopheles'))
    print(get_abbreviation('aedes'))
    print(get_abbreviation('unknown species'))
    print(fullname(spp=34))
    print(fullname(model_class=34))

    load_database()
    load_database(subset='spp68')

    ind = load_indices()
    print(ind)

    classes.iloc[0, 0] = 100
    print(classes)

