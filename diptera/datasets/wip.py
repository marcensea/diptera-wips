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
    def fullname(cls, x):
        """ Get the fullname from class integer """
        return f"[{cls.classes.loc[cls.classes['Classes'] == x, 'Genera'].iloc[0]} class:{x}]"


# =====================================
# References to class methods
DATA_PATH = WIP.DATA_PATH
abbreviations = WIP.abbreviations
classes = WIP.classes
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


def load_pictures_names():
    # Import a list of valid picture names (no missing image/label).
    return pd.read_csv(DATA_PATH / 'valid_pictures.csv', header=None).squeeze("columns")


def open_dataset(dataset='spp68'):
    # Import image data
    # Must be closed with dset.close() when done
    return h5py.File(DATA_PATH / dataset / 'image_data.hdf5', 'r')


if __name__ == '__main__':

    print(DATA_PATH)

    # Testing 'get_abbreviation'
    print(get_abbreviation('culex'))
    print(get_abbreviation('anopheles'))
    print(get_abbreviation('aedes'))
    print(get_abbreviation('unknown species'))
    print(fullname(34))

    load_database()
    load_database(subset='spp68')

    ind = load_indices()
    print(ind)

    classes.iloc[0, 0] = 100
    print(classes)
