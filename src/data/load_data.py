# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from features.process_features import standardize

PROJECT_DIR = os.getcwd()

RAW_DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'raw')
DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')

# Raw data directories for Malawi and Indonesia poverty data
MWI_DIR = os.path.join(RAW_DATA_DIR, 'KCP2017_MP', 'KCP_ML_MWI')
IDN_DIR = os.path.join(RAW_DATA_DIR, 'KCP2017_MP', 'KCP_ML_IDN')
RIAU_DIR = os.path.join(RAW_DATA_DIR, 'SSN2020_RIAU')

MWI_HOUSEHOLD = os.path.join(MWI_DIR, 'MWI_2012_household.dta')
MWI_INDIVIDUAL = os.path.join(MWI_DIR, 'MWI_2012_individual.dta')

IDN_HOUSEHOLD = os.path.join(IDN_DIR, 'IDN_2012_household.dta')
IDN_INDIVIDUAL = os.path.join(IDN_DIR, 'IDN_2012_individual.dta')
IDN_EXPENDITURE = os.path.join(RAW_DATA_DIR, 'IDN_2012', 'IDN2012_expenditure.dta')

# additional data for testing idn models

IDN_2011_DIR = os.path.join(RAW_DATA_DIR, 'IDN_2011')
IDN_2011_HOUSEHOLD = os.path.join(IDN_2011_DIR, 'IDN2011_household.dta')
IDN_2011_INDIVIDUAL = os.path.join(IDN_2011_DIR, 'IDN2011_individual.dta')

IDN_2013_DIR = os.path.join(RAW_DATA_DIR, 'IDN_2013')
IDN_2013_HOUSEHOLD = os.path.join(IDN_2013_DIR, 'IDN2013_household.dta')
IDN_2013_INDIVIDUAL = os.path.join(IDN_2013_DIR, 'IDN2013_individual.dta')

IDN_2014_DIR = os.path.join(RAW_DATA_DIR, 'IDN_2014')
IDN_2014_HOUSEHOLD = os.path.join(IDN_2014_DIR, 'IDN2014_household.dta')
IDN_2014_INDIVIDUAL = os.path.join(IDN_2014_DIR, 'IDN2014_individual.dta')

COUNTRY_LIST = ['riau', 'mwi', 'mwi-competition', 'idn', 'idn-2011', 'idn-2013', 'idn-2014']
CATEGORY_LIST = ['train', 'test', 'questions']


def get_country_filepaths(country):
    if country not in COUNTRY_LIST:
        raise ValueError("{} not one of the countries we cover, which are {}".format(country, COUNTRY_LIST))

    country_dir = os.path.join(DATA_DIR, country)

    if not os.path.exists(country_dir):
        os.makedirs(country_dir)

    return (os.path.join(country_dir, 'train.csv'),
            os.path.join(country_dir, 'test.csv'))


def split_features_labels_weights(path,
                                  weights=['weind', 'fwt', 'renum'],
                                  weights_col=['weind'],
                                  label_col=['miskin']):
    '''Split data into features, labels, and weights dataframes'''
    data = pd.read_csv(path)
    return (data.drop(weights + label_col, axis=1),
            data[label_col],
            data[weights_col])


def load_data(path, selected_columns=None, ravel=True, standardize_columns='numeric'):
    X, y, w = split_features_labels_weights(path)
    if selected_columns is not None:
        X = X[[col for col in X.columns.values if col in selected_columns]]
    if standardize_columns == 'numeric':
        standardize(X)
    elif standardize_columns == 'all':
        standardize(X, numeric_only=False)
    if ravel is True:
        y = np.ravel(y)
        w = np.ravel(w)
    return (X, y, w)
