import os
import pandas as pd
from sklearn.externals import joblib


def load_data(file_name):
    data_path = os.path.join(os.getcwd(), 'data', file_name)
    return pd.read_csv(data_path)


def save_model(model, file_name):
    model_dir = os.path.join(os.getcwd(), 'model')
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    model_path = os.path.join(model_dir, file_name)
    joblib.dump(model, model_path)