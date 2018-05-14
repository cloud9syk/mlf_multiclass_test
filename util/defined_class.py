from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
import os
import datetime
import pandas as pd
import numpy as np
import re
from numpy import dtype
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))


# pandas df에서 data_info를 뽑아냄
def extract_data_info_from_dataframe(df):
    data_info = dict()
    dtypes_dict = df.dtypes.to_dict()
    data_info['names'] = df.dtypes.index.tolist()
    data_info['dtypes'] = df.dtypes.tolist()
    return data_info


# data_info에서, dtype에 속하는 컬럼만 택하여 그 컬럼 인덱스 리스트를 리턴
# flip=True일 경우 해당 dtype에 속하지 않는 컬럼을 택함
def get_indices_by_dtype(data_info, dtype, flip=False):
    import numpy as np

    if not flip:
        filtered = filter(lambda x: np.issubdtype(x[1], np.number),
                          enumerate(data_info['dtypes']))
    else:
        filtered = filter(lambda x: not np.issubdtype(x[1], np.number),
                          enumerate(data_info['dtypes']))

    indices = list(map(lambda x: x[0], filtered))

    return indices


# 숫자인 컬럼 인덱스 리스트를 리턴 (flip=True일 경우 숫자 아닌(카테고리) 컬럼 인덱스 리스트를 리턴)
def get_indices_dtype_number(data_info, flip=False):
    import numpy as np
    return get_indices_by_dtype(data_info, np.number, flip)


# ndarray에 data_info가 있으면 dataframe을 만들 수 있음
def convert_ndarray_to_dataframe(X, data_info=None):
    from pandas import DataFrame

    X = DataFrame(X)

    if data_info is None:
        return X

    X.columns = data_info['names']
    dtypes_dict = dict(zip(data_info['names'], data_info['dtypes']))
    X = X.astype(dtypes_dict)
    return X


class TextProcess(BaseEstimator, TransformerMixin):
    def __init__(self, index_num):
        self.index_num = index_num

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if type(X) == pd.core.frame.DataFrame:
            pass
        else:
            X = pd.DataFrame(X)
        text = X.iloc[:, self.index_num]
        processed = text.apply(lambda x: re.sub(r'[^\w\s]', '',x.lower()))
        length = processed.apply(lambda x: len(x))
        words_not_stopwords = processed.apply(lambda x: len([t for t in x.split(' ') if t not in stopWords]))
        commas = processed.apply(lambda x: x.count(','))
        mean_length = processed.apply(lambda x: np.mean([len(w) for w in str(x).split()]))
        new = np.column_stack((processed, length, words_not_stopwords, commas, mean_length))
        new_df = pd.DataFrame(new, columns=['text', 'length', 'words_not_stopwords', 'commas', 'mean_length'])
        new_df[['length', 'words_not_stopwords', 'commas', 'mean_length']] = new_df[['length',
                                                                                     'words_not_stopwords',
                                                                                     'commas',
                                                                                     'mean_length']].apply(pd.to_numeric)
        return new_df




class NumericSelector(BaseEstimator, TransformerMixin):
    def __init__(self, data_info, flip=False, array=False):
        self.data_info = data_info
        self.flip = flip
        self.array = array

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        indices = get_indices_dtype_number(self.data_info, self.flip)
        if self.array == False:
            return X.iloc[:, indices]
        else:
            return np.array(X.iloc[:, indices])


class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data for the purpose of ma
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]

