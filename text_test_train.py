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


### Glabal variables
RANDOM_STATE = 42
TEST_SIZE = 0.3
stopWords = set(stopwords.words('english'))
Featured_data_info = {'dtypes': [dtype('O'), dtype('int64'), dtype('int64'), dtype('int64'),dtype('int64')],
                      'names': ['processed', 'length', 'words_not_stopwords', 'commas','mean_length']}
author_mapping_dict = {'EAP': 0, 'HPL': 1, 'MWS': 2}


### User-defined functions & class
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
    """
    process text data with making new features
    """
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
        processed = text.apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
        length = processed.apply(lambda x: len(x))
        words_not_stopwords = processed.apply(lambda x: len([t for t in x.split(' ') if t not in stopWords]))
        commas = text.apply(lambda x: x.count(','))
        mean_length = processed.apply(lambda x: np.mean([len(w) for w in str(x).split()]))
        new = np.column_stack((processed, length, words_not_stopwords, commas, mean_length))
        new_df = pd.DataFrame(new, columns=['processed', 'length', 'words_not_stopwords', 'commas', 'mean_length'])
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


def data_preparer():
    from util.io_functions import load_data

    df = load_data('train_text.csv')
    df = df.drop(['id'],axis=1)
    response = 'author'

    df_X = df.drop(response, axis=1)
    df_y = df[response].copy()
    data_info = extract_data_info_from_dataframe(df_X)
    return df_X, df_y, data_info


df_X, df_y, data_info = data_preparer()



def model_trainer(df_X, df_y, data_info):
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import log_loss
    import xgboost as xgb
    from util.io_functions import save_model

    X = convert_ndarray_to_dataframe(df_X, data_info)
    y = convert_ndarray_to_dataframe(df_y)

    y = y['author'].map(author_mapping_dict)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_STATE)

    for train_index, test_index in split.split(X, y):
        X_train = X.loc[train_index]
        y_train = y.loc[train_index].values.ravel()
        X_test = X.loc[test_index]
        y_test = y.loc[test_index].values.ravel()

    text_pipe = Pipeline([
        ('selector', TextSelector('processed')),
        ('vectorizer', CountVectorizer()),
        ('tfidf_trans', TfidfTransformer())
    ])

    numeric_pipe = Pipeline([
        ('selector', NumericSelector(Featured_data_info)),
        ('scaler', StandardScaler())
    ])

    union = FeatureUnion([
        ('text', text_pipe),
        ('numeric', numeric_pipe)
    ])

    preprocess_pipe = Pipeline([
        ('feature_eng', TextProcess(index_num=0)),
        ('union', union)
    ])

    full_pipeline = Pipeline([
        ('preprocess', preprocess_pipe),
        ('xgb_clf', xgb.XGBClassifier(random_state=RANDOM_STATE))
    ])

    param_grids = {
        'xgb_clf__max_depth': [8],
        'xgb_clf__learning_rate': [0.35],
        'xgb_clf__objective': ['multi:softprob'],
        'xgb_clf__silent': [1],
        'xgb_clf__min_child_weight': [1],
        'xgb_clf__subsample': [0.8],
        'xgb_clf__colsample_bytree': [0.8]
    }

    model = GridSearchCV(full_pipeline, param_grids, cv=3, scoring='neg_log_loss')
    model.fit(X_train, y_train)

    best_model = model.best_estimator_

    pred_train = best_model.predict_proba(X_test)
    mlog_loss = log_loss(y_test, pred_train, labels=sorted(np.unique(y)))

    X = np.r_[X_train, X_test]
    y = np.r_[y_train, y_test]

    best_model.fit(X,y)
    feat_importance = best_model.named_steps['xgb_clf'].feature_importances_
    best_params = model.best_params_

    preprocess_pipe.fit_transform(X)
    vec_feature_names=preprocess_pipe.named_steps['union'].transformer_list[0][1].named_steps['vectorizer'].get_feature_names()

    model_info = {
        'model': best_model,
        'preprocess':best_model.steps,
        'metric': {
            'log_loss': mlog_loss
        },
        'interpretation': {
            'feature_importances': list(sorted(zip(vec_feature_names, feat_importance),key=lambda x:x[1],reverse=True))
        }
    }

    deployed_model = best_model

    save_model(deployed_model, 'xgb_clf.pkl')

    return model_info, deployed_model


model_info, d_model = model_trainer(df_X, df_y, data_info)

print(model_info)
print(d_model)
