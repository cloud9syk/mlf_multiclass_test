from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
import os
import datetime
import pandas as pd
import numpy as np
import re
from numpy import dtype
from nltk.corpus import stopwords

### Glabal variables
RANDOM_STATE = 42
TEST_SIZE = 0.3
stopWords = set(stopwords.words('english'))
Featured_data_info = {'dtypes': [dtype('O'), dtype('int64'), dtype('int64'), dtype('int64'),dtype('int64')],
                      'names': ['text', 'length', 'words_not_stopwords', 'commas','mean_length']}
author_mapping_dict = {'EAP': 0, 'HPL': 1, 'MWS': 2}

def data_preparer():
    from util.io_functions import load_data
    from util.defined_class import extract_data_info_from_dataframe
    from sklearn.model_selection import StratifiedShuffleSplit

    df = load_data('train_text.csv')
    df = df.drop(['id'],axis=1)
    response = 'author'

    df_X = df.drop(response, axis=1)
    df_y = df[response].copy()
    data_info = extract_data_info_from_dataframe(df_X)
    #split = StratifiedShuffleSplit(n_splits=3, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    return df_X, df_y, data_info

df_X, df_y, data_info = data_preparer()



def model_trainer(df_X, df_y, data_info):
    from util.defined_class import convert_ndarray_to_dataframe, TextSelector, NumericSelector
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import log_loss
    import lightgbm as lgb
    from util.io_functions import save_model
    from util.defined_class import TextProcess


    X = convert_ndarray_to_dataframe(df_X, data_info)
    y = convert_ndarray_to_dataframe(df_y)

    y = y['author'].map(author_mapping_dict)

    split = StratifiedShuffleSplit(n_splits=3, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    for train_index, test_index in split.split(X, y):
        X_train = X.loc[train_index]
        y_train = y.loc[train_index].values.ravel()
        X_test = X.loc[test_index]
        y_test = y.loc[test_index].values.ravel()

    text_pipe = Pipeline([
        ('selector', TextSelector('text')),
        ('vectorizer', CountVectorizer()),
        ('tfidf_trans', TfidfTransformer())
    ])

    numeric_pipe = Pipeline([
        ('selector', NumericSelector(Featured_data_info)),
        ('scaler', StandardScaler())
    ])
    """
    union = FeatureUnion([
        ('text',text_pipe),
        ('numeric',numeric_pipe)
    ])

    preprocess_pipe = Pipeline([
        ('feature_eng', TextProcess(index_num=0)),
        ('union',union)
    ])
    """

    preprocess_pipe = Pipeline([
        ('feature_eng', TextProcess(index_num=0)),
        ('union', FeatureUnion([
            ('txt', text_pipe),
            ('numeric', numeric_pipe)
        ]))
    ])


    full_pipeline = Pipeline([
        ('preprocess', preprocess_pipe),
        ('lgb_clf', lgb.LGBMClassifier(random_state=RANDOM_STATE))
    ])


    param_grids = {
        'lgb_clf__boosting_type':['gbdt'],
        'lgb_clf__objective':['multiclass'],
        'lgb_clf__num_leaves':[50],
        'lgb_clf__min_child_samples':[10],
        'lgb_clf__max_depth':[50],
        'lgb_clf__learning_rate':[0.1,0.15]
    }

    lgb_model = GridSearchCV(full_pipeline, param_grids, cv=3, scoring='neg_log_loss')
    lgb_model.fit(X_train, y_train)

    best_model = lgb_model.best_estimator_

    pred_train = best_model.predict_proba(X_test)
    mlog_loss = log_loss(y_test, pred_train, labels=sorted(np.unique(y)))

    feat_importances = best_model.named_steps['lgb_clf'].feature_importances_
    best_params = lgb_model.best_params_

    preprocess_pipe.fit_transform(X)
    vec_feature_names = preprocess_pipe.named_steps['union'].transformer_list[0][1].named_steps[
        'vectorizer'].get_feature_names()
    zipped = list(zip(vec_feature_names, feat_importances))
    filtered = list(filter(lambda x: x[1] >= 1, zipped))

    model_info = {
        'model': best_model,
        'preprocess': best_model.steps,
        'metric': {
            'log_loss': mlog_loss
        },
        'interpretation': {
            'feature_importances': list(sorted(filtered,key=lambda x:x[1],reverse=True))
        }
    }

    deployed_model = best_model

    save_model(deployed_model, 'lgb_clf.pkl')


    return model_info, deployed_model

model_in, dep_model = model_trainer(df_X, df_y, data_info)

print(model_in)
