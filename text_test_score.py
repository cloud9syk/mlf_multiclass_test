from sklearn.externals import joblib
from sklearn.metrics import log_loss
import os
import util.defined_class
from util.defined_class import TextProcess, TextSelector, NumericSelector, extract_data_info_from_dataframe, convert_ndarray_to_dataframe
from util.io_functions import load_data
from numpy import dtype
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

RANDOM_STATE = 42
TEST_SIZE = 0.3
stopWords = set(stopwords.words('english'))
Featured_data_info = {'dtypes': [dtype('O'), dtype('int64'), dtype('int64'), dtype('int64'),dtype('int64')],
                      'names': ['processed', 'length', 'words_not_stopwords', 'commas','mean_length']}


def data_preparer():
    df = load_data('test_text.csv')
    df = df.drop(['id'], axis=1)
    data_info = extract_data_info_from_dataframe(df)
    return df, data_info

test, test_data_info = data_preparer()


def scorer(data, data_info):
    data_path = os.path.join(os.getcwd(), 'model', 'xgb_clf.pkl')
    model = joblib.load(data_path)
    print(model)
    scoring = model.predict_proba(data)
    return scoring

scoring = scorer(test, test_data_info)

print(scoring)
