#importing packages.

import seaborn as sns
import pandas as pd
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
scaler = StandardScaler()
#scaler = MinMaxScaler()
label_e=preprocessing.LabelEncoder()
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
# from xgboost import XGBRegressor,XGBClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_log_error
from sklearn import preprocessing
from scipy.stats import uniform, randint
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import time
#sns.set()

def datagenerator(train_path,test_path,start_date,end_date):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X = train.filter(['store', 'product'])
    Y = train.filter(["number_sold"])
    test_X = train.filter((train['Date']>=start_date) and (train['Date']<=end_date))#test.filter(['store', 'product'])
    x_train,x_test, y_train, y_test = train_test_split(X,Y,test_size=0.33)
    return x_train, x_test, y_train, y_test,test_X

def get_interval(train_path):
    train = pd.read_csv(train_path)
    return train['Date'].tolist()[0], train['Date'].tolist()[-1]

def models_list():
    models = {}
    models["KNeighbors"] = KNeighborsRegressor(n_neighbors=13)
    models["DecisionTree"] = DecisionTreeRegressor()
    models["Gradient Boosting"] = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    models["MLP"] = MLPRegressor()
    models["RandomForest"] = RandomForestRegressor()
    models["AdaBoost"] = AdaBoostRegressor(n_estimators=100, random_state=0)
    models["Grad Boosting"] = GradientBoostingRegressor(n_estimators=120, learning_rate=0.05,max_depth=8, random_state=0)
    return models

def test_models(model,x_train,y_train,x_test,y_test):
    np.seterr(divide='ignore', invalid='ignore')
    train_score = []
    test_score = []
    t1 = time.time()
    m = model.fit(x_train,y_train)
    train_score = r2_score(y_train,m.predict(x_train))
    test_score = r2_score(y_test,m.predict(x_test))
    t2 = time.time()
    T= t2 - t1
    return train_score,test_score,T 

def csv_to_json(l1,l2):
    mydict = {}
    if len(l1) == len(l2):
        for rank in range(len(l1)):
            mydict[l1[rank]] = l2[rank]
        return mydict
    else:
        return {"data":"no data"}

def serialized_data(mymodel,X,Y,test_X,test):
    mym = mymodel.fit(X,Y) 
    prediction = mym.predict(test_X)
    prediction.tolist() 
    #pred = pd.DataFrame({"id":test['id'],"Machine failure":prediction.tolist()})
    f = csv_to_json([elt.strftime("%Y-%m-%d %H:%M:%S") for elt in test["Date"].tolist()],prediction)
    return f
    
            