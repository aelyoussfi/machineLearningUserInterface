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
import joblib

import time
#sns.set()


def csv_to_json(l1,l2):
    mydict = {}
    if len(l1) == len(l2):
        for rank in range(len(l1)):
            mydict[l1[rank]] = l2[rank]
        return mydict
    else:
        return {"data":"no data"}
    

class MLFact():
    def __init__(self):
        train = pd.read_csv('brain/train.csv')
        test = pd.read_csv('brain/test.csv')
        X = train.filter(['store', 'product'])
        Y = train.filter(["number_sold"])
        self.x_train,self.x_test, self.y_train, self.y_test = train_test_split(X,Y,test_size=0.25)
        
    def get_interval(self,train_path):
        train = pd.read_csv(train_path)
        return train['Date'].tolist()[0], train['Date'].tolist()[-1]
    
    def get_test_X(self,start_date,end_date):
        train = pd.read_csv('brain/train.csv')
        train["Date"] = pd.to_datetime(train["Date"])
        print(train.dtypes)
        print(train)
        test_X = train[(train['Date']>= pd.to_datetime(start_date)) & (train['Date']<= pd.to_datetime(end_date))]
        self.test = test_X
        self.test_Y = test_X.filter(["number_sold"])
        self.test_X = test_X.filter(['store', 'product'])
        self.test_Y = test_X.filter(["number_sold"])
        print('test_Y :',self.test_Y)
        return self.test_X,self.test_Y,self.test

    def get_models_list(self):
        self.models = {}
        self.models["KNeighbors"] = KNeighborsRegressor(n_neighbors=5)
        self.models["DecisionTree"] = DecisionTreeRegressor()
        self.models["Gradient Boosting"] = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
        self.models["MLP"] = MLPRegressor()
        self.models["RandomForest"] = RandomForestRegressor()
        self.models["AdaBoost"] = AdaBoostRegressor(n_estimators=100, random_state=0)
        self.models["Grad Boosting"] = GradientBoostingRegressor(n_estimators=120, learning_rate=0.05,max_depth=8, random_state=0)
        return self.models
    
    def fitting(self,mlmodel,x_train, y_train):
        t1 = time.time()
        list_models = self.get_models_list()
        m = list_models[mlmodel].fit(x_train,y_train.values.ravel()) 
        self.train_score = r2_score(y_train,m.predict(x_train))
        t2 = time.time()
        self.T = t2 - t1
        joblib.dump(m, 'brain/modelscontainer/model'+str(self.models[mlmodel])+'.pkl')
        print('training of : '+str(self.models[mlmodel])+' is done!')
        return self.T, self.train_score
    
    def inference(self,mlmodel,start_date,end_date):
        list_models = self.get_models_list()
        self.test_X,self.test_Y,self.test = self.get_test_X(start_date,end_date)
        
        loaded_model = joblib.load('brain/modelscontainer/model'+str(list_models[mlmodel])+'.pkl')

        prediction = loaded_model.predict(self.test_X).flatten()
        prediction.tolist() 
        print('prediction :',prediction)
        print('test_y :',self.test_Y)
        self.test_score = r2_score(self.test_Y,prediction)
        #pred = pd.DataFrame({"id":test['id'],"Machine failure":prediction.tolist()})
        self.f = csv_to_json([elt.strftime("%Y-%m-%d") for elt in self.test["Date"].tolist()],prediction)
        return self.f,self.test_score 