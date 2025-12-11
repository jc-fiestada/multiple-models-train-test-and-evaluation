from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

import os, joblib
import LoadData as ld

def BuildPipelines():
    LinearRegressionModel = Pipeline([
        ('LinearRegressScaler', StandardScaler),
        ('LinearRegressModel', LinearRegression())
    ])

    LogisticRegressionModel = Pipeline([
        ('LogRegressScaler', StandardScaler()),
        ('LogRegressModel', LogisticRegression())
    ])

    DecisionTreeClassifierModel = Pipeline([
        ('DTCScaler', StandardScaler()),
        ('DTCModel', DecisionTreeClassifier())
    ])

    DecisionTreeRegressorModel = Pipeline([
        ('DTRScaler', StandardScaler()),
        ('DTCModel', DecisionTreeRegressor())
    ])

    KNNClassifierModel = Pipeline([
        ('KNNClassificationScaler', StandardScaler()),
        ('KNNClassificationModel', KNeighborsClassifier())
    ])

    KNNRegressorModel = Pipeline([
        ('KNNRegScaler', StandardScaler()),
        ('KNNRegModel', KNeighborsRegressor())
    ])

    return {'LinearRegressionModel' : LinearRegressionModel, 
            'LogisticRegressionModel' : LogisticRegressionModel,
            'DecisionTreeClassifierModel' : DecisionTreeClassifierModel, 
            'DecisionTreeRegressorModel' : DecisionTreeRegressorModel,
            'KNNClassifierModel' : KNNClassifierModel,
            'KNNRegressorModel' : KNNRegressorModel
            }

def TrainAndSaveModels():

    return
