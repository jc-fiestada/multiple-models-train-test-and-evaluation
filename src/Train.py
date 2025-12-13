from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

import os, joblib
from . import LoadData as ld

def RegressionPipelines():
    LinearRegressionModel = Pipeline([
        ('LinearRegressScaler', StandardScaler()),
        ('LinearRegressModel', LinearRegression())
    ])

    DecisionTreeRegressorModel = Pipeline([
        ('DTRScaler', StandardScaler()),
        ('DTCModel', DecisionTreeRegressor())
    ])

    KNNRegressorModel = Pipeline([
        ('KNNRegScaler', StandardScaler()),
        ('KNNRegModel', KNeighborsRegressor())
    ])

    return {'LinearRegressionModel' : LinearRegressionModel, 
            'DecisionTreeRegressorModel' : DecisionTreeRegressorModel,
            'KNNRegressorModel' : KNNRegressorModel}

def ClassificationPipelines():
    LogisticRegressionModel = Pipeline([
        ('LogRegressScaler', StandardScaler()),
        ('LogRegressModel', LogisticRegression())
    ])

    DecisionTreeClassifierModel = Pipeline([
        ('DTCScaler', StandardScaler()),
        ('DTCModel', DecisionTreeClassifier())
    ])

    KNNClassifierModel = Pipeline([
        ('KNNClassificationScaler', StandardScaler()),
        ('KNNClassificationModel', KNeighborsClassifier())
    ])

    return {'LogisticRegressionModel' : LogisticRegressionModel,
            'DecisionTreeClassifierModel' : DecisionTreeClassifierModel, 
            'KNNClassifierModel' : KNNClassifierModel}


def TrainAndSaveModels(test_size: float=0.25, random_state: int=42):

    dataPrice = ld.LoadDataPrice() #continuous
    dataLikeability = ld.LoadDataLikeability() # classification
    valueMapping = ld.LoadMapping()

    # split x and y values
    X_price = dataPrice[['rarity', 'age', 'condition', 'popularity']]
    y_price = dataPrice['price']

    X_likeability = dataLikeability[['age', 'plays_games', 'owns_cards', 'watches_anime']]
    y_likeability = dataLikeability['likes_pokemon']

    # split train and test values
    X_price_train, X_price_test, y_price_train, y_price_test = train_test_split(X_price, y_price, test_size=test_size, random_state=random_state)
    X_likeability_train, X_likeability_test, y_likeability_train, y_likeability_test = train_test_split(X_likeability, y_likeability, test_size=test_size, random_state=random_state)

    CPipelines = ClassificationPipelines()
    RPipelines = RegressionPipelines()

    for models in CPipelines.values():
        models.fit(X_likeability_train, y_likeability_train)

    for models in RPipelines.values():
        models.fit(X_price_train, y_price_train)

    os.makedirs('../models/classification', exist_ok=True)
    os.makedirs('../models/regression', exist_ok=True)

    for modelName, model in CPipelines.items():
        filename = f'{modelName}.pkl'
        filepath = os.path.join('../models/classification/', filename)
        joblib.dump(model, filepath)
    
    for modelName, model in RPipelines.items():
        filename = f'{modelName}.pkl'
        filepath = os.path.join('../models/regression/', filename)
        joblib.dump(model, filepath)

    return valueMapping, (X_likeability_test, y_likeability_test), (X_price_test, y_price_test)
