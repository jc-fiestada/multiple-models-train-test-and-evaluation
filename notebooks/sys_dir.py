import os, sys

base_dir = os.path.dirname(os.path.dirname(__file__))
models_dir = os.path.join(base_dir, 'models')

classification_dir = os.path.join(models_dir, 'classification')
regression_dir = os.path.join(models_dir, 'regression')

# classification file path
DTCM = os.path.join(classification_dir, 'DecisionTreeClassifierModel.pkl')
KNNC = os.path.join(classification_dir, 'KNNClassifierModel.pkl')
LogRM = os.path.join(classification_dir, 'LogisticRegressionModel.pkl')

# regression file path
DTRM = os.path.join(regression_dir, 'DecisionTreeRegressorModel.pkl')
KNNR = os.path.join(regression_dir, 'KNNRegressorModel.pkl')
LinRM = os.path.join(regression_dir, 'LinearRegressionModel.pkl')