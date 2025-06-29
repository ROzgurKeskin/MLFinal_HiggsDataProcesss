import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from xgboost import XGBClassifier
from library import getSelectedFeatureDataFrame
from ModelUtils import evaluate_and_plot

def main():
    #data = getSelectedFeatureDataFrame() 
    data = getSelectedFeatureDataFrame().sample(n=10000, random_state=42) #sonuçların daha hızlı gelmesi için 
    X = data.drop(columns=['Limit'])
    y = data['Limit']
    classes = np.unique(y)
    n_classes = len(classes)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('classifier', XGBClassifier(eval_metric='logloss', random_state=42))
    ])
    param_grid = {
        'feature_selection__k': [5], # sonuçların daha hızlı gelmesi için 
        #'feature_selection__k': [5,10,15], 
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [3, 6]
    }
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring='accuracy', n_jobs=-1)

    evaluate_and_plot("XGB", classes, n_classes, outer_cv, grid, X, y)

if __name__ == "__main__":
    main()