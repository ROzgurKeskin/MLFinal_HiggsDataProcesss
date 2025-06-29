import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from library import getSelectedFeatureDataFrame
from ModelUtils import evaluate_and_plot

def main():
    data = getSelectedFeatureDataFrame().sample(n=10000, random_state=42)
    #data = getSelectedFeatureDataFrame() Sonuçlarn daha hızlı gelmesi için
    X = data.drop(columns=['Limit'])
    y = data['Limit']
    classes = np.unique(y)
    n_classes = len(classes)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('classifier', SVC(probability=True))
    ])
    param_grid = {
        'feature_selection__k': [5],
        'classifier__C': [0.1],
        'classifier__kernel': ['linear']
        #'feature_selection__k': [5,10,15],
        #'classifier__C': [0.1,1,10],onuçlarn daha hızlı gelmesi için
        #'classifier__kernel': ['linear', 'rbf'] Sonuçlarn daha hızlı gelmesi için
        
    }
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring='accuracy', n_jobs=-1)

    evaluate_and_plot("SVM", classes, n_classes, outer_cv, grid, X, y)

if __name__ == "__main__":
    main()