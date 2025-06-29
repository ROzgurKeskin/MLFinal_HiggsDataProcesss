from matplotlib import pyplot as plt
import pandas as pd
from sklearn.base import accuracy_score
from sklearn.calibration import label_binarize
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import SVC
from xgboost import XGBClassifier

from library import getSelectedFeatureDataFrame

# Load the dataset generated after feature selection

# Main function for nested cross-validation
def nested_cv():
    # Load data
    data = getSelectedFeatureDataFrame()
    X = data.drop(columns=['Limit'])
    y = data['Limit']
    classes = np.unique(y)
    y_bin = label_binarize(y, classes=classes)
    n_classes = y_bin.shape[1]

    # Define the model and pipeline
   # Model ve parametreler
    models = {
        'KNN': (
            KNeighborsClassifier(),
            {
                'feature_selection__k': [5, 10, 15],
                'classifier__n_neighbors': list(range(3, 12, 2))
            }
        ),
        'SVM': (
            SVC(),
            {
                'feature_selection__k': [5, 10, 15],
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf']
            }
        ),
        'MLP': (
            MLPClassifier(max_iter=300, random_state=42),
            {
                'feature_selection__k': [5, 10, 15],
                'classifier__hidden_layer_sizes': [(50,), (100,)],
                'classifier__activation': ['relu', 'tanh']
            }
        ),
        'XGBoost': (
            XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            {
                'feature_selection__k': [5, 10, 15],
                'classifier__n_estimators': [50, 100],
                'classifier__max_depth': [3, 6]
            }
        )
    }
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for model_name, (clf, param_grid) in models.items():
        print(f"\nModel: {model_name}")
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(score_func=f_classif)),
            ('classifier', clf)
        ])
        grid = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring='accuracy', n_jobs=-1)
         # Metrikleri toplamak için listeler
        accs, precs, recalls, f1s, aucs = [], [], [], [], []
        all_fpr = np.linspace(0, 1, 100)
        mean_tprs = np.zeros((n_classes, 100))
        for train_idx, test_idx in outer_cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            y_test_bin = label_binarize(y_test, classes=classes)

            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)

            accs.append(accuracy_score(y_test, y_pred))
            precs.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            recalls.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            f1s.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            aucs.append(roc_auc_score(y_test_bin, y_proba, average='weighted', multi_class='ovr'))

            # ROC eğrileri için
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                mean_tprs[i] += np.interp(all_fpr, fpr, tpr)
                
        # Ortalama ROC eğrisi
        mean_tprs /= outer_cv.get_n_splits()
        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            plt.plot(all_fpr, mean_tprs[i], label=f'Class {classes[i]} (AUC = {np.mean(aucs):.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (OVA) - {model_name}')
        plt.legend(loc='lower right')
        plt.show()

        print(f"Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
        print(f"Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
        print(f"F1 Score:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"ROC-AUC:   {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")


if __name__ == "__main__":
    nested_cv()