import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def evaluate_and_plot(model_name, classes, n_classes, outer_cv, grid, X, y):
    accs, precs, recalls, f1s, aucs = [], [], [], [], []
    all_fpr = np.linspace(0, 1, 100)
    mean_tprs = np.zeros((n_classes if n_classes > 2 else 1, 100))
    # Confusion matrix için gerçek ve tahmin edilenleri biriktir
    all_y_true = []
    all_y_pred = []
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        y_test_bin = label_binarize(y_test, classes=classes)

        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        recalls.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1s.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

        # ROC-AUC hesaplama
        if n_classes == 2:
            aucs.append(roc_auc_score(y_test, y_proba[:, 1]))
            fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba[:, 1])
            mean_tprs[0] += np.interp(all_fpr, fpr, tpr)
        else:
            aucs.append(roc_auc_score(y_test_bin, y_proba, average='weighted', multi_class='ovr'))
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                mean_tprs[i] += np.interp(all_fpr, fpr, tpr)
    mean_tprs /= outer_cv.get_n_splits()

    # ROC eğrisi ve AUC boxplot birlikte çiz
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # ROC eğrisi
    if n_classes == 2:
        ax1.plot(all_fpr, mean_tprs[0], label=f'Class {classes[1]} (Mean AUC = {np.mean(aucs):.2f})')
    else:
        for i in range(n_classes):
            ax1.plot(all_fpr, mean_tprs[i], label=f'Class {classes[i]} (Mean AUC = {np.mean(aucs):.2f})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'ROC Curve (OVA) - {model_name}')
    ax1.legend(loc='lower right')

    # AUC boxplot
    ax2.boxplot(aucs, vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax2.set_title(f'AUC Scores Distribution - {model_name}')
    ax2.set_ylabel('AUC Score')
    ax2.set_xticks([1])
    ax2.set_xticklabels([model_name])
    ax2.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(f'ROC_AUC_{model_name}.png')
    plt.close()

    # Confusion matrix çizimi
    cm = confusion_matrix(all_y_true, all_y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax_cm, cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'ConfusionMatrix_{model_name}.png')
    plt.close()

    print(f"Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"F1 Score:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"ROC-AUC:   {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # Yorumlama çıktısı
    print(f"\nModel: {model_name} için AUC skorlarının dağılımı: {aucs}")
    if np.mean(aucs) > 0.8:
        print("Model çok iyi ayırt edici güce sahip (AUC > 0.8).")
    elif np.mean(aucs) > 0.7:
        print("Model iyi ayırt edici güce sahip (AUC > 0.7).")
    else:
        print("Modelin ayırt edici gücü düşük (AUC <= 0.7).")

    with open(f"results_{model_name}.txt", "w") as f:
        f.write(f"Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}\n")
        f.write(f"Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}\n")
        f.write(f"Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}\n")
        f.write(f"F1 Score:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n")
        f.write(f"ROC-AUC:   {np.mean(aucs):.4f} ± {np.std(aucs):.4f}\n")
        f.write(f"AUC Scores: {aucs}\n")