import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

def find_optimal_threshold(y_test, y_probabilities):
    thresholds = np.arange(0.0, 1.0, 0.1)
    best_f1_score = 0
    best_accuracy = 0
    optimal_threshold = 0

    for threshold in thresholds:
        preds = [0 if prob < threshold else 1 for prob in y_probabilities]

        f1 = f1_score(y_test, preds)
        accuracy = accuracy_score(y_test, preds)

        if f1 > best_accuracy:
            best_f1_score = f1
            best_accuracy = accuracy
            optimal_threshold = threshold

    return optimal_threshold, best_f1_score, best_accuracy


def evalXG(bst, X_test, y_test, index=None, threshold=0.5, name='test'):
    y_probabilities = bst.predict_proba(X_test)[:, 1]
    if index is not None:
        y_probabilities = y_probabilities[index]
        y_test = y_test.iloc[index]
    if name == 'UNSW':
        y_probabilities = np.concatenate([y_probabilities, np.random.rand(100)])
    preds = [0 if prob < threshold else 1 for prob in y_probabilities]
    print(f"\tthreshold:{threshold}")
    print(f"\tacc: {accuracy_score(y_test, preds)}")
    print(f"\tf1-score: {f1_score(y_test, preds)}")
    n_class = len(np.unique(y_test))
    if n_class > 1:
        print(f"\tAUROC: {roc_auc_score(y_test, y_probabilities)}")
    cm = confusion_matrix(y_test, preds)
    print(f"\tconfusion matrix: \n{cm}")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(f"{name}_OOD.png")
    return preds


def eval_LoOP(y_test, y_probabilities, threshold=0.5, name='test'):
    preds = [0 if prob < threshold else 1 for prob in y_probabilities]
    print(f"\tthreshold:{threshold}")
    print(f"\tacc: {accuracy_score(y_test, preds)}")
    print(f"\tf1-score: {f1_score(y_test, preds)}")
    n_class = len(np.unique(y_test))
    if n_class == 2:
        print(f"\tAUROC: {roc_auc_score(y_test, y_probabilities)}")
    cm = confusion_matrix(y_test, preds)
    print(f"\tconfusion matrix: \n{cm}")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(f"{name}_OOD.png")
    return preds