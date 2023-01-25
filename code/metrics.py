import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer, cohen_kappa_score

# Recall, Sensitivity and True Positive Ratio (TPR) are the same metric.
# Specificity and True Negative Ratio (TNR) are the same metric.
# Precision and Positive Predictive Value (PPV) are the same metric.

# Calculates TN, FP, FN, TP.
def confusion_matrix_values(y_true, y_pred):
    y_true = np.round(y_true, 0).astype(int)
    y_pred = np.round(y_pred, 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn, fp, fn, tp

# Calculates True Negative (TN).
def true_negative_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix_values(y_true, y_pred)
    return tn

# Calculates False Negative (FN).
def false_negative_loss(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix_values(y_true, y_pred)
    return fn

# Calculates True Positive (TP).
def true_positive_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix_values(y_true, y_pred)
    return tp

# Calculates False Positive (FP).
def false_positive_loss(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix_values(y_true, y_pred)
    return fp

# Calculates Specificity/True Negative Ratio (TNR):
def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix_values(y_true, y_pred)
    specificity = tn/(tn+fp)
    return specificity

# Sets the scoring_functions object for GridSearchCV:
def internal_validation_metrics():
    fn_loss = make_scorer(true_negative_score, greater_is_better=False)
    fp_loss = make_scorer(false_positive_loss, greater_is_better=False)
    tn_score = make_scorer(false_negative_loss, greater_is_better=True)
    tp_score = make_scorer(true_positive_score, greater_is_better=True)
    specificity_score = make_scorer(specificity, greater_is_better=True)
    ck_score = make_scorer(cohen_kappa_score, greater_is_better=True)

    scoring_functions = {
        'int_MCC':'matthews_corrcoef', 'int_F1-Score':'f1', 'int_ROC-AUC': 'roc_auc',
        'int_ACC':'accuracy', 'int_bACC':'balanced_accuracy',
        'int_PPV':'precision', 'int_TPR':'recall', 'int_TNR':specificity_score,
        'int_TN':tn_score, 'int_FP':fp_loss, 'int_FN': fn_loss, 'int_TP':tp_score,
        'int_CohenKappa':ck_score
    }
    return scoring_functions


def external_validation_metrics():
    fn_loss = make_scorer(true_negative_score, greater_is_better=False)
    fp_loss = make_scorer(false_positive_loss, greater_is_better=False)
    tn_score = make_scorer(false_negative_loss, greater_is_better=True)
    tp_score = make_scorer(true_positive_score, greater_is_better=True)
    specificity_score = make_scorer(specificity, greater_is_better=True)
    ck_score = make_scorer(cohen_kappa_score, greater_is_better=True)

    scoring_functions = {
        'ext_MCC':'matthews_corrcoef', 'ext_F1-Score':'f1', 'ext_ROC-AUC': 'roc_auc',
        'ext_ACC':'accuracy', 'ext_bACC':'balanced_accuracy',
        'ext_PPV':'precision', 'ext_TPR':'recall', 'ext_TNR':specificity_score,
        'ext_TN':tn_score, 'ext_FP':fp_loss, 'ext_FN': fn_loss, 'ext_TP':tp_score,
        'ext_CohenKappa':ck_score
    }
    return scoring_functions