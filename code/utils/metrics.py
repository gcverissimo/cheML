import warnings
import numpy as np

import pandas as pd
import rdkit.ML.Scoring.Scoring as rd

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (confusion_matrix, matthews_corrcoef,
                             f1_score, roc_auc_score, accuracy_score,
                             balanced_accuracy_score, precision_score,
                             recall_score, cohen_kappa_score, roc_curve, auc)

# Recall, Sensitivity and True Positive Ratio (TPR) are the same metric.
# Specificity and True Negative Ratio (TNR) are the same metric.
# Precision and Positive Predictive Value (PPV) are the same metric.


class Classification_Metrics:
    def __init__(self, y_true, y_pred, y_score):
        self.y_true = np.round(y_true, 0).astype(int)
        self.y_pred = np.round(y_pred, 0).astype(int)
        self.y_score = y_score
        warnings.simplefilter('ignore', UndefinedMetricWarning)

        # Calculates TN, FP, FN, TP.
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(
            self.y_true, self.y_pred).ravel()

    def confusion_matrix_values(self):
        # Calculates the confusion matrix.
        return self.tn, self.fp, self.fn, self.tp

    def true_negative_score(self):
        # Calculates True Negative (TN).
        return self.tn

    def false_negative_loss(self):
        # Calculates False Negative (FN).
        return self.fn

    def true_positive_score(self):
        # Calculates True Positive (TP).
        return self.tp

    def false_positive_loss(self):
        # Calculates False Positive (FP).
        return self.fp

    def mcc(self):
        # Calculates Matthews Correlation Coefficient:
        return matthews_corrcoef(self.y_true, self.y_pred)

    def f1(self):
        # Calculates F1 Score:
        return f1_score(self.y_true, self.y_pred)

    def roc_auc(self):
        # Calculates ROC-AUC:
        if np.any(np.isnan(self.y_score)):
            return 0.0000
        else:
            return roc_auc_score(self.y_true, self.y_score)

    def rdkit_roc_auc(self):
        if np.any(np.isnan(self.y_score)):
            return 0.0000
        else:
            # Calculates ROC-AUC using RDKit:
            df = pd.DataFrame({"y_true": self.y_true,
                               "y_score": self.y_score})
            score_column = "y_score"
            binary_column = "y_true"
            df[binary_column] = df[binary_column].fillna(0)
            binary_index = list(df.columns).index("y_true")
            df = df.sort_values(score_column, axis=0,
                                ascending=False, ignore_index=True)
            roc_auc = rd.CalcAUC(df.values, col=binary_index)
            return roc_auc

    def acc(self):
        # Calculates Accuracy:
        return accuracy_score(self.y_true, self.y_pred)

    def b_acc(self):
        # Calculates Balanced Accuracy:
        return balanced_accuracy_score(self.y_true, self.y_pred)

    def ppv(self):
        # Calculates Precision/Positive Predictive Value (PPV):
        return precision_score(self.y_true, self.y_pred)

    def tpr(self):
        # Calculates Recall/Sensitivity/True Positive Ratio (TPR):
        return recall_score(self.y_true, self.y_pred)

    def tnr(self):
        # Calculates Specificity/True Negative Ratio (TNR):
        specificity = self.tn/(self.tn+self.fp)
        return specificity

    def ck(self):
        # Calculates Cohen Kappa Score:
        return cohen_kappa_score(self.y_true, self.y_pred)

    def _dict_metrics(self):
        # Calculates all metrics.
        dict_metrics = {
            'MCC': self.mcc(),
            'F1-Score': self.f1(),
            'ROC-AUC': self.roc_auc(),
            'rdROC-AUC': self.rdkit_roc_auc(),
            'ACC': self.acc(),
            'bACC': self.b_acc(),
            'PPV': self.ppv(),
            'TPR': self.tpr(),
            'TNR': self.tnr(),
            'TN': self.tn,
            'FP': self.fp,
            'FN': self.fn,
            'TP': self.tp,
            'CohenKappa': self.ck()
        }
        return dict_metrics

    def _failed_dict_metrics(self):
        dict_metrics = {
            'MCC': 0.0,
            'F1-Score': 0.0,
            'ROC-AUC': 0.0,
            'rdROC-AUC': 0.0,
            'ACC': 0.0,
            'bACC': 0.0,
            'PPV': 0.0,
            'TPR': 0.0,
            'TNR': 0.0,
            'TN': 0.0,
            'FP': 0.0,
            'FN': 0.0,
            'TP': 0.0,
            'CohenKappa': 0.0
        }
        return dict_metrics

    def internal_metrics(self):
        # Defines metric keys as internal metrics.
        dict_metrics = self._dict_metrics()
        int_metrics = {}

        for key, value in dict_metrics.items():
            int_metrics['int_'+key] = value

        return int_metrics

    def external_metrics(self):
        # Defines metric keys as external metrics.
        dict_metrics = self._dict_metrics()
        ext_metrics = {}

        for key, value in dict_metrics.items():
            ext_metrics['ext_'+key] = value

        return ext_metrics

    def failed_internal_metrics(self):
        # Defines metric keys as internal metrics.
        dict_metrics = self._failed_dict_metrics()
        int_metrics = {}

        for key, value in dict_metrics.items():
            int_metrics['int_'+key] = value

        return int_metrics

    def failed_external_metrics(self):
        # Defines metric keys as external metrics.
        dict_metrics = self._failed_dict_metrics()
        ext_metrics = {}

        for key, value in dict_metrics.items():
            ext_metrics['ext_'+key] = value

        return ext_metrics
