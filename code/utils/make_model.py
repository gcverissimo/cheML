import warnings
import copy
import pickle

import numpy as np

from sklearn.model_selection import (StratifiedKFold, PredefinedSplit)
from utils.metrics import Classification_Metrics
from sklearn.exceptions import ConvergenceWarning


class Model:
    def __init__(self, X_train, Y_train, X_test, Y_test, clf, params):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.clf = clf
        self.params = params

    def val_5fold(self):
        # Does the internal validation and returns the metric values.

        # Sets the 5-Fold stratifier.
        split = StratifiedKFold(n_splits=5, shuffle=False)
        split_indices = split.split(self.X_train, self.Y_train)
        fold5_metrics = {}
        have_convergence_warn = False
        have_runtime_warns = False

        for i, (train5_index, test5_index) in enumerate(split_indices):
            # (5-fold) Split iteration for training set.
            cv5_X_train = self.X_train[train5_index]
            cv5_Y_train = self.Y_train[train5_index]

            # (5-fold) Split iteration for test set.
            cv5_X_test = self.X_train[test5_index]
            cv5_Y_test = self.Y_train[test5_index]

            # Catch warnings during model fit and predict.
            with warnings.catch_warnings(record=True) as caught_warnings:
                # Model fit and predict.
                clf_5fold = copy.deepcopy(self.clf)
                try:
                    clf_5fold.fit(cv5_X_train, cv5_Y_train)
                    cv5_predY_test = clf_5fold.predict(cv5_X_test)
                    cv5_scoreY_test = clf_5fold.predict_proba(cv5_X_test)[:, 1]
                except ValueError:
                    # Returns ValueError without breaking the code.
                    # Only useful for MLP using SGD solver.
                    # Due to solver producing non-finite parameter weights
                    # Otherwise, look for problems with your data.
                    metrics = Classification_Metrics(y_true=[0, 0],
                                                     y_pred=[1, 1],
                                                     y_score=[1, 1])
                    internal_val = metrics.failed_internal_metrics()
                    mean_std_5_fold = {}
                    for key, value in internal_val.items():
                        key_mean = 'mean_'+key
                        mean_std_5_fold[key_mean] = value

                        key_std = 'std_'+key
                        mean_std_5_fold[key_std] = value
                    mean_std_5_fold['have_int_convergence_warn'] = True
                    mean_std_5_fold['have_int_runtime_warns'] = 'ValueError'
                    return mean_std_5_fold

            for warn in caught_warnings:
                if warn.category == ConvergenceWarning:
                    have_convergence_warn = True
                elif warn.category == RuntimeWarning:
                    have_runtime_warns = True
                else:
                    print(warn.category, warn.message)

            # 5-fold metrics calculation.
            metrics = Classification_Metrics(y_true=cv5_Y_test,
                                             y_pred=cv5_predY_test,
                                             y_score=cv5_scoreY_test)
            internal_val = metrics.internal_metrics()
            for key, value in internal_val.items():
                key_split = 'split_'+key
                if key_split not in fold5_metrics:
                    fold5_metrics[key_split] = []
                    fold5_metrics[key_split].append(value)
                else:
                    fold5_metrics[key_split].append(value)

        # Mean and std calculation for internal validation metrics.
        mean_std_5_fold = {}
        for key in fold5_metrics.keys():
            key_mean = key.replace('split_', 'mean_')
            mean = np.mean(fold5_metrics[key])
            mean_std_5_fold[key_mean] = mean

            key_std = key.replace('split_', 'std_')
            std = np.std(fold5_metrics[key])
            mean_std_5_fold[key_std] = std

        # Keep track of any Convergence warnings
        # that occurred during model training and,
        # if any, reset the value of metrics.
        if have_convergence_warn is True:
            for key in mean_std_5_fold.keys():
                mean_std_5_fold[key] = 0
            mean_std_5_fold['have_int_convergence_warn'] = True
        else:
            mean_std_5_fold['have_int_convergence_warn'] = False

        # Keep track of any RunTime warnings. This part was
        # implemented due to matmul, reduce and add warnings.
        # Examples: "overflow encountered in matmul" and
        # "invalid value encountered in matmul".
        # This was observed with 'sgd' solver for MLPClassifier.
        if have_runtime_warns is True:
            mean_std_5_fold['have_int_runtime_warns'] = True
        else:
            mean_std_5_fold['have_int_runtime_warns'] = False
        return mean_std_5_fold

    def val_ext(self, save_model=False):
        # Does the external validation and returns the metric values.
        X_dataset = np.concatenate((self.X_train, self.X_test))
        Y_dataset = np.concatenate((self.Y_train, self.Y_test))

        train_indices = np.full((len(self.X_train), ), -1, dtype=int)
        test_indices = np.full((len(self.X_test), ), 0, dtype=int)
        dataset_indices = np.concatenate((train_indices, test_indices))

        ps = PredefinedSplit(dataset_indices)
        ext_metrics = {}
        have_convergence_warn = False
        have_runtime_warns = False

        for i, (trainext_index, testext_index) in enumerate(ps.split()):
            # (External) Split iteration for training set.
            ext_X_train = X_dataset[trainext_index]
            ext_Y_train = Y_dataset[trainext_index]

            # (External) Split iteration for test set.
            ext_X_test = X_dataset[testext_index]
            ext_Y_test = Y_dataset[testext_index]

            # Catch warnings during model fit and predict.
            with warnings.catch_warnings(record=True) as caught_warnings:
                # Model fit and predict.
                ext_clf = copy.deepcopy(self.clf)
                try:
                    ext_clf.fit(ext_X_train, ext_Y_train)
                    ext_predY_test = ext_clf.predict(ext_X_test)
                    ext_scoreY_test = ext_clf.predict_proba(ext_X_test)[:, 1]
                except ValueError:
                    # Returns ValueError without breaking the code.
                    # Only useful for MLP using SGD solver.
                    # Due to solver producing non-finite parameter weights
                    # Otherwise, look for problems with your data.
                    metrics = Classification_Metrics(y_true=[0, 0],
                                                     y_pred=[1, 1],
                                                     y_score=[1, 1])
                    external_val = metrics.failed_external_metrics()
                    ext_metrics = {}
                    for key, value in external_val.items():
                        key_mean = 'mean_'+key
                        ext_metrics[key_mean] = value

                        key_std = 'std_'+key
                        ext_metrics[key_std] = value
                    ext_metrics['have_ext_convergence_warn'] = True
                    ext_metrics['have_ext_runtime_warns'] = 'ValueError'
                    return ext_metrics

            for warn in caught_warnings:
                if warn.category == ConvergenceWarning:
                    have_convergence_warn = True
                elif warn.category == RuntimeWarning:
                    have_runtime_warns = True
                else:
                    print(warn.category, warn.message)

            # External validation metrics calculation.
            metrics = Classification_Metrics(y_true=ext_Y_test,
                                             y_pred=ext_predY_test,
                                             y_score=ext_scoreY_test)
            external_val = metrics.external_metrics()
            for key, value in external_val.items():
                key_split = 'split_'+key
                if key_split not in ext_metrics:
                    ext_metrics[key_split] = []
                    ext_metrics[key_split].append(value)
                else:
                    ext_metrics[key_split].append(value)

        # Mean and std calculation for internal validation metrics.
        mean_std_ext = {}
        for key in ext_metrics.keys():
            key_mean = key.replace('split_', 'mean_')
            mean = np.mean(ext_metrics[key])
            mean_std_ext[key_mean] = mean

            key_std = key.replace('split_', 'std_')
            std = np.std(ext_metrics[key])
            mean_std_ext[key_std] = std

        # Keep track of any convergence warnings
        # that occurred during model training and,
        # if any, reset the value of metrics.
        if have_convergence_warn is True:
            for key in mean_std_ext.keys():
                mean_std_ext[key] = 0
            mean_std_ext['have_ext_convergence_warn'] = True
        else:
            mean_std_ext['have_ext_convergence_warn'] = False

        # Keep track of any RunTime warnings. This part was
        # implemented due to matmul, reduce and add warnings.
        # Examples: "overflow encountered in matmul" and
        # "invalid value encountered in matmul".
        # This was observed with 'sgd' solver for MLPClassifier.
        if have_runtime_warns is True:
            mean_std_ext['have_ext_runtime_warns'] = True
        else:
            mean_std_ext['have_ext_runtime_warns'] = False

        if save_model is True:
            # This option should be accessible only
            # through the save_model function.
            return ext_clf
        else:
            return mean_std_ext

    def int_ext_val(self):
        # Do the internal and external validation.
        mean_std_ext = self.val_ext()
        mean_std_5_fold = self.val_5fold()
        all_metrics = {**mean_std_5_fold, **mean_std_ext}
        return all_metrics

    def model_fit_params(self):
        params = self.params
        metrics = self.int_ext_val()

        params_metrics = {**params, **metrics}
        return params_metrics

    def save_model(self, file_path):
        # Changes the behavior of val_ext to save the model
        # trained with training set.
        model = self.val_ext(save_model=True)
        with open(file_path, 'wb') as handle:
            pickle.dump(model, handle)
