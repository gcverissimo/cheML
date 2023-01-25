import warnings
import numpy as np

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from metrics import internal_validation_metrics, external_validation_metrics



def model_5cv(clf, search_space, X_train, Y_train, N_THREADS):
    scoring_functions = internal_validation_metrics()
    gs_5cv = GridSearchCV(estimator=clf,
                    param_grid=search_space,
                    scoring = scoring_functions,
                    refit = 'int_MCC',
                    cv = 5,
                    verbose = 1,
                    n_jobs=N_THREADS
                    )
    
    warnings.filterwarnings(action='error', category = ConvergenceWarning, module='sklearn')
    gs_5cv.fit(X_train, Y_train)
    return gs_5cv

def model_extv(clf, search_space, X_train, Y_train, X_test, Y_test, N_THREADS):
    X_dataset = np.concatenate((X_train, X_test))
    Y_dataset = np.concatenate((Y_train, Y_test))
    
    train_indices = np.full((len(X_train), ), -1, dtype=int)
    test_indices = np.full((len(X_test), ), 0, dtype=int)
    dataset_indices = np.concatenate((train_indices, test_indices))

    ps = PredefinedSplit(dataset_indices)
    scoring_functions = external_validation_metrics()
    gs_ext = GridSearchCV(estimator=clf,
                    param_grid=search_space,
                    scoring = scoring_functions,
                    refit = 'ext_MCC',
                    cv = ps,
                    verbose = 1,
                    n_jobs=N_THREADS
                    )
    
    warnings.filterwarnings(action='error', category = ConvergenceWarning, module='sklearn')
    gs_ext.fit(X_dataset, Y_dataset)
    return gs_ext

def remove_splits(gs_cv_results):
    without_splits = {key: value for key, value in gs_cv_results.items() if "split" not in key}
    return without_splits