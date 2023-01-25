import os
import numpy as np
import pandas as pd

from sklearn.svm import SVC

from make_model import model_5cv, model_extv, remove_splits
from file_manager import (directory_out_generation, file_to_arrays,
                          get_LUNA_results, output_generation)



# SVC_Poly
model_type = 'SVC_Poly'
N_THREADS = 20

# Getting the input files in the input directory.
pca_dir = "D:/Gabriel/9_ML_new/1_Fingerprint_transformation/2_Standardization_PCA/2_results"
dict_pca_paths = get_LUNA_results(pca_dir)

# Setting the output directory.
ml_dir = "D:/Gabriel/9_ML_new/2_MLs"
results_dir = ml_dir + '/' + model_type
os.makedirs(results_dir, exist_ok=True)

# Setting the search space.
search_space = {
'C': [0.0001, 0.001, 0.01, 0.1, 0.5,
        1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
        100, 150, 200, 250, 500, 750, 1000, 1250, 1500],
'gamma': [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1,
            2, 2.5, 3, 3.5, 4.5, 5, 10, 15, 20, 25, 30, 35,
            40, 45, 50, 100, 150, 200, 250, 300],
'kernel': ['poly'], 
'degree': [2, 3, 4, 5],
'coef0': [0, 0.0001, 0.001, 0.005, 0.01, 0.05,
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

for path, name in dict_pca_paths.items():
    name = name.replace('PCA', model_type)
    output_directory = directory_out_generation(name, model_type, results_dir)
    if output_directory == 'continue':
        continue

    X_train, Y_train, X_test, Y_test = file_to_arrays(path)

    # Setting the classifier.
    clf = SVC(random_state=2023)

    # Runs the GridSearch.
    gs_5cv = model_5cv(clf, search_space, X_train, Y_train, N_THREADS)
    gs_ext = model_extv(clf, search_space, X_train, Y_train, X_test, Y_test, N_THREADS)

    # Results treatment.
    gs_5cv_results = remove_splits(gs_5cv.cv_results_)
    gs_ext_results = remove_splits(gs_ext.cv_results_)

    # Generates outputs.
    df_internal = pd.DataFrame(gs_5cv_results)
    df_external = pd.DataFrame(gs_ext_results)
    result_file = output_generation(df_internal, df_external, name, output_directory)