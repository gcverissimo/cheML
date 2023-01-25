import os
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier

from make_model import model_5cv, model_extv, remove_splits
from file_manager import (directory_out_generation, file_to_arrays,
                          get_LUNA_results, output_generation)



# MLP_LBFGS
model_type = 'MLP_lbfgs'
N_THREADS = 20

# Getting the input files in the input directory.
pca_dir = "D:/Gabriel/9_ML_new/1_Fingerprint_transformation/2_Standardization_PCA/2_results"
dict_pca_paths = get_LUNA_results(pca_dir)

# Setting the output directory.
ml_dir = "D:/Gabriel/9_ML_new/2_MLs"
results_dir = ml_dir + '/' + model_type
os.makedirs(results_dir, exist_ok=True)

# Setting the search space.
with open('layers.txt', 'r') as f:
    layers = eval((f.read()))

search_space = {
'hidden_layer_sizes': layers,
'solver': ['lbfgs'],
'activation': ['logistic', 'tanh', 'relu'],
'max_iter': [300, 500, 1000]
}


for path, name in dict_pca_paths.items():
    name = name.replace('PCA', model_type)
    output_directory = directory_out_generation(name, model_type, results_dir)
    if output_directory == 'continue':
        continue

    X_train, Y_train, X_test, Y_test = file_to_arrays(path)

    # Setting the classifier.
    clf = MLPClassifier(random_state=2023)

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