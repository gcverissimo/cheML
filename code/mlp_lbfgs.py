import os
import pandas as pd
from sklearn.neural_network import MLPClassifier

from utils.grid_search import Grid_cheML
from utils.file_manager import (directory_out_generation, file_to_arrays,
                                get_LUNA_results, output_generation)


# MLP_LBFGS
model_type = 'MLP_LBFGS'
N_THREADS = -1

# Getting the input files in the input directory.
pca_dir = "/home/PCA_input"
dict_pca_paths = get_LUNA_results(pca_dir)

# Setting the output directory.
ml_dir = "/home/output"
results_dir = ml_dir + '/' + model_type
os.makedirs(results_dir, exist_ok=True)

# Setting the search space.
# Alpha = https://doi.org/10.1021/acs.molpharmaceut.9b00182.
with open('layers.txt', 'r') as f:
    layers = eval((f.read()))

search_space = {
    'hidden_layer_sizes': layers,
    'solver': ['lbfgs'],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'alpha': [0.00001, 0.0001, 0.001, 0.01],
    'max_iter': [500]
}

for path, name in dict_pca_paths.items():
    name = name.replace('PCA', model_type)
    output_directory = directory_out_generation(name, model_type, results_dir)
    if output_directory == 'continue':
        continue

    X_train, Y_train, X_test, Y_test = file_to_arrays(path)

    clf = MLPClassifier(random_state=2023)
    grid = Grid_cheML(X_train, Y_train,
                      X_test, Y_test,
                      search_space, clf)

    results = grid.fit_models(N_THREADS=N_THREADS)

    df = pd.DataFrame(results)
    result_file = output_generation(df, name, output_directory)
