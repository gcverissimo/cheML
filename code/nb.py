import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import numpy as np

from utils.grid_search import Grid_cheML
from utils.file_manager import (directory_out_generation, file_to_arrays,
                                get_LUNA_results, output_generation)


# NB
model_type = 'NB'
N_THREADS = -1

# Getting the input files in the input directory.
pca_dir = "/home/PCA_input"
dict_pca_paths = get_LUNA_results(pca_dir)

# Setting the output directory.
ml_dir = "/home/output"
results_dir = ml_dir + '/' + model_type
os.makedirs(results_dir, exist_ok=True)

# Defining the search space of priors.
prob_a = np.arange(0.01, 1.00, 0.01)
prob = []
for a in prob_a:
    a = round(a, 2)
    b = round(1.00 - a, 2)
    prob.append(np.array((a, b)))
prob.append(None)

# Setting the search space.
search_space = {
    'var_smoothing': np.logspace(0, -10, num=50),
    'priors': prob
}

for path, name in dict_pca_paths.items():
    name = name.replace('PCA', model_type)
    output_directory = directory_out_generation(name, model_type, results_dir)
    if output_directory == 'continue':
        continue

    X_train, Y_train, X_test, Y_test = file_to_arrays(path)

    clf = GaussianNB()
    grid = Grid_cheML(X_train, Y_train,
                      X_test, Y_test,
                      search_space, clf)

    results = grid.fit_models(N_THREADS=N_THREADS)

    df = pd.DataFrame(results)
    result_file = output_generation(df, name, output_directory)
