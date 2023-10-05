import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from utils.grid_search import Grid_cheML
from utils.file_manager import (directory_out_generation, file_to_arrays,
                                get_LUNA_results, output_generation)


# DT
model_type = 'DT'
N_THREADS = -1

# Getting the input files in the input directory.
pca_dir = "/home/PCA_input"
dict_pca_paths = get_LUNA_results(pca_dir)

# Setting the output directory.
ml_dir = "/home/output"
results_dir = ml_dir + '/' + model_type
os.makedirs(results_dir, exist_ok=True)

# Setting the search space.
search_space = {
    'criterion': ['gini'],
    'splitter': ['best'],
    'min_samples_split': [2, 4, 6, 8, 10, 12,
                          14, 16, 18, 20, 22,
                          24, 26, 28, 30, 32,
                          34, 36, 38, 40],
    'max_features': ['sqrt', 'log2', None]
}

for path, name in dict_pca_paths.items():
    name = name.replace('PCA', model_type)
    output_directory = directory_out_generation(name, model_type, results_dir)
    if output_directory == 'continue':
        continue

    X_train, Y_train, X_test, Y_test = file_to_arrays(path)

    clf = DecisionTreeClassifier(random_state=2023)
    grid = Grid_cheML(X_train, Y_train,
                      X_test, Y_test,
                      search_space, clf)

    results = grid.fit_models(N_THREADS=N_THREADS)

    df = pd.DataFrame(results)
    result_file = output_generation(df, name, output_directory)
