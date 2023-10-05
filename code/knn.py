import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from utils.grid_search import Grid_cheML
from utils.file_manager import (directory_out_generation, file_to_arrays,
                                get_LUNA_results, output_generation)


# kNN
model_type = 'kNN'
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
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13,
                    15, 17, 19, 21, 23, 25,
                    27, 29, 31, 33, 35, 37, 39],
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree'],
    'p': [2],
    'leaf_size': [15, 30, 45, 60],
    'metric': ['minkowski']
}

for path, name in dict_pca_paths.items():
    name = name.replace('PCA', model_type)
    output_directory = directory_out_generation(name, model_type, results_dir)
    if output_directory == 'continue':
        continue

    X_train, Y_train, X_test, Y_test = file_to_arrays(path)

    clf = KNeighborsClassifier()
    grid = Grid_cheML(X_train, Y_train,
                      X_test, Y_test,
                      search_space, clf)

    results = grid.fit_models(N_THREADS=N_THREADS)

    df = pd.DataFrame(results)
    result_file = output_generation(df, name, output_directory)
