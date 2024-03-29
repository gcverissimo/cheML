import os
import pandas as pd
from sklearn.svm import SVC

from utils.grid_search import Grid_cheML
from utils.file_manager import (directory_out_generation, file_to_arrays,
                                get_LUNA_results, output_generation)


# SVC_Linear
model_type = 'SVC_Linear'
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
    'C': [0.0001, 0.001, 0.01, 0.1, 0.5,
          1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
          100, 150, 200, 250, 500, 750, 1000, 1250, 1500],
    'kernel': ['linear'],
}

for path, name in dict_pca_paths.items():
    name = name.replace('PCA', model_type)
    output_directory = directory_out_generation(name, model_type, results_dir)
    if output_directory == 'continue':
        continue

    X_train, Y_train, X_test, Y_test = file_to_arrays(path)

    clf = SVC(random_state=2023, probability=True)
    grid = Grid_cheML(X_train, Y_train,
                      X_test, Y_test,
                      search_space, clf)

    results = grid.fit_models(N_THREADS=N_THREADS)

    df = pd.DataFrame(results)
    result_file = output_generation(df, name, output_directory)
