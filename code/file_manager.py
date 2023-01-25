import os
import numpy as np
import pandas as pd


def get_LUNA_results(LUNA_transformed_results):
    # Creates a dictionary from all LUNA files in the directory.
    # Dictionary of file_path:name_of_the_LUNA_protocol.
    dict_fingerprints = {}
    for root, dirs, files in os.walk(LUNA_transformed_results):
        for file in files:
            if "csv" in file:
                path = root+'/'+file
                name_file = file.replace('.csv', '')
                dict_fingerprints[path] = name_file
    return dict_fingerprints

def file_to_arrays(path):
    # Input: path for the .csv.
    # Works with a DataFrame with training and test samples.
    # Returns: np.array for X_train, Y_train, X_test, Y_test.
    df = pd.read_csv(path)

    df_train = df.loc[df['Set'] == 'training']
    df_test = df.loc[df['Set'] == 'test']

    X_train = df_train.drop(columns=['ID', 'Set', 'Labels']).values
    Y_train = df_train[['Labels']].values.ravel()

    X_test = df_test.drop(columns=['ID', 'Set', 'Labels']).values
    Y_test = df_test[['Labels']].values.ravel()
    return X_train, Y_train, X_test, Y_test

def directory_out_generation(name, model_type, results_dir):
    # Generates the output directory.
    name_wo_modeltype = name.replace('_'+model_type, '')
    output_directory = results_dir + '/' + name_wo_modeltype
    try:
        os.makedirs(output_directory)
        return output_directory
    except FileExistsError:
        return "continue"

def output_generation(df_internal, df_external, name, output_directory):
    # Returns the DataFrame with validation results.
    metrics_csv = output_directory + '/' + name + '.csv'
    metrics_csv_5fold = metrics_csv.replace('.csv', '_intval.csv')
    metrics_csv_external = metrics_csv.replace('.csv', '_extval.csv')
    
    df_internal.to_csv(metrics_csv_5fold, sep=',', index=False)
    df_external.to_csv(metrics_csv_external, sep=',', index=False)
    return metrics_csv