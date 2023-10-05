import copy
import itertools
import multiprocessing as mp
from tqdm import tqdm

from utils.make_model import Model


class Grid_cheML:
    def __init__(self,
                 X_train, Y_train,
                 X_test, Y_test,
                 search_space, estimator):

        self.search_space = search_space
        self.estimator = estimator
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def _generate_grid(self):
        # Generates all combinations of hyperparameters.
        keys = list(self.search_space.keys())
        combinations = list(itertools.product(*self.search_space.values()))

        result = [dict(zip(keys, combination))
                  for combination in combinations]

        return result

    def _generate_estimators(self):
        # Generates estimators for all combinations of hyperparameters.
        params_dict = self._generate_grid()
        estimators = []
        for params in params_dict:
            self.estimator.set_params(**params)
            actual_estimator = copy.deepcopy(self.estimator)
            estimators.append(actual_estimator)
        return estimators

    def _fit_model(self, model):
        # Fit each individual model.
        model_params = model.get_params()
        search_keys = self.search_space.keys()

        model_params = {k: model_params[k]
                        for k in search_keys
                        if k in model_params}

        new_params = {}
        for k, v in model_params.items():
            new_params['param_'+k] = v

        new_params['params'] = model_params

        model = Model(self.X_train, self.Y_train,
                      self.X_test, self.Y_test,
                      model, new_params)

        result = model.model_fit_params()
        return result

    def fit_models(self, N_THREADS):
        # Fit all models.
        models = self._generate_estimators()
        message = f"Running {len(models)} models with {N_THREADS} threads."
        print(message)

        with mp.Pool(processes=N_THREADS) as pool:
            results = list(tqdm(pool.imap(self._fit_model, models), total=len(models)))
        return results
