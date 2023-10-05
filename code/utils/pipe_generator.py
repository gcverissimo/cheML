import pickle
from sklearn.pipeline import Pipeline


class Pipeready:
    def __init__(self, preprocess, clf):
        self.pipe = Pipeline([
            ('preprocessing', preprocess),
            ('clf', clf)])

    def transform(self, X):
        X_t = self.pipe["preprocessing"].transform(X)
        return X_t

    def predict(self, X_t):
        Y_pred = self.pipe["clf"].predict(X_t)
        return Y_pred

    def predict_proba(self, X_t):
        Y_score = self.pipe["clf"].predict_proba(X_t)
        return Y_score

    def predprob_class(self, X_t):
        Y_score = self.pipe["clf"].predict_proba(X_t)
        return Y_score[:, 1]

    def transform_predict(self, X):
        X_t = self.pipe["preprocessing"].transform(X)
        Y_pred = self.pipe["clf"].predict(X_t)
        return Y_pred

    def transform_predprob(self, X):
        X_t = self.pipe["preprocessing"].transform(X)
        Y_score = self.pipe["clf"].predict_proba(X_t)
        return Y_score

    def transform_predclass(self, X):
        X_t = self.pipe["preprocessing"].transform(X)
        Y_score = self.pipe["clf"].predict_proba(X_t)
        return Y_score[:, 1]

    def get_pipe(self):
        return self.pipe

    def save_pipe(self, file_path):
        with open(file_path, 'wb') as handle:
            pickle.dump(self, handle)
        string = f"Model was saved as: {file_path}"
        return string
