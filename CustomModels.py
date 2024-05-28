import pickle

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
# SKLearnEstimator is derived from BaseEstimator
from flaml.automl.model import SKLearnEstimator
from flaml import tune
from sklearn.ensemble import BaggingRegressor


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    """
    x_max = np.max(x)
    e_x = np.exp(x - x_max)  # Subtracting the maximum value for numerical stability
    return e_x / e_x.sum(axis=0)  # Calculating softmax along the specified axis


class HeterogeneousEnsenbleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, pkl_file_names, use_weight=False, **config):
        self.use_weight = use_weight
        self.automl = []
        for f in pkl_file_names:
            automl = pickle.load(open(f, 'rb'))
            self.automl.append(automl)

    def fit(self, X, y):
        pass

    def predict(self, X):
        predictions = []
        weight = []
        for i, automl in enumerate(self.automl):
            # predictions = automl.model.predict(X) * (1 - automl.best_loss)
            predictions.append(automl.model.predict(X))
            weight.append(1 - automl.best_loss if self.use_weight else 1.0)
            # weight.append(1.0)
        # softmax_loss = softmax(weight)
        # softmax_loss = np.array(weight)
        return np.sum(np.array(predictions).T * np.array(weight), axis=1)/len(self.automl) if len(self.automl) > 0 else np.zeros(X.shape[0])



class BaggingWithGradientBoostingRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, n_estimators=10, learning_rate=0.1, n_estimators_for_bagging=10, **config):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.n_estimators_for_bagging = n_estimators_for_bagging
        # self.max_depth_for_bagging = max_depth_for_bagging
        self.trees = []

    def fit(self, X, y):
        residuals = y.copy()
        for i in range(self.n_estimators):
            tree = BaggingRegressor(n_estimators=self.n_estimators_for_bagging)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Update residuals
            predictions = tree.predict(X)
            residuals -= self.learning_rate * predictions

    def predict(self, X):
        # predictions = np.zeros(len(X))
        predictions = []
        for tree in self.trees:
            # predictions += self.learning_rate * tree.predict(X)
            predictions.append(self.learning_rate * tree.predict(X))
        return np.sum(np.array(predictions), axis=0)


class BagGradientBoostingRegressor(SKLearnEstimator, RegressorMixin):
    def __init__(self, task="binary", **config):
        super().__init__(task, **config)
        self.estimator_class = BaggingWithGradientBoostingRegressor

    @classmethod
    def search_space(cls, data_size, task):
        # Define search space for hyperparameters
        space = {
            "n_estimators": {"domain": tune.qrandint(lower=10, upper=100, q=10), "low_cost_init_value": 10},
            "learning_rate": {"domain": tune.quniform(lower=0.1, upper=1, q=0.1), "low_cost_init_value": 0.2},
            "n_estimators_for_bagging": {"domain": tune.qrandint(lower=10, upper=100, q=10), "low_cost_init_value": 10},
        }
        return space