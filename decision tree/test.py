from sklearn.metrics import make_scorer, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from collections import Counter
from sklearn.metrics import roc_curve


class Model:
    def __init__(self, n_jobs=1, depth=10, trees=100, cv=5):
        self.n_jobs = n_jobs
        self.depth = depth
        self.trees = trees
        self.cv = cv
        self.My_score()
        self.param = {'n_estimators': range(
            100, 111, 10), 'max_depth': range(5, 8, 1)}

    def _convert_data(self, y_pred_prob):
        import numpy as np
        y_pred = [prob[1] for prob in y_pred_prob]
        return np.array(y_pred)

    def calc_max_ks(self, y_true, y_pred, sample_weight=None):
        if len(y_pred[0]) == 2:
            y_pred = self._convert_data()
        fpr, tpr, thre = roc_curve(y_true, y_pred, sample_weight=sample_weight)
        ks = tpr - fpr
        return max(ks)

    def _calc_max_ks(self, estimator, X, y):
        if hasattr(estimator, 'predict_proba'):
            y_pred_prob = estimator.predict_proba(X)
            y_pred = [prob[1] for prob in y_pred_prob]
        else:
            y_pred = estimator.predict(X)
        print(y_pred, len(y_pred))
        fpr, tpr, thre = roc_curve(y, y_pred)
        ks = tpr - fpr
        return max(ks)

    def score_func(self, truth, pred):
        print(confusion_matrix(truth, pred))
        TP = confusion_matrix(truth, pred)[0, 0]
        FP = confusion_matrix(truth, pred)[0, 1]
        FN = confusion_matrix(truth, pred)[1, 0]
        TN = confusion_matrix(truth, pred)[1, 1]
        Precision = TP / (TP + FN)
        Recall = FP / (FP + TN)
        ks = Precision - Recall
        return ks

    def My_score(self):
        self.KS = make_scorer(self.score_func, greater_is_better=True)

    def fit(self, data, target):
        rf = RandomForestClassifier(
            n_estimators=self.trees, max_depth=self.depth, verbose=2, n_jobs=self.n_jobs)
        score = cross_val_score(rf, data, target, scoring=self.KS, cv=self.cv)
        print(score)
        return score

    def fit_cross(self, data, target):
        cnt = Counter(target)
        sample_weight = {}
        sample_weight[0] = 1
        sample_weight[1] = int(cnt.get(0) / cnt.get(1) + 0.5)
        self.sample_weight = sample_weight
        rf = RandomForestClassifier(
            n_estimators=self.trees,
            max_depth=self.depth,
            verbose=2,
            n_jobs=self.n_jobs,
            class_weight=self.sample_weight)
        score = cross_val_score(
            rf,
            data,
            target,
            scoring=self._calc_max_ks,
            cv=self.cv)
        return score

    def fit_GridSearch(self, data, target):
        rf = RandomForestClassifier(verbose=1)
        gs = GridSearchCV(rf, self.param, scoring=self.KS,
                          cv=self.cv, verbose=1)
        gs.fit(data, target)
        #gs.best_estimator_.fit(test_data, test_target)
        #print(gs.predict(test_data), test_target)
        return gs
