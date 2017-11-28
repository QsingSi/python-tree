import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from collections import Counter
from sklearn.metrics import confusion_matrix


class Config:
    xgb_best_param = {'learning_rate': 0.25, 'max_depth': 5,
                      'objective': 'binary:logistic', 'eval_metric': 'error',
                      'num_boost_round': 90, 'n_estimators': 90}
    lgb_best_param = dict(subsample=0.7,
                          reg_lambda=0.1,
                          reg_alpha=0.05,
                          num_leaves=63,
                          n_estimators=280,
                          min_child_weight=4,
                          min_child_samples=80,
                          max_bin=200,
                          learning_rate=0.05,
                          colsample_bytree=0.9)
    rf_best_param = dict(n_estimators=90,
                         max_depth=10)


class RandomXgb(Config):
    def __init__(self, cv=5, n_estimators=50):
        super(RandomXgb, self).__init__()
        self.cv = cv
        self.n_estimators = n_estimators

    def BuildUnit(self, data, target):
        lg = lgb.LGBMClassifier(**(self.lgb_best_param))
        lg.fit(data, target)
        return lg

    def MajorClass(self, lab):
        cnt = Counter(lab)
        print(cnt)
        cnt0, cnt1 = cnt.get(0, 0), cnt.get(1, 0)
        print(cnt0, cnt1)
        return 0 if cnt0 > cnt1 else 1

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

    def fit(self, data, target):
        '''
        Only accept DataFrame
        '''
        Model = []
        sub_feature_name = []
        for step in range(self.n_estimators):
            sub_data = data.sample(frac=0.2, axis=1)
            sub_feature_name.append(sub_data.columns)
            sub_model = self.BuildUnit(sub_data.values, target.values)
            Model.append(sub_model)
        label = []
        for i in range(data.shape[0]):
            tot_label_i = []
            # print(sub_feature_name[Model.index(es)])
            for es in Model:
                tot_label_i.append(es.predict(
                    data[sub_feature_name[Model.index(es)]].iloc[i].values)[0])
            print(tot_label_i)
            label.append(self.MajorClass(tot_label_i))
        print(self.score_func(target, label))
        return self.score_func(target, label)
