import os
import logging
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, ShuffleSplit, train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, chi2, f_classif, mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import Imputer, imputation


class train:
    def __init__(self, dataset):
        self.dataset = dataset
        self.train_data, self.test_data, self.train_target, self.test_target = train_test_split(
            dataset.iloc[:, 3:].values, dataset.iloc[:, 2].values.astype(int), test_size=0.3)

    def get_sample_weight(self):
        label = self.dataset['label'].astype(int)
        length = len(label)
        cnt = 0
        for flag in label:
            if flag == 1:
                cnt += 1
        cnt1 = length - cnt
        sample_weight = {}
        sample_weight[0] = 1.0
        sample_weight[1] = int((length - cnt) / cnt + 0.5)
        return sample_weight

    def calRatio(self, target, prediction):
        TP = FP = TN = FN = 0
        for i in range(len(prediction)):
            if target[i] == 1:
                if prediction[i] == 1:
                    TP += 1
                else:
                    FN += 1
            else:
                if prediction[i] == 0:
                    TN += 1
                else:
                    FP += 1
        Precison = TP * 1.0 / (TP + FN)
        Recall = FP * 1.0 / (FP + TN)
        return Precison, Recall

    def model(self, num_feature):
        dataImputer = Imputer(missing_values='NaN',
                              strategy="most_frequent", axis=0)
        feature_filter = SelectKBest(mutual_info_classif, num_feature)
        clf = RandomForestClassifier(
            n_estimators=90, max_depth=10, class_weight=self.get_sample_weight())
        '''
        rf = Pipeline(steps=[('Imputer', dataImputer),
                             ('feature', feature_filter), ('randomF', clf)])
        '''
        pipe = Pipeline(steps=[('Imputer', dataImputer),
                               ('feature', feature_filter)])
        '''
        rf.set_params(Imputer__verbose=2, randomF__verbose=2, randomF__n_jobs=12).fit(
            self.train_data, self.target)
        '''
        dataDeal = pipe.set_params(Imputer__verbose=10).fit(
            self.train_data, self.train_target)
        self.train_prediction = cross_val_predict(
            estimator=clf, X=self.train_data, y=self.train_target, cv=5, n_jobs=10)
        #self.prediction = rf.predict(self.train_data)
        self.train_P, self.train_R = self.calRatio(
            self.train_target, self.train_prediction)
        self.train_ks = self.train_P - self.train_R
        print('result under {} features on train data:'.format(num_feature))
        print('查杀率 = {}, 误杀率 = {}, ks = {}'.format(
            self.train_P, self.train_R, self.train_ks))
        self.test_prediction = clf.predict(self.test_data)
        self.test_P, self.test_R = self.calRatio(
            self.test_target, self.test_prediction)
        self.test_ks = self.test_P - self.test_R
        print('result under {} features on test data:'.format(num_feature))
        print('查杀率 = {}, 误杀率 = {}, ks ={}'.format(
            self.test_P, self.test_R, self.test_ks))


def main():
    file_name = 'df_plain_104'
    path = os.path.join('../data/', file_name)
    df = pd.read_pickle(path)
    test = train(df)
    test.model(num_feature=50)


if __name__ == '__main__':
    main()
