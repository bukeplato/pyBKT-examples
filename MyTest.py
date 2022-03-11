import datetime
import pickle
import sys
import json
import numpy as np
from pyBKT.models import Model
import pandas as pd
import joblib

pd.set_option('display.max_rows', 50)  #最大行数
pd.set_option('display.max_columns', None)    #最大列数
pd.set_option('display.width', None)        #页面宽度

np.seterr(divide='ignore', invalid='ignore')
sys.path.append('../')


def train_model():
    # Initialize the model with an optional seed
    # model = Model(seed=42, num_fits=1)
    # model.fit(data_path='data/ct.csv', skills=["Plot imperfect radical",
    #                                            "Plot pi"],
    #           multigs=True, forgets=True,
    #           multilearn=True)
    #
    # print(model.params())

    # ct_df = pd.read_csv('data/ct.csv', encoding='latin')
    # print(ct_df.columns)
    # ct_df.head(5)


    model = Model(seed=42, num_fits=10)
    # model.coef_ = {'Plot imperfect radical': {'prior': 0.1, 'learns': np.array([0.1])}}

    model.fit(data_path='data/ct_myfix.csv', forgets=True, multigs=True, multilearn='Anon Student Id')

    file_name = 'test1'
    fits = 10
    model.save('./output/forgets_n_%s_fits_0%s.pkl' % (str(file_name), str(fits)))

    preds = model.predict(data_path='data/ct_myfix.csv')
    print(preds)
    # print(model.params())

    # print(preds[['Anon Student Id', 'KC(Default)', 'Correct First Attempt',
    #              'correct_predictions', 'state_predictions']])

    preds.to_csv('./output/prediction2.csv')     # 文件会产生在远程机器上
    # 每个模型拟合结束后，将评估结果保存

def predict_stu():
    model111 = Model()
    model111.load('./output/forgets_n_test1_fits_010.pkl')
    # Pkl_Filename = './output/forgets_n_test1_fits_010.pkl'
    # with open(Pkl_Filename, 'rb') as file:
    #     Pickled_LR_Model = pickle.load(file)

    preds = model111.predict(data_path='data/ct_myfix.csv')
    print(preds)
    preds.to_csv('./output/predic111_from_pkl.csv')  # 文件会产生在远程机器上

    preds2 = model111.predict(data_path='data/ct_myfix.csv')
    print(preds2)
    preds2.to_csv('./output/predic112_from_pkl.csv')  # 文件会产生在远程机器上


def main():
    # train_model()
    predict_stu()


if __name__ == '__main__':
    sys.exit(main())
