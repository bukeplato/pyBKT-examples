import sys
import numpy as np
from pyBKT.models import Model
import pandas as pd

pd.set_option('display.max_rows', 50)  #最大行数
pd.set_option('display.max_columns', None)    #最大列数
pd.set_option('display.width', None)        #页面宽度


np.seterr(divide='ignore', invalid='ignore')

sys.path.append('../')


def main():
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

    model = Model(seed=42, num_fits=1)
    model.fit(data_path='data/ct.csv', forgets = True, skills=".*fraction.*")
    preds = model.predict(data_path='data/ct.csv')
    # print(preds)
    # print(model.fit_model)
    print(preds[['Anon Student Id', 'KC(Default)',
                 'correct_predictions', 'state_predictions']])
    # preds.to_csv('./output/prediction.csv')     # 文件会产生在远程机器上

if __name__ == '__main__':
    sys.exit(main())
