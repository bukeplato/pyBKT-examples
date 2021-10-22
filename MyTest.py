import sys
import numpy as np
from pyBKT.models import Model
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')

sys.path.append('../')


def main():
    # Initialize the model with an optional seed
    model = Model(seed=42, num_fits=1)
    model.fit(data_path='data/ct.csv', skills=["Plot imperfect radical",
                                               "Plot pi"],
              multigs=True, forgets=True,
              multilearn=True)

    print(model.params())


if __name__ == '__main__':
    # sys.exit(main())

    ct_df = pd.read_csv('data/ct.csv', encoding='latin')
    print(ct_df.columns)
    # ct_df.head(5)

    model = Model(seed=42, num_fits=1)
    model.fit(data_path='data/ct.csv', skills=".*fraction.*")
    preds = model.predict(data_path='data/ct.csv')
    preds[['Anon Student Id', 'KC(Default)', 'Correct First Attempt',
           'correct_predictions', 'state_predictions']].head(5)