import sys
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from pyBKT.models import Model

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
    sys.exit(main())