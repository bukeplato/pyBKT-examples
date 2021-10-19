import sys
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from pyBKT.models import Model

sys.path.append('../')


def main():

    # model = Model(seed=0, num_fits=20)
    # model.fit(data_path="data/builder_train_preprocessed.csv")
    # print("Standard BKT:", model.evaluate(data_path="data/builder_test_preprocessed.csv", metric="auc"))
    model2 = Model(seed=0, num_fits=20)
    model2.fit(data_path="data/builder_train_preprocessed.csv", forgets=True)
    print("BKT+Forgets:", model2.evaluate(data_path="data/builder_test_preprocessed.csv", metric="auc"))


if __name__ == '__main__':
    sys.exit(main())
