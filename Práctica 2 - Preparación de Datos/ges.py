
import pandas
import numpy
from chefboost import Chefboost

def loadDataset():
    data = pandas.read_excel("data/accidentes_reduced.xls")
    return data


if __name__ == "__main__":
    data = loadDataset()
    print(data)