
import pandas
import numpy
from chefboost import Chefboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.metrics import accuracy_score
import pydotplus
import matplotlib.pyplot as plot
from PIL import Image
import graphviz
from caimcaim import CAIMD

def loadDataset():
    data = pandas.read_excel("data/accidentes_reduced.xls")
    features = data.columns[0:len(data.columns)-3]
    x = pandas.DataFrame(data[features])
    y = pandas.DataFrame(data.PRPTYDMG_CRASH)
    return x, y


def remove_columns(data, elements):
    for element in elements:
        data = data.drop(element, axis="columns")
    return data

if __name__ == "__main__":
    x, y = loadDataset()
    x = remove_columns(x, ["WEEKDAY", "HOUR", "REL_JCT", "ALIGN", "PROFILE", "SUR_COND", "TRAF_CON",
    "SPD_LIM", "LGHT_CON", "WEATHER", "ALCOHOL"])

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6)

    ctree = DecisionTreeClassifier(random_state=1)
    ctree.fit(x_train, y_train)

    y_pred = ctree.predict(x_test)

    print("Accuracy: " + str(accuracy_score(y_test, y_pred)))

    x_2_discr = x_train[["WKDY_I", "HOUR_I", "RELJCT_I"]]
    discr = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="uniform")
    x_disc = x_train
    t = discr.fit_transform(x_2_discr)
    x_disc[["WKDY_I", "HOUR_I", "RELJCT_I"]] = t #

    ctree.fit(x_disc, y_train)
    y_pred = ctree.predict(x_test)

    print("Accuracy CAIM: " + str(accuracy_score(y_test, y_pred)))
