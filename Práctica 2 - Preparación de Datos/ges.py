
import pandas
import numpy
from chefboost import Chefboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score

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
    x = remove_columns(x, ["WEEKDAY", "HOUR", "MAN_COL", "REL_JCT", "ALIGN", "PROFILE", "SUR_COND", "TRAF_CON",
    "SPD_LIM", "LGHT_CON", "WEATHER", "ALCOHOL"])

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6)

    ctree = DecisionTreeClassifier(random_state=1)
    ctree.fit(x_train, y_train)

    y_pred = ctree.predict(x_test)

    print("Accuracy: " + str(accuracy_score(y_test, y_pred)))

    elements = ["WKDY_I", "HOUR_I", "RELJCT_I", "MANCOL_I", "RELJCT_I", "ALIGN_I", "PROFIL_I", "SURCON_I", "TRFCON_I",
    "SPDLIM_H", "LGTCON_I", "WEATHR_I", "ALCHL_I"]

    for element in elements:
        discr = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
        x_train[element] = discr.fit_transform(x_train[element].values.reshape(-1, 1))

    ctree.fit(x_train, y_train)
    y_pred = ctree.predict(x_test)

    print("Accuracy CAIM: " + str(accuracy_score(y_test, y_pred)))
