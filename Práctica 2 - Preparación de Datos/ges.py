
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

def discretize(x_train):
    x_train = remove_columns(x_train, imputed)
    x_train_disc = x_train

    # ["WKDY_I", "HOUR_I", "RELJCT_I", "MANCOL_I", "RELJCT_I", "ALIGN_I", "PROFIL_I", "SURCON_I", "TRFCON_I",
    #"SPDLIM_H", "LGTCON_I", "WEATHR_I", "ALCHL_I"]
    elements = ["HOUR_I", "SPDLIM_H", "WKDY_I"]
    bins = [2, 5, 4]

    accuracy_mean_discr = 0
    accuracy_mean = 0

    ctree = DecisionTreeClassifier(random_state=1)
    n = 10

    for variable, bin_i in zip(elements, bins):
        discr = KBinsDiscretizer(n_bins=bin_i, encode="ordinal", strategy="uniform")

        for i in range(n):
            ctree.fit(x_train, y_train)
            y_pred = ctree.predict(x_test)
            accuracy_mean += accuracy_score(y_test, y_pred)

            x_train_disc[variable] = discr.fit_transform(x_train[variable].values.reshape(-1, 1))
            ctree.fit(x_train_disc, y_train)
            y_pred = ctree.predict(x_test)
            accuracy_mean_discr += accuracy_score(y_test, y_pred)

    print("Accuracy: " + str(accuracy_mean / (n * len(elements))))
    print("Accuracy CAIM: " + str(accuracy_mean_discr / (n * len(elements))))


def loss_values():
    x_train_imputed = remove_columns(x_train, not_imputed)
    x_test_imputed = remove_columns(x_test, not_imputed)

    ctree = DecisionTreeClassifier(random_state=1)
    ctree.fit(x_train_imputed, y_train)
    y_pred = ctree.predict(x_test_imputed)
    print("Accuracy Imputed: " + str(accuracy_score(y_test, y_pred)))

    x_train_not_imputed = remove_columns(x_train, imputed)
    x_test_not_imputed = remove_columns(x_test, imputed)

    ctree = DecisionTreeClassifier(random_state=1)
    ctree.fit(x_train_not_imputed, y_train)
    y_pred = ctree.predict(x_test_not_imputed)
    print("Accuracy Not Imputed: " + str(accuracy_score(y_test, y_pred)))

    #c) Imputar valores por media o moda
    unknown = [9, 99, 9, 19, 99, 9, 9, 9, 29, 49, 99, 99, 9, 9, 9]

    for variable, value in zip(not_imputed, unknown):
        x_train_not_imputed[variable] =  x_train_not_imputed[variable].replace(value, None)

        mean = x_train_not_imputed[variable].mean()
        x_train_not_imputed[variable].fillna(value=mean)

    tree = DecisionTreeClassifier(random_state=1)
    ctree.fit(x_train_not_imputed, y_train)
    y_pred = ctree.predict(x_test_not_imputed)
    print("Accuracy Not Imputed Mean: " + str(accuracy_score(y_test, y_pred)))

if __name__ == "__main__":
    x, y = loadDataset()
    not_imputed = ["WEEKDAY", "HOUR", "MAN_COL", "REL_JCT", "ALIGN", "PROFILE", "SUR_COND", "TRAF_CON",
    "SPD_LIM", "LGHT_CON", "WEATHER", "ALCOHOL"]
    imputed =  ["WKDY_I", "HOUR_I", "MANCOL_I", "RELJCT_I", "ALIGN_I", "PROFIL_I", "SURCON_I", "TRFCON_I",
    "SPDLIM_H", "LGTCON_I", "WEATHR_I", "ALCHL_I"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6)

    #Ejercicio 1
    #discretize(x_train)
    #Ejercicio 2
    #a
    loss_values()
