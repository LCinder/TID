import numpy as np
import pandas
import numpy
from chefboost import Chefboost
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector as sequential_feature
from sklearn.linear_model import LinearRegression, LogisticRegression


def load_dataset():
    data = pandas.read_excel("data/accidentes_reduced.xls")
    features = data.columns[0:len(data.columns)-3]
    x = pandas.DataFrame(data[features])
    y = pandas.DataFrame(data.PRPTYDMG_CRASH)
    return x, y


def remove_columns(data, elements):
    for element in elements:
        data = data.drop(element, axis="columns")
    return data


def pred(x_train_arg, y_train_arg, x_test_arg, y_test_arg):
    ctree = DecisionTreeClassifier(random_state=1)
    ctree.fit(x_train_arg, y_train_arg)
    y_pred = ctree.predict(x_test_arg)
    return round(accuracy_score(y_test_arg, y_pred), 3)


def discretize(x_train_arg, x_test_arg):
    x_train_arg = remove_columns(x_train_arg, not_imputed)
    x_test_arg = remove_columns(x_test_arg, not_imputed)
    x_train_disc = x_train_arg

    # ["WKDY_I", "HOUR_I", "RELJCT_I", "MANCOL_I", "RELJCT_I", "ALIGN_I", "PROFIL_I", "SURCON_I", "TRFCON_I",
    #"SPDLIM_H", "LGTCON_I", "WEATHR_I", "ALCHL_I"]
    elements = ["HOUR_I", "SPDLIM_H", "WKDY_I"]
    bins = [2, 5, 4]

    accuracy_mean_discr = 0
    accuracy_mean = 0
    n = 10

    for variable, bin_i in zip(elements, bins):
        discr = KBinsDiscretizer(n_bins=bin_i, encode="ordinal", strategy="uniform")

        for i in range(n):
            accuracy_mean += pred(x_train, y_train, x_test, y_test)
            x_train_disc[variable] = discr.fit_transform(x_train_arg[variable].values.reshape(-1, 1))
            accuracy_mean_discr += pred(x_train_disc, y_train, x_test_arg, y_test)

    print("Accuracy: " + str(round(accuracy_mean / (n * len(elements)), 3)))
    print("Accuracy Discretize: " + str(round(accuracy_mean_discr / (n * len(elements)), 3)))

    return x_train_disc


def loss_values():
    x_train_imputed = remove_columns(x_train, not_imputed)
    x_test_imputed = remove_columns(x_test, not_imputed)
    x_train_not_imputed = remove_columns(x_train, imputed)
    x_test_not_imputed = remove_columns(x_test, imputed)
    res_imputed = 0
    res_not_imputed = 0
    res_not_imputed_mean = 0
    res_not_imputed_mode = 0
    n = 10

    x_train_not_imputed_mode = remove_columns(x_train, imputed)
    x_train_not_imputed_mean = remove_columns(x_train, imputed)
    unknown = [9, 99, 9, 19, 99, 9, 9, 9, 29, 49, 99, 99, 9, 9, 9]

    for i in range(n):
        res_imputed += pred(x_train_imputed, y_train, x_test_imputed, y_test)
        res_not_imputed += pred(x_train_not_imputed, y_train, x_test_not_imputed, y_test)

        for variable, value in zip(not_imputed, unknown):
            x_train_not_imputed_mean[variable] = x_train_not_imputed_mean[variable].replace(value, None)

            mean = x_train_not_imputed_mean[variable].mean()
            x_train_not_imputed_mean[variable].fillna(value=mean)

            mode = x_train_not_imputed_mode[variable].mode()[0]
            x_train_not_imputed_mode[variable].replace(value, mode, inplace=True)

        res_not_imputed_mean += pred(x_train_not_imputed_mean, y_train, x_test_not_imputed, y_test)
        res_not_imputed_mode += pred(x_train_not_imputed_mode, y_train, x_test_not_imputed, y_test)

    x_train_imputed_instances = x_train_imputed
    x_test_imputed_instances = x_test_imputed

    for i in x_train_imputed_instances.columns:
        x_train_imputed_instances[(x_train_imputed_instances[i] != 9) & (x_train_imputed_instances[i] != 19)
        & (x_train_imputed_instances[i] != 29) & (x_train_imputed_instances[i] != 49) & (x_train_imputed_instances[i] != 99)]
        
        x_test_imputed_instances[(x_test_imputed_instances[i] != 9) & (x_test_imputed_instances[i] != 19)
        & (x_test_imputed_instances[i] != 29) & (x_test_imputed_instances[i] != 49) & (
        x_test_imputed_instances[i] != 99)]

    acc_imputed_removed_instances = pred(x_train_imputed, y_train, x_test_imputed, y_test)

    x_train_imputed = remove_columns(x_train, not_imputed)
    x_train_imputed = remove_columns(x_train_imputed, imputed)
    x_test_imputed = remove_columns(x_test_imputed, imputed)
    acc_imputed_removed_characteristics = pred(x_train_imputed, y_train, x_test_imputed, y_test)

    acc_imputed = round(res_imputed/n, 3)
    acc_not_imputed = round(res_not_imputed/n, 3)
    acc_not_imputed_mean = round(res_not_imputed_mean/n, 3)
    acc_not_imputed_mode = round(res_not_imputed_mode/n, 3)

    print("Accuracy Imputed: " + str(acc_imputed))
    print("Accuracy Not Imputed: " + str(acc_not_imputed))
    print("Accuracy Not Imputed Mean: " + str(acc_not_imputed_mean))
    print("Accuracy Not Imputed Mode: " + str(acc_not_imputed_mode))
    print("Accuracy Imputed Removed Instances: " + str(acc_imputed_removed_instances))
    print("Accuracy Imputed Removed Characteristics: " + str(acc_imputed_removed_characteristics))

    best = {acc_imputed : "Imputed",
    acc_not_imputed : "Not Imputed",
    acc_not_imputed_mean : "Not Imputed Mean",
    acc_not_imputed_mode : "Not Imputed Mode",
    acc_imputed_removed_instances : "Imputed Removed Instances",
    acc_imputed_removed_characteristics : "Imputed Removed Characteristics"}

    sort = sorted(best.items(), reverse=True)[0]


    if sort[1] == "Imputed":
        return x_train_imputed, "Imputed"
    elif sort[1] == "Not Imputed":
        return x_train, "Not Imputed"
    elif sort[1] == "Not Imputed Mean":
        return x_train_not_imputed_mean, "Not Imputed Mean"
    elif sort[1] == "Not Imputed Mode":
        return x_train_not_imputed_mode, "Not Imputed Mode"
    elif sort[1] == "Imputed Removed Instances":
        return x_train_imputed_instances, "Imputed Removed Instances"
    elif sort[1] == "Imputed Removed Characteristics":
        return x_train_imputed, "Imputed Removed Characteristics"


#https://www.analyticsvidhya.com/blog/2021/04/backward-feature-elimination-and-its-implementation/
def selection(x_sel, x_test_sel):
    #x_test_sel = remove_columns(x_test_sel, not_imputed)
    all = pred(x_sel, y_train, x_test_sel, y_test)

    s = sequential_feature(GaussianNB(), k_features=5, forward=False)
    s = s.fit(x_sel, np.ravel(y_train))
    features = list(s.k_feature_names_)

    not_selected = list(set(x_sel.columns) - set(features))

    x_sel = remove_columns(x_sel, not_selected)
    x_test_sel = remove_columns(x_test_sel, not_selected)
    removed_features = pred(x_sel, y_train, x_test_sel, y_test)

    print("Accuracy All: " + str(all))
    print("Features: " + str(features))
    print("Accuracy Removed Features: " + str(removed_features))

    return x_sel, x_test_sel


if __name__ == "__main__":
    x, y = load_dataset()

    not_imputed = ["WEEKDAY", "HOUR", "MAN_COL", "REL_JCT", "ALIGN", "PROFILE", "SUR_COND", "TRAF_CON",
    "SPD_LIM", "LGHT_CON", "WEATHER", "ALCOHOL"]
    imputed =  ["WKDY_I", "HOUR_I", "MANCOL_I", "RELJCT_I", "ALIGN_I", "PROFIL_I", "SURCON_I", "TRFCON_I",
    "SPDLIM_H", "LGTCON_I", "WEATHR_I", "ALCHL_I"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6)

    #Ejercicio 1
    x_train_discr = discretize(x_train, x_test)

    #Ejercicio 2
    best, mode  = loss_values()

    print("El mejor es " + mode)

    #Ejercicio 3
    x_test_not_imputed = remove_columns(x_test, not_imputed)
    x_sel_char, x_test_sel_char = selection(x_train_discr, x_test_not_imputed)
    x_sample = pandas.DataFrame(x_sel_char)
    x_sample = x_sample.sample(frac=0.3, random_state=1)
    y_train = y_train.loc[x_sample.index]
    x_test_sample = remove_columns(x_test, set(x_train.columns) - set(x_sample.columns))

    print("Muestreo del 30%")
    acc_res_sample = 0
    acc_res_sample += pred(x_sample, y_train, x_test_sample, y_test)
    print("Accuracy sample: " + str(acc_res_sample))


