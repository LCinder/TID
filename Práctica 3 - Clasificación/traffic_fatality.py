import numpy
import numpy as np
import pandas
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, CategoricalNB, ComplementNB
import matplotlib.pyplot as plot
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector as sequential_feature
from imblearn.over_sampling import SMOTE


def load_dataset():
    data = pandas.read_csv("data/traffic_fatality.csv", delimiter=";")
    return data


def remove_columns(data_arg, elements):
    data = data_arg[:]
    for element in elements:
        data = data.drop(element, axis="columns")
    return data


def pred(x_arg, y_arg, tipo, roc=False):
    if tipo == "NB":
        predictor = GaussianNB()
    elif tipo == "KNC":
        predictor = KNeighborsClassifier()
    elif tipo == "DTC":
        predictor = DecisionTreeClassifier()
    elif tipo == "LSVC":
        predictor = LinearSVC()
    elif tipo == "RFC":
        predictor = RandomForestClassifier()
    elif tipo == "ABC":
        predictor = AdaBoostClassifier()
############################################################################
    x_train, x_test, y_train, y_test = train_test_split(x_arg, y_arg, train_size=0.6)

    y_arg = numpy.ravel(y_arg)
    y_test = numpy.ravel(y_test)
    y_train = numpy.ravel(y_train)

    predictor.fit(x_train, y_train)
    y_pred = predictor.predict(x_test)
    y_pred = numpy.ravel(y_pred)

    if roc:
        fpr, tpr, threshold = roc_curve(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        plot.plot(fpr, tpr)
        plot.title("AUC: " + str(auc))
        disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
        disp.plot()
        plot.show()
        print("Cross Val: " + str(round(cross_val_score(predictor, x_arg, y_arg).mean(), 3)))
    return round(cross_val_score(predictor, x_arg, numpy.ravel(y_arg)).mean(), 3)


def discretize(x_train_arg, x_test_arg):
    elements = ["Age", "Alcohol_Results", "weekday"]
    bins = [2, 4, 5, 10]

    accuracy_mean = pred(x_train_arg, y_train, x_test_arg, y_test)
    print("Accuracy: " + str(round(accuracy_mean, 3)))

    for variable in elements:
        for bin_i in bins:
            x_train_disc = x_train_arg[:]

            x_train_disc[variable] = pandas.cut(x_train_disc[variable], labels=range(bin_i), bins=bin_i)
            accuracy_mean_discr = pred(x_train_disc, y_train, x_test_arg, y_test)

            print("Accuracy Discretize with variable/bin: " + variable + "/"
            + str(bin_i) + " : " + str(round(accuracy_mean_discr, 3)))

    return x_train_disc

def preprocess(data):
    encoder = LabelEncoder()
    for column in data.columns:
        if isinstance(data[column][0], str):
            data[column] = encoder.fit_transform(data[column])

    data.replace("\\N", numpy.nan, inplace=True)
    data.replace("nan", numpy.nan, inplace=True)
    data["Alcohol_Results"] = data["Alcohol_Results"].astype(float)
    indexes = data[(data.Age < 16)].index
    data.drop(list(indexes), axis=0, inplace=True)
    #data.dropna(axis=0, how="any", inplace=True)

    features = data.columns[0:len(data.columns) - 1]
    x = pandas.DataFrame(data[features])
    y = pandas.DataFrame(data.Fatality)
    y.replace("no_fatal", 0, inplace=True)
    y.replace("fatal", 1, inplace=True)
    
    return x, y


def loss_values(tipo):
    x_mean = x.copy()
    x_mode = x.copy()
    y_mean = y.copy()
    y_mode = y.copy()
    x_characteristics = x.copy()
    y_characteristics = y.copy()

    variables = ["Alcohol_Results", "Age"]
    #res_init = pred(x_train, y_train, x_test_test, y_test, tipo)

    for variable in variables:
        mean = x_mean[variable].mean()
        if variable == "Age":
            x_mean[variable].replace(numpy.nan, int(mean), inplace=True)
        else:
            x_mean[variable].replace(numpy.nan, mean, inplace=True)

        mode = x_mode[variable].mode()[0]
        x_mode[variable].replace(numpy.nan, mode, inplace=True)

    x_characteristics = remove_columns(x_characteristics, ["Age", "Alcohol_Results"])

    res_not_imputed_mean = pred(x_mean, y_mean, tipo)
    res_not_imputed_mode = pred(x_mode, y_mode, tipo)
    res_not_imputed_characteristics = pred(x_characteristics, y_characteristics, tipo)

    #acc = round(res_init, 3)
    acc_not_imputed_mean = round(res_not_imputed_mean, 3)
    acc_not_imputed_mode = round(res_not_imputed_mode, 3)
    acc_imputed_removed_characteristics = round(res_not_imputed_characteristics, 3)

    #print("Accuracy: " + str(acc))
    print("Accuracy Mean: " + str(acc_not_imputed_mean))
    print("Accuracy Mode: " + str(acc_not_imputed_mode))
    print("Accuracy Removed Characteristics: " + str(acc_imputed_removed_characteristics))

    best = {acc_not_imputed_mean: "Mean",
            acc_not_imputed_mode: "Mode",
            acc_imputed_removed_characteristics: "Removed Characteristics"}
    sort = sorted(best.items(), reverse=True)[0]

    if sort[1] == "Mean":
        return x_mean, y_mean, "Mean"
    elif sort[1] == "Mode":
        return x_mode, y_mode, "Mode"
    elif sort[1] == "Removed Characteristics":
        return x_characteristics, y_characteristics, "Removed Characteristics"


if __name__ == "__main__":
    data = load_dataset()
    x, y = preprocess(data)

    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")
    print("------------------------Practica 2--------------------------------")
    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")
    print("Selecciona una opcion:")
    print("1.-Naive-Bayes, 2.-KNeighborsClassifier, 3.-DecissionTreeClasifier, 4.-LinearSVC, "
    " 5.-RandomForestClassifier, 6.-AdaBoostClassifier")
    n = input()

    tipo = ""
    if n == "1":
        tipo = "NB"
    elif n == "2":
        tipo = "KNC"
    elif n == "3":
        tipo = "DTC"
    elif n == "4":
        tipo = "LSVC"
    elif n == "5":
        tipo = "RFC"
    elif n == "6":
        tipo = "ABC"

    print("------------------------------------------------------------------")
    x_res, y_res, best = loss_values(tipo)
    print(best)
    print("------------------------------------------------------------------")
    x_train, y_train = SMOTE().fit_resample(x_res, y_res)
    accuracy = pred(x_res, y_res, tipo, True)
    print("Accuracy SMOTE: " + str(accuracy))



