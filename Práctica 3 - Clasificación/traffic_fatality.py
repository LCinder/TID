import numpy
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB, ComplementNB
import matplotlib.pyplot as plot
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve, plot_confusion_matrix
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


def pred(x_train_arg, y_train_arg, x_test_arg, y_test_arg, roc=False):
    predictor = GaussianNB()
    predictor.fit(x_train_arg, y_train_arg)
    y_pred = predictor.predict(x_test_arg)
    if roc:
        fpr, tpr, threshold = roc_curve(y_test_arg, y_pred)
        auc = roc_auc_score(y_test_arg, y_pred)
        plot.plot(fpr, tpr)
        plot.title("AUC: " + str(auc))
        plot_confusion_matrix(predictor, x_test_arg, y_test_arg)
        plot.show()
    return round(accuracy_score(y_test_arg, y_pred), 3)


def discretize(x_train_arg, x_test_arg):
    elements = ["Age", "Alcohol_Results"]
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
    data.dropna(axis=0, how="any", inplace=True)

    features = data.columns[0:len(data.columns) - 1]
    x = pandas.DataFrame(data[features])
    y = pandas.DataFrame(data.Fatality)
    y.replace("no_fatal", 0, inplace=True)
    y.replace("fatal", 1, inplace=True)
    
    return x, y


def loss_values():
    x_train_mean = x_train.copy()
    x_train_mode = x_train.copy()
    x_test_test = x_test[:]
    variables = ["Alcohol_Results"]
    res_init = pred(x_train, y_train, x_test_test, y_test)

    for variable in variables:
        mean = x_train_mean[variable].mean()
        x_train_mean[variable].replace(numpy.nan, mean, inplace=True)

        mode = x_train_mode[variable].mode()[0]
        x_train_mode[variable].replace(mode, inplace=True)

    res_not_imputed_mean = pred(x_train_mean, y_train, x_test_test, y_test)
    res_not_imputed_mode = pred(x_train_mode, y_train, x_test_test, y_test)

    x_train_characteristics = x_train.copy()
    x_test_characteristics = x_test_test.copy()

    x_train_characteristics = remove_columns(x_train_characteristics, ["Age", "Alcohol_Results"])
    x_test_characteristics = remove_columns(x_test_characteristics, ["Age", "Alcohol_Results"])

    acc = round(res_init, 3)
    acc_not_imputed_mean = round(res_not_imputed_mean, 3)
    acc_not_imputed_mode = round(res_not_imputed_mode, 3)
    acc_imputed_removed_characteristics = pred(x_train_characteristics, y_train, x_test_characteristics, y_test)

    print("Accuracy: " + str(acc))
    print("Accuracy Mean: " + str(acc_not_imputed_mean))
    print("Accuracy Mode: " + str(acc_not_imputed_mode))
    print("Accuracy Removed Characteristics: " + str(acc_imputed_removed_characteristics))

    best = {acc_not_imputed_mean: "Mean",
            acc_not_imputed_mode: "Mode",
            acc_imputed_removed_characteristics: "Removed Characteristics"}
    sort = sorted(best.items(), reverse=True)[0]

    if sort[1] == "Mean":
        return x_train_mean, "Mean"
    elif sort[1] == "Mode":
        return x_train_mode, "Mode"
    elif sort[1] == "Removed Characteristics":
        return x_train_characteristics, "Removed Characteristics"


if __name__ == "__main__":
    data = load_dataset()
    x, y = preprocess(data)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6)

    print("------------------------------------------------------------------")
    x_train, best = loss_values()
    print(best)
    print("------------------------------------------------------------------")
    #x_train, y_train = SMOTE().fit_resample(x_train, y_train)
    accuracy = pred(x_train, y_train, x_test, y_test, True)
    print("Accuracy: " + str(accuracy))
