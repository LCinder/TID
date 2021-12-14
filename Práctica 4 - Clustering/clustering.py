import numpy
import pandas
import matplotlib.pyplot as plot
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, MeanShift
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

def load_dataset():
    data = pandas.read_csv("data/CustomerSegmentation.csv", delimiter=",")
    return data


def remove_columns(data_arg, elements):
    data = data_arg[:]
    for element in elements:
        data = data.drop(element, axis="columns")
    return data


def preprocess(data):
    x = pandas.DataFrame(data).copy()
    x = remove_columns(x, ["CUST_ID", "PURCHASES"])
    x = x.dropna(how="any", axis=0)

    #########################################################################
    # Normalizar
    #########################################################################
    for column in x.columns:
        x[column] = x[column].astype(float)
        x[column] = x[column] / x[column].max()

    return x


def loss_values():
    x_mean = x.copy()

    variables = ["MINIMUM_PAYMENTS", "CREDIT_LIMIT"]

    for variable in variables:
        mean = x_mean[variable].mean()
        x_mean[variable].replace(numpy.nan, mean, inplace=True)

    return x_mean


def plot_dendogram():
    plot.figure(figsize=(7, 7))
    dendrogram_plot = dendrogram(linkage(x_pre, method="average"), labels=x_pre.index, truncate_mode="lastp", p=50)
    plot.xticks(rotation=90)
    plot.xlabel("Cluster size")
    plot.ylabel("Distance")
    plot.title("Dendogram with optimal clusters")
    plot.savefig("dendogram.png")
    plot.show()


def plot_clusters():
    x_value = "PURCHASES_FREQUENCY"
    y_value = "BALANCE_FREQUENCY"
    COLORS = {0: "r", 1: "b", 2: "g", 3: "y", 4: "o"}
    colors = [COLORS[i] for i in y_pred]
    plot.xlabel(x_value)
    plot.ylabel(y_value)
    plot.scatter(x_pre[x_value], x_pre[y_value], c=colors)
    plot.show()


def choose_clusters(type1):
    dist = []
    for n in range(1, 10):
        cluster = pred(x_pre, n, type1)
        dist.append(cluster.inertia_)

    plot.plot(range(1, 10), dist, "ro-")
    plot.xlabel("Steps")
    plot.ylabel("Distances")
    plot.show()


def pred(x_arg, n, type1):
    if type1 == "kmeans":
        cluster = KMeans(n_clusters=n).fit(x_arg)
    elif type1 == "dbscan":
        cluster = DBSCAN().fit(x_arg)
    elif type1 == "spectralclustering":
        cluster = SpectralClustering().fit(x_arg)
    elif type1 == "meanshift":
        cluster = MeanShift().fit(x_arg)
    return cluster


if __name__ == "__main__":

    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")
    print("------------------------Practica 4--------------------------------")
    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")

    print("Elige una opcion:")
    print("1.-K-Means, 2.- DBSCAN, 3.-SpectralClustering, 4.-MeanShift:")
    option1 = input()
    type1 = ""
    if option1 == "1":
        type1 = "kmeans"
    elif option1 == "2":
        type1 = "dbscan"
    elif option1 == "3":
        type1 = "spectralclustering"
    elif option1 == "4":
        type1 = "meanshift"
    ##########################################################################
    # Preprocess
    ##########################################################################
    data = load_dataset()
    x = preprocess(data)
    x_pre = loss_values()
    ##########################################################################
    # Plot matrix characteristics
    ##########################################################################
    #pairplot(x)
    ##########################################################################
    ##########################################################################
    if type1 == "kmeans":
        choose_clusters(type1)
    ##########################################################################
    # Make Cluster
    ##########################################################################
    cluster = pred(x_pre, 3, type1)
    y_pred = numpy.ravel(cluster.labels_)
    ##########################################################################
    # Plot clusters
    ##########################################################################
    if type1 == "kmeans":
        plot_clusters()
    ##########################################################################
    # Plot dendogram
    ##########################################################################
    plot_dendogram()
    ##########################################################################
    # Metrics
    ##########################################################################
    accuracy = silhouette_score(x_pre, y_pred)
    print("Accuracy: " + str(round(accuracy, 3)))

    #x_pre["y"] = y_pred
    #pairplot(x, vars=x_pre["y"])

    #https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f
    #https://medium.com/analytics-vidhya/clustering-on-iris-dataset-in-python-using-k-means-4735b181affe


