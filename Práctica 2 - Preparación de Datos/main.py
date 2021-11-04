
import pandas
import numpy
from caimcaim import CAIMD
from chefboost import Chefboost

def loadDataset():
    data = pandas.read_excel("data/autos.xls")
    return data


if __name__=="__main__":
    data = loadDataset()

    #Elimina columnas
    data = data.drop("normalized-losses", axis="columns")

    #Elimina columnas con un 10% de valores perdidos
    lost = int(((100/ 10)/100)*data.shape[0] + 1)
    data = data.dropna(axis="columns", thresh=lost)

    # Rellenar valores columna con media
    data["bore"] = data["bore"].replace("?", None)
    data["bore"] = (data["bore"].astype(float)).astype(int)

    mean = data["bore"].mean()
    data["bore"].fillna(value=mean)

    # Rellenar otros valores como strings con mas frecuente, moda
    for column in data.columns.values:
        mode = data[column].mode()[0]
        data[column].replace("?", mode, inplace=True)

    #Realizar muestreo 30%
    data = data.sample(frac=0.3, random_state=1)

    #Convertir primera columna symboling de entero a string
    #data["symboling"] = data["symboling"].astype(str)

    #Discretizar con caim
    caim = CAIMD()
    x = data["wheel-base"]
    y = data["symboling"]
    #data_discr = caim.fit_transform(x, y)



    