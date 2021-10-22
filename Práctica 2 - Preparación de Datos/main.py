
import pandas

def loadDataset():
    data = pandas.read_excel("data/autos.xls")
    pandas
    return data


if __name__=="__main__":
    data = loadDataset()
    print(data)

    #Elimina columnas, 1 porque es una columna
    data = data.drop("normalized-losses", axis="columns")

    #Elimina columnas con un 10% de valores perdidos
    lost = int(((100/ 10)/100)*data.shape[0] + 1)
    data = data.dropna(axis="columns", thresh=lost)

    # Rellenar valores columa con media
    mean = data["bore"].mean()
    data["bore"].fillna(value=mean)
    print(data)

    # Rellenar otros valores como strings con mas frecuente, moda
    # iloc devuelve la i-esima columna
    for i in range(data.shape[0]):
        data.iloc(i).mode()
    print(data)
