
import matplotlib.pyplot as plot
import numpy as np
import pandas
from pandas.plotting import parallel_coordinates
from pandas import read_excel

def ejercicio1():
    x = ["150-155", "155-160", "160-165", "165-170", "170-175", "175-180", "180-185", "185-190"]
    y = [2, 18, 24, 42, 56, 32, 20, 5]

    plot.figure(figsize=[10, 8])
    plot.bar(x, y, color="r")
    plot.xlabel("Altura (cm)")
    plot.ylabel("Nº Alumnos")
    plot.title("Altura alumnos")
    plot.xticks(rotation=45)
    plot.savefig("ejercicio1.png")
    plot.show()


def ejercicio2():
    x = [[23, 12, 43, 54, 12, 1, 13, 0], [12, 12, 12, 23, 23, 23, 23, 12],
    [34, 23, 15, 65, 78, 45, 23, 41], [34, 21, 20, 21, 24, 57, 56, 60], [23, 23, 23, 23, 12, 56, 34, 34]]

    plot.boxplot(x, patch_artist=True)
    plot.xlabel("Dias")
    plot.ylabel("Nº Llamadas")
    plot.title("Nº Llamadas/Dia")
    plot.savefig("ejercicio2.png")
    plot.show()


def ejercicio3():
    x = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 1, 0, 2, 0, 0, 2, 0,
    0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0,
    1, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0,
    1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y = [0, 1, 2]

    plot.hist(x, color="r")
    plot.xticks(range(3))
    plot.title("Nº Ocurrencias decodificador en histograma")
    plot.savefig("ejercicio3_hist.png")
    plot.show()

    x_pie = [np.count_nonzero(x == 0), np.count_nonzero(x == 1), np.count_nonzero(x == 2)]
    plot.pie(x_pie, labels=y)
    plot.savefig("ejercicio3_pie.png")
    plot.show()


def ejercicio4():
    y_mat = [4.5, 2.8, 6.75, 1.25, 9.8, 10, 5.5, 5, 3.4, 2.1, 0, 8.7, 6.25]
    y_fis = [2, 1.2, 5, 4.5, 8, 9.9, 1.75, 4.75, 5, 1.8, 0, 9.75, 4.5]

    plot.scatter(range(len(y_mat)), y_mat, color="b", label="Matemáticas")
    plot.scatter(range(len(y_fis)), y_fis, color="r", label="Física")
    plot.xlabel("Nº Alumno")
    plot.ylabel("Nota obtenida")
    plot.title("Nº Alumnos y notas obtenidas")
    plot.legend()
    plot.savefig("ejercicio4.png")
    plot.show()

def ejercicio5():
    d = [["Paciente 1", 0.34, 0.2, 0.56, 0.12, 0.9, 0.8], ["Paciente 2", 0.5, 0.8, 0.9, 0.7, 0.5, 0.9], ["Paciente 3",0.12, 0.15, 0.23, 0.45, 0.6, 0.1], ["Paciente 4",0.1, 0.6, 0.1, 0.4, 0.3, 0.9]]
    d2 = [["Paciente 1", "Tabaquismo", 0.34, 0.5, 0.12, 0.1], ["Paciente 2", "Obesidad", 0.2, 0.8, 0.15, 0.6], ["Paciente 3", "Tensión", 0.56, 0.9, 0.23, 0.1], ["Paciente 4", "Pulsaciones", 0.12, 0.7, 0.45, 0.4], ["Paciente 5", "Edad", 0.9, 0.5, 0.6, 0.3], ["Paciente 6", "Hierro", 0.8, 0.9, .01, 0.9]]
    data = pandas.DataFrame(d, columns=["Paciente", "Tabaquismo", "Obesidad", "Tensión", "Pulsaciones", "Edad", "Hierro"])
    print(data)
    parallel_coordinates(data, class_column="Paciente", color=["b", "g", "c", "r"])
    plot.xlabel("Enfermedad")
    plot.ylabel("Índice")
    plot.title("Enfermedad vs Índice")
    plot.legend()
    plot.savefig("ejercicio5.png")
    plot.show()

def ejercicio6():
    data = read_excel("prestamo.xls", "datos")
    creditCard = data["CreditCard"].map({0: "No", 1: "Si"})
    familySize = data["Family"]
    age = data["Age"]
    experience = data["Experience"]

    plot.hist(creditCard, color="r")
    plot.title("Nº Personas Tarjeta Crédito")
    plot.savefig("ejercicio6_hist.png")
    plot.show()

    plot.scatter(range(len(familySize)), familySize, alpha=0.5, s=500)
    plot.yticks(range(max(familySize)+1))
    plot.title("Nº Miembros Familia")
    plot.savefig("ejercicio6_bubble.png")
    plot.show()

    plot.scatter(age, experience, alpha=0.5, c="r")
    plot.xlabel("Edad (años)")
    plot.ylabel("Experiencia (años)")
    plot.title("Edad vs Experiencia (años)")
    plot.savefig("ejercicio6_age_experience.png")
    plot.show()

    xLabel = "Income"
    yLabel = "Education"
    x = data[xLabel]
    y = data[yLabel]
    plot.title(xLabel + " vs " + yLabel)
    plot.scatter(x, y, alpha=0.5, c="r")
    plot.xlabel(xLabel)
    plot.ylabel(yLabel)
    plot.savefig("ejercicio6_" + xLabel + "_" + yLabel + ".png")
    plot.show()

if __name__ == '__main__':
    ejercicio5()
