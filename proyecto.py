import pandas as pd
import numpy as np
import math as mt
import scipy.stats as st
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os


def leer_archivo( archivo ):
    df = pd.read_csv( archivo )
    df = df.fillna(0)
    df = df.drop(df.index[14:23])
    if df.iloc[0, 0] == 0:
        df = df.drop(df.index[0])
    df = df.reset_index(drop=True)
    df = df.iloc[0:14, 0:12]
    df = df.rename(columns={'Unnamed: 0': 'Type of case'})
    for i in range(1, 12):
        df.iloc[:, i] = pd.to_numeric(df.iloc[:, i], errors='coerce')
    for i in range(1, 12):
        if '\xa0%' in df.columns[i]:
            df.columns.values[i] = df.columns[i].replace('\xa0%', ' %')
            for j in range(0, 13):
                df.iloc[j, i] = (df.iloc[j, i-1] / df.iloc[j, 1]) * 100
                df.iloc[j, i] = round(df.iloc[j, i], 2)

    return df

def media_por_columna( df ):
    media = []
    for i in range(1, len(df.columns)):
        if " %" in df.columns[i]:
            continue
        else :
            #append the title of the column
            media.append(df.columns[i])
            #append the mean of the column
            media.append(round(df.iloc[:, i].mean(),2))
    media = np.array(media).reshape(-1, 2) #reshape para tener una matriz de dos columnas
    return media

def media_por_fila(df):
    medias = []
    for i in range(1, len(df)):  # Asegurémonos de iterar sobre las filas correctamente
        # skip if the cell has a % sign
        if "%" not in str(df.iloc[i, 1:11].values).lower():
            # append the title of the column
            medias.append(df.iloc[i, 0])
            # append the mean of the column
            medias.append(round(df.iloc[i, 1:11].mean(),2))

    medias = np.array(medias).reshape(-1, 2)  # reshape para tener una matriz de dos columnas
    return medias

def suma_por_columna(df):
    suma = []
    for i in range(1, len(df.columns)):
        if " %" in df.columns[i]:
            continue
        else:
            # append the title of the column
            suma.append(df.columns[i])
            # append the mean of the column
            suma.append(round(df.iloc[:, i].sum(), 2))
    suma = np.array(suma).reshape(-1, 2)  # reshape para tener una matriz de dos columnas
    return suma

def suma_por_fila(df):
    sumas = []
    for i in range(1, len(df)):  # Asegurémonos de iterar sobre las filas correctamente
        # skip if the cell has a % sign
        if "%" not in str(df.iloc[i, 1:11].values).lower():
            # append the title of the column
            sumas.append(df.iloc[i, 0])
            # append the mean of the column
            sumas.append(round(df.iloc[i, 1:11].sum(), 2))

    sumas = np.array(sumas).reshape(-1, 2)  # reshape para tener una matriz de dos columnas
    return sumas

def mediana_por_columna(df):
    mediana = []
    for i in range(1, len(df.columns)):
        if " %" in df.columns[i]:
            continue
        else:
            # append the title of the column
            mediana.append(df.columns[i])
            # append the mean of the column
            mediana.append(round(df.iloc[:, i].median(), 2))
    mediana = np.array(mediana).reshape(-1, 2)  # reshape para tener una matriz de dos columnas
    return mediana

def mediana_por_fila(df):
    medianas = []
    for i in range(1, len(df)):  # Asegurémonos de iterar sobre las filas correctamente
        # skip if the cell has a % sign
        if "%" not in str(df.iloc[i, 1:11].values).lower():
            # append the title of the column
            medianas.append(df.iloc[i, 0])
            # append the mean of the column
            medianas.append(round(df.iloc[i, 1:11].median(), 2))

    medianas = np.array(medianas).reshape(-1, 2)  # reshape para tener una matriz de dos columnas
    return medianas

def moda_por_columna(df):
    moda = []
    for i in range(1, len(df.columns)):
        if " %" in df.columns[i]:
            continue
        else:
            # append the title of the column
            moda.append(df.columns[i])
            # append the mean of the column
            moda.append(round(df.iloc[:, i].mode(), 2))
    moda = np.array(moda).reshape(-1, 2)  # reshape para tener una matriz de dos columnas
    return moda

def moda_por_fila(df):
    modas = []
    for i in range(1, len(df)):  # Asegurémonos de iterar sobre las filas correctamente
        # skip if the cell has a % sign
        if "%" not in str(df.iloc[i, 1:11].values).lower():
            # append the title of the column
            modas.append(df.iloc[i, 0])
            # append the mean of the column
            modas.append(round(df.iloc[i, 1:11].mode(), 2))

    modas = np.array(modas).reshape(-1, 2)  # reshape para tener una matriz de dos columnas
    return modas

def desviacion_tipica_por_columna(df):
    desviacion_tipica = []
    for i in range(1, len(df.columns)):
        if " %" in df.columns[i]:
            continue
        else:
            # append the title of the column
            desviacion_tipica.append(df.columns[i])
            # append the mean of the column
            desviacion_tipica.append(round(df.iloc[:, i].std(), 2))
    desviacion_tipica = np.array(desviacion_tipica).reshape(-1, 2)  # reshape para tener una matriz de dos columnas
    return desviacion_tipica

def desviacion_tipica_por_fila(df):
    desviaciones_tipicas = []
    for i in range(1, len(df)):  # Asegurémonos de iterar sobre las filas correctamente
        # skip if the cell has a % sign
        if "%" not in str(df.iloc[i, 1:11].values).lower():
            # append the title of the column
            desviaciones_tipicas.append(df.iloc[i, 0])
            # append the mean of the column
            desviaciones_tipicas.append(round(df.iloc[i, 1:11].std(), 2))

    desviaciones_tipicas = np.array(desviaciones_tipicas).reshape(-1, 2)  # reshape para tener una matriz de dos columnas
    return desviaciones_tipicas

def varianza_por_columna(df):
    varianza = []
    for i in range(1, len(df.columns)):
        if " %" in df.columns[i]:
            continue
        else:
            # append the title of the column
            varianza.append(df.columns[i])
            # append the mean of the column
            varianza.append(round(df.iloc[:, i].var(), 2))
    varianza = np.array(varianza).reshape(-1, 2)  # reshape para tener una matriz de dos columnas
    return varianza

def varianza_por_fila(df):
    varianzas = []
    for i in range(1, len(df)):  # Asegurémonos de iterar sobre las filas correctamente
        # skip if the cell has a % sign
        if "%" not in str(df.iloc[i, 1:11].values).lower():
            # append the title of the column
            varianzas.append(df.iloc[i, 0])
            # append the mean of the column
            varianzas.append(round(df.iloc[i, 1:11].var(), 2))

    varianzas = np.array(varianzas).reshape(-1, 2)  # reshape para tener una matriz de dos columnas
    return varianzas

def chi_cuadrada(df):


    # Obtenemos el valor total de cases abiertos en la fila Chroma
    total = df.iloc[12, 1]
    print("Total de cases abiertos en la fila Chroma: ", total)

    # Obtenemos las skins rojas de la fila Chroma
    rojas = df.iloc[12, 10]
    print("Skins rojas obtenidas en la fila Chroma: ", rojas)

    # Obtenemos las skins rojas no obtenidas de la fila Chroma
    no_rojas = total - rojas

    # Creamos la tabla de contingencia
    observed = np.array([[rojas, total - rojas], [no_rojas, total - no_rojas]])

    print("Tabla de contingencia: ")
    print(observed)

    # Obtenemos el valor de chi cuadrado
    chi2, p_value, _, _ = chi2_contingency(observed)

    # Impresión de resultados
    print("Valor de Chi cuadrado: ", chi2)
    print(f"Valor p: {p_value:.10f}")

    # Interpretación de resultados
    alpha = 0.05  # Nivel de significancia
    if p_value < alpha:
        print("La diferencia es estadísticamente significativa. Rechazamos la hipótesis nula.")
    else:
        print("No hay evidencia suficiente para rechazar la hipótesis nula.")

def grafica_de_dispercion(df):
   
    # porcentaje de skins rojas
    porcentaje = df.iloc[:, 9]
    # cases abiertos
    Totales = df.iloc[:, 1]

    model = LinearRegression()

    # reshape para tener una matriz de dos columnas
    porcentaje = np.array(porcentaje).reshape(-1, 1)
    Totales = np.array(Totales).reshape(-1, 1)

    model.fit(porcentaje, Totales)

    # prediccion

    y_pred = model.predict(porcentaje)

    # grafica de dispersion
    plt.scatter(porcentaje, Totales, color='red', label='Datos')
    plt.plot(porcentaje, y_pred, color='blue', linewidth=3, label='Regresión lineal')
    plt.title('Regresión lineal')
    plt.xlabel('Porcentaje de skins rojas')
    plt.ylabel('Cases abiertos')
    plt.legend()
    plt.show()  
    

## MENU ##

def menu():
    
    print("_"*100, "\n")
    print("__________MENU__________")
    print("1. Media por columna")
    print("2. Media por fila")
    print("3. Suma por columna")
    print("4. Suma por fila")
    print("5. Mediana por columna")
    print("6. Mediana por fila")
    print("7. Moda por columna")
    print("8. Moda por fila")
    print("9. Desviación típica por columna")
    print("10. Desviación típica por fila")
    print("11. Varianza por columna")
    print("12. Varianza por fila")
    print("13. Hipotesis (chi cuadrado)")
    print("14. Grafica de dispercion")
    print("15. Conjuntos de entrenamiento y prueba")
    print("16. Regresión lineal")
    print("17. Prediccion y evaluacion del modelo")
    print("18. Grafica de dispercion con regresion lineal")
    print("19. Interpretacion de resultados")
    print("0. Salir")
    print("_"*100, "\n")

    opcion = int(input("Opción: "))
    # clear the screen 
    os.system('cls')
    return opcion

## MAIN ##

df = leer_archivo("cases - Total.csv")

while True:
    opcion = menu()
    if opcion == 1:
        print("Media por columna")
        print(media_por_columna(df)) 
        # pause the screen until the user press a key
        os.system('pause')

    elif opcion == 2:
        print("Media por fila")
        print(media_por_fila(df))
        # pause the screen until the user press a key
        os.system('pause')

    elif opcion == 3:
        print("Suma por columna")
        print(suma_por_columna(df))
        # pause the screen until the user press a key
        os.system('pause')

    elif opcion == 4:
        print("Suma por fila")
        print(suma_por_fila(df))
        # pause the screen until the user press a key
        os.system('pause')

    elif opcion == 5:
        print("Mediana por columna")
        print(mediana_por_columna(df))
        # pause the screen until the user press a key
        os.system('pause')

    elif opcion == 6:
        print("Mediana por fila")
        print(mediana_por_fila(df))
        # pause the screen until the user press a key
        os.system('pause')

    elif opcion == 7:
        print("Moda por columna")
        print(moda_por_columna(df))
        # pause the screen until the user press a key
        os.system('pause')

    elif opcion == 8:
        print("Moda por fila")
        print(moda_por_fila(df))
        # pause the screen until the user press a key
        os.system('pause')

    elif opcion == 9:
        print("Desviación típica por columna")
        print(desviacion_tipica_por_columna(df))
        # pause the screen until the user press a key
        os.system('pause')

    elif opcion == 10:
        print("Desviación típica por fila")
        print(desviacion_tipica_por_fila(df))
        # pause the screen until the user press a key
        os.system('pause')

    elif opcion == 11:
        print("Varianza por columna")
        print(varianza_por_columna(df))
        # pause the screen until the user press a key
        os.system('pause')

    elif opcion == 12:
        print("Varianza por fila")
        print(varianza_por_fila(df))
        # pause the screen until the user press a key
        os.system('pause')

    elif opcion == 13:
        print("Hipotesis (chi cuadrado)")
        print("Nuestra hipotesis nula La probabilidad de sacar una skin de categoría roja es igual al 1.2% (p = 0.012)")
        print("H0: mu1 = mu2")
        print("Nuestra hipotesis alternativa La probabilidad de sacar una skin de categoría roja no es igual al 1.2%")
        print("H1: mu1 != mu2")
        print("Tomaremos la fila con mas cases abiertos, en este caso la fila Chroma, con un total de 942 cases abiertos")

        os.system('pause')

        chi_cuadrada(df)
        
        # pause the screen until the user press a key
        os.system('pause')
        
    elif opcion == 14:
        print("Grafica de dispercion")
        print("Variable dependiente: Cases abiertos")
        print("Variable independiente: Porcentaje de skins rojas")

        # pause the screen until the user press a key
        os.system('pause')

        grafica_de_dispercion(df)

    elif opcion == 15:
        
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        print("Conjunto de entrenamiento")
        print(train_df)
        print("Conjunto de prueba")
        print(test_df)

        # pause the screen until the user press a key
        os.system('pause')

    elif opcion == 16:
        print("Variable dependiente: Cases abiertos")
        print("Variable independiente: Porcentaje de skins rojas")

        # pause the screen until the user press a key
        os.system('pause')

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        X_train = train_df.iloc[:, 9].values.reshape(-1, 1)
        y_train = train_df.iloc[:, 1].values.reshape(-1, 1)
        X_test = test_df.iloc[:, 9].values.reshape(-1, 1)
        y_test = test_df.iloc[:, 1].values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X_train, y_train)
        b0 = model.intercept_
        b1 = model.coef_


        print(f'Ecuación de regresión: Y = {b0[0]:.4f} + {b1[0][0]:.4f} * x1')

        # pause the screen until the user press a key
        os.system('pause')

    elif opcion == 17:
        print("Variable dependiente: Cases abiertos")
        print("Variable independiente: Porcentaje de skins rojas")

        # pause the screen until the user press a key
        os.system('pause')

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        X_train = train_df.iloc[:, 9].values.reshape(-1, 1)
        y_train = train_df.iloc[:, 1].values.reshape(-1, 1)
        X_test = test_df.iloc[:, 9].values.reshape(-1, 1)
        y_test = test_df.iloc[:, 1].values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Predicción")
        print(y_pred)
        print("Evaluación del modelo")
        print(f'Error cuadrático medio: {mean_squared_error(y_test, y_pred):.2f}')

        # pause the screen until the user press a key
        os.system('pause')

    elif opcion == 18:
        print("Variable dependiente: Cases abiertos")
        print("Variable independiente: Porcentaje de skins rojas")

        # pause the screen until the user press a key
        os.system('pause')

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        X_train = train_df.iloc[:, 9].values.reshape(-1, 1)
        y_train = train_df.iloc[:, 1].values.reshape(-1, 1)
        X_test = test_df.iloc[:, 9].values.reshape(-1, 1)
        y_test = test_df.iloc[:, 1].values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # grafica de dispersion
        plt.scatter(X_test, y_test, color='red', label='Datos')
        plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regresión lineal')
        plt.title('Regresión lineal')
        plt.xlabel('Porcentaje de skins rojas')
        plt.ylabel('Cases abiertos')
        plt.legend()
        plt.show()

        # pause the screen until the user press a key
        os.system('pause')

    elif opcion == 19:

        print("Variable dependiente: Cases abiertos")
        print("Variable independiente: Porcentaje de skins rojas")
        print("La ecuación de regresión es: Y = 0.55 + 0.01 * x1")
        print("Interpretación de los coeficientes:")
        print("b0 = 0.55: Es el valor promedio de cases abiertos cuando el porcentaje de skins rojas es igual a 0%")
        print("b1 = 0.01: Por cada incremento de 1'%' en el porcentaje de skins rojas, el número de cases abiertos aumenta en 0.01")
        print("Interpretación del coeficiente de determinación:")
        print("R2 = 0.0001: El modelo explica el 0.01'%' de la variabilidad de la variable dependiente")

        # pause the screen until the user press a key
        os.system('pause')
    elif opcion == 0:
        print("Saliendo...")
        break
    else:
        print("Opción incorrecta")


