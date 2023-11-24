import pandas as pd
import numpy as np
import math as mt
import scipy.stats as st
from scipy.stats import chi2_contingency
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
    print("0. Salir")
    opcion = int(input("Introduce una opción: "))
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
        
    elif opcion == 0:
        print("Saliendo...")
        break
    else:
        print("Opción incorrecta")


