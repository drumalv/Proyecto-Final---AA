#####
# Proyecto Final: Census data Income
# Authors: Alvaro Beltran Camacho - Yabir Garcia Benchakhtir
# Aprendije automatico 2020
#####

# Importamos las librerias que necesitamos. Entre ellas tenemos
# pandas para tener un control amplio sobre los datos, matplotlib
# para realizar gráficos y sklearn que incluye los métodos de 
# aprendizaje que vamos a desarrollar

import pandas as pd
import numpy as np 

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn import model_selection 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, LassoCV, Perceptron, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier


# Establecemos la semilla para los procesos aleatorios

SEED = 1
np.random.seed(SEED)

# Definimos la carpeta donde se encuentran los datos

data_folder = "data/"

data_training = data_folder + "adult.data"
data_test = data_folder + "adult.test"

# Fijamos una lista de las columnas que tienen nuestros datos 
# de esta manera tenemos mas informaicion en los dataframes de 
# pandas

headers = [
    "age", 
    "workclass", 
    "fnlwgt", 
    "education", 
    "education-num", 
    "marital-status", 
    "occupation", 
    "relationship", 
    "race", 
    "sex", 
    "capital-gain",
    "captial-loss", 
    "hours-per-week", 
    "native-country", 
    "income"
]

# leemos los datos desde el archivo de datos. Este tiene irregularidades
# en el formato, ya que hemos visto que a veces las variables se separan 
# con espacios y otras no. Hemos optados por usar una expresión regular para solverntar
# este problema

df_train = pd.read_csv(data_training, index_col=False, delimiter=",[ ]*", names=headers)
df_test = pd.read_csv(data_test, index_col=False, delimiter=",[ ]*", names=headers)

# combinamos los datos que tenemos en un solo conjunto para realizar
# el limpiado de datos

df = pd.concat([df_train,df_test])


# Limpiamos los datos
# en primer lugar vamos a sustituir los simbolos de interrogacion 
# que hay en el dataset por valores NaN de numpy 

df = df.replace(' ?', np.NaN)
df = df.replace('?', np.NaN)

# mostramos el numero de valores null que hay en el dataset
print("Numero de valores perdidos en el conjunto de datos: {}".format(
    df.isnull().sum(axis=0).sort_values(ascending = False).head(30)))

# sustituimos los valores de income por etiquetas 0 y 1

df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

# mostramos la cantidad de cada tipo de datos segun la etiqueta income

print("Numero de datos de cada clase")
print(df.income.value_counts())

# eliminamos las columnas que no vamos a utilizar en el analisis

df = df.drop(['fnlwgt'], axis=1)
df = df.drop(['education-num'], axis=1)

# Realizamos la coficacion de las variables categoricas

# en primer lugar listamos las variables que son categoricas

cols_with_categories = [
    'workclass', 
    'education',
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]

# Trabajar con los datos perdidos

print("Datos perdidos por columnas: ")
print(df.isnull().sum(axis=0).sort_values(ascending = False).head(15))
print("\nDatos perdidos por filas: ")
print(df.isnull().sum(axis=1).sort_values(ascending = False).head(15))

# En primer lugar eliminamos los datos que tengan mas de un 10 % de 
# valores perdidos

df.dropna(thresh=14, inplace=True, axis=0)

# Para los valores con menos de 10% de los valores perdidos aproximamos por 
# una multinomial con el resto de valores en la columna
# En este caso las columnas con valores perdidos que quedan son
# native-country y occupation

s = df["native-country"].value_counts(normalize=True)
missing = df["native-country"].isnull()
df.loc[missing,"native-country"] = np.random.choice(s.index, size=len(df[missing]),p=s.values)

s = df["occupation"].value_counts(normalize=True)
missing = df["occupation"].isnull()
df.loc[missing,"occupation"] = np.random.choice(s.index, size=len(df[missing]),p=s.values)

print("Datos perdidos por columnas despues del procesado: ")
print(df.isnull().sum(axis=0).sort_values(ascending = False).head(30))
print("Datos perdidos por filas despues del procesado: ")
print(df.isnull().sum(axis=1).sort_values(ascending = False).head(30))

# sustituimos las variables categoricas por una codificacion de 1s y 0s
print("Tamaño antes del conjunto de datos antes de recodificar las variables: {}".format(df.shape))
df = pd.get_dummies(data=df, columns=cols_with_categories)
print("Tamaño antes del conjunto de datos despues de recodificar las variables: {}".format(df.shape))


#https://stackoverflow.com/questions/44867219/pandas-filling-na-values-to-be-filled-based-on-distribution-of-existing-values

# creamos los conjuntos de training y de test
X, y = df[df.columns.difference(['income'])], df['income']
X, y = shuffle(X, y, random_state=SEED)
train_x, test_x, train_y, test_y = train_test_split(X,y, test_size=0.3, stratify=y)


# creamos la pipeline de preprocesado

preproc = [
    ("var", VarianceThreshold(0.1)),   
    ("standardize", StandardScaler()),      
    ("lasso", SelectFromModel(estimator=LassoCV(tol=0.001))),
    ("poly",PolynomialFeatures(1)), 
    ("var2", VarianceThreshold(0.1)),   
]