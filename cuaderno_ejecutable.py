#!/usr/bin/env python
# coding: utf-8

# In[1]:


#####
# Proyecto Final: Census data Income
# Authors: Alvaro Beltran Camacho - Yabir Garcia Benchakhtir
# Aprendije automatico 2020
#####

# Importamos las librerias que necesitamos. Entre ellas tenemos
# pandas para tener un control amplio sobre los datos, matplotlib
# para realizar gráficos y sklearn que incluye los métodos de 
# aprendizaje que vamos a desarrollar


# In[2]:


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

from sklearn.metrics import confusion_matrix


# In[3]:


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

df_train = pd.read_csv(data_training, index_col=False, delimiter=",", names=headers)
df_test = pd.read_csv(data_test, index_col=False, delimiter=",", names=headers)

# combinamos los datos que tenemos en un solo conjunto para realizar
# el limpiado de datos

df = pd.concat([df_train,df_test], ignore_index=True)


# Limpiamos los datos
# en primer lugar vamos a sustituir los simbolos de interrogacion 
# que hay en el dataset por valores NaN de numpy 



df = df.replace('?', np.NaN)
df = df.replace(' ?', np.NaN)

# mostramos el numero de valores null que hay en el dataset
print("Numero de valores perdidos en el conjunto de datos: {}".format(
    df.isnull().sum(axis=0).sort_values(ascending = False).head(30)))
input("\n--- Pulsar tecla para continuar ---\n")
# sustituimos los valores de income por etiquetas 0 y 1


df['income'] = df['income'].str.strip()
df['income'] = df['income'].str.replace(".", "")
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})


# In[8]:


# mostramos el numero de valores null que hay en el dataset
print("Numero de valores perdidos en el conjunto de datos: {}".format(
    df.isnull().sum(axis=0).sort_values(ascending = False).head(30)))

input("\n--- Pulsar tecla para continuar ---\n")
# In[9]:


print("Numero de datos de cada clase")
print(df.income.value_counts())

input("\n--- Pulsar tecla para continuar ---\n")

# In[10]:


# Mostramos el número de muestras de cada clase


# In[11]:


plot = df.income.value_counts().plot(kind="bar", title="Numero de muestras de cada clase", legend=True, figsize=(10,10))
plot.set_xlabel("Clase")
plot.set_ylabel("Número de muestras")
plot.set_xticklabels( ('<=50K', '>50K'), rotation=1)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# In[12]:


# eliminamos las columnas que no vamos a utilizar en el analisis
df = df.drop(['fnlwgt'], axis=1)
df = df.drop(['education-num'], axis=1)


# In[13]:


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


# In[14]:


# Trabajar con los datos perdidos

print("Datos perdidos por columnas: ")
print(df.isnull().sum(axis=0).sort_values(ascending = False).head(15))
print("\nDatos perdidos por filas: ")
print(df.isnull().sum(axis=1).sort_values(ascending = False).head(15))

input("\n--- Pulsar tecla para continuar ---\n")
# In[15]:

print("Descripción del dataset")
df2 = df.copy()
print(df2.shape)

input("\n--- Pulsar tecla para continuar ---\n")
# En primer lugar eliminamos los datos que tengan mas de un 10 % de 
# valores perdidos
df.dropna(thresh=df.shape[1]-1, inplace=True, axis=0)

print("Eliminamos los datos con más de un 10 por ciento de valores perdidos")
print("\nDatos perdidos por filas: ")
print(df.isnull().sum(axis=1).sort_values(ascending = False))

input("\n--- Pulsar tecla para continuar ---\n")

print("\nDatos perdidos por columnas: ")
print(df.isnull().sum(axis=0).sort_values(ascending = False).head(30))

input("\n--- Pulsar tecla para continuar ---\n")
# In[16]:


# Para los valores con menos de 10% de los valores perdidos aproximamos por 
# una multinomial con el resto de valores en la columna
# En este caso las columnas con valores perdidos que quedan son
# native-country y occupation

print("Procedemos a aplicar una multinomial para aproximar los valores que faltan: ")
s = df["native-country"].value_counts(normalize=True)
missing = df["native-country"].isnull()
df.loc[missing,"native-country"] = np.random.choice(s.index, size=len(df[missing]),p=s.values)

s = df["occupation"].value_counts(normalize=True)
missing = df["occupation"].isnull()
df.loc[missing,"occupation"] = np.random.choice(s.index, size=len(df[missing]),p=s.values)

print("Datos perdidos por filas despues del procesado: ")
print(df.isnull().sum(axis=1).sort_values(ascending = False))

input("\n--- Pulsar tecla para continuar ---\n")
print("\nDatos perdidos por columnas despues del procesado: ")
print(df.isnull().sum(axis=0).sort_values(ascending = False).head(30))

input("\n--- Pulsar tecla para continuar ---\n")
# In[17]:


# sustituimos las variables categoricas por una codificacion de 1s y 0s
print("Tamaño antes del conjunto de datos antes de recodificar las variables: {}".format(df.shape))
df = pd.get_dummies(data=df, columns=cols_with_categories)
print("Tamaño antes del conjunto de datos despues de recodificar las variables: {}".format(df.shape))

input("\n--- Pulsar tecla para continuar ---\n")


# creamos los conjuntos de training y de test
X, y = df[df.columns.difference(['income'])], df['income']
X, y = shuffle(X, y, random_state=SEED)
train_x, test_x, train_y, test_y = train_test_split(X,y, test_size=0.8, stratify=y)


# creamos la pipeline de preprocesado
preproc = [
    ("var", VarianceThreshold(0.01)),   
    ("standardize", StandardScaler()),      
    ("lasso", SelectFromModel(estimator=LassoCV(tol=0.01))),
]


# In[21]:


p = Pipeline(preproc)


# In[22]:


x_train_prep = p.fit_transform(train_x, train_y)
print("Descripción de los datos antes y después del preprocesado")
print("Antes: {}".format(train_x.shape))
print("Despues: {}".format(x_train_prep.shape))

input("\n--- Pulsar tecla para continuar ---\n")
# In[23]:


# Modelo lineal

preproc_lin = [
    ("var", VarianceThreshold(0.01)),   
    ("standardize", StandardScaler()),      
    ("poly",PolynomialFeatures(1)), 
    ("lasso", SelectFromModel(estimator=LassoCV(tol=0.01))),
]

pipe_lineal = Pipeline(steps=preproc_lin+[('estimator', LogisticRegression())])
params_lineal = {
    'estimator':[LogisticRegression(max_iter=500)],
    'estimator__solver':['lbfgs'],
    'estimator__C': np.logspace(-6, 6, 4),
    'estimator__penalty': ['l2'],
    'poly__degree': [1,2],
    'estimator__tol': [1e-3, 1e-4, 1e-2]
}
best_clf_lin = GridSearchCV(pipe_lineal, params_lineal, scoring = 'f1',cv = 5, n_jobs = -1, verbose=1)



# Entrenamos
best_clf_lin.fit(train_x, train_y)


# In[25]:


print("\nMejores parámetros para LR: ")
best_clf_lin.best_params_

print("F1 en training:", round(100.0 * best_clf_lin.score(train_x, train_y), 5))
print("F1 en test: ", round(100.0 * best_clf_lin.score(test_x, test_y),5))
input("\n--- Pulsar tecla para continuar ---\n")

# In[27]:


# Random Forest
pipe_lineal = Pipeline(steps=preproc+[('estimator', RandomForestClassifier(random_state = SEED))])
params_lineal = {
    'estimator':[RandomForestClassifier(random_state = SEED)],
    'estimator__criterion': ['gini','entropy'],
    'estimator__max_features': ['sqrt'],
    'estimator__bootstrap':['True'],
    'estimator__min_samples_split': [2,3,4,5]
}
best_clf_random = GridSearchCV(pipe_lineal, params_lineal, scoring = 'f1',cv = 5, n_jobs = -1, verbose=1)


# In[28]:


best_clf_random.fit(train_x, train_y)


# In[29]:


print("\nMejores parámetros para RF: ")
print(best_clf_random.best_params_)


# In[30]:


print("F1 en training:", round(100.0 * best_clf_random.score(train_x, train_y),5))
print("F1 en test: ", round(100.0 * best_clf_random.score(test_x, test_y),5))
input("\n--- Pulsar tecla para continuar ---\n")

# In[31]:


# Perceptron
pipe_perceptron = Pipeline(steps=preproc_lin+[('estimator', Perceptron(random_state = SEED))])
params_perceptron = {
    'estimator':[Perceptron(random_state = SEED)],
    'estimator__penalty': ['l1'],
    'estimator__alpha':[1.0, 1e-2, 1e-3, 1e-4, 2, 5],
    'estimator__max_iter':[2000],
    'estimator__tol': np.logspace(-6, 1, 3),
    'estimator__shuffle': [True],
    'poly__degree': [1,2]
}
best_clf_perceptron = GridSearchCV(pipe_perceptron, params_perceptron, scoring = 'f1',cv = 5, n_jobs = -1, verbose=1)# 


# In[32]:


best_clf_perceptron.fit(train_x, train_y)


# In[33]:

print("\nMejores parámetros para Perceptron: ")
print(best_clf_perceptron.best_params_)
print("F1 en training:", round(100.0 * best_clf_perceptron.score(train_x, train_y),5))
print("F1 en test: ", round(100.0 * best_clf_perceptron.score(test_x, test_y),5))


input("\n--- Pulsar tecla para continuar ---\n")
# In[35]:


# MLP
pipe_MLP = Pipeline(steps=preproc+[('estimator', MLPClassifier(random_state = SEED))])
params_MLP = {
    'estimator__activation': ['logistic', 'tanh', 'relu'],
    'estimator__solver': ['lbfgs'],
    'estimator__alpha': [1.0, 1e-2, 1e-3, 1e-4, 2, 5, 10],
    'estimator__max_fun': [20000]
}
best_clf_mlp = GridSearchCV(pipe_MLP, params_MLP, scoring = 'f1',cv = 5, n_jobs = -1, verbose=1)# 


# In[36]:


best_clf_mlp.fit(train_x, train_y)


# In[37]:


print("\nMejores parámetros para MLP: ")
print(best_clf_mlp.best_params_)

print("F1 en training:", round(100.0 * best_clf_mlp.score(train_x, train_y),5))
print("F1 en test: ", round(100.0 * best_clf_mlp.score(test_x, test_y),5))

input("\n--- Pulsar tecla para continuar ---\n")
# In[39]:


# SVM
pipe_SVM = Pipeline(steps=preproc+[('estimator', SVC(gamma = "scale", kernel="rbf"))])
params_SVM = {
    'estimator__C': [0.1, 1, 2, 5, 7, 10],
}
best_clf_svm = GridSearchCV(pipe_SVM, params_SVM, scoring = 'f1',cv = 5, n_jobs = -1, verbose=1)# 


# In[40]:


best_clf_svm.fit(train_x, train_y)


# In[41]:


print("\nMejores parámetros para SVM: ")
print(best_clf_svm.best_params_)


# In[42]:


print("F1 en training:", round(100.0 * best_clf_svm.score(train_x, train_y),5))
print("F1 en test: ", round(100.0 * best_clf_svm.score(test_x, test_y),5))
input("\n--- Pulsar tecla para continuar ---\n")

# In[43]:


# Una vez elegidos los mejores  parámetros elegimos el mejor modelo atendiendo al 
# mismo conjunto de valiación cruzada


# In[44]:


params_grid = [ {
                'estimator':[LogisticRegression(max_iter=500)],
                'estimator__solver':['lbfgs'],
                'estimator__C': [100],
                'estimator__penalty': ['l2'],
                'estimator__tol': [0.001],
                'poly__degree': [2]
                },
                {
                'estimator': [Perceptron(random_state = SEED)],
                'estimator__penalty': ['l1'],
                'estimator__alpha': [0.0001],
                'estimator__max_iter': [2000],
                'estimator__shuffle': [True],
                'estimator__tol': [10],
                'poly__degree': [2]
                },
                {
                'estimator': [RandomForestClassifier(random_state = SEED)],
                'estimator__criterion': ['entropy'],
                'estimator__max_features': ['sqrt'],
                'estimator__bootstrap':['True'],
                'estimator__min_samples_split': [5]
                },
                {
                'estimator': [MLPClassifier(random_state = SEED)],
                'estimator__activation': ['logistic'],
                'estimator__solver': ['lbfgs'],
                'estimator__alpha': [5],
                'estimator__max_fun': [20000]
                },
                {
                'estimator': [SVC()],
                'estimator__C': [7],
                'estimator__kernel': ['rbf'],
                'estimator__gamma': ['scale']
                }
               # {'estimator':[Any_other_estimator_you_want],
               #  'estimator__valid_param_of_your_estimator':[valid_values]

]


# In[45]:


# Entrenamiento del mejor modelo
pipe_model = Pipeline(steps=preproc_lin+[('estimator', SVC(gamma = "scale", kernel="rbf"))])

best_clf = GridSearchCV(pipe_model, params_grid, scoring = 'f1',cv = 5, n_jobs = -1, verbose=1)# 


# In[46]:


best_clf.fit(train_x, train_y)


# In[47]:


print("Mejor modelo obtenido")
print(best_clf.best_params_)

input("\n--- Pulsar tecla para continuar ---\n")
# In[48]:


# A continuación vamos a crear dataframes con los resultados para mostrar gráficas


# In[49]:


algs = ["LR", "Perceptron", "RandomForest", "MLP", "SVC"]


# In[50]:


results = [best_clf.cv_results_["split{}_test_score".format(i)] for i in range(5)]


# In[52]:


cv_results = {"Modelo": algs, "f1": results}


# In[53]:


cv_results_pandas = pd.DataFrame(results)


# In[54]:


cv_results_pandas["Modelo"] = algs


# In[56]:

print("Grafico de tipo box and whiskers de los errores de validación cruzada")
sns.set_style('ticks')
fig, ax = plt.subplots()
# the size of A4 paper
ax.set_title("Errores obtenidos en CV")
ax.set_ylabel("F1 en CV")
ax.set_xlabel("Algoritmo")
fig.set_size_inches(11.7, 8.27)
sns.boxplot(x=algs, y=results)  
sns.despine()
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")

# In[57]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc


# In[58]:


# Procedemos a obtener las gráficas de preccision vs recall


# In[59]:


modelos = [best_clf_lin, best_clf_random, best_clf_mlp]


# In[60]:

print("Mostramos graficas de precission vs recall")



for m, model in zip(["RL", "Random Forest", "MLP"], modelos):
    lr_probs = model.predict_proba(test_x)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # predict class values
    yhat = model.predict(test_x)
    lr_precision, lr_recall, _ = precision_recall_curve(test_y, lr_probs)
    lr_f1, lr_auc = f1_score(test_y, yhat), auc(lr_recall, lr_precision)
    # summarize scores
    print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    no_skill = len(test_y[test_y==1]) / len(test_y)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.', label=m)
    # axis labels
    plt.title("Curva precission-recall para {}".format(m))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
input("\n--- Pulsar tecla para continuar ---\n")

# In[61]:


# Obtenemos las matrices de confusión para cada modelo obtenido


# In[62]:


modelos = [("Regresión logística",best_clf_lin), ("Perceptrón", best_clf_perceptron), ("Random Forest",best_clf_random), ("Perceptrón multicapa", best_clf_mlp), ("SVC",best_clf_svm)]


# In[63]:

print("Mostramos las matrices de confusión de los modelos")

for name, model in modelos:
    print("Matriz de confusión para {}".format(name))
    y_pred = model.predict(test_x)
    print(confusion_matrix(test_y, y_pred))
input("\n--- Pulsar tecla para continuar ---\n")

# In[64]:

print("Usamos accuracy para estimar el etest")
for name, model in modelos:
    print("Accuracy para el modelo {}: {}.\nE_test:{}\n".format(name, round(accuracy_score(test_y, y_pred),4), round(1-accuracy_score(test_y, y_pred),4)))
    y_pred = model.predict(test_x)

input("\n--- Pulsar tecla para continuar ---\n")