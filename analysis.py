import json

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

SEED = 1

np.random.seed(SEED)

data_folder = "data/"


headers = ["age", "workclass", "fnlwgt", "education", "education-num",
"marital-status", "occupation", "relationship", "race", "sex", "capital-gain",
"captial-loss", "hours-per-week", "native-country", "income"]

df = pd.read_csv(data_folder + "adult.data", index_col=False, delimiter=",[ ]*", names=headers)

df['income']=df['income'].map({'<=50K': 0, '>50K': 1})

# mostrar info
print(df.head())
print(len(df))

# contar el numero que faltan
df = df.replace(' ?', np.NaN)
df = df.replace('?', np.NaN)
print(df.isnull().sum(axis=0).sort_values(ascending = False).head(30))


#print(set(df["occupation"]))

print(df.income.value_counts())

df_test = pd.read_csv(data_folder + "adult.test", index_col=False, delimiter=",[ ]*", names=headers)

print(len(df_test))
df_test = df_test.replace('?', np.NaN)
df_test = df_test.replace(' ?', np.NaN)
print(df_test.isnull().sum(axis=0).sort_values(ascending = False).head(30))
df_test['income']=df_test['income'].map({'<=50K.': 0, '>50K.': 1})


print(df.income.value_counts())
print(df_test.income.value_counts())

#uniendo los conjuntos data y test en dataSet
dataSet=pd.concat([df,df_test])

print(dataSet.shape)
#vemos cuantos valores perdidos
print(dataSet.isnull().sum(axis=0).sort_values(ascending = False).head(30))

#eliminamos valores perdidos
dataSet = dataSet.dropna()
print(dataSet.isnull().sum(axis=0).sort_values(ascending = False).head(30))
print(dataSet.shape)

#vemos las proporciones de etiquetar post eliminar
print(dataSet.income.value_counts())
print(dataSet.income.value_counts()/45222)

#eliminamos fnlwgt por que no aporta información
dataSet = dataSet.drop(['fnlwgt'], axis=1)
dataSet = dataSet.drop(['education-num'], axis=1)


#Encoding categorical values
""" dataSet["workclass"] = dataSet["workclass"].astype('category')
dataSet["education"] = dataSet["education"].astype('category')
dataSet["marital-status"] = dataSet["marital-status"].astype('category')
dataSet["occupation"] = dataSet["occupation"].astype('category')
dataSet["relationship"] = dataSet["relationship"].astype('category')
dataSet["race"] = dataSet["race"].astype('category')
dataSet["sex"] = dataSet["sex"].astype('category')
dataSet["native-country"] = dataSet["native-country"].astype('category')


dataSet["workclass"] = dataSet["workclass"].cat.codes
dataSet["education"] = dataSet["education"].cat.codes
dataSet["marital-status"] = dataSet["marital-status"].cat.codes
dataSet["occupation"] = dataSet["occupation"].cat.codes
dataSet["relationship"] = dataSet["relationship"].cat.codes
dataSet["race"] = dataSet["race"].cat.codes
dataSet["sex"] = dataSet["sex"].cat.codes
dataSet["native-country"] = dataSet["native-country"].cat.codes """

cols_with_categories = ['workclass', 'education',"marital-status","occupation","relationship","race","sex","native-country"]

for col in cols_with_categories:
    print("Before: {} total of categories in {}".format(len(set(dataSet[col])), col))


# convertir variables categoricas a dummies

dataSet = pd.get_dummies(data=dataSet, columns=cols_with_categories)

for asd in cols_with_categories:
    cols = [col for col in dataSet if col.startswith(asd)]
    print("After: {} total of categories in {}".format( len(cols), asd))

print("==============")

print(dataSet.head())

print("==============")
#Creamos las particiones train y test
X, y = dataSet[dataSet.columns.difference(['income'])], dataSet['income']

X, y = shuffle(X, y, random_state=SEED)

train_x, test_x, train_y, test_y = train_test_split(X,y, test_size=0.3, stratify=y)

#vemos las proporciones de etiquetar post eliminar

print(test_y.value_counts()/len(test_y))
print(train_y.value_counts()/len(train_y))
print(train_x.shape, test_x.shape)#vemos las proporciones de etiquetar post eliminar
print(dataSet.income.value_counts()/45222)

#################
#  Preprocesado #
#################

correlation_matrix = train_x.corr()
#print (correlation_matrix)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(correlation_matrix, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

#plt.show()

preproc=[
            ("var", VarianceThreshold(0.01)),   
            ("standardize", StandardScaler()),      
            ("lasso", SelectFromModel(estimator=LassoCV(tol=0.001))),
            ("poly",PolynomialFeatures(1)), 
            ("var2", VarianceThreshold(0.01)),   
        ]

# estadisticas preprocesado

pipe_pre=Pipeline(preproc)
x_train_prep = pipe_pre.fit_transform(train_x, train_y)
print("Descripción de los datos antes y después del preprocesado")
print("Antes: {}".format(train_x.shape))
print("Despues: {}".format(x_train_prep.shape))

#creamos un pipeline de sklearn donde añadiremos uno de los modelos a estudiar
pipe = Pipeline(steps=preproc+[('estimator', LogisticRegression())])

# Añadimos los estimadores que vamos a utilizar y los parametros que vamos a estudiar:
  
params_grid = [ {
                'estimator':[LogisticRegression(max_iter=500)],
                'estimator__solver':['lbfgs'],
                'estimator__C': np.logspace(-4, 4, 3),
                'estimator__penalty': ['l1'],
                'estimator__tol': [1e-3, 1e-4]
                },
                {
                'estimator': [Perceptron(random_state = SEED)],
                'estimator__alpha':[1.0,1e-3, 1e-4],
                'estimator__max_iter':[2000],
                'estimator__tol': [1e-3, 1e-4],
                'estimator__shuffle': [True]
                },
                {
                'estimator': [RandomForestClassifier(random_state = SEED)],
                'estimator__criterion': ['gini','entropy'],
                'estimator__max_features': ['sqrt'],
                'estimator__bootstrap':['True']
                },
                {
                'estimator': [MLPClassifier(random_state = SEED)],
                'estimator__activation': ['logistic', 'tanh', 'relu'],
                'estimator__solver': ['lbfgs'],
                'estimator__alpha': [1.0,1e-3, 1e-4],
                'estimator__max_fun': [20000]
                },
                {
                'estimator': [SVC()],
                'estimator__C': np.logspace(-4, 4, 3),
                'estimator__kernel': ['rbf'],
                'estimator__gamma': ['scale']
                }
               # {'estimator':[Any_other_estimator_you_want],
               #  'estimator__valid_param_of_your_estimator':[valid_values]

]

print("CON PREPROCESADO Y REGULARIZACION: \n")

# entrenamos con crossvalidation y sacamos el mejor con sus parámetros.
best_clf = GridSearchCV(pipe, params_grid, scoring = 'accuracy',cv = 5, n_jobs = -1, verbose=1)
best_clf.fit(train_x, train_y)

results=pd.DataFrame(best_clf.cv_results_)

print("Mejor modelo:\n",best_clf.best_params_)
print("Precisión en training:", 100.0 * best_clf.score(train_x, train_y))
print("Precisión en test: ",100.0 * best_clf.score(test_x, test_y))

print(results)

with open("results.json", "w") as f:
    json.dump(best_clf.cv_results_, f, default=str)


