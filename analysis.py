import pandas as pd
import numpy as np 

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

SEED = 1

np.random.seed(SEED)

data_folder = "data/"


headers = ["age", "workclass", "fnlwgt", "education", "education-num",
"marital-status", "occupation", "relationship", "race", "sex", "capital-gain",
"captial-loss", "hours-per-week", "native-country", "income"]

df = pd.read_csv(data_folder + "adult.data", index_col=False, delimiter=",", names=headers)

df['income']=df['income'].map({' <=50K': 0, ' >50K': 1})

# mostrar info
print(df.head())
print(len(df))

# contar el numero que faltan
df = df.replace(' ?', np.NaN)
print(df.isnull().sum(axis=0).sort_values(ascending = False).head(30))


#print(set(df["occupation"]))

print(df.income.value_counts())

df_test = pd.read_csv(data_folder + "adult.test", index_col=False, delimiter=",", names=headers)

print(len(df_test))
df_test = df_test.replace(' ?', np.NaN)
print(df_test.isnull().sum(axis=0).sort_values(ascending = False).head(30))
df_test['income']=df_test['income'].map({' <=50K.': 0, ' >50K.': 1})


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

#Encoding categorical values
dataSet["workclass"] = dataSet["workclass"].astype('category')
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
dataSet["native-country"] = dataSet["native-country"].cat.codes

print(dataSet.head())

#Creamos las particiones train y test
X, y = dataSet[dataSet.columns.difference(['income'])], dataSet['income']

X, y = shuffle(X, y, random_state=SEED)

train_x, test_x, train_y, test_y = train_test_split(X,y, test_size=0.3, stratify=y)

#vemos las proporciones de etiquetar post eliminar

print("=====")
print(test_y.value_counts()/len(test_y))
print(train_y.value_counts()/len(train_y))
print("=====")
print(train_x.shape, test_x.shape)#vemos las proporciones de etiquetar post eliminar
print(dataSet.income.value_counts()/45222)

print(set(dataSet["capital-gain"]))