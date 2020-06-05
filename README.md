# Proyecto-Final---AA

Proyecto Final de la asignatura de aprendizaje automático

## Estudio del Problema



## TO DO

- Mirar si estan balanceadas las clases -> OK 75/25
- Las clases con valores perdidos son : occupation (1843) , workclass (1836) y native-country (583). En test tenemos: occupation (967), workclass (964) ,native-country (275) y luego menos age todos tienen un valor perdido. Vamos a tomar la decisión de eliminar las instancias con valores perdidos.
- fnlwgt: número de personas representadas en esa instancia. Poco interesante para el estudio.
- Creamos los conjuntos de test y training 0.3 y 0.7 respectivamente. Mantenemos las proporciones 75/25
- Había un problema con los espacios en las columnas y hemos solucionado esto leyendo los datos con una expresión regular y buscando valores
  perdidos en dos casos
- Eliminamos education-num por que es una representación numérica de el atributo education.


## Preprocesado

- [StandadScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- Hemos estudiado la matriz de correlaciones y la mayotia de los valores son ceros por lo que no podemos determinar ninguna correlación entre variables.
- [VarianceThreshold](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html)
- Decisiones tomadas con respecto a los valores perdidos
- Los dos atributos eliminados

## Regularización

- Usamos Lasso por que nuestros datos no están fuertemente correlados y con la decisión tomada con las variables dummy parece interesante usar Lasso

## Modelos 

Lineales:

- [Regresión logistica](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)

No lineales:

- [Perceptron Multicapa](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [SVD](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

# Preguntas
- Argumentar sobre la idoneidad de la función regularización usada 
- Outliyers
- hemos visto que un modelo que gana en CV saca peor Etest que otro que pierde en CV