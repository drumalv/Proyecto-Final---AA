# Proyecto-Final---AA

Proyecto Final de la asignatura de aprendizaje automático

## TO DO

- Mirar si estan balanceadas las clases -> OK 75/25
- Las clases con valores perdidos son : occupation (1843) , workclass (1836) y native-country (583). En test tenemos: occupation (967), workclass (964) ,native-country (275) y luego menos age todos tienen un valor perdido. Vamos a tomar la decisión de eliminar las instancias con valores perdidos.
- fnlwgt: número de personas representadas en esa instancia. Poco interesante para el estudio.
- Creamos los conjuntos de test y training 0.3 y 0.7 respectivamente. Mantenemos las proporciones 75/25


# Modelos 

Lineales:

- [Regresión logistica](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)

No lineales:

- [Perceptron Multicapa](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [SVD](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

# Preguntas
- Argumentar sobre la idoneidad de la función regularización usada 
- Eliminación datos perdidos.