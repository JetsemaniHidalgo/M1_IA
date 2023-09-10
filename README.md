# M1_IA

Este proyecto utiliza para analizar y predecir la decisión de comprar o alquilar una vivienda utilizando un modelo de Naive Bayes y se utilizan diversas bibliotecas de Python para cargar, explorar y modelar los datos.

Requisitos 
Las siguientes bibliotecas de Python son necesarias para ejecutar el código:

pandas
matplotlib
seaborn
statsmodels
numpy
scikit-learn (sklearn)

De forma general el código realiza lo siguiente:

Carga de Datos: 
Carga un conjunto de datos desde el archivo 'comprar_alquilar.csv' utilizando la biblioteca pandas.

Exploración de Datos:
Crea histogramas para visualizar la distribución de varias variables independientes.
Calcula y muestra una matriz de correlación que ilustra las relaciones entre las variables.
Identifica las 5 variables más correlacionadas con la variable objetivo ('comprar').

Evaluación de Multicolinealidad:
Calcula el Factor de Inflación de la Varianza (VIF) para cada variable independiente para evaluar la multicolinealidad en el conjunto de datos.

Selección de Características:
Utiliza la selección de características para elegir las 5 mejores variables independientes para el modelo.
Modelado de Naive Bayes:

Crea dos modelos de Naive Bayes: 
Uno utilizando todas las variables y otro utilizando las 5 mejores variables seleccionadas.
Divide los datos en conjuntos de entrenamiento y prueba.
Entrena y evalúa ambos modelos utilizando la precisión como métrica.
Predicciones:

Crea un DataFrame con nuevos datos de entrada para realizar predicciones.
Utiliza el mejor modelo (todas las variables o las 5 mejores variables) para predecir si se debe comprar o no una vivienda basándose en los nuevos datos.
Muestra los resultados de las predicciones.

Observaciones generales 
