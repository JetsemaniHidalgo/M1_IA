# M1_IA

Este proyecto se utiliza para analizar y predecir la decisión de comprar o alquilar una vivienda utilizando un modelo de Naive Bayes y se aplican diversas bibliotecas de Python para cargar, explorar y modelar los datos.

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

Observaciones generales:

Desde mi perspectiva, desarrollar este código resultó algo complejo debido a que es la primera vez que interactúo con Python. Sin embargo, encontré el curso muy atractivo y los temas son comprensibles en general. Más que nada, considero que es necesario investigar a fondo para poder llevar a cabo el proyecto. Por ejemplo, al abordar la multicolinealidad utilizando el Factor de Inflación de la Varianza (VIF), encontré que requería una investigación adicional, entre otro temas.

Por otro lado, es fundamental analizar minuciosamente los datos que se cargan en el proyecto. Durante el proyecto, desarolle una práctica de un código pequeño de Gaussian Naive Bayes con el objetiv de entender su estructura y funcionamiento y observé una peculiaridad que no comprendí del todo: el valor del estado civil podía ser 0, 1 o 2. Esta variabilidad en los datos me desconcertó, ya que afectó las predicciones en el programa con que estaba practicando, lo que probablemente si repercute al programa final.

Reconozco que aún tengo margen para mejorar la organización y estructura de mi programa. En resumen, el proceso de desarrollo de este código me ha enseñado valiosas lecciones y ha demostrado la importancia de la investigación y la comprensión profunda de los datos en proyectos de este tipo ademas me brinda otra perspectiva de los que son las I.A. con respecto a lo que esta actualmente.
