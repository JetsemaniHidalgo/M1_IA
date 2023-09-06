import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# Cargar el dataset
dataset = pd.read_csv('comprar_alquilar.csv')

# Variables a describir (excluyendo 'comprar')
variables_to_describe = ['ingresos', 'gastos_comunes', 'pago_coche', 'gastos_otros', 'ahorros', 'vivienda', 'estado_civil', 'hijos', 'trabajo']

# Crear histogramas para cada variable
for variable in variables_to_describe:
    plt.figure(figsize=(8, 6))
    plt.hist(dataset[variable], bins=20, edgecolor='k')
    plt.title(f'Histograma de {variable}')
    plt.xlabel(variable)
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.show()

# Calcular la matriz de correlación
correlation_matrix = dataset.corr()

# Gráfico de correlación de todas las variables
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlación de todas las variables')
plt.show()

# Gráfico de correlación con la variable a predecir
plt.figure(figsize=(8, 6))
corr_with_target = correlation_matrix['comprar'].drop('comprar')  # Excluimos la variable a predecir
top_corr_with_target = corr_with_target.abs().sort_values(ascending=False).head(5)  # Tomar las 5 más correlacionadas (por valor absoluto)
top_corr_with_target.plot(kind='bar', color='blue')
plt.title('Correlación con la variable a predecir (Top 5)')
plt.ylabel('Correlación (valor absoluto)')
plt.xlabel('Variables')
plt.xticks(rotation=45)
plt.show()

# Calcular VIF para cada variable independiente
independent_vars = dataset.drop(columns=['comprar'])
vif_data = pd.DataFrame()
vif_data["Variable"] = independent_vars.columns
vif_data["VIF"] = [variance_inflation_factor(independent_vars.values, i) for i in range(independent_vars.shape[1])]

print("Factor de Inflación de la Varianza (VIF) para cada variable independiente:")
print(vif_data)

# Evaluación de multicolinealidad
max_vif = vif_data["VIF"].max()
print("\nMáximo Factor de Inflación de la Varianza (VIF):", max_vif)
if max_vif > 10:
    print("Puede haber problemas de multicolinealidad en el modelo.")
else:
    print("No parece haber problemas significativos de multicolinealidad.")

# Ordenar las correlaciones con la variable a predecir en orden descendente
top_corr_with_target = corr_with_target.sort_values(ascending=False)

# Mostrar las 5 variables más correlacionadas con la variable a predecir
print("\nLas 5 variables más correlacionadas con la variable a predecir ('comprar'):")
print(top_corr_with_target.head(5))

# Separar las características (X) y la variable a predecir (y)
X = dataset.drop(columns=['comprar'])
y = dataset['comprar']

# Dividir el conjunto de datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar la selección de características para obtener las 5 mejores variables
selector = SelectKBest(score_func=f_classif, k=5)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Crear un modelo de Naive Bayes utilizando todas las variables
naive_bayes_model_all = GaussianNB()
naive_bayes_model_all.fit(X_train, y_train)
y_pred_all = naive_bayes_model_all.predict(X_test)
accuracy_all = accuracy_score(y_test, y_pred_all)
print("\nPrecisión del modelo con todas las variables:", accuracy_all)
print("Precisión del modelo Naive Bayes con todas las variables: {:.2f}%".format(accuracy_all * 100))

# Crear un modelo de Naive Bayes utilizando las 5 mejores variables
naive_bayes_model_selected = GaussianNB()
naive_bayes_model_selected.fit(X_train_selected, y_train)
y_pred_selected = naive_bayes_model_selected.predict(X_test_selected)
accuracy_selected = accuracy_score(y_test, y_pred_selected)
print("\nPrecisión del modelo con las 5 mejores variables:", accuracy_selected)
print("Precisión del modelo Naive Bayes con las 5 mejores variables: {:.2f}%".format(accuracy_selected * 100))

# Comparar y elegir el mejor modelo en términos de precisión
if accuracy_selected > accuracy_all:
    print("\nEl modelo con las 5 mejores variables es mejor en términos de precisión.")
else:
    print("\nEl modelo con todas las variables es mejor en términos de precisión.")

 #Crear un DataFrame con los nuevos datos
nuevos_datos = pd.DataFrame({
    'ingresos': [2000, 6000],
    'gastos_comunes': [944, 944],
    'pago_coche': [0, 0],
    'gastos_otros': [245, 245],
    'ahorros': [5000, 34000],
    'vivienda': [200000, 320000],
    'estado_civil': [2,2],
    'hijos': [0, 2],
    'trabajo': [1,1]
})

# Realizar predicciones con el mejor modelo
predicciones = naive_bayes_model_all.predict(nuevos_datos)

# Mapear las predicciones a "comprar" o "no comprar"
resultado = ['comprar' if p == 1 else 'no comprar' for p in predicciones]

# Mostrar los resultados para ambos casos
for i, r in enumerate(resultado):
    print(f"Caso {i+1}: {r}")


# Mostrar las 5 variables más correlacionadas con la variable a predecir
print("\nLas 5 variables más correlacionadas con la variable a predecir ('comprar'):")
print(top_corr_with_target.head(5))
