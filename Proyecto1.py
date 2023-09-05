import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar el dataset
dataset = pd.read_csv('comprar_alquilar.csv')

# Variables a describir (excluyendo 'comprar')
variables_to_describe = ['ingresos', 'gastos_comunes', 'pago_coche', 'gastos_otros', 'ahorros', 'vivienda', 'estado_civil', 'hijos', 'trabajo']

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
