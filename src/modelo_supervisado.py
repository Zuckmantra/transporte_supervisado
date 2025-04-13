import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

# Cargar el dataset
try:
    df = pd.read_csv("data/datset_transporte.csv")  # Asegúrate de que la ruta sea correcta
    print("Dataset cargado correctamente.")
    print(df.head())  # Muestra las primeras filas para verificar
except Exception as e:
    print("Error al cargar el dataset:", e)
    exit()

# Verificar columnas
print("Columnas del dataset:", df.columns)

# Codificar variables categóricas
df["tipo_transporte"] = df["tipo_transporte"].map({"Autobus": 0, "Metro": 1, "Tren": 2})
df["zona"] = df["zona"].map({"Centro": 0, "Periferia": 1, "Residencial": 2})

# Verificar valores únicos
print("Valores únicos de 'tipo_transporte':", df["tipo_transporte"].unique())
print("Valores únicos de 'zona':", df["zona"].unique())

# Definir características y variable objetivo
X = df[["num_pasajeros", "distancia_estaciones", "hora_pico", "tipo_transporte", "zona"]]
y = df["satisfaccion"]

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar el modelo de árbol de decisión
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = clf.predict(X_test)

# Evaluar el modelo
print("Precision:", accuracy_score(y_test, y_pred))
print("Reporte de clasificacion:\n", classification_report(y_test, y_pred))

# Visualizar el árbol de decisión
plt.figure(figsize=(14, 10))
tree.plot_tree(
    clf, 
    feature_names=X.columns, 
    class_names=["Insatisfecho", "Satisfecho"], 
    filled=True, 
    rounded=True,
    impurity=True,
    node_ids=True,
    fontsize=10,
    proportion=True
)
plt.gca().set_facecolor("#f9f9f9")
plt.tight_layout()
plt.show()