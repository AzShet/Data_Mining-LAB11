"""
Funciones reutilizables para el proyecto de análisis de datos de publicidad con algoritmos genéticos.

Autor: César Diego Ruelas Flores
Fecha: 28-may-2025
"""
import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt

def cargar_datos(url):
    """Carga los datos desde una URL y devuelve un DataFrame de Polars."""
    return pl.read_csv(url)

def tratar_outliers_iqr(df, columnas=None):
    """Elimina outliers utilizando el método del rango intercuartílico (IQR)."""
    if columnas is None:
        columnas = df.columns

    df_tratado = df.clone()
    for col in columnas:
        if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            mediana = df[col].median()

            df_tratado = df_tratado.with_columns(
                pl.when((pl.col(col) < limite_inferior) | (pl.col(col) > limite_superior))
                .then(mediana)
                .otherwise(pl.col(col))
                .alias(col)
            )
    return df_tratado

def escalar_datos(df, columnas=None, metodo='minmax'):
    """Escala los datos usando MinMaxScaler o StandardScaler."""
    if columnas is None:
        columnas = df.columns

    df_escalado = df.clone()

    if metodo == 'minmax':
        for col in columnas:
            if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                min_val = df[col].min()
                max_val = df[col].max()
                df_escalado = df_escalado.with_columns(
                    ((pl.col(col) - min_val) / (max_val - min_val)).alias(col)
                )

    return df_escalado

def separar_variables(df, variable_objetivo):
    """Separa las variables predictoras de la variable objetivo."""
    X = df.drop(variable_objetivo)
    y = df.select(variable_objetivo)
    return X, y

def convertir_a_numpy(df):
    """Convierte un DataFrame de Polars a array de numpy."""
    return df.to_numpy()

def crear_fitness_function(X, y):
    """Crea la función de fitness para el algoritmo genético."""
    def evaluar_individuo(individual):
        selected_features = [i for i, bit in enumerate(individual) if bit == 1]

        if len(selected_features) == 0:
            return (0.0,)  # Si no hay características seleccionadas, fitness = 0

        X_selected = X[:, selected_features]

        modelo = LinearRegression()
        scores = cross_val_score(modelo, X_selected, y.ravel(), cv=5, scoring='r2')

        return (np.mean(scores),)

    return evaluar_individuo

def configurar_algoritmo_genetico(n_features):
    """Configura el algoritmo genético usando DEAP."""
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    return toolbox

def ejecutar_algoritmo_genetico(X, y, n_pop=50, cx_pb=0.7, mut_pb=0.2, n_gen=50):
    """Ejecuta el algoritmo genético para selección de características."""
    n_features = X.shape[1]

    toolbox = configurar_algoritmo_genetico(n_features)

    fitness_func = crear_fitness_function(X, y)
    toolbox.register("evaluate", fitness_func)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=n_pop)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cx_pb, mutpb=mut_pb, 
                                       ngen=n_gen, stats=stats, verbose=True)

    return pop, logbook

def obtener_mejor_individuo(population):
    """Obtiene el mejor individuo de la población."""
    return tools.selBest(population, k=1)[0]

def entrenar_modelo_final(X, y, selected_features, test_size=0.2, random_state=42):
    """Entrena el modelo final con las características seleccionadas."""
    X_selected = X[:, selected_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y.ravel(), test_size=test_size, random_state=random_state
    )

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    return modelo, r2_train, r2_test

def graficar_evolucion(logbook):
    """Grafica la evolución del algoritmo genético."""
    gen = logbook.select("gen")
    avg_fitness = logbook.select("avg")
    max_fitness = logbook.select("max")

    plt.figure(figsize=(10, 6))
    plt.plot(gen, avg_fitness, label='Promedio', linewidth=2)
    plt.plot(gen, max_fitness, label='Máximo', linewidth=2)
    plt.xlabel('Generación')
    plt.ylabel('R² Score')
    plt.title('Evolución del Coeficiente de Determinación (R²)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def mostrar_caracteristicas_seleccionadas(mejor_individuo, nombres_columnas):
    """Muestra las características seleccionadas por el algoritmo genético."""
    selected_features = [i for i, bit in enumerate(mejor_individuo) if bit == 1]
    selected_names = [nombres_columnas[i] for i in selected_features]

    print("Características seleccionadas:")
    for i, name in enumerate(selected_names):
        print(f"  {i+1}. {name}")

    return selected_features, selected_names

def analizar_outliers_multivariado(df, umbral=3):
    """Detecta outliers multivariados usando la distancia de Mahalanobis."""
    data = df.to_numpy()

    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)

    try:
        inv_cov = np.linalg.inv(cov)
        mahal_dist = []

        for row in data:
            diff = row - mean
            mahal_dist.append(np.sqrt(diff.T @ inv_cov @ diff))

        mahal_dist = np.array(mahal_dist)
        outliers = mahal_dist > umbral

        return outliers, mahal_dist
    except np.linalg.LinAlgError:
        print("Error: Matriz de covarianza singular. No se pueden detectar outliers multivariados.")
        return np.zeros(len(data), dtype=bool), np.zeros(len(data))

def probar_multiples_configuraciones(X, y, configuraciones):
    """Prueba múltiples configuraciones del algoritmo genético."""
    resultados = []

    for i, config in enumerate(configuraciones):
        print(f"\nProbando configuración {i+1}/{len(configuraciones)}")
        print(f"N_POP: {config['n_pop']}, CX_PB: {config['cx_pb']}, MUT_PB: {config['mut_pb']}, N_GEN: {config['n_gen']}")

        pop, logbook = ejecutar_algoritmo_genetico(
            X, y, 
            n_pop=config['n_pop'],
            cx_pb=config['cx_pb'],
            mut_pb=config['mut_pb'],
            n_gen=config['n_gen']
        )

        mejor_individuo = obtener_mejor_individuo(pop)
        mejor_fitness = mejor_individuo.fitness.values[0]

        resultados.append({
            'configuracion': config,
            'mejor_fitness': mejor_fitness,
            'mejor_individuo': mejor_individuo,
            'logbook': logbook
        })

    return resultados

def graficar_valores_reales_vs_predichos(y_real, y_predicho):
    """
    Grafica una comparación entre valores reales y predichos escalados.

    Parameters:
    - y_real (array): valores reales escalados
    - y_predicho (array): valores predichos por el modelo escalados
    """
    plt.figure(figsize=(10, 5))
    plt.plot(y_real, label='Valor real (escalado)', color='blue', linewidth=2)
    plt.plot(y_predicho, label='Predicción (escalada)', color='orange', linestyle='--', linewidth=2)
    plt.title('Comparación: Valores reales vs Predicciones (Escalados)')
    plt.xlabel('Observaciones')
    plt.ylabel('Ventas escaladas')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
