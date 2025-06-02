import pytest
import polars as pl
import numpy as np
from sklearn.datasets import make_regression
from utils import *

# Definición de variables globales
URL = "https://raw.githubusercontent.com/AzShet/Data_Mining-LAB11/refs/heads/develop/Advertising-1.csv"
COLUMNAS = ["TV", "Radio", "Newspaper", "Sales"]

# Generar un DataFrame de prueba
def generar_dataframe_prueba():
    X, y = make_regression(n_samples=100, n_features=3, noise=0.1)
    df = pl.DataFrame({
        "TV": X[:, 0],
        "Radio": X[:, 1],
        "Newspaper": X[:, 2],
        "Sales": y
    })
    return df

def test_cargar_datos():
    df = cargar_datos(URL)
    assert df.shape[0] > 0  # Verifica que se carguen filas
    assert set(COLUMNAS).issubset(df.columns)  # Verifica que las columnas esperadas estén presentes

def test_tratar_outliers_iqr():
    df = generar_dataframe_prueba()
    df_sin_outliers = tratar_outliers_iqr(df)
    assert df_sin_outliers.shape[0] <= df.shape[0]  # Verifica que se eliminen filas

def test_escalar_datos():
    df = generar_dataframe_prueba()
    df_escalado = escalar_datos(df)
    assert (df_escalado.select(pl.all()).to_numpy().min() >= 0).all()  # Verifica que los valores estén en el rango [0, 1]
    assert (df_escalado.select(pl.all()).to_numpy().max() <= 1).all()

def test_separar_variables():
    df = generar_dataframe_prueba()
    X, y = separar_variables(df, 'Sales')
    assert X.shape[1] == 3  # Verifica que se devuelvan 3 características
    assert y.shape[1] == 1  # Verifica que se devuelva 1 variable objetivo

def test_convertir_a_numpy():
    df = generar_dataframe_prueba()
    np_array = convertir_a_numpy(df)
    assert np_array.shape == (100, 4)  # Verifica que el tamaño del array sea correcto

def test_crear_fitness_function():
    df = generar_dataframe_prueba()
    X, y = separar_variables(df, 'Sales')
    fitness_func = crear_fitness_function(X.to_numpy(), y.to_numpy())
    individual = [1, 0, 1]  # Selecciona las características 0 y 2
    fitness = fitness_func(individual)
    assert isinstance(fitness, tuple)  # Verifica que la función devuelva una tupla

def test_configurar_algoritmo_genetico():
    df = generar_dataframe_prueba()
    X, y = separar_variables(df, 'Sales')
    toolbox = configurar_algoritmo_genetico(X.shape[1])
    assert toolbox.population(10)  # Verifica que se pueda crear una población

def test_ejecutar_algoritmo_genetico():
    df = generar_dataframe_prueba()
    X, y = separar_variables(df, 'Sales')
    pop, logbook = ejecutar_algoritmo_genetico(X.to_numpy(), y.to_numpy(), n_pop=10, n_gen=5)
    assert len(pop) == 10  # Verifica que la población tenga el tamaño correcto
    assert len(logbook) == 6  # Verifica que el logbook tenga el número correcto de generaciones

def test_obtener_mejor_individuo():
    df = generar_dataframe_prueba()
    X, y = separar_variables(df, 'Sales')
    pop, _ = ejecutar_algoritmo_genetico(X.to_numpy(), y.to_numpy(), n_pop=10, n_gen=5)
    mejor_individuo = obtener_mejor_individuo(pop)
    assert mejor_individuo is not None  # Verifica que se obtenga un mejor individuo

def test_entrenar_modelo_final():
    df = generar_dataframe_prueba()
    X, y = separar_variables(df, 'Sales')
    modelo, r2_train, r2_test = entrenar_modelo_final(X.to_numpy(), y.to_numpy(), selected_features=[0, 1], test_size=0.2)
    assert modelo is not None  # Verifica que se entrene un modelo
    assert r2_train >= 0  # Verifica que el R² en entrenamiento sea no negativo
    assert r2_test >= 0  # Verifica que el R² en prueba sea no negativo

def test_graficar_evolucion():
    df = generar_dataframe_prueba()
    X, y = separar_variables(df, 'Sales')
    pop, logbook = ejecutar_algoritmo_genetico(X.to_numpy(), y.to_numpy(), n_pop=10, n_gen=5)
    try:
        graficar_evolucion(logbook)  # Verifica que no se produzca un error al graficar
    except Exception as e:
        assert False, f"Error al graficar evolución: {e}"

def test_mostrar_caracteristicas_seleccionadas():
    df = generar_dataframe_prueba()
    X, y = separar_variables(df, 'Sales')
    pop, logbook = ejecutar_algoritmo_genetico(X.to_numpy(), y.to_numpy(), n_pop=10, n_gen=5)
    mejor_individuo = obtener_mejor_individuo(pop)
    caracteristicas_seleccionadas, nombres_seleccionados = mostrar_caracteristicas_seleccionadas(mejor_individuo, COLUMNAS)
    assert len(caracteristicas_seleccionadas) <= len(COLUMNAS)  # Verifica que no se seleccionen más características de las disponibles

def test_analizar_outliers_multivariado():
    df = generar_dataframe_prueba()
    outliers, distancias = analizar_outliers_multivariado(df)
    assert len(outliers) == df.shape[0]  # Verifica que el número de outliers coincida con el número de filas

def test_graficar_valores_reales_vs_predichos():
    """Test visual para asegurar que la función de graficado se ejecuta sin errores."""

    # Datos de prueba (valores escalados)
    y_real = np.array([0.1, 0.4, 0.6, 0.9, 1.0])
    y_pred = np.array([0.2, 0.5, 0.5, 0.85, 0.95])

    try:
        graficar_valores_reales_vs_predichos(y_real, y_pred)
    except Exception as e:
        pytest.fail(f"La función lanzó una excepción inesperada: {e}")


def test_probar_multiples_configuraciones():
    df = generar_dataframe_prueba()
    X, y = separar_variables(df, 'Sales')
    configuraciones = [
        {'n_pop': 10, 'cx_pb': 0.6, 'mut_pb': 0.1, 'n_gen': 5},
        {'n_pop': 20, 'cx_pb': 0.7, 'mut_pb': 0.2, 'n_gen': 10}
    ]
    resultados = probar_multiples_configuraciones(X.to_numpy(), y.to_numpy(), configuraciones)
    assert len(resultados) == len(configuraciones)  # Verifica que se devuelvan resultados para cada configuración
