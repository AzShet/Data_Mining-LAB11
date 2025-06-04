"""
Funciones reutilizables para el proyecto de análisis de datos de publicidad con algoritmos genéticos.

Autor: César Diego Ruelas Flores
Fecha: 28-may-2025
"""
import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# MinMaxScaler no se usa directamente, pero se deja por si se quiere extender 'escalar_datos'
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Optional, Dict, Any
import warnings
import sklearn.exceptions # Added for UndefinedMetricWarning

def cargar_datos(url: str) -> pl.DataFrame:
    try:
        df = pl.read_csv(url, raise_if_empty=False)
        if df.is_empty():
            warnings.warn(f"El archivo CSV cargado desde {url} está vacío.", UserWarning)
        return df
    except pl.exceptions.NoDataError:
        warnings.warn(f"El archivo CSV cargado desde {url} está vacío o no contiene datos.", UserWarning)
        return pl.DataFrame()
    except Exception as e:
        raise pl.exceptions.PolarsError(f"No se pudo cargar el archivo CSV desde {url}: {e}")

def tratar_outliers_iqr(df: pl.DataFrame, columnas: Optional[List[str]] = None) -> pl.DataFrame:
    if df.is_empty():
        warnings.warn("El DataFrame de entrada está vacío. No se realizará tratamiento de outliers.", UserWarning)
        return df
    df_tratado: pl.DataFrame = df.clone()
    columnas_a_procesar: List[str] = []
    if columnas is None:
        columnas_a_procesar = [col for col in df.columns if df[col].dtype in [pl.Float32, pl.Float64, pl.Int16, pl.Int32, pl.Int64]]
        if not columnas_a_procesar:
            warnings.warn("No se encontraron columnas numéricas para el tratamiento de outliers.", UserWarning)
            return df
    else:
        for col_nombre in columnas:
            if col_nombre not in df.columns:
                raise ValueError(f"La columna '{col_nombre}' especificada no existe en el DataFrame.")
            if df[col_nombre].dtype not in [pl.Float32, pl.Float64, pl.Int16, pl.Int32, pl.Int64]:
                warnings.warn(f"La columna '{col_nombre}' no es de tipo numérico y será ignorada.", UserWarning)
            else:
                columnas_a_procesar.append(col_nombre)
        if not columnas_a_procesar:
            warnings.warn("Ninguna de las columnas especificadas es numérica o válida. No se realizará tratamiento de outliers.", UserWarning)
            return df

    for col in columnas_a_procesar:
        current_col_data = df[col]

        is_all_nan_float = False
        if current_col_data.dtype in [pl.Float32, pl.Float64]:
            numpy_array = current_col_data.to_numpy()
            if np.isnan(numpy_array).all():
                is_all_nan_float = True

        if is_all_nan_float:
            warnings.warn(
                f"NumPyDetect: Columna '{col}' es todo NaNs (detectado con NumPy). Se omitirá en tratamiento de outliers.", UserWarning
            )
            df_tratado = df_tratado.with_columns(pl.lit(None, dtype=current_col_data.dtype).alias(col))
            continue

        col_sin_nulos = current_col_data.drop_nulls()
        if col_sin_nulos.is_empty():
            warnings.warn(
                f"Columna '{col}' está vacía después de quitar NaNs (originalmente podría ser todo NaNs). "
                "Se dejará como está (todos NaNs).", UserWarning
            )
            if current_col_data.dtype in [pl.Float32, pl.Float64]:
                 df_tratado = df_tratado.with_columns(pl.lit(None, dtype=current_col_data.dtype).alias(col))
            continue

        Q1: Optional[float] = col_sin_nulos.quantile(0.25, interpolation='linear')
        Q3: Optional[float] = col_sin_nulos.quantile(0.75, interpolation='linear')
        mediana: Optional[float] = col_sin_nulos.median()

        if Q1 is None or Q3 is None or mediana is None:
            warnings.warn(f"No se pudieron calcular Q1, Q3 o mediana para la columna '{col}'. Se omite el tratamiento.", UserWarning)
            continue
        IQR: float = Q3 - Q1
        if IQR == 0:
            continue
        limite_inferior: float = Q1 - 1.5 * IQR
        limite_superior: float = Q3 + 1.5 * IQR
        df_tratado = df_tratado.with_columns(
            pl.when((pl.col(col) < limite_inferior) | (pl.col(col) > limite_superior))
            .then(pl.lit(mediana, dtype=df[col].dtype))
            .otherwise(pl.col(col))
            .alias(col)
        )
    return df_tratado

def escalar_datos(df: pl.DataFrame, columnas: Optional[List[str]] = None, metodo: str = 'minmax') -> pl.DataFrame:
    if df.is_empty():
        warnings.warn("El DataFrame de entrada está vacío. No se realizará escalado.", UserWarning)
        return df
    if metodo not in ['minmax']:
        raise ValueError(f"Método de escalado '{metodo}' no reconocido. Use 'minmax'.")
    df_escalado: pl.DataFrame = df.clone()
    columnas_a_procesar: List[str] = []
    if columnas is None:
        columnas_a_procesar = [col for col in df.columns if df[col].dtype in [pl.Float32, pl.Float64, pl.Int16, pl.Int32, pl.Int64]]
        if not columnas_a_procesar:
            warnings.warn("No se encontraron columnas numéricas para el escalado.", UserWarning)
            return df
    else:
        for col_nombre in columnas:
            if col_nombre not in df.columns:
                raise ValueError(f"La columna '{col_nombre}' especificada no existe en el DataFrame.")
            if df[col_nombre].dtype not in [pl.Float32, pl.Float64, pl.Int16, pl.Int32, pl.Int64]:
                warnings.warn(f"La columna '{col_nombre}' no es de tipo numérico y será ignorada para el escalado.", UserWarning)
            else:
                columnas_a_procesar.append(col_nombre)
        if not columnas_a_procesar:
            warnings.warn("Ninguna de las columnas especificadas es numérica o válida. No se realizará escalado.", UserWarning)
            return df

    if metodo == 'minmax':
        for col in columnas_a_procesar:
            current_col_data = df[col] # Use original df for checks

            is_all_nan_float = False
            if current_col_data.dtype in [pl.Float32, pl.Float64]:
                numpy_array = current_col_data.to_numpy()
                if np.isnan(numpy_array).all():
                    is_all_nan_float = True

            if is_all_nan_float:
                warnings.warn(
                    f"NumPyDetect: Columna '{col}' es todo NaNs (detectado con NumPy). Se escalará a todos NaNs (Float64).", UserWarning
                )
                df_escalado = df_escalado.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
                continue

            col_sin_nulos = current_col_data.drop_nulls()
            if col_sin_nulos.is_empty():
                warnings.warn(
                    f"Columna '{col}' está vacía después de quitar NaNs (originalmente podría ser todo NaNs). "
                    "Se escalará a todos NaNs (dtype Float64).", UserWarning
                )
                df_escalado = df_escalado.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
                continue

            min_val: Optional[float] = col_sin_nulos.min()
            max_val: Optional[float] = col_sin_nulos.max()

            if min_val is None or max_val is None:
                 warnings.warn(f"No se pudieron calcular min/max para la columna '{col}'. Se omite escalado.", UserWarning)
                 df_escalado = df_escalado.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
                 continue

            if min_val == max_val:
                warnings.warn(f"La columna '{col}' tiene todos los valores iguales ({min_val}). Se escalará a 0.0.", UserWarning)
                df_escalado = df_escalado.with_columns(
                    pl.when(pl.col(col).is_not_null()).then(0.0).otherwise(None).cast(pl.Float64).alias(col)
                )
            else:
                df_escalado = df_escalado.with_columns(
                    pl.when(pl.col(col).is_not_null())
                    .then(((pl.col(col) - min_val) / (max_val - min_val)))
                    .otherwise(None)
                    .cast(pl.Float64).alias(col)
                )
    return df_escalado

def separar_variables(df: pl.DataFrame, variable_objetivo: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
    if df.is_empty(): raise ValueError("El DataFrame de entrada está vacío.")
    if variable_objetivo not in df.columns: raise ValueError(f"La variable objetivo '{variable_objetivo}' no se encuentra en las columnas del DataFrame: {df.columns}")
    X: pl.DataFrame = df.drop(variable_objetivo); y: pl.DataFrame = df.select(variable_objetivo)
    return X, y

def convertir_a_numpy(df: pl.DataFrame) -> np.ndarray:
    if df.is_empty(): raise ValueError("No se puede convertir un DataFrame vacío a NumPy array.")
    return df.to_numpy()

def crear_fitness_function(X: np.ndarray, y: np.ndarray) -> Callable[[List[int]], Tuple[float]]:
    if X.shape[0] == 0 or X.shape[1] == 0: raise ValueError("El array X de características no puede estar vacío o no tener características.")
    if y.shape[0] == 0: raise ValueError("El array y de variable objetivo no puede estar vacío.")
    if X.shape[0] != y.shape[0]: raise ValueError(f"Incompatibilidad de formas: X tiene {X.shape[0]} filas y y tiene {y.shape[0]} filas.")
    def evaluar_individuo(individual: List[int]) -> Tuple[float]:
        selected_features_indices: List[int] = [i for i, bit in enumerate(individual) if bit == 1]
        if not selected_features_indices: return (0.0,)
        X_selected: np.ndarray = X[:, selected_features_indices]
        if X_selected.shape[0] == 0 or X_selected.shape[1] == 0:
             warnings.warn("X_selected terminó vacío, esto no debería suceder si X original es válido.", UserWarning); return (0.0,)
        modelo: LinearRegression = LinearRegression()
        cv_folds = min(5, X_selected.shape[0])
        if X_selected.shape[0] >= 2 and cv_folds < 2: cv_folds = 2
        if X_selected.shape[0] < cv_folds and X_selected.shape[0] > 0 :
             warnings.warn(f"Muy pocas muestras ({X_selected.shape[0]}) para validación cruzada con {cv_folds} folds. Retornando fitness de -1.0.", UserWarning); return (-1.0,)
        try:
            with warnings.catch_warnings(record=True) as caught_sklearn_warnings:
                warnings.simplefilter("always", sklearn.exceptions.UndefinedMetricWarning)
                scores: np.ndarray = cross_val_score(modelo, X_selected, y.ravel(), cv=cv_folds, scoring='r2')

            sklearn_undefined_metric_triggered = any(
                issubclass(w.category, sklearn.exceptions.UndefinedMetricWarning) for w in caught_sklearn_warnings
            )

            if sklearn_undefined_metric_triggered:
                warnings.warn("CustomWarning: SKLEARN UndefinedMetricWarning capturada en cross_val_score (R² no calculable). Fitness es -1.", UserWarning); return (-1.0,)

            if np.isnan(scores).any():
                warnings.warn("CustomWarning: NaN scores obtenidos de cross_val_score. Fitness es -1.", UserWarning); return (-1.0,)
            return (np.mean(scores),)
        except ValueError as e:
            warnings.warn(f"CustomWarning: ValueError en cross_val_score ({e}). Fitness es -1.", UserWarning); return (-1.0,)
        except Exception as e:
            warnings.warn(f"CustomWarning: Excepción inesperada en cross_val_score ({e}). Fitness es -1.", UserWarning); return (-1.0,)
    return evaluar_individuo

def configurar_algoritmo_genetico(n_features: int) -> base.Toolbox:
    if n_features <= 0: raise ValueError("El número de características (n_features) debe ser positivo.")
    if hasattr(creator, "FitnessMax"): del creator.FitnessMax
    if hasattr(creator, "Individual"): del creator.Individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,)); creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox: base.Toolbox = base.Toolbox(); toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return toolbox

def ejecutar_algoritmo_genetico(X: np.ndarray, y: np.ndarray, n_pop: int = 50,
                                cx_pb: float = 0.7, mut_pb: float = 0.2, n_gen: int = 50,
                                indpb_mutation: float = 0.05, tournsize_selection: int = 3
                               ) -> Tuple[List["creator.Individual"], tools.Logbook]:
    if not (isinstance(n_pop, int) and n_pop > 0): raise ValueError(f"n_pop debe ser un entero positivo, pero es {n_pop}.")
    if not (0.0 <= cx_pb <= 1.0): raise ValueError(f"cx_pb debe estar entre 0.0 y 1.0, pero es {cx_pb}.")
    if not (0.0 <= mut_pb <= 1.0): raise ValueError(f"mut_pb debe estar entre 0.0 y 1.0, pero es {mut_pb}.")
    if not (isinstance(n_gen, int) and n_gen >= 0): raise ValueError(f"n_gen debe ser un entero no negativo, pero es {n_gen}.")
    if not (0.0 <= indpb_mutation <= 1.0): raise ValueError(f"indpb_mutation debe estar entre 0.0 y 1.0, pero es {indpb_mutation}.")
    if not (isinstance(tournsize_selection, int) and tournsize_selection > 0): raise ValueError(f"tournsize_selection debe ser un entero positivo, pero es {tournsize_selection}.")
    if X.shape[1] == 0: raise ValueError("X no tiene características (columnas).")
    n_features: int = X.shape[1]; toolbox: base.Toolbox = configurar_algoritmo_genetico(n_features)
    fitness_func: Callable[[List[int]], Tuple[float]] = crear_fitness_function(X, y)
    toolbox.register("evaluate", fitness_func); toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=indpb_mutation); toolbox.register("select", tools.selTournament, tournsize=tournsize_selection)
    pop: List[creator.Individual] = toolbox.population(n=n_pop)
    stats: tools.Statistics = tools.Statistics(lambda ind: ind.fitness.values); stats.register("avg", np.mean); stats.register("max", np.max); stats.register("min", np.min)
    try:
        final_pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cx_pb, mutpb=mut_pb, ngen=n_gen, stats=stats, verbose=False)
    except Exception as e:
        warnings.warn(f"Error durante la ejecución de DEAP algorithms.eaSimple: {e}", RuntimeWarning); return pop, tools.Logbook()
    return final_pop, logbook

def obtener_mejor_individuo(population: List["creator.Individual"]) -> Optional["creator.Individual"]:
    if not population: return None
    return tools.selBest(population, k=1)[0]

def entrenar_modelo_final(X: np.ndarray, y: np.ndarray, selected_features_indices: List[int],
                          test_size: float = 0.2, random_state: Optional[int] = 42
                         ) -> Tuple[LinearRegression, float, float]:
    if not selected_features_indices: raise ValueError("La lista selected_features_indices no puede estar vacía.")
    X_selected: np.ndarray = X[:, selected_features_indices]
    if X_selected.shape[0] == 0: raise ValueError("X_selected (datos con características seleccionadas) no tiene filas.")
    y_ravel: np.ndarray = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_ravel, test_size=test_size, random_state=random_state)
    modelo: LinearRegression = LinearRegression(); modelo.fit(X_train, y_train)
    y_pred_train: np.ndarray = modelo.predict(X_train); y_pred_test: np.ndarray = modelo.predict(X_test)
    r2_train: float = r2_score(y_train, y_pred_train); r2_test: float = r2_score(y_test, y_pred_test)
    return modelo, r2_train, r2_test

def graficar_evolucion(logbook: tools.Logbook) -> None:
    if not logbook: warnings.warn("Logbook vacío o nulo, no se puede graficar la evolución.", UserWarning); return
    if logbook.header is None or "gen" not in logbook.header or not logbook.select("gen"):
        warnings.warn("Logbook no contiene datos de 'gen' o está vacío (header missing or no 'gen' data). No se puede graficar.", UserWarning); return
    gen: List[int] = logbook.select("gen"); avg_fitness: List[float] = logbook.select("avg"); max_fitness: List[float] = logbook.select("max")
    plt.figure(figsize=(10, 6)); plt.plot(gen, avg_fitness, label='Promedio R²', linewidth=2); plt.plot(gen, max_fitness, label='Máximo R²', linewidth=2)
    plt.xlabel('Generación'); plt.ylabel('R² Score'); plt.title('Evolución del Coeficiente de Determinación (R²)')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); plt.show()

def mostrar_caracteristicas_seleccionadas(mejor_individuo: Optional["creator.Individual"],
                                        nombres_columnas: List[str]) -> Tuple[List[int], List[str]]:
    if mejor_individuo is None: print("\nNo hay un mejor individuo para mostrar (posiblemente la población inicial estaba vacía o hubo un error)."); return [], []
    selected_indices: List[int] = [i for i, bit in enumerate(mejor_individuo) if bit == 1]
    valid_selected_indices = [i for i in selected_indices if i < len(nombres_columnas)]
    if len(valid_selected_indices) != len(selected_indices):
        warnings.warn("Algunos índices de características seleccionadas estaban fuera de rango para los nombres de columnas proporcionados.", UserWarning); selected_indices = valid_selected_indices
    selected_names: List[str] = [nombres_columnas[i] for i in selected_indices]
    print("\nCaracterísticas seleccionadas por el Algoritmo Genético:")
    if selected_names:
        for i, name in enumerate(selected_names): print(f"  {i+1}. {name}")
    else: print("  Ninguna característica fue seleccionada o los nombres no pudieron ser mapeados.")
    return selected_indices, selected_names

def analizar_outliers_multivariado(df: pl.DataFrame, umbral: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    if df.is_empty():
        warnings.warn("DataFrame de entrada para analizar_outliers_multivariado está vacío. Retornando arrays vacíos.", UserWarning); return np.array([], dtype=bool), np.array([])
    df_numeric = df.select([pl.col(c) for c in df.columns if df[c].dtype in [pl.Float32, pl.Float64, pl.Int16, pl.Int32, pl.Int64]])
    if df_numeric.is_empty() or df_numeric.width == 0:
        warnings.warn("DataFrame no contiene columnas numéricas para análisis multivariado. Retornando arrays vacíos.", UserWarning); return np.array([], dtype=bool), np.array([])
    has_any_null = False
    for col_name in df_numeric.columns:
        if np.isnan(df_numeric[col_name].to_numpy()).any(): has_any_null = True; break
    has_any_inf = False
    for col_name in df_numeric.columns:
        if np.isinf(df_numeric[col_name].to_numpy()).any(): has_any_inf = True; break
    if has_any_null or has_any_inf:
        raise ValueError("DataFrame contiene NaNs o Infinitos en columnas numéricas. No se pueden calcular distancias de Mahalanobis.")
    data: np.ndarray = df_numeric.to_numpy()
    if data.shape[0] < data.shape[1]:
        warnings.warn(f"Hay menos filas ({data.shape[0]}) que columnas ({data.shape[1]}). La matriz de covarianza será singular. No se pueden calcular distancias de Mahalanobis.", UserWarning); return np.zeros(len(data), dtype=bool), np.zeros(len(data))
    if data.shape[0] == 0: return np.zeros(0, dtype=bool), np.zeros(0)
    mean: np.ndarray = np.mean(data, axis=0); mahal_dist_list: List[float] = []
    try:
        if np.any(np.std(data, axis=0) < 1e-9):
             warnings.warn("Al menos una característica tiene varianza cercana a cero. La matriz de covarianza podría ser singular. Se devuelven no outliers.", UserWarning); return np.zeros(len(data), dtype=bool), np.zeros(len(data))
        cov: np.ndarray = np.cov(data.T)
        if cov.ndim == 0:
            if cov == 0:
                warnings.warn("La única característica numérica tiene varianza cero. No se pueden calcular distancias de Mahalanobis.", UserWarning); return np.zeros(len(data), dtype=bool), np.zeros(len(data))
            inv_cov_diag = 1.0 / cov; mahal_dist_list = [np.sqrt(((row - mean)**2).sum() * inv_cov_diag) for row in data]
        else:
            if np.linalg.matrix_rank(cov) < cov.shape[0]:
                warnings.warn("Matriz de covarianza es singular (rango deficiente). No se pueden calcular distancias de Mahalanobis.", UserWarning); return np.zeros(len(data), dtype=bool), np.zeros(len(data))
            inv_cov: np.ndarray = np.linalg.inv(cov)
            for row_idx in range(data.shape[0]):
                diff: np.ndarray = data[row_idx] - mean; mahal_dist_list.append(np.sqrt(diff.T @ inv_cov @ diff))
    except (np.linalg.LinAlgError, ValueError) as e:
        warnings.warn(f"Error en cálculo de Mahalanobis (datos problemáticos o matriz singular): {e}", UserWarning); return np.zeros(len(data), dtype=bool), np.zeros(len(data))
    if not mahal_dist_list and data.size > 0:
        warnings.warn("mahal_dist_list no fue populada, retornando no outliers.", UserWarning); return np.zeros(len(data), dtype=bool), np.zeros(len(data))
    mahal_dist: np.ndarray = np.array(mahal_dist_list) if mahal_dist_list else np.array([]); outliers: np.ndarray = mahal_dist > umbral if mahal_dist.size > 0 else np.array([], dtype=bool)
    return outliers, mahal_dist

def probar_multiples_configuraciones(X: np.ndarray, y: np.ndarray,
                                   configuraciones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if X.ndim != 2 or X.shape[1] == 0: raise ValueError("X debe ser un array 2D con al menos una característica.")
    if y.ndim != 1: raise ValueError("y debe ser un array 1D.")
    if X.shape[0] != y.shape[0]: raise ValueError(f"Incompatibilidad de formas: X tiene {X.shape[0]} filas y y tiene {y.shape[0]} elementos.")
    resultados: List[Dict[str, Any]] = []
    for i, config in enumerate(configuraciones):
        print(f"\n--- Probando configuración {i+1}/{len(configuraciones)} ---")
        expected_keys = ['n_pop', 'cx_pb', 'mut_pb', 'n_gen']
        if not all(key in config for key in expected_keys):
            warnings.warn(f"Configuración {i+1} incompleta. Esperadas: {expected_keys}. Recibidas: {config.keys()}. Saltando esta configuración.", UserWarning); continue
        print(f"Parámetros: N_POP={config['n_pop']}, CX_PB={config['cx_pb']}, MUT_PB={config['mut_pb']}, N_GEN={config['n_gen']}")
        try:
            pop, logbook = ejecutar_algoritmo_genetico(X, y, n_pop=config['n_pop'], cx_pb=config['cx_pb'], mut_pb=config['mut_pb'], n_gen=config['n_gen'])
        except ValueError as e:
            warnings.warn(f"Error al ejecutar algoritmo genético para config {i+1}: {e}. Saltando esta configuración.", UserWarning); continue
        mejor_individuo: Optional["creator.Individual"] = obtener_mejor_individuo(pop)
        if mejor_individuo is not None and mejor_individuo.fitness.valid:
            mejor_fitness: float = mejor_individuo.fitness.values[0]; print(f"Mejor Fitness (R²) para config {i+1}: {mejor_fitness:.4f}")
        else:
            mejor_fitness = -float('inf'); print(f"No se pudo obtener un mejor individuo válido o fitness para config {i+1}.")
        resultados.append({'configuracion': config, 'mejor_fitness': mejor_fitness, 'mejor_individuo': mejor_individuo, 'logbook': logbook})
    print("\n--- Fin de pruebas de configuraciones ---")
    return resultados

def graficar_valores_reales_vs_predichos(y_real: np.ndarray, y_predicho: np.ndarray, titulo: str = "Comparación") -> None:
    if y_real.shape != y_predicho.shape: raise ValueError(f"y_real (shape {y_real.shape}) e y_predicho (shape {y_predicho.shape}) deben tener la misma forma.")
    if y_real.ndim != 1: warnings.warn(f"Se esperaba un array 1D para y_real, pero tiene {y_real.ndim} dimensiones. Se intentará aplanar.", UserWarning); y_real = y_real.ravel()
    if y_predicho.ndim != 1: warnings.warn(f"Se esperaba un array 1D para y_predicho, pero tiene {y_predicho.ndim} dimensiones. Se intentará aplanar.", UserWarning); y_predicho = y_predicho.ravel()
    is_scaled: bool = (np.all(y_real >= -0.001) and np.all(y_real <= 1.001) and np.all(y_predicho >= -0.001) and np.all(y_predicho <= 1.001))
    escala_info: str = "(Escalados)" if is_scaled else "(Originales)"; titulo_completo: str = f"{titulo}: Valores Reales vs Predicciones {escala_info}"
    plt.figure(figsize=(12, 6)); plt.plot(y_real, label=f'Valor Real {escala_info}', color='dodgerblue', linewidth=2, alpha=0.8); plt.plot(y_predicho, label=f'Predicción {escala_info}', color='orangered', linestyle='--', linewidth=2, alpha=0.8)
    plt.title(titulo_completo, fontsize=15); plt.xlabel('Observaciones', fontsize=12); plt.ylabel('Valor de la Variable Objetivo', fontsize=12); plt.legend(fontsize=10); plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout(); plt.show()

if __name__ == '__main__':
    try:
        sample_data_dict = {
            'TV': [100.0, 120.0, 150.0, 110.0, 90.0, 200.0, 50.0, 75.0, 180.0, 130.0, 130.0, 130.0],
            'Radio': [20.0, 25.0, 30.0, 22.0, 18.0, 35.0, 10.0, 15.0, 32.0, 28.0, 28.0, 28.0],
            'Newspaper': [10.0, 12.0, 15.0, 11.0, 9.0, 20.0, 5.0, 8.0, 18.0, 13.0, 0.0, -5.0],
            'Sales': [10.1, 11.5, 13.8, 11.2, 9.5, 18.0, 6.0, 8.5, 16.0, 12.5, 12.5, 12.5],
            'NonNumeric': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
        }
        df_adv = pl.DataFrame(sample_data_dict)
        print("DataFrame de ejemplo creado:"); print(df_adv.head())
        print("\n--- Análisis de Outliers Multivariado ---")
        df_para_mahalanobis = df_adv.select(['TV', 'Radio', 'Newspaper'])
        if not df_para_mahalanobis.is_empty() and df_para_mahalanobis.width > 0:
            try:
                outliers_maha, distancias_maha = analizar_outliers_multivariado(df_para_mahalanobis)
                print(f"\nOutliers multivariados detectados (Mahalanobis > 3): {np.sum(outliers_maha)} de {len(df_para_mahalanobis)}")
            except ValueError as e: print(f"\nError en análisis multivariado: {e}")
        else: print("\nDataFrame para análisis multivariado está vacío o no tiene columnas.")
    except Exception as e:
        print(f"Ocurrió un error inesperado en el bloque __main__: {e}")
        import traceback
        traceback.print_exc()
