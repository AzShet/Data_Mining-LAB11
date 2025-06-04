import pytest
import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sklearn.exceptions
from deap import base, creator, tools, algorithms
import warnings # Python's built-in warnings
import random
import re # Import re for regex search in warnings
import matplotlib.pyplot as plt

# Import functions from utils.py
from utils import *

# --- Fixtures ---

@pytest.fixture
def sample_df_features() -> pl.DataFrame:
    return pl.DataFrame({
        "TV": [100.0, 150.0, 50.0, 200.0, 120.0, -10.0, 250.0],
        "Radio": [20.0, 30.0, 10.0, 40.0, 25.0, 5.0, 35.0],
        "Newspaper": [10.0, 15.0, 5.0, 20.0, 12.0, 2.0, 18.0],
        "Sales": [10.1, 15.2, 5.3, 20.4, 12.5, 3.0, 22.1]
    })

@pytest.fixture
def empty_df() -> pl.DataFrame:
    return pl.DataFrame()

@pytest.fixture
def df_col_all_nans() -> pl.DataFrame: # Simplified as per prompt
    return pl.DataFrame({
        "A": [1.0],
        "B": [np.nan],
        "C": [4.0]
    }, schema={"A": pl.Float64, "B": pl.Float64, "C": pl.Float64})

@pytest.fixture
def df_with_nans_in_numeric() -> pl.DataFrame:
    return pl.DataFrame({
        "A": [1.0, np.nan, 3.0],
        "B": [4.0, 5.0, 6.0],
        "C": ["x", "y", "z"]
    }, schema={"A": pl.Float64, "B": pl.Float64, "C": pl.Utf8})


@pytest.fixture
def df_constant_value_col() -> pl.DataFrame:
    return pl.DataFrame({"A": [5.0, 5.0, 5.0], "B": [1.0, 2.0, 3.0]})

@pytest.fixture
def df_non_numeric() -> pl.DataFrame:
    return pl.DataFrame({"A": ["x", "y", "z"], "B": [1, 2, 3]})

@pytest.fixture
def df_truly_non_numeric() -> pl.DataFrame:
    return pl.DataFrame({"A": ["x", "y", "z"], "C": ["u", "v", "w"]})

@pytest.fixture
def sample_X_y_numpy(sample_df_features: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X_df, y_df = separar_variables(sample_df_features, "Sales")
    return X_df.to_numpy(), y_df.to_numpy().ravel()

@pytest.fixture
def xy_datos_y_constante() -> Tuple[np.ndarray, np.ndarray]:
    X = np.random.rand(20, 3)
    y = np.ones(20)
    return X, y

@pytest.fixture(scope="function")
def deap_toolbox_setup():
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return toolbox

def create_mock_csv(tmp_path, content: str, filename: str = "test.csv") -> str:
    file_path = tmp_path / filename
    file_path.write_text(content)
    return str(file_path)

# --- Tests for cargar_datos ---
def test_cargar_datos_happy_path(tmp_path):
    csv_content = "col1,col2\n1,a\n2,b"
    file_path = create_mock_csv(tmp_path, csv_content)
    expected_df = pl.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    df = cargar_datos(file_path)
    assert_frame_equal(df, expected_df)

def test_cargar_datos_invalid_url_raises_error():
    invalid_url = "nonexistent://file.csv"
    with pytest.raises(pl.exceptions.PolarsError, match="No se pudo cargar el archivo CSV"):
        cargar_datos(invalid_url)

def test_cargar_datos_empty_csv_warns(tmp_path):
    csv_content = ""
    file_path = create_mock_csv(tmp_path, csv_content, "empty.csv")
    with pytest.warns(UserWarning, match="El archivo CSV cargado desde .* está vacío."):
        df = cargar_datos(file_path)
    assert df.is_empty()

# --- Tests for tratar_outliers_iqr ---
def test_tratar_outliers_iqr_col_all_nans_warns(df_col_all_nans: pl.DataFrame):
    expected_msg = "NumPyDetect: Columna 'B' es todo NaNs \\(detectado con NumPy\\)\\. Se omitirá en tratamiento de outliers\\."
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", UserWarning)
        df_result = tratar_outliers_iqr(df_col_all_nans.clone(), columnas=["B"]) # Test specific all-NaN column "B"

    assert np.isnan(df_result["B"].to_numpy()).all(), f"Column B was not all NaNs (checked with numpy). Content: {df_result['B']}"
    assert_frame_equal(df_result.drop("B"), df_col_all_nans.drop("B"))

    found_warning = False
    for w in record:
        if issubclass(w.category, UserWarning) and re.search(expected_msg, str(w.message)):
            found_warning = True
            break
    assert found_warning, f"Expected warning '{expected_msg}' not found. Recorded: {[str(wr.message) for wr in record]}"


# --- Tests for escalar_datos ---
def test_escalar_datos_col_all_nans_warns_and_fills_nan(df_col_all_nans: pl.DataFrame):
    expected_msg = "NumPyDetect: Columna 'B' es todo NaNs \\(detectado con NumPy\\)\\. Se escalará a todos NaNs \\(Float64\\)\\."
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", UserWarning)
        df_scaled = escalar_datos(df_col_all_nans.clone(), columnas=["B"])

    assert np.isnan(df_scaled["B"].to_numpy()).all(), f"Column B was not all NaNs (checked with numpy). Content: {df_scaled['B']}"
    assert df_scaled["B"].dtype == pl.Float64
    assert_frame_equal(df_scaled.drop("B"), df_col_all_nans.drop("B"))

    found_warning = False
    for w in record:
        if issubclass(w.category, UserWarning) and re.search(expected_msg, str(w.message)):
            found_warning = True
            break
    assert found_warning, f"Expected warning '{expected_msg}' not found. Recorded: {[str(wr.message) for wr in record]}"


# --- Tests for crear_fitness_function & evaluar_individuo ---
def test_evaluar_individuo_cross_val_score_handles_undefined_metric_warns(xy_datos_y_constante: Tuple[np.ndarray, np.ndarray]):
    X, y = xy_datos_y_constante
    eval_func = crear_fitness_function(X, y)
    individual = [1, 1, 1]

    expected_custom_warning_message = ("CustomWarning: SKLEARN UndefinedMetricWarning capturada en cross_val_score "
                                       "\\(R² no calculable\\)\\. Fitness es -1\\.")

    with pytest.warns(UserWarning, match=expected_custom_warning_message) as record:
        fitness = eval_func(individual)

    assert fitness == (-1.0,)
    assert any(re.search(expected_custom_warning_message, str(w.message)) for w in record.list), \
           f"Expected warning not found. Recorded: {[str(w.message) for w in record.list]}"


# --- Other existing tests (abbreviated for brevity but included in overwrite) ---
def test_tratar_outliers_iqr_happy_path(sample_df_features: pl.DataFrame):
    df = sample_df_features.with_columns(pl.Series("TV", [100.0, 150.0, 50.0, 800.0, 120.0, -200.0, 250.0]))
    df_treated = tratar_outliers_iqr(df.clone(), columnas=["TV"])
    assert 120.0 in df_treated["TV"].to_list()

def test_escalar_datos_happy_path(sample_df_features: pl.DataFrame):
    df = sample_df_features.select(["TV", "Radio"])
    df_scaled = escalar_datos(df.clone())
    assert df_scaled["TV"].min() >= 0.0

def test_separar_variables_happy_path(sample_df_features: pl.DataFrame):
    X_df, y_df = separar_variables(sample_df_features.clone(), "Sales")
    assert "Sales" not in X_df.columns

def test_convertir_a_numpy_happy_path(sample_df_features: pl.DataFrame):
    np_array = convertir_a_numpy(sample_df_features.select(["TV", "Sales"]))
    assert isinstance(np_array, np.ndarray)

def test_configurar_algoritmo_genetico_happy_path():
    toolbox = configurar_algoritmo_genetico(5)
    assert hasattr(toolbox, "individual")

def test_ejecutar_algoritmo_genetico_happy_path(sample_X_y_numpy: Tuple[np.ndarray, np.ndarray], deap_toolbox_setup):
    X, y = sample_X_y_numpy
    pop, logbook = ejecutar_algoritmo_genetico(X, y, n_pop=10, n_gen=2)
    assert len(pop) == 10

def test_obtener_mejor_individuo_happy_path(deap_toolbox_setup):
    pop = deap_toolbox_setup.population(n=5)
    for i, ind in enumerate(pop): ind.fitness.values = (float(i),)
    assert obtener_mejor_individuo(pop).fitness.values == (4.0,)

def test_entrenar_modelo_final_happy_path(sample_X_y_numpy: Tuple[np.ndarray, np.ndarray]):
    X, y = sample_X_y_numpy
    model, _, _ = entrenar_modelo_final(X, y, [0, 1])
    assert isinstance(model, LinearRegression)

def test_graficar_evolucion_happy_path(deap_toolbox_setup):
    logbook = tools.Logbook(); logbook.record(gen=0, avg=0.5, max=0.8)
    try: graficar_evolucion(logbook); plt.close()
    except Exception as e: pytest.fail(f"graficar_evolucion raised: {e}")

def test_mostrar_caracteristicas_seleccionadas_happy_path(deap_toolbox_setup):
    ind = deap_toolbox_setup.individual(); ind[:] = [1,0,1]
    _, names = mostrar_caracteristicas_seleccionadas(ind, ["TV", "Radio", "Newspaper"])
    assert names == ["TV", "Newspaper"]

def test_analizar_outliers_multivariado_happy_path(sample_df_features: pl.DataFrame):
    df_numeric = sample_df_features.select(["TV", "Radio", "Newspaper"])
    outliers, _ = analizar_outliers_multivariado(df_numeric.clone())
    assert len(outliers) == df_numeric.height

def test_analizar_outliers_multivariado_df_with_nans_raises_error(df_with_nans_in_numeric: pl.DataFrame):
    with pytest.raises(ValueError, match="DataFrame contiene NaNs o Infinitos en columnas numéricas\\. No se pueden calcular distancias de Mahalanobis\\."):
        analizar_outliers_multivariado(df_with_nans_in_numeric)

def test_probar_multiples_configuraciones_happy_path(sample_X_y_numpy: Tuple[np.ndarray, np.ndarray], deap_toolbox_setup):
    X, y = sample_X_y_numpy
    configs = [{'n_pop': 5, 'cx_pb': 0.5, 'mut_pb': 0.1, 'n_gen': 1}]
    results = probar_multiples_configuraciones(X, y, configs)
    assert len(results) == 1

def test_graficar_valores_reales_vs_predichos_happy_path():
    y_real = np.array([1.0, 2.0, 3.0]); y_pred = np.array([1.1, 1.9, 3.2])
    try: graficar_valores_reales_vs_predichos(y_real, y_pred); plt.close()
    except Exception as e: pytest.fail(f"graficar_valores_reales_vs_predichos raised: {e}")

def teardown_module(module):
    plt.close('all')
