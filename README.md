# Sesión 11 – Algoritmos Genéticos, Selección de Variables y Regresión Lineal  
**Big Data y Ciencia de Datos – [TECSUP](https://www.tecsup.edu.pe/)**

---

## Tabla de Contenidos
1. [Descripción General](#descripción-general)  
2. [Objetivos del Laboratorio](#objetivos-del-laboratorio)  
3. [Descripción del Conjunto de Datos](#descripción-del-conjunto-de-datos)  
4. [Metodología y Flujo de Trabajo](#metodología-y-flujo-de-trabajo)  
5. [Estructura del Repositorio](#estructura-del-repositorio)  
6. [Requisitos de Software](#requisitos-de-software)  
7. [Instalación y Puesta en Marcha](#instalación-y-puesta-en-marcha)  
8. [Uso Paso a Paso](#uso-paso-a-paso)  
9. [Detalles de Implementación](#detalles-de-implementación)  
10. [Pruebas Unitarias](#pruebas-unitarias)  
11. [Resultados Principales](#resultados-principales)  
12. [Visualizaciones Generadas](#visualizaciones-generadas)  
13. [Buenas Prácticas y Estilo de Código](#buenas-prácticas-y-estilo-de-código)  
14. [Contribuciones](#contribuciones)  
15. [Créditos y Reconocimientos](#créditos-y-reconocimientos)  
16. [Licencia](#licencia)  

---

## Descripción General
Este proyecto demuestra la aplicación de **Polars** y **scikit-learn** para resolver un problema de predicción de ventas a partir de inversiones publicitarias en distintos medios. El flujo completo incluye:

- Limpieza y exploración inicial de datos.  
- Detección y tratamiento de _outliers_ (univariados y multivariados).  
- Escalamiento de variables mediante normalización _Min-Max_ implementada manualmente.  
- Selección de variables con **algoritmos genéticos** (DEAP).  
- Entrenamiento y evaluación de **regresión lineal** multivariable.  
- Visualizaciones analíticas y comparativas.  
- Suite completa de **pruebas unitarias** con `pytest`.

Se entrega un **notebook totalmente reproducible**, código modularizado en `utils.py`, y un conjunto robusto de pruebas para garantizar la estabilidad del pipeline.

---

## Objetivos del Laboratorio
1. **Aplicar Polars** en lugar de pandas para un rendimiento superior en DataFrames.  
2. Desarrollar un flujo **end-to-end** que abarque preprocesamiento → modelado → evaluación.  
3. Implementar **algoritmos genéticos** para encontrar subconjuntos óptimos de variables explicativas.  
4. Construir un **modelo de regresión lineal** y compararlo frente al modelo completo de referencia.  
5. Asegurar la **reproducibilidad** y la **calidad del software** mediante pruebas automatizadas.

---

## Descripción del Conjunto de Datos
| Fuente | GitHub – Rama `develop` |
| ------ | ---------------------- |
| URL    | `https://raw.githubusercontent.com/AzShet/Data_Mining-LAB11/refs/heads/develop/Advertising-1.csv` |

| Variable   | Descripción                              |
| ---------- | ---------------------------------------- |
| **TV**     | Presupuesto invertido en anuncios de TV  |
| **Radio**  | Presupuesto invertido en anuncios de radio |
| **Newspaper** | Presupuesto invertido en anuncios impresos |
| **Sales**  | Ventas generadas (variable objetivo)     |

---

## Metodología y Flujo de Trabajo


```
            ┌──────────┐
            │  Datos   │
            └────┬─────┘
                 │
  ┌──────────────┴──────────────┐
  │  Exploración y Descripción   │
  └──────────┬──────────┬───────┘
             │          │
 Outliers Univ.  Outliers Multiv.
             │          │
      ┌──────┴───────┐  │
      │  Escalamiento│  │
      └──────┬───────┘  │
             │          │
      ┌──────┴──────────┴─────────┐
      │ Algoritmo Genético (DEAP) │
      └──────┬──────────┬─────────┘
             │          │
Conjunto-óptimo   Conjunto completo
             │          │
     Regresión lineal   │
             │          │
     Métricas y Comparación
             │
       Visualizaciones
```

---

## Estructura del Repositorio
```

├── LAB11-RUELAS.ipynb      # Notebook principal
├── utils.py                # Módulo con funciones reutilizables
├── test_utils.py           # Pruebas unitarias con pytest
├── requirements.txt        # Dependencias de Python
├── README.md               # Documentación (este archivo)
└── .gitignore              # Exclusiones de Git

````

---

## Requisitos de Software
| Componente        | Versión mínima |
| ----------------- | -------------- |
| Python            | 3.9           |
| polars            | 0.20          |
| numpy             | 1.26          |
| scikit-learn      | 1.4           |
| deap              | 1.4           |
| matplotlib        | 3.8           |
| pytest            | 8.1           |

> **Nota**: Todas las versiones exactas se encuentran fijadas en `requirements.txt`.

---

## Instalación y Puesta en Marcha
```bash
# 1. Clonar el repositorio
git clone https://github.com/<usuario>/<repositorio>.git
cd <repositorio>

# 2. Crear y activar entorno virtual (opcional pero recomendado)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Lanzar JupyterLab / VS Code / o ejecutar directamente el notebook
jupyter lab
```

---

## Uso Paso a Paso

1. Abrir `LAB11-RUELAS.ipynb`.
2. Ejecutar las celdas **en orden** para reproducir todo el flujo.
3. Editar los parámetros del algoritmo genético en la sección *“configuraciones”* si se desea experimentación.
4. Los resultados y gráficos se generarán de forma interactiva.

---

## Detalles de Implementación

### Preprocesamiento

* **Tratamiento de NaN**: No se detectaron valores faltantes; se documenta la revisión.
* **Outliers univariados**: Método del rango inter-cuartílico (IQR) con reemplazo por la mediana para cada variable numérica.
* **Outliers multivariados**: Distancia de Mahalanobis con umbral configurable (por defecto = 3).
* **Escalamiento**: Normalización *Min-Max* manual sobre todas las columnas numéricas (100 % Polars, sin `MinMaxScaler` de `sklearn`).

### Selección de Variables

Algoritmo genético configurado con:

* **Población inicial**: 30–100 individuos (configurable).
* **Operadores**:

  * Cruce `cxTwoPoint` (prob. 0.6–0.9).
  * Mutación `mutFlipBit` (prob. 0.1–0.3).
* **Fitness**: Promedio del **R²** obtenido por validación cruzada (CV = 5).
* **Selección**: Torneo de tamaño = 3.
* **Estadísticas**: Se registra promedio, máximo y mínimo en cada generación.

### Modelado y Evaluación

* **Regresión lineal** multivariable (`LinearRegression` de `sklearn`).
* Comparación de métricas **R²** en entrenamiento/prueba para:

  1. Conjunto de variables seleccionadas.
  2. Conjunto completo de variables.
* **Visualizaciones**:

  * Evolución del fitness del GA.
  * Valores reales vs. predichos (escalados).
  * Recta de regresión univariante (TV → Sales).

---

## Pruebas Unitarias

Las funciones críticas están cubiertas por `pytest`:

| Test                                        | Objetivo Principal                                    |
| ------------------------------------------- | ----------------------------------------------------- |
| `test_cargar_datos`                         | Verifica carga remota y columnas esperadas            |
| `test_tratar_outliers_iqr`                  | Confirma tratamiento de outliers univariados          |
| `test_escalar_datos`                        | Asegura rango $0, 1$ tras escalamiento                |
| `test_crear_fitness_function`               | Comprueba salida tipo `tuple` del fitness             |
| `test_ejecutar_algoritmo_genetico`          | Valida tamaño de población y logbook                  |
| `test_entrenar_modelo_final`                | Garantiza `R² ≥ 0` y modelo entrenado                 |
| `test_graficar_valores_reales_vs_predichos` | Confirma que la función visual se ejecuta sin errores |

Ejecutar tests:

```bash
pytest test_utils.py -v
```

---

## Resultados Principales

| Métrica                    | Conjunto Completo         | Conjunto Seleccionado |
| -------------------------- | ------------------------- | --------------------- |
| **R² (train)**             | 0.xx                      | 0.yy                  |
| **R² (test)**              | 0.xx                      | 0.yy                  |
| **Reducción de variables** | n\_original → n\_opt      | \~ ZZ %               |
| **Mejor configuración GA** | Ver *output* del notebook |                       |

> Los valores concretos se generan dinámicamente en la sección “Resumen de Resultados” del notebook.

---

## Visualizaciones Generadas

1. **Evolución del fitness (GA)** – Línea de promedio y máximo por generación.
2. **Valores reales vs. predicciones** – Comparación temporal tras escalamiento.
3. **Recta de regresión simple (TV vs Sales)** – Dispersión y línea ajustada.

Todas las figuras se renderizan con **Matplotlib** y permanecen embebidas en el notebook para análisis posterior.

---

## Buenas Prácticas y Estilo de Código

* Demarcación estricta entre **celdas Markdown** (documentación) y **celdas de código**.
* **Docstrings** detallados en cada función de `utils.py`.
* Variables **globales en mayúsculas**, funciones **snake\_case**.
* Evitada la dependencia de `pandas`; **100 % Polars** para manipulación de datos.
* Convenciones PEP 8 respetadas (longitud de línea, espacios, etc.).
* Uso de **tipado** donde agrega valor.

---

## Contribuciones

Se aceptan *pull requests* con:

1. Nuevos experimentos de configuración del GA.
2. Métricas adicionales (MAE, RMSE, Adjusted R²).
3. Integración de otras técnicas de selección de variables (e.g., *Recursive Feature Elimination*).

Por favor abrir un *issue* antes de enviar grandes cambios para coordinación.

---

## Créditos y Reconocimientos

* **Profesor**: [Luis Paraguay Arzapalo](https://github.com/luispar90) – Curso *Minería de Datos*.
* **DEAP**: Proyecto de evolución algorítmica en Python.
* **Polars**: Motor de DataFrames optimizado en Rust.
* Comunidad de **scikit-learn** por las herramientas de modelado.

---

## Licencia

Este repositorio se publica bajo la licencia **MIT**.
Revisar el archivo `LICENSE` para más detalles.

---
