"""
cross_bin_evaluator.py

Evalúa tablas cruzadas (contingencia) entre dos grupos de variables numéricas
binarizadas de forma óptima (OptBinning), calculando el Gini global de cada
cruce y el recall de cobertura de las celdas que superan un umbral de
defaults.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from optbinning import OptimalBinning
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def _orientar_por_riesgo(bin_idx: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Reorienta los índices de bin de una variable para que un índice mayor
    siempre corresponda a mayor riesgo (mayor tasa de default).

    OptBinning garantiza que cada variable, por sí sola, es monotónica
    respecto al target -- pero no garantiza en qué DIRECCIÓN (podría ser
    "a mayor bin, mayor riesgo" o "a mayor bin, menor riesgo",
    dependiendo del signo natural de la variable). Si dos variables con
    direcciones opuestas se combinan sin ajustar, se forma una matriz en
    "diagonal invertida" (ej. bin_a sube el riesgo, bin_b lo baja) que es
    perfectamente válida y ordenada, pero que el código posicional
    fila*n_columnas+columna o el score de monotonicidad podrían interpretar
    incorrectamente como desorden.

    Esta función mide la correlación entre el índice de bin y el target;
    si es negativa, invierte el orden de los índices (0 <-> max, 1 <-> max-1,
    etc.) para que la dirección quede alineada ("mayor índice = mayor
    riesgo") de forma consistente entre variables.
    """
    bin_idx = bin_idx.astype(int)
    if len(np.unique(y)) < 2 or len(np.unique(bin_idx)) < 2:
        return bin_idx

    corr = np.corrcoef(bin_idx, y)[0, 1]
    if np.isnan(corr) or corr >= 0:
        return bin_idx

    max_idx = bin_idx.max()
    return max_idx - bin_idx


def _gini_logistico(
    bin_a: np.ndarray,
    bin_b: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
) -> float:
    """
    Calcula el Gini de la tabla cruzada ajustando una regresión logística
    de UNA sola variable: el código posicional (bijectivo) de cada celda.

    En vez de usar bin_a y bin_b como dos predictores separados, ni un
    producto simple fila*columna (que puede colisionar -> celdas distintas
    con el mismo código, ej. fila=2,col=3 y fila=3,col=2 podrían dar el
    mismo valor en algunas grillas), se codifica cada celda como:

        n_columnas = número de bins distintos de la variable B
        codigo = fila * n_columnas + columna

    donde fila = índice de bin_a + 1 y columna = índice de bin_b + 1
    (1-indexado). Esto equivale a "aplanar" la matriz 2D en un índice 1D
    -- es matemáticamente inyectivo (nunca hay dos celdas con el mismo
    código dado un número fijo de columnas), y conserva una relación
    mayormente monotónica con el riesgo: fila domina el orden y columna
    actúa como desempate.

    IMPORTANTE: antes de construir el código, ambos ejes se pasan por
    `_orientar_por_riesgo` para que "índice mayor" signifique siempre
    "mayor riesgo" en las dos variables. Sin este paso, una variable que
    sube el riesgo y otra que lo baja (diagonal invertida) se cancelarían
    parcialmente al combinarse en un solo código, deteriorando el Gini
    aunque los datos sean muy predictivos.

    Ventajas frente al enfoque de dos predictores:
    - Sigue siendo un modelo paramétrico (no memoriza celdas individuales),
      por lo que conserva la resistencia al sobreajuste.
    - Captura el efecto conjunto fila-columna en una sola dimensión
      ordinal, sin colisiones entre celdas distintas.
    - Se evalúa con validación cruzada (out-of-fold) para evitar el sesgo
      optimista de medir el AUC sobre los mismos datos de ajuste.
    """
    if len(np.unique(y)) < 2:
        return np.nan

    bin_a_or = _orientar_por_riesgo(bin_a, y)
    bin_b_or = _orientar_por_riesgo(bin_b, y)

    fila = bin_a_or.astype(int) + 1
    columna = bin_b_or.astype(int) + 1
    n_columnas = int(bin_b_or.astype(int).max()) + 1
    codigo = (fila * n_columnas + columna).astype(float).reshape(-1, 1)

    n_minoritaria = np.bincount(y.astype(int)).min()
    n_splits = min(cv_folds, n_minoritaria)

    try:
        if n_splits < 2:
            # Muy pocos defaults para hacer CV -> se ajusta en la muestra
            # completa (menos honesto, pero evita fallar).
            modelo = LogisticRegression()
            modelo.fit(codigo, y)
            proba = modelo.predict_proba(codigo)[:, 1]
        else:
            skf = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            modelo = LogisticRegression()
            proba = cross_val_predict(
                modelo, codigo, y, cv=skf, method="predict_proba"
            )[:, 1]
        auc = roc_auc_score(y, proba)
        return 2 * auc - 1
    except Exception:
        return np.nan


def _monotonicity_score(pivot_rates: pd.DataFrame) -> float:
    """
    Mide qué tan monotónica es la matriz de tasas de default en ambas
    dimensiones (filas y columnas), usando Spearman por fila/columna.

    Devuelve un score entre 0 (desordenada, tipo Matriz B) y 1
    (perfectamente monotónica en ambas direcciones -- sin importar si
    ambos ejes suben el riesgo en la misma dirección o en direcciones
    opuestas, ej. "diagonal invertida": bin_a sube el riesgo, bin_b lo
    baja. Ambos casos son igual de válidos/ordenados).
    """
    vals = pivot_rates.values.astype(float)
    n_rows, n_cols = vals.shape

    row_corrs = []
    for i in range(n_rows):
        fila = vals[i, :]
        valid = ~np.isnan(fila)
        if valid.sum() > 1 and np.unique(fila[valid]).size > 1:
            corr, _ = spearmanr(np.arange(n_cols)[valid], fila[valid])
            if not np.isnan(corr):
                row_corrs.append(corr)

    col_corrs = []
    for j in range(n_cols):
        col = vals[:, j]
        valid = ~np.isnan(col)
        if valid.sum() > 1 and np.unique(col[valid]).size > 1:
            corr, _ = spearmanr(np.arange(n_rows)[valid], col[valid])
            if not np.isnan(corr):
                col_corrs.append(corr)

    if not row_corrs or not col_corrs:
        return np.nan

    avg_row = float(np.mean(row_corrs))
    avg_col = float(np.mean(col_corrs))

    # NOTA: no se penaliza que avg_row y avg_col tengan signos opuestos.
    # Una variable puede subir el riesgo al aumentar su bin y la otra
    # bajarlo -- eso es una "diagonal invertida" perfectamente válida y
    # ordenada (ej. bin_a fuertemente creciente en riesgo, bin_b
    # fuertemente decreciente), no un desorden. Lo único que importa es
    # que cada eje, por separado, sea consistentemente monotónico
    # (magnitud alta de |avg_row| y |avg_col|), sin importar la dirección
    # relativa entre ambos.
    score = (abs(avg_row) + abs(avg_col)) / 2
    return float(np.clip(score, 0, 1))


def evaluate_cross_bins(
    df: pd.DataFrame,
    vars_grupo_a: list,
    vars_grupo_b: list,
    target_col: str = "target",
    umbral_default_x: int = 30,
    optb_params: dict = None,
    dtype_vars: dict = None,
    cv_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evalúa cruces entre variables de vars_grupo_a y vars_grupo_b tras
    binarizarlas óptimamente con OptBinning.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con las variables y el target (0/1).
    vars_grupo_a : list[str]
        Variables del primer grupo (ej. ventana de tiempo A).
    vars_grupo_b : list[str]
        Variables del segundo grupo (ej. ventana de tiempo B).
    target_col : str
        Nombre de la columna target binaria (0/1).
    umbral_default_x : int
        Umbral mínimo de defaults por celda para considerarla "seleccionada".
    optb_params : dict, opcional
        Parámetros adicionales para OptimalBinning (ej. {"max_n_bins": 5}).
    dtype_vars : dict, opcional
        Mapeo {variable: "numerical"/"categorical"} si hay variables
        categóricas. Por defecto todas se asumen numéricas.
    cv_folds : int
        Número de folds para la validación cruzada del Gini (se ajusta
        automáticamente hacia abajo si hay pocos defaults en el cruce).
    random_state : int
        Semilla para la partición de folds.

    Retorna
    -------
    pd.DataFrame
        Una fila por combinación (var_a, var_b) con:
        - gini_tabla_cruzada : Gini calculado con una regresión logística
          de una sola variable (código fila*columna del bin de cada
          celda) evaluada por validación cruzada. Menos propenso a
          sobreajuste que el Gini "empírico" por celda.
        - orden_monotonico : score 0-1 que mide si la grilla de tasas de
          default (empíricas) respeta un orden consistente por filas y
          columnas (1 = totalmente ordenada tipo "Matriz A", 0 =
          desordenada tipo "Matriz B"). Diagnóstico complementario: con
          regresión logística el desorden ya se penaliza en gran parte
          dentro del propio gini_tabla_cruzada, pero esta columna ayuda a
          interpretar *por qué*.
        - gini_ajustado : gini_tabla_cruzada * orden_monotonico. Úsalo
          como criterio principal de selección.
        - n_celdas_totales / n_celdas_seleccionadas
        - poblacion_acumulada
        - defaults_acumulados
        - recall_cobertura
    """
    optb_params = optb_params or {}
    dtype_vars = dtype_vars or {}

    y = df[target_col].values
    total_defaults_global = df[target_col].sum()

    # ---------- 1. Binarización óptima de todas las variables ----------
    all_vars = list(dict.fromkeys(list(vars_grupo_a) + list(vars_grupo_b)))
    binned_data = {}

    for var in all_vars:
        try:
            x = df[var].values
            dtype = dtype_vars.get(var, "numerical")
            optb = OptimalBinning(name=var, dtype=dtype, **optb_params)
            optb.fit(x, y)
            # metric="indices" -> devuelve el índice de bin por observación
            # (incluye bins especiales/missing como índices negativos)
            binned_data[var] = optb.transform(x, metric="indices")
        except Exception as e:
            warnings.warn(f"No se pudo binarizar la variable '{var}': {e}")

    results = []

    # ---------- 2-5. Cruce de variables ----------
    for var_a, var_b in itertools.product(vars_grupo_a, vars_grupo_b):
        if var_a == var_b:
            continue
        if var_a not in binned_data or var_b not in binned_data:
            continue

        temp = pd.DataFrame(
            {"bin_a": binned_data[var_a], "bin_b": binned_data[var_b], "target": y}
        )

        # Tabla cruzada: población y defaults por celda
        cross_total = temp.groupby(["bin_a", "bin_b"]).size().rename("poblacion")
        cross_defaults = (
            temp.groupby(["bin_a", "bin_b"])["target"].sum().rename("defaults")
        )
        cross_table = pd.concat([cross_total, cross_defaults], axis=1).reset_index()
        cross_table["default_rate"] = (
            cross_table["defaults"] / cross_table["poblacion"]
        )

        # ---------- 3. Gini de la tabla cruzada vía regresión logística ----------
        # En vez de usar la tasa de default empírica de cada celda como
        # score (enfoque in-sample, propenso a sobreajuste en celdas
        # pequeñas/ruidosas), se ajusta una regresión logística de una
        # sola variable: el código fila*columna del bin de cada celda
        # (ej. fila 3, columna 3 -> código 9).
        gini = _gini_logistico(
            temp["bin_a"].values,
            temp["bin_b"].values,
            temp["target"].values,
            cv_folds=cv_folds,
            random_state=random_state,
        )

        # ---------- 3b. Diagnóstico de desorden/no-monotonicidad ----------
        # Aunque la regresión logística ya penaliza estructuras erráticas
        # (no puede fitear bien un patrón no-monotónico con predictores
        # ordinales lineales), se mantiene esta métrica explícita como
        # diagnóstico interpretable de qué tan ordenada está la grilla de
        # tasas empíricas de default, y para reforzar la penalización vía
        # gini_ajustado.
        pivot_rates = cross_table.pivot(index="bin_a", columns="bin_b", values="default_rate")
        orden_monotonico = _monotonicity_score(pivot_rates)
        gini_ajustado = (
            gini * orden_monotonico
            if not np.isnan(gini) and not np.isnan(orden_monotonico)
            else np.nan
        )

        # ---------- 4. Filtrado de celdas por umbral de defaults ----------
        seleccionadas = cross_table[cross_table["defaults"] >= umbral_default_x]

        # ---------- 5. Métricas agregadas del subconjunto ----------
        poblacion_acumulada = int(seleccionadas["poblacion"].sum())
        defaults_acumulados = int(seleccionadas["defaults"].sum())
        recall_cobertura = (
            defaults_acumulados / total_defaults_global
            if total_defaults_global > 0
            else np.nan
        )

        results.append(
            {
                "var_a": var_a,
                "var_b": var_b,
                "gini_tabla_cruzada": gini,
                "orden_monotonico": orden_monotonico,
                "gini_ajustado": gini_ajustado,
                "n_celdas_totales": len(cross_table),
                "n_celdas_seleccionadas": len(seleccionadas),
                "poblacion_acumulada": poblacion_acumulada,
                "defaults_acumulados": defaults_acumulados,
                "recall_cobertura": recall_cobertura,
            }
        )

    # ---------- 6. Output estructurado ----------
    df_resultados = pd.DataFrame(results)
    if not df_resultados.empty:
        df_resultados = df_resultados.sort_values(
            "gini_ajustado", ascending=False
        ).reset_index(drop=True)

    return df_resultados





"""
verificacion_par_var_a_var_b.py

Script plano (sin envolver en función) para verificar, paso a paso, que la
construcción de la tabla cruzada y el Gini estén correctos para UN solo par
de variables: var_a_3m y var_b_3m, sobre df_demo.

Reutiliza las funciones auxiliares del archivo cross_bin_evaluator.py
(deben estar en el mismo directorio, o ajusta el import según tu proyecto):
- _orientar_por_riesgo : alinea la dirección de cada eje (mayor bin =
  mayor riesgo), necesario para que el código posicional funcione bien
  en matrices de "diagonal invertida".
- _gini_logistico       : Gini vía regresión logística de 1 variable
  (código fila*n_columnas+columna, ya orientado por riesgo), con CV.
- _monotonicity_score   : score 0-1 de qué tan ordenada está la grilla
  de tasas de default, sin penalizar direcciones opuestas.
"""

import numpy as np
import pandas as pd
from optbinning import OptimalBinning

# ---------------------------------------------------------------------
# Parámetros de la verificación (ajusta a tu caso)
# ---------------------------------------------------------------------
var_a = "var_a_3m"
var_b = "var_b_3m"
target_col = "target"
umbral_default_x = 15

y = df_demo[target_col].values
total_defaults_global = int(y.sum())

# ---------------------------------------------------------------------
# 1. Binarización óptima de cada variable
# ---------------------------------------------------------------------
optb_a = OptimalBinning(name=var_a, dtype="numerical")
optb_a.fit(df_demo[var_a].values, y)
bin_a = optb_a.transform(df_demo[var_a].values, metric="indices")

optb_b = OptimalBinning(name=var_b, dtype="numerical")
optb_b.fit(df_demo[var_b].values, y)
bin_b = optb_b.transform(df_demo[var_b].values, metric="indices")

print(f"Puntos de corte óptimos de {var_a}: {optb_a.splits}")
print(f"Puntos de corte óptimos de {var_b}: {optb_b.splits}")

# ---------------------------------------------------------------------
# 2. Tabla cruzada (contingencia): población y defaults por celda
# ---------------------------------------------------------------------
temp = pd.DataFrame({"bin_a": bin_a, "bin_b": bin_b, "target": y})

cross_total = temp.groupby(["bin_a", "bin_b"]).size().rename("poblacion")
cross_defaults = temp.groupby(["bin_a", "bin_b"])["target"].sum().rename("defaults")
cross_table = pd.concat([cross_total, cross_defaults], axis=1).reset_index()
cross_table["default_rate"] = cross_table["defaults"] / cross_table["poblacion"]

print("\n--- Tabla cruzada completa (formato largo) ---")
print(cross_table)

# Vista en formato matriz (igual a la imagen: filas = bin_a, columnas = bin_b)
pivot_rate = cross_table.pivot(index="bin_a", columns="bin_b", values="default_rate")
pivot_pop = cross_table.pivot(index="bin_a", columns="bin_b", values="poblacion")
pivot_def = cross_table.pivot(index="bin_a", columns="bin_b", values="defaults")

print(f"\n--- Matriz de tasa de default (fila={var_a}, columna={var_b}) ---")
print(pivot_rate.round(3))

print("\n--- Matriz de población por celda ---")
print(pivot_pop)

print("\n--- Matriz de defaults por celda ---")
print(pivot_def)

# ---------------------------------------------------------------------
# 3. Chequeo de orientación de riesgo por eje (diagnóstico)
# ---------------------------------------------------------------------
bin_a_orientado = _orientar_por_riesgo(bin_a, y)
bin_b_orientado = _orientar_por_riesgo(bin_b, y)

fue_invertido_a = not np.array_equal(bin_a, bin_a_orientado)
fue_invertido_b = not np.array_equal(bin_b, bin_b_orientado)

print(f"\n¿{var_a} fue invertido para alinear con el riesgo?: {fue_invertido_a}")
print(f"¿{var_b} fue invertido para alinear con el riesgo?: {fue_invertido_b}")

# ---------------------------------------------------------------------
# 4. Gini de la tabla cruzada vía regresión logística (código posicional)
# ---------------------------------------------------------------------
gini = _gini_logistico(bin_a, bin_b, y, cv_folds=5, random_state=42)
print(f"\nGini tabla cruzada (regresión logística, CV): {gini:.4f}")

# ---------------------------------------------------------------------
# 5. Score de monotonicidad + Gini ajustado
# ---------------------------------------------------------------------
orden_monotonico = _monotonicity_score(pivot_rate)
gini_ajustado = (
    gini * orden_monotonico
    if not np.isnan(gini) and not np.isnan(orden_monotonico)
    else np.nan
)

print(f"Orden monotónico: {orden_monotonico:.4f}")
print(f"Gini ajustado: {gini_ajustado:.4f}")

# ---------------------------------------------------------------------
# 6. Filtrado de celdas por umbral de defaults
# ---------------------------------------------------------------------
seleccionadas = cross_table[cross_table["defaults"] >= umbral_default_x]
print(f"\n--- Celdas con defaults >= {umbral_default_x} ---")
print(seleccionadas)

# ---------------------------------------------------------------------
# 7. Métricas agregadas del subconjunto seleccionado
# ---------------------------------------------------------------------
poblacion_acumulada = int(seleccionadas["poblacion"].sum())
defaults_acumulados = int(seleccionadas["defaults"].sum())
recall_cobertura = (
    defaults_acumulados / total_defaults_global if total_defaults_global > 0 else np.nan
)

print(f"\nPoblación acumulada (celdas seleccionadas): {poblacion_acumulada}")
print(f"Defaults acumulados (celdas seleccionadas): {defaults_acumulados}")
print(f"Total defaults global del DataFrame: {total_defaults_global}")
print(f"Recall de cobertura: {recall_cobertura:.4f}")