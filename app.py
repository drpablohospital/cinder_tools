import re
import sqlite3
import warnings
from datetime import datetime
import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import unidecode
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, ttest_ind
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud
import plotly.io as pio
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

DB_PATH = "toxicologia.db"


# ------------------ Autenticación ------------------ #
def check_password():
    """Devuelve True si el usuario ha iniciado sesión correctamente."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    # Formulario de login
    st.title("🔐 Acceso restringido")
    st.markdown("Por favor, inicia sesión para continuar.")
    with st.form("login_form"):
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Ingresar")
        if submitted:
            if username == "cinder" and password == "Cinder26?":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos")
    return False


# ------------------ Helper Functions ------------------ #
def normalize_header(name: str) -> str:
    name = str(name).strip()
    name = unidecode.unidecode(name)
    name = name.lower()
    name = re.sub(r"[\s\.\-\/\\]+", "_", name)
    name = re.sub(r"[^\w]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to match database schema and tolerate human-edited CSVs."""
    df = df.copy()
    df.columns = [normalize_header(c) for c in df.columns]

    mapping = {
        "no_entrada": "no_entrada",
        "no_entrada_": "no_entrada",
        "no__entrada": "no_entrada",
        "no_de_entrada": "no_entrada",
        "numero_entrada": "no_entrada",
        "num_entrada": "no_entrada",
        "n_entrada": "no_entrada",
        "noentrada": "no_entrada",
        "no_entrada_paciente": "no_entrada",
        "no_de_entrada_paciente": "no_entrada",

        "identificador": "identificador",
        "id_paciente": "identificador",
        "paciente": "identificador",
        "nombre": "identificador",

        "fecha_atencion": "fecha_atencion",
        "fecha_de_atencion": "fecha_atencion",
        "fecha": "fecha_atencion",

        "hora_atencion": "hora_atencion",
        "hora_de_atencion": "hora_atencion",
        "hora": "hora_atencion",

        "anos_cumplidos": "años_cumplidos",
        "ano_cumplidos": "años_cumplidos",
        "edad": "años_cumplidos",

        "genero": "genero",
        "sexo": "genero",

        "entidad": "entidad",
        "derivacion": "derivacion",
        "derivacion_2": "derivacion2",
        "derivacion2": "derivacion2",

        "especialidad": "especialidad",
        "nivel_atencion": "nivel_atencion",
        "motivo_atencion": "motivo_atencion",
        "motivo_de_atencion": "motivo_atencion",
        "motivo_consulta": "motivo_atencion",

        "impresion_diagnostica": "impresion_diagnostica",
        "diagnostico": "impresion_diagnostica",

        "ecg": "ecg",
        "glasgow": "ecg",

        "tas": "tas",
        "ta_sistolica": "tas",

        "tad": "tad",
        "ta_diastolica": "tad",

        "fc": "fc",
        "fr": "fr",

        "t": "t",
        "temp": "t",
        "temperatura": "t",

        "sao2": "sao2",
        "spo2": "sao2",
        "sat_o2": "sao2",
        "saturacion": "sao2",

        "gluc": "gluc",
        "glucosa": "gluc",

        "news2": "news2_score",
        "news_2": "news2_score",
        "news2_score": "news2_score",

        "atendio": "atendio",

        "tox_benzodiacepina": "tox_benzodiacepina",
        "tox_antidepresivo": "tox_antidepresivo",
        "tox_antipsicotico": "tox_antipsicotico",
        "tox_analgesico": "tox_analgesico",
        "tox_alcohol": "tox_alcohol",
        "tox_droga_ilegal": "tox_droga_ilegal",
        "tox_antiepileptico": "tox_antiepileptico",
        "tox_plaguicida": "tox_plaguicida",
        "tox_animal": "tox_animal",
        "tox_producto_de_limpieza": "tox_producto_de_limpieza",
        "tox_antihipertensivo": "tox_antihipertensivo",
        "tox_hipoglucemiante": "tox_hipoglucemiante",
        "tox_antihistaminico": "tox_antihistaminico",
        "tox_hidrocarburos": "tox_hidrocarburos",
        "tox_natural": "tox_natural",

        "intencional": "intencional",
        "num_farmacos": "num_farmacos",
        "con_alcohol": "con_alcohol",
        "tipo_toxico_principal": "tipo_toxico_principal",
        "sitio_de_procedencia": "sitio_de_procedencia",

        "bh": "bh",
        "qs": "qs",
        "es": "es",
        "gaso": "gaso",
        "pfh": "pfh",
        "tipo_descontaminacion": "tipo_descontaminacion",
        "tiempo_desde_consumo": "tiempo_desde_consumo",
        "tiempo_desde_llegada": "tiempo_desde_llegada",
        "destino": "destino",
        "tiempo_al_alta": "tiempo_al_alta",
        "observaciones": "observaciones",
    }

    df = df.rename(columns={c: mapping.get(c, c) for c in df.columns})
    return df


def normalizar_texto(texto):
    """Normalize text and tolerate accidental Series caused by duplicate columns."""
    if isinstance(texto, pd.Series):
        texto = next((x for x in texto.tolist() if pd.notna(x) and str(x).strip() != ""), "")

    if texto is None:
        return ""

    try:
        if pd.isna(texto):
            return ""
    except Exception:
        pass

    texto = unidecode.unidecode(str(texto)).upper().strip()
    return " ".join(texto.split())


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def get_required_defaults():
    return {
        "no_entrada": None,
        "identificador": "",
        "fecha_atencion": "",
        "hora_atencion": "",
        "años_cumplidos": None,
        "genero": "",
        "entidad": "",
        "derivacion": "",
        "especialidad": "",
        "nivel_atencion": "",
        "motivo_atencion": "",
        "impresion_diagnostica": "",
        "ecg": None,
        "tas": None,
        "tad": None,
        "fc": None,
        "fr": None,
        "t": None,
        "sao2": None,
        "gluc": None,
        "news2_score": None,
        "atendio": "",
        "tox_benzodiacepina": 0,
        "tox_antidepresivo": 0,
        "tox_antipsicotico": 0,
        "tox_analgesico": 0,
        "tox_alcohol": 0,
        "tox_droga_ilegal": 0,
        "tox_antiepileptico": 0,
        "tox_plaguicida": 0,
        "tox_animal": 0,
        "tox_producto_de_limpieza": 0,
        "tox_antihipertensivo": 0,
        "tox_hipoglucemiante": 0,
        "tox_antihistaminico": 0,
        "tox_hidrocarburos": 0,
        "tox_natural": 0,
        "intencional": 0,
        "num_farmacos": 0,
        "con_alcohol": 0,
        "tipo_toxico_principal": "",
        "sitio_de_procedencia": "",
        "derivacion2": "",
        "bh": "",
        "qs": "",
        "es": "",
        "gaso": "",
        "pfh": "",
        "tipo_descontaminacion": "",
        "tiempo_desde_consumo": None,
        "tiempo_desde_llegada": None,
        "destino": "",
        "tiempo_al_alta": None,
        "observaciones": "",
    }


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS pacientes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            no_entrada INTEGER,
            identificador TEXT,
            fecha_atencion TEXT,
            hora_atencion TEXT,
            años_cumplidos REAL,
            genero TEXT,
            entidad TEXT,
            derivacion TEXT,
            especialidad TEXT,
            nivel_atencion TEXT,
            motivo_atencion TEXT,
            impresion_diagnostica TEXT,
            ecg REAL,
            tas REAL,
            tad REAL,
            fc REAL,
            fr REAL,
            t REAL,
            sao2 REAL,
            gluc REAL,
            news2_score REAL,
            atendio TEXT,
            tox_benzodiacepina INTEGER,
            tox_antidepresivo INTEGER,
            tox_antipsicotico INTEGER,
            tox_analgesico INTEGER,
            tox_alcohol INTEGER,
            tox_droga_ilegal INTEGER,
            tox_antiepileptico INTEGER,
            tox_plaguicida INTEGER,
            tox_animal INTEGER,
            tox_producto_de_limpieza INTEGER,
            tox_antihipertensivo INTEGER,
            tox_hipoglucemiante INTEGER,
            tox_antihistaminico INTEGER,
            tox_hidrocarburos INTEGER,
            tox_natural INTEGER,
            intencional INTEGER,
            num_farmacos INTEGER,
            con_alcohol INTEGER,
            tipo_toxico_principal TEXT,
            sitio_de_procedencia TEXT,
            derivacion2 TEXT,
            bh TEXT,
            qs TEXT,
            es TEXT,
            gaso TEXT,
            pfh TEXT,
            tipo_descontaminacion TEXT,
            tiempo_desde_consumo REAL,
            tiempo_desde_llegada REAL,
            destino TEXT,
            tiempo_al_alta REAL,
            observaciones TEXT
        )"""
    )
    conn.commit()
    conn.close()


def insert_paciente(data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    columns = ", ".join(data.keys())
    placeholders = ":" + ", :".join(data.keys())
    sql = f"INSERT INTO pacientes ({columns}) VALUES ({placeholders})"
    cursor.execute(sql, data)
    conn.commit()
    conn.close()


def update_paciente(id_value, data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    set_clause = ", ".join([f"{k}=:{k}" for k in data.keys()])
    sql = f"UPDATE pacientes SET {set_clause} WHERE id=:id"
    data["id"] = id_value
    cursor.execute(sql, data)
    conn.commit()
    conn.close()


def delete_paciente(id_value):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM pacientes WHERE id=?", (id_value,))
    conn.commit()
    conn.close()


def clear_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM pacientes")
    conn.commit()
    conn.close()


def load_all():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM pacientes ORDER BY no_entrada, id", conn)
    conn.close()
    return df


# ------------------ Import from CSV ------------------ #
def import_from_csv(file):
    try:
        df = pd.read_csv(file, encoding="utf-8", sep=None, engine="python")
    except Exception:
        file.seek(0)
        try:
            df = pd.read_csv(file, encoding="latin-1", sep=None, engine="python")
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, encoding="utf-8", sep=";", engine="python")

    original_columns = list(df.columns)
    df = clean_column_names(df)

    duplicated_cols = []
    if df.columns.duplicated().any():
        duplicated_cols = list(df.columns[df.columns.duplicated()])
        df = df.loc[:, ~df.columns.duplicated()]

    st.write("Número de columnas detectadas:", len(df.columns))
    with st.expander("Diagnóstico de importación CSV", expanded=False):
        st.write("Columnas originales:", original_columns)
        st.write("Columnas normalizadas:", list(df.columns))
        if duplicated_cols:
            st.warning(
                f"Se detectaron columnas duplicadas tras normalizar y se conservó la primera aparición: {duplicated_cols}"
            )

    if len(df.columns) <= 3:
        st.error("El CSV parece haberse leído mal. Probablemente el separador no coincide.")
        st.stop()

    object_cols = list(df.select_dtypes(include=["object"]).columns)
    for col in object_cols:
        serie = df[col]
        if isinstance(serie, pd.DataFrame):
            serie = serie.iloc[:, 0]
        df[col] = serie.apply(normalizar_texto)

    required_defaults = get_required_defaults()
    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default

    numeric_cols = [
        "no_entrada", "años_cumplidos", "ecg", "tas", "tad", "fc", "fr", "t",
        "sao2", "gluc", "news2_score", "num_farmacos",
        "tiempo_desde_consumo", "tiempo_desde_llegada", "tiempo_al_alta"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    binary_cols = [
        "tox_benzodiacepina", "tox_antidepresivo", "tox_antipsicotico",
        "tox_analgesico", "tox_alcohol", "tox_droga_ilegal",
        "tox_antiepileptico", "tox_plaguicida", "tox_animal",
        "tox_producto_de_limpieza", "tox_antihipertensivo",
        "tox_hipoglucemiante", "tox_antihistaminico",
        "tox_hidrocarburos", "tox_natural", "intencional", "con_alcohol"
    ]
    for col in binary_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["num_farmacos"] = pd.to_numeric(df["num_farmacos"], errors="coerce").fillna(0).astype(int)

    # remove impossible ages
    df.loc[(df["años_cumplidos"] < 0) | (df["años_cumplidos"] > 120), "años_cumplidos"] = np.nan

    with st.expander("Vista previa del CSV listo para insertar", expanded=False):
        st.dataframe(df.head(), use_container_width=True)
        st.write(f"Filas totales: {len(df)}")
        st.write(f"Edad: valores no nulos = {df['años_cumplidos'].notna().sum()}")
        st.write(f"NEWS-2: valores no nulos = {df['news2_score'].notna().sum()}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    sql = """
        INSERT INTO pacientes (
            no_entrada, identificador, fecha_atencion, hora_atencion, años_cumplidos,
            genero, entidad, derivacion, especialidad, nivel_atencion, motivo_atencion,
            impresion_diagnostica, ecg, tas, tad, fc, fr, t, sao2, gluc, news2_score,
            atendio, tox_benzodiacepina, tox_antidepresivo, tox_antipsicotico, tox_analgesico,
            tox_alcohol, tox_droga_ilegal, tox_antiepileptico, tox_plaguicida, tox_animal,
            tox_producto_de_limpieza, tox_antihipertensivo, tox_hipoglucemiante,
            tox_antihistaminico, tox_hidrocarburos, tox_natural, intencional, num_farmacos,
            con_alcohol, tipo_toxico_principal, sitio_de_procedencia, derivacion2,
            bh, qs, es, gaso, pfh, tipo_descontaminacion, tiempo_desde_consumo,
            tiempo_desde_llegada, destino, tiempo_al_alta, observaciones
        ) VALUES (
            :no_entrada, :identificador, :fecha_atencion, :hora_atencion, :años_cumplidos,
            :genero, :entidad, :derivacion, :especialidad, :nivel_atencion, :motivo_atencion,
            :impresion_diagnostica, :ecg, :tas, :tad, :fc, :fr, :t, :sao2, :gluc, :news2_score,
            :atendio, :tox_benzodiacepina, :tox_antidepresivo, :tox_antipsicotico, :tox_analgesico,
            :tox_alcohol, :tox_droga_ilegal, :tox_antiepileptico, :tox_plaguicida, :tox_animal,
            :tox_producto_de_limpieza, :tox_antihipertensivo, :tox_hipoglucemiante,
            :tox_antihistaminico, :tox_hidrocarburos, :tox_natural, :intencional, :num_farmacos,
            :con_alcohol, :tipo_toxico_principal, :sitio_de_procedencia, :derivacion2,
            :bh, :qs, :es, :gaso, :pfh, :tipo_descontaminacion, :tiempo_desde_consumo,
            :tiempo_desde_llegada, :destino, :tiempo_al_alta, :observaciones
        )
    """

    inserted = 0
    errors = []

    for idx, row in df.iterrows():
        data = row.to_dict()

        for k, v in list(data.items()):
            if pd.isna(v):
                data[k] = None

        if data.get("no_entrada") is not None:
            try:
                data["no_entrada"] = int(float(data["no_entrada"]))
            except Exception:
                data["no_entrada"] = None

        try:
            cursor.execute(sql, data)
            inserted += 1
        except Exception as e:
            errors.append(f"Fila {idx}: {e}")

    conn.commit()
    conn.close()

    st.success(f"Importación terminada. Filas insertadas: {inserted}")
    if errors:
        st.warning(f"Filas con error: {len(errors)}")
        with st.expander("Ver errores de importación", expanded=False):
            for err in errors[:100]:
                st.write(err)


# ------------------ Analysis ------------------ #
def build_analysis_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_candidates = ["años_cumplidos", "news2_score", "fc", "fr", "tas", "tad", "sao2", "num_farmacos", "t", "gluc"]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "años_cumplidos" in df.columns:
        df.loc[(df["años_cumplidos"] < 0) | (df["años_cumplidos"] > 120), "años_cumplidos"] = np.nan
        edad_clean = df["años_cumplidos"].dropna()
        if not edad_clean.empty:
            df["grupo_edad"] = pd.cut(
                df["años_cumplidos"],
                bins=[0, 18, 30, 45, 60, 120],
                labels=["0-18", "19-30", "31-45", "46-60", "60+"],
                include_lowest=True,
            )
        else:
            df["grupo_edad"] = np.nan

    if "news2_score" in df.columns:
        df["news2_alto"] = np.where(
            df["news2_score"].notna(),
            (df["news2_score"] >= 5).astype(int),
            np.nan,
        )

    cluster_vars = [v for v in ["años_cumplidos", "news2_score", "fc", "sao2", "num_farmacos"] if v in df.columns]
    df["cluster"] = np.nan

    if len(cluster_vars) >= 3:
        df_cluster = df[cluster_vars].dropna().copy()
        if len(df_cluster) >= 10:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_cluster)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            df_cluster["cluster_temp"] = clusters
            perfiles = df_cluster.groupby("cluster_temp")[cluster_vars].mean()

            idx_grave = perfiles["news2_score"].idxmax() if "news2_score" in perfiles.columns else perfiles.mean(axis=1).idxmax()
            idx_poli = perfiles["num_farmacos"].idxmax() if "num_farmacos" in perfiles.columns else list(perfiles.index)[0]
            remaining = [i for i in perfiles.index if i not in [idx_grave, idx_poli]]
            idx_leve = remaining[0] if remaining else idx_poli

            cluster_map = {
                idx_poli: "Cluster 0: Joven · Polifarmacia · Riesgo moderado",
                idx_leve: "Cluster 1: Menor carga fisiológica",
                idx_grave: "Cluster 2: Mayor gravedad fisiológica",
            }
            df.loc[df_cluster.index, "cluster"] = [cluster_map.get(c, f"Cluster {c}") for c in clusters]

    return df


def render_interpretation(df: pd.DataFrame):
    st.subheader("📊 Interpretación general de los datos")

    total = len(df)
    edad_media = df["años_cumplidos"].mean() if "años_cumplidos" in df and df["años_cumplidos"].notna().any() else np.nan
    news_media = df["news2_score"].mean() if "news2_score" in df and df["news2_score"].notna().any() else np.nan
    poli_media = df["num_farmacos"].mean() if "num_farmacos" in df and df["num_farmacos"].notna().any() else np.nan

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de casos", total)
    col2.metric("Edad media", f"{edad_media:.1f}" if not np.isnan(edad_media) else "NA")
    col3.metric("NEWS-2 medio", f"{news_media:.1f}" if not np.isnan(news_media) else "NA")
    col4.metric("Nº fármacos medio", f"{poli_media:.1f}" if not np.isnan(poli_media) else "NA")


def find_gravity_rule(df: pd.DataFrame, target='news2_alto'):
    """Busca combinaciones de variables que predicen mejor target usando árbol de decisión."""
    if target not in df.columns or df[target].isna().all():
        st.warning("No hay suficientes datos con NEWS-2 para entrenar regla.")
        return

    # Seleccionar variables relevantes
    numeric_vars = ['años_cumplidos', 'fc', 'fr', 'tas', 'tad', 'sao2', 'num_farmacos']
    numeric_vars = [v for v in numeric_vars if v in df.columns]
    cat_vars = ['genero'] + [c for c in df.columns if c.startswith('tox_') and c != 'tox_principal']
    cat_vars = [v for v in cat_vars if v in df.columns]

    data = df[numeric_vars + cat_vars + [target]].dropna().copy()
    if len(data) < 20:
        st.warning("Datos insuficientes para entrenar regla (mínimo 20 casos).")
        return

    # Codificar categóricas
    for var in cat_vars:
        if data[var].dtype == 'object':
            data[var] = data[var].astype('category').cat.codes
        else:
            data[var] = data[var].astype(int)

    X = data[numeric_vars + cat_vars]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    clf = DecisionTreeClassifier(max_depth=3, random_state=42, min_samples_leaf=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Importancia de características
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.markdown("### 🌳 Variables más importantes para predecir NEWS ≥ 5")
    st.dataframe(importances.to_frame(name="Importancia").round(3), use_container_width=True)

    # Extraer reglas del árbol
    from sklearn.tree import _tree
    tree_ = clf.tree_
    feature_names = X.columns.tolist()

    def recurse(node, depth, rule_text):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            left_child = tree_.children_left[node]
            right_child = tree_.children_right[node]
            recurse(left_child, depth+1, rule_text + f"({name} ≤ {threshold:.2f}) and ")
            recurse(right_child, depth+1, rule_text + f"({name} > {threshold:.2f}) and ")
        else:
            value = tree_.value[node][0]
            total = value.sum()
            pos = value[1]
            prop = pos / total if total > 0 else 0
            if total > 0 and prop > 0.5:
                st.write(f"Regla: {rule_text[:-5]} → NEWS≥5 en {prop:.1%} ({int(pos)}/{int(total)} casos)")

    st.markdown("**Reglas del árbol de decisión (hojas con mayoría de NEWS≥5)**")
    recurse(0, 0, "")
    st.write(f"Precisión en test: {acc:.2%}")


def render_toxic_combinations(df: pd.DataFrame):
    """Explora combinaciones de tóxicos y su relación con NEWS."""
    st.subheader("🔬 Asociaciones de combinaciones de tóxicos")

    tox_cols = [c for c in df.columns if c.startswith('tox_') and c != 'tox_principal']
    if not tox_cols:
        st.info("No hay columnas de tóxicos binarios.")
        return

    tox_data = df[tox_cols].fillna(0).astype(int)

    # Pares frecuentes
    from itertools import combinations
    pair_counts = {}
    for tox1, tox2 in combinations(tox_cols, 2):
        count = ((tox_data[tox1] == 1) & (tox_data[tox2] == 1)).sum()
        if count >= 5:
            pair_counts[f"{tox1.replace('tox_','')} + {tox2.replace('tox_','')}"] = count

    if pair_counts:
        st.markdown("**Pares de tóxicos más frecuentes (>5 casos)**")
        df_pairs = pd.DataFrame(list(pair_counts.items()), columns=['Combinación', 'Frecuencia']).sort_values('Frecuencia', ascending=False)
        st.dataframe(df_pairs, use_container_width=True)

    # NEWS promedio por par
    if 'news2_score' in df.columns and len(pair_counts) > 0:
        st.markdown("**NEWS-2 promedio por pares de tóxicos**")
        pair_counts_orig = {}
        for tox1, tox2 in combinations(tox_cols, 2):
            count = ((tox_data[tox1] == 1) & (tox_data[tox2] == 1)).sum()
            if count >= 5:
                pair_counts_orig[f"{tox1} + {tox2}"] = count
        pair_news = {}
        for pair, count in pair_counts_orig.items():
            tox1, tox2 = pair.split(' + ')
            mask = (tox_data[tox1] == 1) & (tox_data[tox2] == 1)
            news_mean = df.loc[mask, 'news2_score'].mean()
            if not pd.isna(news_mean):
                pair_news[pair] = news_mean
        if pair_news:
            df_news = pd.DataFrame(list(pair_news.items()), columns=['Combinación', 'NEWS-2 medio']).sort_values('NEWS-2 medio', ascending=False)
            st.dataframe(df_news, use_container_width=True)

    # Asociación individual
    st.markdown("**Tóxicos individuales y su asociación con NEWS-2 (prueba Mann-Whitney)**")
    tox_news = {}
    for tox in tox_cols:
        grupo1 = df[df[tox]==1]['news2_score'].dropna()
        grupo0 = df[df[tox]==0]['news2_score'].dropna()
        if len(grupo1)>=5 and len(grupo0)>=5:
            try:
                stat, p = mannwhitneyu(grupo1, grupo0, alternative='two-sided')
                tox_news[tox] = p
            except:
                pass
    if tox_news:
        df_tox_p = pd.DataFrame(tox_news.items(), columns=['Tóxico', 'p-value']).sort_values('p-value')
        st.dataframe(df_tox_p)
    else:
        st.write("No hay suficientes datos para comparación.")

    # Boxplot para un tóxico seleccionado
    st.markdown("**Selecciona un tóxico para ver distribución de NEWS-2**")
    selected_tox = st.selectbox("Tóxico", tox_cols, key="toxic_comb")
    if selected_tox in df.columns:
        fig = px.box(df, x=selected_tox, y='news2_score', title=f"NEWS-2 según presencia de {selected_tox}", points='all')
        st.plotly_chart(fig, use_container_width=True)


def render_basic_stats(df: pd.DataFrame):
    st.subheader("Resumen descriptivo")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        desc = df[numeric_cols].describe().T.round(2)
        st.dataframe(desc, use_container_width=True)

    cat_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    if cat_cols:
        selected_cat = st.selectbox("Variable categórica para frecuencias", cat_cols)
        freq = df[selected_cat].fillna("VACÍO").value_counts(dropna=False).reset_index()
        freq.columns = [selected_cat, "n"]
        st.dataframe(freq, use_container_width=True)


def render_overview_figures(df: pd.DataFrame):
    st.subheader("Vista general")

    c1, c2 = st.columns(2)

    with c1:
        if "news2_score" in df.columns and df["news2_score"].notna().any():
            fig = px.histogram(df, x="news2_score", nbins=20, title="Distribución de NEWS-2")
            st.plotly_chart(fig, use_container_width=True)
            # Botón descarga PNG
            with st.container():
                if st.button("📸 Descargar PNG", key="png_news"):
                    img_bytes = fig.to_image(format="png")
                    b64 = base64.b64encode(img_bytes).decode()
                    href = f'<a href="data:image/png;base64,{b64}" download="news_dist.png">Descargar</a>'
                    st.markdown(href, unsafe_allow_html=True)

    with c2:
        if "tipo_toxico_principal" in df.columns:
            top = (
                df["tipo_toxico_principal"]
                .fillna("VACÍO")
                .replace("", "VACÍO")
                .value_counts()
                .head(15)
                .reset_index()
            )
            if not top.empty:
                top.columns = ["tipo", "n"]
                fig = px.bar(top, x="tipo", y="n", title="Tóxicos más frecuentes")
                st.plotly_chart(fig, use_container_width=True)
                with st.container():
                    if st.button("📸 Descargar PNG", key="png_tox"):
                        img_bytes = fig.to_image(format="png")
                        b64 = base64.b64encode(img_bytes).decode()
                        href = f'<a href="data:image/png;base64,{b64}" download="tox_freq.png">Descargar</a>'
                        st.markdown(href, unsafe_allow_html=True)

    if "news2_score" in df.columns and "tipo_toxico_principal" in df.columns:
        temp = df[["tipo_toxico_principal", "news2_score"]].dropna()
        if not temp.empty:
            resumen = temp.groupby("tipo_toxico_principal")["news2_score"].mean().sort_values(ascending=False).head(15).reset_index()
            fig = px.bar(resumen, x="tipo_toxico_principal", y="news2_score", title="NEWS-2 promedio por tóxico")
            st.plotly_chart(fig, use_container_width=True)
            with st.container():
                if st.button("📸 Descargar PNG", key="png_news_tox"):
                    img_bytes = fig.to_image(format="png")
                    b64 = base64.b64encode(img_bytes).decode()
                    href = f'<a href="data:image/png;base64,{b64}" download="news_by_tox.png">Descargar</a>'
                    st.markdown(href, unsafe_allow_html=True)


def render_relationships(df: pd.DataFrame):
    """Versión enfocada en variables clínicamente relevantes."""
    st.subheader("Relaciones clínicas")

    # Definir variables clínicamente relevantes
    clinical_numeric = ['años_cumplidos', 'news2_score', 'fc', 'fr', 'tas', 'tad', 'sao2', 'num_farmacos']
    clinical_numeric = [c for c in clinical_numeric if c in df.columns]
    clinical_cat = ['genero', 'tipo_toxico_principal']
    clinical_cat = [c for c in clinical_cat if c in df.columns]
    # También incluir tóxicos binarios como categóricas
    tox_cols = [c for c in df.columns if c.startswith('tox_') and c != 'tox_principal']
    clinical_cat = clinical_cat + tox_cols
    clinical_cat = list(set(clinical_cat))

    # Mostrar correlación entre variables numéricas
    if len(clinical_numeric) >= 2:
        st.markdown("**Correlación de Spearman entre variables clínicas**")
        corr = df[clinical_numeric].corr(method='spearman').round(2)
        fig = px.imshow(corr, text_auto=True, aspect='auto', title='Matriz de correlación')
        st.plotly_chart(fig, use_container_width=True)
        with st.container():
            if st.button("📸 Descargar PNG", key="png_corr_clin"):
                img_bytes = fig.to_image(format="png")
                b64 = base64.b64encode(img_bytes).decode()
                href = f'<a href="data:image/png;base64,{b64}" download="corr_clinical.png">Descargar</a>'
                st.markdown(href, unsafe_allow_html=True)

    # Relación numérica vs categórica
    if clinical_numeric and clinical_cat:
        col1, col2 = st.columns(2)
        with col1:
            num_var = st.selectbox("Variable numérica", clinical_numeric, key="rel_num_clin")
        with col2:
            cat_var = st.selectbox("Variable categórica", clinical_cat, key="rel_cat_clin")

        temp = df[[num_var, cat_var]].dropna()
        if not temp.empty and temp[cat_var].nunique() >= 2:
            fig = px.box(temp, x=cat_var, y=num_var, points='all', title=f"{num_var} por {cat_var}")
            st.plotly_chart(fig, use_container_width=True)

            groups = [g[num_var].dropna().values for _, g in temp.groupby(cat_var)]
            if len(groups) == 2:
                try:
                    _, p = ttest_ind(groups[0], groups[1], equal_var=False, nan_policy='omit')
                    st.write(f"T-test Welch p = {p:.4g}")
                except:
                    pass
                try:
                    _, p = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                    st.write(f"Mann-Whitney p = {p:.4g}")
                except:
                    pass

    # Tabla de contingencia para dos categóricas
    if len(clinical_cat) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            cat_a = st.selectbox("Categoría A", clinical_cat, key="cat_a_clin")
        with col2:
            cat_b = st.selectbox("Categoría B", clinical_cat, key="cat_b_clin")

        temp = df[[cat_a, cat_b]].dropna()
        if not temp.empty:
            try:
                temp_a = temp[cat_a].astype(str)
                temp_b = temp[cat_b].astype(str)
                tab = pd.crosstab(temp_a, temp_b)
                st.dataframe(tab, use_container_width=True)
                if tab.shape[0] >= 2 and tab.shape[1] >= 2:
                    try:
                        _, p, _, _ = chi2_contingency(tab)
                        st.write(f"Chi² p = {p:.4g}")
                    except:
                        pass
                    if tab.shape == (2,2):
                        try:
                            _, p = fisher_exact(tab.values)
                            st.write(f"Fisher exacta p = {p:.4g}")
                        except:
                            pass
            except Exception as e:
                st.error(f"Error en tabla de contingencia: {e}")


def render_clusters(df: pd.DataFrame):
    st.subheader("Fenotipos / clusters")
    if "cluster" not in df.columns or df["cluster"].dropna().empty:
        st.info("No hay suficientes datos completos para clustering.")
        return

    counts = df["cluster"].value_counts(dropna=False).rename_axis("cluster").reset_index(name="n")
    st.dataframe(counts, use_container_width=True)

    cluster_vars = [v for v in ["años_cumplidos", "news2_score", "fc", "sao2", "num_farmacos"] if v in df.columns]
    dfx = df[cluster_vars + ["cluster"]].dropna()
    if len(dfx) >= 5 and len(cluster_vars) >= 2:
        # PCA para visualización
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(dfx[cluster_vars])
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_scaled)
        plot_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "cluster": dfx["cluster"].values})
        fig = px.scatter(plot_df, x="PC1", y="PC2", color="cluster", title="Proyección PCA de clusters")
        st.plotly_chart(fig, use_container_width=True)
        with st.container():
            if st.button("📸 Descargar PNG", key="png_cluster"):
                img_bytes = fig.to_image(format="png")
                b64 = base64.b64encode(img_bytes).decode()
                href = f'<a href="data:image/png;base64,{b64}" download="cluster_pca.png">Descargar</a>'
                st.markdown(href, unsafe_allow_html=True)

        # Perfiles medios
        profile = dfx.groupby("cluster")[cluster_vars].mean().round(2)
        st.dataframe(profile, use_container_width=True)

        # Gráfico de radar (spider)
        st.markdown("**Perfil de los clusters (variables normalizadas)**")
        # Normalizar las variables para el radar
        scaler_radar = StandardScaler()
        profile_scaled = pd.DataFrame(scaler_radar.fit_transform(profile), index=profile.index, columns=profile.columns)
        categories = profile_scaled.columns.tolist()
        fig_radar = go.Figure()
        for cluster in profile_scaled.index:
            values = profile_scaled.loc[cluster].values.tolist()
            # Cerrar el polígono
            values += values[:1]
            categories_radar = categories + [categories[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories_radar,
                fill='toself',
                name=cluster
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-2, 2])),
            showlegend=True,
            title="Perfil de clusters (valores Z)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Listar pacientes por cluster
        with st.expander("Ver pacientes por cluster"):
            for cluster_name in df["cluster"].dropna().unique():
                pacientes = df[df["cluster"] == cluster_name][["id", "identificador", "no_entrada"]].dropna()
                st.write(f"**{cluster_name}** ({len(pacientes)} pacientes)")
                st.dataframe(pacientes, use_container_width=True)


def render_wordcloud(df: pd.DataFrame):
    st.subheader("Exploración semántica")
    text_cols = [c for c in ["motivo_atencion", "impresion_diagnostica", "tipo_toxico_principal", "observaciones"] if c in df.columns]
    if not text_cols:
        st.info("No hay columnas de texto para explorar.")
        return

    corpus = " ".join(
        df[text_cols].fillna("").astype(str).apply(lambda row: " ".join(row), axis=1).tolist()
    ).strip()

    if not corpus:
        st.info("No hay texto suficiente para nube de palabras.")
        return

    wc = WordCloud(width=1200, height=500, background_color="white").generate(corpus)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


# ------------------ Reporte HTML completo ------------------ #
def export_full_report(df):
    """Genera un reporte HTML con todas las tablas y gráficos clave."""
    # Usamos una copia para no alterar el DataFrame original
    df_report = df.copy()
    # Asegurar variables necesarias
    if "news2_alto" not in df_report.columns and "news2_score" in df_report.columns:
        df_report["news2_alto"] = (df_report["news2_score"] >= 5).astype(int)

    # Iniciar HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Reporte Toxicología</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .figure {{ margin: 20px 0; text-align: center; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
    <h1>Reporte de análisis de toxicología</h1>
    <p>Generado el {fecha}</p>
    """.format(fecha=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Estadísticas descriptivas
    html += "<h2>Resumen descriptivo</h2>"
    numeric_cols = df_report.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        desc = df_report[numeric_cols].describe().round(2).T
        html += desc.to_html()

    # Frecuencias de tóxicos principales
    if "tipo_toxico_principal" in df_report.columns:
        html += "<h2>Tóxicos principales más frecuentes</h2>"
        freq_tox = df_report["tipo_toxico_principal"].fillna("VACÍO").value_counts().reset_index()
        freq_tox.columns = ["Tóxico", "Frecuencia"]
        html += freq_tox.head(15).to_html(index=False)

    # Correlación (gráfico)
    clinical_numeric = ['años_cumplidos', 'news2_score', 'fc', 'fr', 'tas', 'tad', 'sao2', 'num_farmacos']
    clinical_numeric = [c for c in clinical_numeric if c in df_report.columns]
    if len(clinical_numeric) >= 2:
        corr = df_report[clinical_numeric].corr(method='spearman').round(2)
        fig_corr = px.imshow(corr, text_auto=True, aspect='auto', title='Correlación de Spearman')
        img_corr = fig_corr.to_image(format="png")
        b64_corr = base64.b64encode(img_corr).decode()
        html += f'<div class="figure"><h3>Matriz de correlación</h3><img src="data:image/png;base64,{b64_corr}"></div>'

    # Distribución NEWS-2
    if "news2_score" in df_report.columns:
        fig_hist = px.histogram(df_report, x="news2_score", nbins=20, title="Distribución de NEWS-2")
        img_hist = fig_hist.to_image(format="png")
        b64_hist = base64.b64encode(img_hist).decode()
        html += f'<div class="figure"><h3>Distribución de NEWS-2</h3><img src="data:image/png;base64,{b64_hist}"></div>'

    # Tóxicos más frecuentes (gráfico de barras)
    if "tipo_toxico_principal" in df_report.columns:
        top = df_report["tipo_toxico_principal"].fillna("VACÍO").value_counts().head(15).reset_index()
        top.columns = ["tipo", "n"]
        fig_bar = px.bar(top, x="tipo", y="n", title="Tóxicos más frecuentes")
        img_bar = fig_bar.to_image(format="png")
        b64_bar = base64.b64encode(img_bar).decode()
        html += f'<div class="figure"><h3>Tóxicos más frecuentes</h3><img src="data:image/png;base64,{b64_bar}"></div>'

    # NEWS promedio por tóxico
    if "news2_score" in df_report.columns and "tipo_toxico_principal" in df_report.columns:
        news_by_tox = df_report.groupby("tipo_toxico_principal")["news2_score"].mean().sort_values(ascending=False).head(15).reset_index()
        news_by_tox.columns = ["Tóxico", "NEWS-2 medio"]
        fig_news_tox = px.bar(news_by_tox, x="Tóxico", y="NEWS-2 medio", title="NEWS-2 promedio por tóxico")
        img_news_tox = fig_news_tox.to_image(format="png")
        b64_news_tox = base64.b64encode(img_news_tox).decode()
        html += f'<div class="figure"><h3>NEWS-2 promedio por tóxico</h3><img src="data:image/png;base64,{b64_news_tox}"></div>'

    # Árbol de decisión (mejores predictores)
    html += "<h2>Predictores de NEWS ≥ 5</h2>"
    try:
        numeric_vars = ['años_cumplidos', 'fc', 'fr', 'tas', 'tad', 'sao2', 'num_farmacos']
        numeric_vars = [v for v in numeric_vars if v in df_report.columns]
        cat_vars = ['genero'] + [c for c in df_report.columns if c.startswith('tox_') and c != 'tox_principal']
        cat_vars = [v for v in cat_vars if v in df_report.columns]
        data = df_report[numeric_vars + cat_vars + ['news2_alto']].dropna().copy()
        if len(data) >= 20:
            for var in cat_vars:
                if data[var].dtype == 'object':
                    data[var] = data[var].astype('category').cat.codes
                else:
                    data[var] = data[var].astype(int)
            X = data[numeric_vars + cat_vars]
            y = data['news2_alto']
            clf = DecisionTreeClassifier(max_depth=3, random_state=42, min_samples_leaf=10)
            clf.fit(X, y)
            importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
            html += "<h3>Importancia de las características</h3>"
            html += importances.to_frame(name="Importancia").round(3).to_html()
            # Reglas (solo las hojas con >50% de NEWS≥5)
            from sklearn.tree import _tree
            tree_ = clf.tree_
            feature_names = X.columns.tolist()
            rules_html = "<ul>"
            def recurse(node, rule_text):
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    name = feature_names[tree_.feature[node]]
                    threshold = tree_.threshold[node]
                    left_child = tree_.children_left[node]
                    right_child = tree_.children_right[node]
                    recurse(left_child, rule_text + f"({name} ≤ {threshold:.2f}) and ")
                    recurse(right_child, rule_text + f"({name} > {threshold:.2f}) and ")
                else:
                    value = tree_.value[node][0]
                    total = value.sum()
                    pos = value[1]
                    prop = pos / total if total > 0 else 0
                    if total > 0 and prop > 0.5:
                        rules_html += f"<li>{rule_text[:-5]} → NEWS≥5 en {prop:.1%} ({int(pos)}/{int(total)} casos)</li>"
            recurse(0, "")
            rules_html += "</ul>"
            html += "<h3>Reglas de clasificación</h3>"
            html += rules_html
            html += f"<p>Precisión en validación: {accuracy_score(y, clf.predict(X)):.2%}</p>"
        else:
            html += "<p>Datos insuficientes para entrenar modelo.</p>"
    except Exception as e:
        html += f"<p>Error al generar el árbol: {e}</p>"

    # Combinaciones de tóxicos
    tox_cols = [c for c in df_report.columns if c.startswith('tox_') and c != 'tox_principal']
    if len(tox_cols) > 0 and 'news2_score' in df_report.columns:
        html += "<h2>Combinaciones de tóxicos</h2>"
        tox_data = df_report[tox_cols].fillna(0).astype(int)
        from itertools import combinations
        pair_counts = {}
        for tox1, tox2 in combinations(tox_cols, 2):
            count = ((tox_data[tox1] == 1) & (tox_data[tox2] == 1)).sum()
            if count >= 5:
                pair_counts[f"{tox1} + {tox2}"] = count
        if pair_counts:
            df_pairs = pd.DataFrame(list(pair_counts.items()), columns=['Combinación', 'Frecuencia']).sort_values('Frecuencia', ascending=False)
            html += "<h3>Pares frecuentes</h3>"
            html += df_pairs.to_html(index=False)
            pair_news = {}
            for pair, count in pair_counts.items():
                tox1, tox2 = pair.split(' + ')
                mask = (tox_data[tox1] == 1) & (tox_data[tox2] == 1)
                news_mean = df_report.loc[mask, 'news2_score'].mean()
                if not pd.isna(news_mean):
                    pair_news[pair] = news_mean
            if pair_news:
                df_news = pd.DataFrame(list(pair_news.items()), columns=['Combinación', 'NEWS-2 medio']).sort_values('NEWS-2 medio', ascending=False)
                html += "<h3>NEWS-2 medio por par</h3>"
                html += df_news.to_html(index=False)

    # Clusters
    if "cluster" in df_report.columns and df_report["cluster"].notna().any():
        html += "<h2>Clusters</h2>"
        counts = df_report["cluster"].value_counts(dropna=False).reset_index()
        counts.columns = ["Cluster", "N"]
        html += counts.to_html(index=False)
        cluster_vars = [v for v in ["años_cumplidos", "news2_score", "fc", "sao2", "num_farmacos"] if v in df_report.columns]
        if len(cluster_vars) >= 2:
            profile = df_report.groupby("cluster")[cluster_vars].mean().round(2)
            html += "<h3>Perfil medio de los clusters</h3>"
            html += profile.to_html()
            # Gráfico de radar
            scaler_radar = StandardScaler()
            profile_scaled = pd.DataFrame(scaler_radar.fit_transform(profile), index=profile.index, columns=profile.columns)
            categories = profile_scaled.columns.tolist()
            fig_radar = go.Figure()
            for cluster in profile_scaled.index:
                values = profile_scaled.loc[cluster].values.tolist()
                values += values[:1]
                categories_radar = categories + [categories[0]]
                fig_radar.add_trace(go.Scatterpolar(r=values, theta=categories_radar, fill='toself', name=cluster))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-2, 2])), showlegend=True)
            img_radar = fig_radar.to_image(format="png")
            b64_radar = base64.b64encode(img_radar).decode()
            html += f'<div class="figure"><h3>Perfil de clusters (valores Z)</h3><img src="data:image/png;base64,{b64_radar}"></div>'

    html += "</body></html>"
    return html


# ------------------ UI ------------------ #
st.set_page_config(page_title="Toxicology Database Manager", layout="wide")

# Verificar autenticación
if not check_password():
    st.stop()

st.title("🧪 Toxicology Database Manager")
st.caption("Importación robusta de CSV + exploración clínica y estadística")

# Inicializar DB y sesión
init_db()
if "form_mode" not in st.session_state:
    st.session_state["form_mode"] = None

with st.sidebar:
    st.header("👤 Usuario")
    st.write("cinder")
    if st.button("Cerrar sesión"):
        st.session_state.authenticated = False
        st.rerun()
    st.markdown("---")
    st.header("📂 Importar datos")
    uploaded_file = st.file_uploader("Sube CSV (original o enriquecido)", type=["csv"])
    if uploaded_file is not None and st.button("Importar a la base de datos"):
        with st.spinner("Importando..."):
            import_from_csv(uploaded_file)
            st.rerun()

    st.markdown("---")
    st.header("🗑️ Base de datos")
    if st.button("Vaciar base de datos"):
        clear_db()
        st.success("Base de datos vaciada.")
        st.rerun()

    st.markdown("---")
    st.header("➕ Nuevo caso")
    if st.button("Agregar registro"):
        st.session_state["form_mode"] = "add"
        st.rerun()

tab1, tab2, tab3 = st.tabs(["📋 Lista de pacientes", "✏️ Editar / Eliminar", "📊 Análisis exploratorio"])

with tab1:
    st.header("Lista de pacientes")
    df = load_all()

    if df.empty:
        st.info("No hay datos. Importa un archivo CSV o agrega un nuevo caso.")
    else:
        cols_show = [c for c in ["id", "no_entrada", "identificador", "fecha_atencion", "genero", "años_cumplidos", "tipo_toxico_principal", "news2_score"] if c in df.columns]
        st.dataframe(df[cols_show], use_container_width=True)

    if st.session_state.get("form_mode") == "add":
        st.subheader("Nuevo caso")
        with st.form("add_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                next_no = 1 if df.empty or df["no_entrada"].dropna().empty else int(df["no_entrada"].dropna().max()) + 1
                no_entrada = st.number_input("No. entrada", min_value=1, step=1, value=next_no)
                identificador = st.text_input("Identificador")
                fecha_atencion = st.date_input("Fecha de atención", value=datetime.today())
                hora_atencion = st.text_input("Hora (HH:MM)")
                años_cumplidos = st.number_input("Edad (años)", min_value=0.0, max_value=120.0, value=0.0)
                genero = st.selectbox("Género", ["", "HOMBRE", "MUJER"])
                entidad = st.text_input("Entidad")

            with col2:
                derivacion = st.text_input("Derivación")
                especialidad = st.text_input("Especialidad")
                nivel_atencion = st.selectbox("Nivel de atención", ["", "I", "II", "III"])
                motivo_atencion = st.text_area("Motivo de atención")
                impresion_diagnostica = st.text_area("Impresión diagnóstica")

            with col3:
                ecg = st.number_input("Glasgow", min_value=3.0, max_value=15.0, value=15.0)
                tas = st.number_input("TAS (mmHg)", value=0.0)
                tad = st.number_input("TAD (mmHg)", value=0.0)
                fc = st.number_input("FC (lpm)", value=0.0)
                fr = st.number_input("FR (rpm)", value=0.0)
                t = st.number_input("Temperatura (°C)", value=0.0, format="%.1f")
                sao2 = st.number_input("SpO₂ (%)", value=0.0)
                gluc = st.number_input("Glucemia (mg/dL)", value=0.0)
                news2_score = st.number_input("NEWS-2", min_value=0.0, max_value=20.0, value=0.0)
                atendio = st.text_input("Atendió")

            st.subheader("Xenobióticos")
            tox_cols = [
                "tox_benzodiacepina", "tox_antidepresivo", "tox_antipsicotico", "tox_analgesico",
                "tox_alcohol", "tox_droga_ilegal", "tox_antiepileptico", "tox_plaguicida",
                "tox_animal", "tox_producto_de_limpieza", "tox_antihipertensivo",
                "tox_hipoglucemiante", "tox_antihistaminico", "tox_hidrocarburos", "tox_natural"
            ]
            tox_vals = {}
            cols = st.columns(4)
            for i, col in enumerate(tox_cols):
                with cols[i % 4]:
                    tox_vals[col] = st.checkbox(col.replace("tox_", "").replace("_", " ").title(), key=f"add_{col}")

            st.subheader("Variables semánticas")
            c1, c2, c3 = st.columns(3)
            with c1:
                intencional = st.checkbox("Intencional")
                con_alcohol = st.checkbox("Con alcohol")
            with c2:
                num_farmacos = st.number_input("Número de fármacos", min_value=0, step=1)
            with c3:
                tipo_toxico_principal = st.text_input("Tipo de tóxico principal")
                sitio_de_procedencia = st.text_input("Sitio de procedencia")
                derivacion2 = st.text_input("Derivación 2")

            st.subheader("Laboratorios y descontaminación")
            c1, c2 = st.columns(2)
            with c1:
                bh = st.text_area("BH")
                qs = st.text_area("QS")
                es = st.text_area("ES")
            with c2:
                gaso = st.text_area("GASO")
                pfh = st.text_area("PFH")

            tipo_descontaminacion = st.text_input("Tipo de descontaminación")
            tiempo_desde_consumo = st.number_input("Tiempo desde consumo (min)", min_value=0.0, value=0.0)
            tiempo_desde_llegada = st.number_input("Tiempo desde llegada (min)", min_value=0.0, value=0.0)
            destino = st.selectbox("Destino", ["", "HOSPITALIZACION", "CHOQUE", "ALTA", "REFERENCIA", "CONTRAREFERENCIA"])
            tiempo_al_alta = st.number_input("Tiempo al alta (días)", min_value=0.0, value=0.0)
            observaciones = st.text_area("Observaciones")

            if st.form_submit_button("Guardar"):
                data = {
                    "no_entrada": int(no_entrada),
                    "identificador": normalizar_texto(identificador),
                    "fecha_atencion": fecha_atencion.strftime("%d%m%y") if fecha_atencion else "",
                    "hora_atencion": normalizar_texto(hora_atencion),
                    "años_cumplidos": años_cumplidos,
                    "genero": normalizar_texto(genero),
                    "entidad": normalizar_texto(entidad),
                    "derivacion": normalizar_texto(derivacion),
                    "especialidad": normalizar_texto(especialidad),
                    "nivel_atencion": normalizar_texto(nivel_atencion),
                    "motivo_atencion": normalizar_texto(motivo_atencion),
                    "impresion_diagnostica": normalizar_texto(impresion_diagnostica),
                    "ecg": ecg,
                    "tas": tas,
                    "tad": tad,
                    "fc": fc,
                    "fr": fr,
                    "t": t,
                    "sao2": sao2,
                    "gluc": gluc,
                    "news2_score": news2_score,
                    "atendio": normalizar_texto(atendio),
                    "intencional": int(intencional),
                    "num_farmacos": int(num_farmacos),
                    "con_alcohol": int(con_alcohol),
                    "tipo_toxico_principal": normalizar_texto(tipo_toxico_principal),
                    "sitio_de_procedencia": normalizar_texto(sitio_de_procedencia),
                    "derivacion2": normalizar_texto(derivacion2),
                    "bh": normalizar_texto(bh),
                    "qs": normalizar_texto(qs),
                    "es": normalizar_texto(es),
                    "gaso": normalizar_texto(gaso),
                    "pfh": normalizar_texto(pfh),
                    "tipo_descontaminacion": normalizar_texto(tipo_descontaminacion),
                    "tiempo_desde_consumo": tiempo_desde_consumo,
                    "tiempo_desde_llegada": tiempo_desde_llegada,
                    "destino": normalizar_texto(destino),
                    "tiempo_al_alta": tiempo_al_alta,
                    "observaciones": normalizar_texto(observaciones),
                }
                for tox in tox_cols:
                    data[tox] = int(tox_vals.get(tox, False))

                insert_paciente(data)
                st.success("Caso agregado")
                st.session_state.pop("form_mode", None)
                st.rerun()

with tab2:
    st.header("Editar o eliminar caso")
    df = load_all()

    if df.empty:
        st.info("No hay datos.")
    else:
        ids = df["id"].tolist()
        selected_id = st.selectbox(
            "Seleccionar ID",
            ids,
            format_func=lambda x: f"ID {x} - {df[df['id'] == x]['identificador'].iloc[0]}"
        )
        row = df[df["id"] == selected_id].iloc[0].to_dict()

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Editar"):
                st.session_state["edit_id"] = selected_id
                st.rerun()
        with c2:
            if st.button("Eliminar"):
                delete_paciente(selected_id)
                st.success("Caso eliminado")
                st.rerun()

        if st.session_state.get("edit_id") == selected_id:
            st.subheader("Editar caso")
            with st.form("edit_form"):
                col1, col2 = st.columns(2)
                with col1:
                    identificador = st.text_input("Identificador", value=row.get("identificador", ""))
                    genero = st.text_input("Género", value=row.get("genero", ""))
                    años_cumplidos = st.number_input("Edad", value=safe_float(row.get("años_cumplidos", 0)))
                    motivo_atencion = st.text_area("Motivo", value=row.get("motivo_atencion", ""))
                    impresion_diagnostica = st.text_area("Impresión diagnóstica", value=row.get("impresion_diagnostica", ""))
                    news2_score = st.number_input("NEWS-2", value=safe_float(row.get("news2_score", 0)))
                    tipo_toxico_principal = st.text_input("Tipo de tóxico principal", value=row.get("tipo_toxico_principal", ""))
                    num_farmacos = st.number_input("Número de fármacos", min_value=0, step=1, value=int(row.get("num_farmacos", 0)))
                    intencional = st.checkbox("Intencional", value=bool(row.get("intencional", 0)))
                    con_alcohol = st.checkbox("Con alcohol", value=bool(row.get("con_alcohol", 0)))
                with col2:
                    st.markdown("**Xenobióticos**")
                    tox_cols = [
                        "tox_benzodiacepina", "tox_antidepresivo", "tox_antipsicotico", "tox_analgesico",
                        "tox_alcohol", "tox_droga_ilegal", "tox_antiepileptico", "tox_plaguicida",
                        "tox_animal", "tox_producto_de_limpieza", "tox_antihipertensivo",
                        "tox_hipoglucemiante", "tox_antihistaminico", "tox_hidrocarburos", "tox_natural"
                    ]
                    tox_vals = {}
                    cols = st.columns(3)
                    for i, tox in enumerate(tox_cols):
                        with cols[i % 3]:
                            tox_vals[tox] = st.checkbox(
                                tox.replace("tox_", "").replace("_", " ").title(),
                                value=bool(row.get(tox, 0)),
                                key=f"edit_{tox}"
                            )

                if st.form_submit_button("Guardar cambios"):
                    update_data = {
                        "identificador": normalizar_texto(identificador),
                        "genero": normalizar_texto(genero),
                        "años_cumplidos": años_cumplidos,
                        "motivo_atencion": normalizar_texto(motivo_atencion),
                        "impresion_diagnostica": normalizar_texto(impresion_diagnostica),
                        "news2_score": news2_score,
                        "tipo_toxico_principal": normalizar_texto(tipo_toxico_principal),
                        "num_farmacos": num_farmacos,
                        "intencional": int(intencional),
                        "con_alcohol": int(con_alcohol),
                    }
                    for tox in tox_cols:
                        update_data[tox] = int(tox_vals.get(tox, False))
                    update_paciente(selected_id, update_data)
                    st.success("Caso actualizado")
                    st.session_state.pop("edit_id", None)
                    st.rerun()

with tab3:
    st.header("Análisis exploratorio")
    df = load_all()

    if df.empty:
        st.info("No hay datos para analizar.")
    else:
        df_analysis = build_analysis_df(df)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Casos", len(df_analysis))
        with c2:
            edad_media = round(df_analysis["años_cumplidos"].dropna().mean(), 1) if "años_cumplidos" in df_analysis.columns and df_analysis["años_cumplidos"].notna().any() else "NA"
            st.metric("Edad media", edad_media)
        with c3:
            news_medio = round(df_analysis["news2_score"].dropna().mean(), 1) if "news2_score" in df_analysis.columns and df_analysis["news2_score"].notna().any() else "NA"
            st.metric("NEWS-2 medio", news_medio)
        with c4:
            poli_media = round(df_analysis["num_farmacos"].dropna().mean(), 1) if "num_farmacos" in df_analysis.columns and df_analysis["num_farmacos"].notna().any() else "NA"
            st.metric("Polifarmacia media", poli_media)

        # Botón para descargar datos completos
        csv_data = df_analysis.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Descargar datos (CSV)", data=csv_data, file_name="toxicologia_datos.csv", mime="text/csv")

        render_interpretation(df_analysis)
        render_overview_figures(df_analysis)
        render_basic_stats(df_analysis)

        st.subheader("Distribuciones")
        dist_num = [c for c in ["news2_score", "años_cumplidos", "fc", "fr", "tas", "tad", "sao2", "num_farmacos"] if c in df_analysis.columns]
        if dist_num:
            selected_num = st.selectbox("Variable numérica para histograma", dist_num)
            fig = px.histogram(df_analysis, x=selected_num, nbins=20, title=f"Distribución de {selected_num}")
            st.plotly_chart(fig, use_container_width=True)

        # Nuevas secciones
        find_gravity_rule(df_analysis)
        render_toxic_combinations(df_analysis)
        render_relationships(df_analysis)
        render_clusters(df_analysis)
        render_wordcloud(df_analysis)

        # Botón para descargar reporte HTML completo
        if st.button("📄 Generar reporte HTML completo"):
            html_report = export_full_report(df_analysis)
            st.download_button("Descargar reporte HTML", data=html_report, file_name="reporte_completo.html", mime="text/html")
