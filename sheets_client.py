"""Google Sheets client for CINDER persistent storage."""
import json
import os
from typing import Any

import gspread
import pandas as pd
import streamlit as st
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def get_spreadsheet_id():
    """Obtiene el ID de la hoja desde Streamlit secrets o variables de entorno."""
    # 1) Streamlit secrets (recomendado para Streamlit Cloud)
    try:
        if "google_sheets" in st.secrets and "spreadsheet_id" in st.secrets["google_sheets"]:
            return st.secrets["google_sheets"]["spreadsheet_id"]
    except Exception:
        pass
    
    # 2) Variable de entorno
    return os.environ.get("CINDER_SHEET_ID", "")


SPREADSHEET_ID = get_spreadsheet_id()

COLUMN_ORDER = [
    "id", "no_entrada", "identificador", "fecha_atencion", "hora_atencion",
    "años_cumplidos", "genero", "entidad", "derivacion", "especialidad",
    "nivel_atencion", "motivo_atencion", "impresion_diagnostica", "ecg",
    "tas", "tad", "fc", "fr", "t", "sao2", "gluc", "news2_score",
    "atendio", "tox_benzodiacepina", "tox_antidepresivo", "tox_antipsicotico",
    "tox_analgesico", "tox_alcohol", "tox_droga_ilegal", "tox_antiepileptico",
    "tox_plaguicida", "tox_animal", "tox_producto_de_limpieza",
    "tox_antihipertensivo", "tox_hipoglucemiante", "tox_antihistaminico",
    "tox_hidrocarburos", "tox_natural", "intencional", "num_farmacos",
    "con_alcohol", "tipo_toxico_principal", "sitio_de_procedencia",
    "derivacion2", "bh", "qs", "es", "gaso", "pfh", "tipo_descontaminacion",
    "tiempo_desde_consumo", "tiempo_desde_llegada", "destino",
    "tiempo_al_alta", "observaciones",
]

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def _load_credentials() -> Credentials:
    """Load service-account credentials from Streamlit secrets ONLY.
    
    IMPORTANTE: Configura las credenciales en Streamlit Cloud:
    Settings -> Secrets -> Agregar [google_service_account] con tus credenciales.
    NO uses variables de entorno CINDER_GCP_CREDENTIALS.
    """
    # 1) Streamlit secrets (recomendado para Streamlit Cloud)
    try:
        if "google_service_account" in st.secrets:
            creds_dict = dict(st.secrets["google_service_account"])
            # Reemplazar \n por saltos reales en la clave privada
            if "private_key" in creds_dict:
                creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
            return Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    except Exception as e:
        st.error(f"Error al cargar credenciales de Streamlit secrets: {e}")
        raise

    # 2) Local dev fallback: archivo JSON (solo para desarrollo local)
    if os.path.exists("service_account.json"):
        return Credentials.from_service_account_file("service_account.json", scopes=SCOPES)

    raise RuntimeError(
        "No se encontraron credenciales de GCP. "
        "Configura [google_service_account] en Streamlit secrets. "
        "NO uses la variable de entorno CINDER_GCP_CREDENTIALS (está deprecada)."
    )


@pd.api.extensions.register_dataframe_accessor("gs")
class GSheetAccessor:
    """Tiny helper to sync a DataFrame with a Google Sheet worksheet."""

    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _client(self):
        creds = _load_credentials()
        if creds.expired:
            creds.refresh(Request())
        return gspread.authorize(creds)

    def _open_worksheet(self, worksheet_name: str = "pacientes"):
        client = self._client()
        spreadsheet = client.open_by_key(SPREADSHEET_ID)
        try:
            ws = spreadsheet.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            ws = spreadsheet.add_worksheet(
                title=worksheet_name, rows=1000, cols=len(COLUMN_ORDER)
            )
            ws.append_row(COLUMN_ORDER)
        return ws

    # ------------------------------------------------------------------
    # READ
    # ------------------------------------------------------------------
    def read(self, worksheet_name: str = "pacientes") -> pd.DataFrame:
        ws = self._open_worksheet(worksheet_name)
        records = ws.get_all_records()
        if not records:
            return pd.DataFrame(columns=COLUMN_ORDER)
        df = pd.DataFrame(records)
        # Coerce numeric types
        for col in (
            "no_entrada años_cumplidos ecg tas tad fc fr t sao2 gluc "
            "news2_score num_farmacos tiempo_desde_consumo tiempo_desde_llegada "
            "tiempo_al_alta id"
        ).split():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in (
            "tox_benzodiacepina tox_antidepresivo tox_antipsicotico tox_analgesico "
            "tox_alcohol tox_droga_ilegal tox_antiepileptico tox_plaguicida "
            "tox_animal tox_producto_de_limpieza tox_antihipertensivo "
            "tox_hipoglucemiante tox_antihistaminico tox_hidrocarburos "
            "tox_natural intencional con_alcohol"
        ).split():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        return df

    # ------------------------------------------------------------------
    # WRITE (replace whole sheet)
    # ------------------------------------------------------------------
    def write(self, worksheet_name: str = "pacientes") -> None:
        ws = self._open_worksheet(worksheet_name)
        # Clear but leave header
        ws.clear()
        ws.append_row(COLUMN_ORDER)
        if self._obj.empty:
            return
        # Reorder / fill missing columns
        df = self._obj.copy()
        for col in COLUMN_ORDER:
            if col not in df.columns:
                df[col] = None
        df = df[COLUMN_ORDER]
        # Replace NaN with empty string for GS API
        df = df.fillna("")
        rows = df.values.tolist()
        ws.append_rows(rows, value_input_option="RAW")

    # ------------------------------------------------------------------
    # INSERT single row
    # ------------------------------------------------------------------
    def append_row(self, row_dict: dict, worksheet_name: str = "pacientes") -> int:
        """Append a single row. Returns the new row number (1-based)."""
        ws = self._open_worksheet(worksheet_name)
        # Assign next ID
        values = []
        for col in COLUMN_ORDER:
            val = row_dict.get(col, "")
            if pd.isna(val):
                val = ""
            values.append(val)
        ws.append_row(values, value_input_option="RAW")
        # Count rows to infer the new ID (header = row 1)
        all_rows = ws.get_all_values()
        new_id = len(all_rows) - 1  # minus header
        # Update the ID cell
        id_cell = f"A{len(all_rows)}"
        ws.update(id_cell, [[new_id]], value_input_option="RAW")
        return new_id

    # ------------------------------------------------------------------
    # UPDATE single row
    # ------------------------------------------------------------------
    def update_row(self, row_id: int, row_dict: dict, worksheet_name: str = "pacientes") -> None:
        ws = self._open_worksheet(worksheet_name)
        all_rows = ws.get_all_values()
        # Find row by ID column (A)
        target_row = None
        for idx, row in enumerate(all_rows, start=1):
            if row and str(row[0]) == str(row_id):
                target_row = idx
                break
        if target_row is None:
            raise ValueError(f"Row with id={row_id} not found in sheet.")
        # Build range and values
        values = []
        for col in COLUMN_ORDER:
            val = row_dict.get(col, "")
            if pd.isna(val):
                val = ""
            values.append(val)
        # Update the whole row
        start = gspread.utils.rowcol_to_a1(target_row, 1)
        end = gspread.utils.rowcol_to_a1(target_row, len(COLUMN_ORDER))
        ws.update(f"{start}:{end}", [values], value_input_option="RAW")

    # ------------------------------------------------------------------
    # DELETE single row
    # ------------------------------------------------------------------
    def delete_row(self, row_id: int, worksheet_name: str = "pacientes") -> None:
        ws = self._open_worksheet(worksheet_name)
        all_rows = ws.get_all_values()
        target_row = None
        for idx, row in enumerate(all_rows, start=1):
            if row and str(row[0]) == str(row_id):
                target_row = idx
                break
        if target_row is None:
            raise ValueError(f"Row with id={row_id} not found in sheet.")
        ws.delete_rows(target_row)

    # ------------------------------------------------------------------
    # CLEAR (keep header)
    # ------------------------------------------------------------------
    def clear(self, worksheet_name: str = "pacientes") -> None:
        ws = self._open_worksheet(worksheet_name)
        ws.clear()
        ws.append_row(COLUMN_ORDER)


# ---------------------------------------------------------------------------
# Convenience module-level helpers for app.py
# ---------------------------------------------------------------------------

_client_instance = None

def get_client():
    global _client_instance
    if _client_instance is None:
        creds = _load_credentials()
        _client_instance = gspread.authorize(creds)
    return _client_instance


def get_worksheet(name: str = "pacientes"):
    client = get_client()
    spreadsheet = client.open_by_key(SPREADSHEET_ID)
    try:
        return spreadsheet.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=name, rows=1000, cols=len(COLUMN_ORDER))
        ws.append_row(COLUMN_ORDER)
        return ws


def load_all() -> pd.DataFrame:
    """Return every record as a DataFrame (same signature as old load_all)."""
    df = pd.DataFrame(columns=COLUMN_ORDER)
    return df.gs.read()


def insert_paciente(data: dict) -> int:
    """Insert a single patient and return the assigned ID."""
    ws = get_worksheet()
    # Determine next ID
    all_values = ws.get_all_values()
    if len(all_values) <= 1:
        next_id = 1
    else:
        # IDs are in column A
        ids = [row[0] for row in all_values[1:] if row]
        next_id = max(int(x) for x in ids if str(x).isdigit()) + 1 if ids else 1

    data["id"] = next_id
    values = [data.get(col, "") for col in COLUMN_ORDER]
    ws.append_row(values, value_input_option="RAW")
    return next_id


def update_paciente(id_value: int, data: dict) -> None:
    ws = get_worksheet()
    all_values = ws.get_all_values()
    target = None
    for idx, row in enumerate(all_values, start=1):
        if row and str(row[0]) == str(id_value):
            target = idx
            break
    if target is None:
        raise ValueError(f"Paciente id={id_value} no encontrado en Sheets.")
    data["id"] = id_value
    values = [data.get(col, "") for col in COLUMN_ORDER]
    start = gspread.utils.rowcol_to_a1(target, 1)
    end = gspread.utils.rowcol_to_a1(target, len(COLUMN_ORDER))
    ws.update(f"{start}:{end}", [values], value_input_option="RAW")


def delete_paciente(id_value: int) -> None:
    ws = get_worksheet()
    all_values = ws.get_all_values()
    target = None
    for idx, row in enumerate(all_values, start=1):
        if row and str(row[0]) == str(id_value):
            target = idx
            break
    if target is None:
        raise ValueError(f"Paciente id={id_value} no encontrado en Sheets.")
    ws.delete_rows(target)


def clear_db() -> None:
    ws = get_worksheet()
    ws.clear()
    ws.append_row(COLUMN_ORDER)
