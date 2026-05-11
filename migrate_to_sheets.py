#!/usr/bin/env python3
"""
Script de migración: Convierte datos de SQLite local a Google Sheets.
Ejecutar UNA VEZ para migrar datos existentes.

Requiere:
  - service_account.json en la raíz (credenciales de Google Cloud)
  - Variable de entorno GOOGLE_SHEETS_SPREADSHEET_ID o modificar SPREADSHEET_ID abajo
  - Archivo toxicologia.db en la raíz (o modificar DB_PATH)
"""

import os
import sqlite3
import sys

import gspread
from google.oauth2.service_account import Credentials
import pandas as pd

# ------------------ Configuración ------------------ #
DB_PATH = "toxicologia.db"
SPREADSHEET_ID = os.environ.get("GOOGLE_SHEETS_SPREADSHEET_ID", "")
WORKSHEET_NAME = "pacientes"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Columnas que espera la hoja de Google Sheets (en orden)
HEADERS = [
    "id", "no_entrada", "identificador", "fecha_atencion", "hora_atencion",
    "años_cumplidos", "genero", "entidad", "derivacion", "especialidad",
    "nivel_atencion", "motivo_atencion", "impresion_diagnostica", "ecg", "tas",
    "tad", "fc", "fr", "t", "sao2", "gluc", "news2_score", "atendio",
    "tox_benzodiacepina", "tox_antidepresivo", "tox_antipsicotico", "tox_analgesico",
    "tox_alcohol", "tox_droga_ilegal", "tox_antiepileptico", "tox_plaguicida",
    "tox_animal", "tox_producto_de_limpieza", "tox_antihipertensivo",
    "tox_hipoglucemiante", "tox_antihistaminico", "tox_hidrocarburos", "tox_natural",
    "intencional", "num_farmacos", "con_alcohol", "tipo_toxico_principal",
    "sitio_de_procedencia", "derivacion2", "bh", "qs", "es", "gaso", "pfh",
    "tipo_descontaminacion", "tiempo_desde_consumo", "tiempo_desde_llegada",
    "destino", "tiempo_al_alta", "observaciones"
]


def get_gspread_client():
    """Autentica con Google Sheets."""
    creds = Credentials.from_service_account_file("service_account.json", scopes=SCOPES)
    return gspread.authorize(creds)


def get_or_create_worksheet(client, spreadsheet_id):
    """Obtiene o crea la hoja 'pacientes'."""
    try:
        spreadsheet = client.open_by_key(spreadsheet_id)
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"ERROR: No se encontró la hoja con ID: {spreadsheet_id}")
        sys.exit(1)

    try:
        worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
        print(f"⚠️  La hoja '{WORKSHEET_NAME}' ya existe. Se limpiará y se recreará.")
        # Limpiar datos existentes
        all_values = worksheet.get_all_values()
        if len(all_values) > 1:
            worksheet.delete_rows(2, len(all_values))
    except gspread.exceptions.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=WORKSHEET_NAME, rows=10000, cols=60)
        worksheet.append_row(HEADERS)
        print(f"✅ Hoja '{WORKSHEET_NAME}' creada con headers.")

    return worksheet


def load_sqlite_data():
    """Carga datos desde SQLite."""
    if not os.path.exists(DB_PATH):
        print(f"ERROR: No se encontró {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM pacientes ORDER BY id", conn)
    conn.close()

    if df.empty:
        print("⚠️  No hay datos en la base de datos SQLite.")
        return df

    print(f"📊 Datos cargados desde SQLite: {len(df)} registros")
    return df


def migrate_data(worksheet, df):
    """Migra los datos de SQLite a Google Sheets."""
    if df.empty:
        print("No hay datos que migrar.")
        return

    # Asegurar que todas las columnas existan
    for col in HEADERS:
        if col not in df.columns:
            df[col] = ""

    # Reordenar según HEADERS
    df = df[HEADERS]

    # Convertir NaN a strings vacíos
    df = df.fillna("")

    # Convertir a lista de listas (más rápido para gspread)
    print("🔄 Preparando datos para subir...")
    values = df.values.tolist()

    # Insertar en lotes de 1000 filas (límite de Google Sheets API)
    batch_size = 1000
    total = len(values)

    for i in range(0, total, batch_size):
        batch = values[i:i + batch_size]
        worksheet.append_rows(batch, value_input_option="USER_ENTERED")
        print(f"  ⬆️  Subidas {min(i + batch_size, total)}/{total} filas...")

    print(f"✅ Migración completa: {total} registros transferidos a Google Sheets")


def main():
    print("=" * 60)
    print("  MIGRACIÓN CINDER: SQLite → Google Sheets")
    print("=" * 60)

    if not SPREADSHEET_ID:
        print("ERROR: Define GOOGLE_SHEETS_SPREADSHEET_ID como variable de entorno")
        print("  export GOOGLE_SHEETS_SPREADSHEET_ID='tu_id_aqui'")
        sys.exit(1)

    if not os.path.exists("service_account.json"):
        print("ERROR: No se encontró service_account.json")
        print("  Descarga las credenciales de tu cuenta de servicio de Google Cloud")
        sys.exit(1)

    print(f"📁 SQLite: {DB_PATH}")
    print(f"📊 Google Sheets ID: {SPREADSHEET_ID}")
    print()

    # Conectar a Google Sheets
    print("🔐 Conectando a Google Sheets...")
    client = get_gspread_client()
    worksheet = get_or_create_worksheet(client, SPREADSHEET_ID)

    # Cargar datos de SQLite
    print("📥 Cargando datos de SQLite...")
    df = load_sqlite_data()

    # Migrar
    print()
    migrate_data(worksheet, df)

    print()
    print("=" * 60)
    print("  ✅ MIGRACIÓN COMPLETADA")
    print("=" * 60)
    print()
    print("Ahora configura los secrets en Streamlit Cloud:")
    print("  1. Ve a Settings → Secrets")
    print("  2. Añade [google_sheets] spreadsheet_id")
    print("  3. Añade [google_service_account] con tus credenciales")


if __name__ == "__main__":
    main()
