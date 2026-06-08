import os
import math
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, make_response
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import pdfplumber

# =========================
# CONFIG
# =========================

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"xlsx", "xls", "pdf", "csv"}

MAX_UPLOAD_MB = 20

STOP_RADIUS_METERS = 25
STOP_MIN_MINUTES = 5
STOP_LONG_DURATION_MINUTES = 180
STOP_MAX_OUTLIERS = 2

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.secret_key = os.environ.get("SECRET_KEY", "cambia_esta_clave_por_una_mas_segura")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

RESULT_STORE: dict[str, dict] = {}
RESULT_STORE_MAX_ENTRIES = 20

# =========================
# UTILIDADES GENERALES
# =========================

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def build_safe_upload_path(original_filename: str) -> tuple[str, str]:
    safe_name = secure_filename(original_filename or "")
    if not safe_name:
        raise ValueError("Nombre de archivo inválido.")

    ext = safe_name.rsplit(".", 1)[1].lower() if "." in safe_name else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError("Formato no permitido. Solo .xlsx, .xls, .pdf y .csv.")

    unique_name = f"{uuid.uuid4().hex}.{ext}"
    return safe_name, os.path.join(app.config["UPLOAD_FOLDER"], unique_name)


def haversine_meters(lat1, lon1, lat2, lon2) -> float:
    r = 6371000

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return r * c


def format_dt(value):
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%d/%m/%Y %H:%M:%S")
    if isinstance(value, datetime):
        return value.strftime("%d/%m/%Y %H:%M:%S")
    return str(value)


def normalize_text(text) -> str:
    text = str(text).strip().lower()
    replacements = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "ñ": "n"
    }
    for a, b in replacements.items():
        text = text.replace(a, b)
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.replace(".", "_").replace("-", "_").replace("/", "_")
    text = "_".join(part for part in text.split())
    return text


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    normalized = [normalize_text(c) for c in df.columns]
    counts = {}
    unique_cols = []
    for col in normalized:
        base = col if col else "col"
        if base not in counts:
            counts[base] = 1
            unique_cols.append(base)
        else:
            counts[base] += 1
            unique_cols.append(f"{base}_{counts[base]}")
    df.columns = unique_cols
    return df


def is_valid_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["fecha_hora", "lat", "lon"]).copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]
    df = df.drop_duplicates(subset=["fecha_hora", "lat", "lon"])
    df = df.sort_values("fecha_hora").reset_index(drop=True)
    return df


def find_first_matching_column(columns, candidates):
    cols = list(columns)
    cands = list(candidates)

    # 1) Match exact column names first.
    for cand in cands:
        if cand in cols:
            return cand

    # 2) Match by token (underscore-separated) to avoid false positives like
    # 'lat' matching 'licenseplate'.
    for col in cols:
        tokens = [t for t in str(col).split("_") if t]
        for cand in cands:
            if cand in tokens:
                return col

    # 3) Match by prefix/suffix (still safer than arbitrary substring match).
    for col in cols:
        for cand in cands:
            if str(col).startswith(cand) or str(col).endswith(cand):
                return col

    return None


def try_parse_datetime_series(series: pd.Series) -> pd.Series:
    """
    Intenta interpretar una columna de fecha/hora.
    Sirve para casos como:
    - 04/09/2024 16:35:00
    - 16:35:00 04/09/2024
    - valores ya datetime de Excel
    """
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    if parsed.notna().sum() == 0:
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True, utc=True)
        if parsed.dt.tz is not None:
            parsed = parsed.dt.tz_localize(None)
    return parsed


def combine_date_time_columns(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    combined = date_series.astype(str).str.strip() + " " + time_series.astype(str).str.strip()
    parsed = pd.to_datetime(combined, errors="coerce", utc=True)
    if parsed.notna().sum() == 0:
        parsed = pd.to_datetime(combined, errors="coerce", dayfirst=True, utc=True)
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


def combine_any_order_datetime_columns(a: pd.Series, b: pd.Series) -> pd.Series:
    ab = combine_date_time_columns(a, b)
    ba = combine_date_time_columns(b, a)

    ab_ok = ab.notna().sum()
    ba_ok = ba.notna().sum()
    return ab if ab_ok >= ba_ok else ba


# =========================
# DETECCIÓN DE CABECERA
# =========================

def detect_header_row(file_path: str, max_rows: int = 25):
    """
    Busca una fila de cabecera plausible en las primeras filas del Excel.
    """
    try:
        raw_df = pd.read_excel(file_path, header=None, nrows=max_rows)
    except Exception:
        raw_df = pd.read_excel(file_path, header=None)

    limit = min(len(raw_df), max_rows)

    header_keywords = [
        "hora", "fecha", "coordenadas", "coordenada",
        "lat", "latitud", "latitude",
        "lon", "lng", "longitud", "longitude"
    ]

    best_row = None
    best_score = -1

    for i in range(limit):
        row = raw_df.iloc[i].fillna("")
        values = [normalize_text(v) for v in row.tolist()]
        row_text = " ".join(values)

        score = 0
        for kw in header_keywords:
            if kw in row_text:
                score += 1

        if score > best_score:
            best_score = score
            best_row = i

    if best_score <= 0:
        return 0

    return best_row


# =========================
# PARSERS
# =========================

def extract_first_two_floats(text: str):
    if text is None:
        return None, None

    s = str(text).strip()
    if not s:
        return None, None

    for sep in [";", "/", "\t", "|"]:
        s = s.replace(sep, ",")
    s = s.replace(" ", ",")

    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) < 2:
        return None, None

    lat = pd.to_numeric(parts[0], errors="coerce")
    lon = pd.to_numeric(parts[1], errors="coerce")
    if pd.isna(lat) or pd.isna(lon):
        return None, None
    return float(lat), float(lon)


def parse_with_coordinates_column(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Caso:
    - una columna con coordenadas tipo 'lat, lon'
    - fecha/hora junta o separada
    """
    df = clean_columns(df)

    coord_col = find_first_matching_column(
        df.columns,
        ["coordenadas", "coordenada", "coords", "coord", "ubicacion", "posicion"]
    )
    if not coord_col:
        return None

    result = df.copy()

    lat_lon = result[coord_col].apply(extract_first_two_floats)
    result["lat"] = pd.to_numeric(lat_lon.apply(lambda x: x[0]), errors="coerce")
    result["lon"] = pd.to_numeric(lat_lon.apply(lambda x: x[1]), errors="coerce")

    # Opción A: fecha/hora en una sola columna
    datetime_col = find_first_matching_column(
        result.columns,
        ["fecha_hora", "fechahora", "datetime", "timestamp", "hora_fecha", "fecha_y_hora"]
    )

    if datetime_col:
        result["fecha_hora"] = try_parse_datetime_series(result[datetime_col])
        result = is_valid_lat_lon(result)
        if not result.empty:
            return result[["fecha_hora", "lat", "lon"]]

    # Opción C: fecha y hora separadas
    date_col = find_first_matching_column(
        result.columns,
        ["fecha", "date"]
    )
    time_col = find_first_matching_column(
        result.columns,
        ["hora", "time"]
    )

    if date_col and time_col and date_col != time_col:
        result["fecha_hora"] = combine_any_order_datetime_columns(result[date_col], result[time_col])
        result = is_valid_lat_lon(result)
        if not result.empty:
            return result[["fecha_hora", "lat", "lon"]]

    # Opción B: una columna que contenga fecha u hora pero realmente venga completa
    generic_dt_col = find_first_matching_column(
        result.columns,
        ["hora", "fecha", "time", "date"]
    )
    if generic_dt_col:
        result["fecha_hora"] = try_parse_datetime_series(result[generic_dt_col])
        result = is_valid_lat_lon(result)
        if not result.empty:
            return result[["fecha_hora", "lat", "lon"]]

    return None


def parse_with_separate_lat_lon(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Caso:
    - coordenadas en columnas separadas
    - fecha/hora junta o separada
    """
    df = clean_columns(df)

    lat_col = find_first_matching_column(
        df.columns,
        ["lat", "latitud", "latitude", "y"]
    )
    lon_col = find_first_matching_column(
        df.columns,
        ["lon", "lng", "long", "longitud", "longitude", "x"]
    )

    if not lat_col or not lon_col or lat_col == lon_col:
        return None

    result = df.copy()
    result["lat"] = pd.to_numeric(result[lat_col], errors="coerce")
    result["lon"] = pd.to_numeric(result[lon_col], errors="coerce")

    # Opción A: fecha/hora en una sola columna
    datetime_col = find_first_matching_column(
        result.columns,
        ["fecha_hora", "fechahora", "datetime", "timestamp", "hora_fecha", "fecha_y_hora"]
    )

    if datetime_col:
        result["fecha_hora"] = try_parse_datetime_series(result[datetime_col])
        result = is_valid_lat_lon(result)
        if not result.empty:
            return result[["fecha_hora", "lat", "lon"]]

    # Opción C: fecha y hora separadas
    date_col = find_first_matching_column(
        result.columns,
        ["fecha", "date"]
    )
    time_col = find_first_matching_column(
        result.columns,
        ["hora", "time"]
    )

    if date_col and time_col and date_col != time_col:
        result["fecha_hora"] = combine_any_order_datetime_columns(result[date_col], result[time_col])
        result = is_valid_lat_lon(result)
        if not result.empty:
            return result[["fecha_hora", "lat", "lon"]]

    # Opción B: una columna que contenga fecha u hora pero realmente venga completa
    generic_dt_col = find_first_matching_column(
        result.columns,
        ["hora", "fecha", "time", "date"]
    )
    if generic_dt_col:
        result["fecha_hora"] = try_parse_datetime_series(result[generic_dt_col])
        result = is_valid_lat_lon(result)
        if not result.empty:
            return result[["fecha_hora", "lat", "lon"]]

    return None


def parse_excel(file_path: str) -> pd.DataFrame:
    """
    Parser multi-formato.
    Soporta:
    - coordenadas juntas en una columna
    - coordenadas separadas
    - fecha/hora juntas
    - fecha y hora separadas
    """
    header_row = detect_header_row(file_path)
    df = pd.read_excel(file_path, header=header_row)
    df = df.dropna(how="all").copy()

    # Primer intento: coordenadas en una sola columna
    parsed = parse_with_coordinates_column(df)
    if parsed is not None and not parsed.empty:
        return parsed

    # Segundo intento: lat/lon separados
    parsed = parse_with_separate_lat_lon(df)
    if parsed is not None and not parsed.empty:
        return parsed

    raise ValueError(
        "Formato de archivo no reconocido. "
        "La app soporta coordenadas en una columna o separadas, y fecha/hora juntas o separadas."
    )


def pdf_tables_to_dataframe(file_path: str) -> pd.DataFrame:
    tables = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_tables()
            if not extracted:
                continue
            for t in extracted:
                if not t or len(t) < 2:
                    continue

                header = [(h if h is not None and str(h).strip() else f"col_{i}") for i, h in enumerate(t[0])]
                rows = t[1:]
                if not header:
                    continue

                df = pd.DataFrame(rows, columns=header)
                df = df.dropna(how="all").copy()
                if not df.empty:
                    tables.append(clean_columns(df))

    if not tables:
        raise ValueError("No se encontraron tablas en el PDF.")

    combined = pd.concat(tables, ignore_index=True)
    combined = combined.dropna(how="all").copy()
    return combined


def parse_pdf(file_path: str) -> pd.DataFrame:
    df = pdf_tables_to_dataframe(file_path)

    parsed = parse_with_coordinates_column(df)
    if parsed is not None and not parsed.empty:
        return parsed

    parsed = parse_with_separate_lat_lon(df)
    if parsed is not None and not parsed.empty:
        return parsed

    raise ValueError(
        "No se pudo reconocer el formato del PDF. "
        "Asegúrate de que contiene una tabla con fecha/hora y coordenadas."
    )


def read_csv_flexible(file_path: str) -> pd.DataFrame:
    last_error = None
    for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(
                file_path,
                sep=None,
                engine="python",
                encoding=encoding,
                on_bad_lines="skip",
            )
        except Exception as e:
            last_error = e
            continue

    raise ValueError(f"No se pudo leer el CSV. Detalle: {last_error}")


def parse_csv(file_path: str) -> pd.DataFrame:
    df = read_csv_flexible(file_path)
    df = df.dropna(how="all").copy()

    parsed = parse_with_coordinates_column(df)
    if parsed is not None and not parsed.empty:
        return parsed

    parsed = parse_with_separate_lat_lon(df)
    if parsed is not None and not parsed.empty:
        return parsed

    raise ValueError(
        "Formato de CSV no reconocido. "
        "Asegúrate de que contiene fecha/hora y coordenadas (juntas o separadas)."
    )


def _detect_real_format(file_path: str) -> str | None:
    """Detecta el formato real del archivo leyendo sus bytes iniciales."""
    try:
        with open(file_path, "rb") as f:
            header = f.read(8)
    except Exception:
        return None

    # ZIP (y por tanto XLSX/XLSM)
    if header[:4] == b"PK\x03\x04":
        return "xlsx"
    # PDF
    if header[:5] == b"%PDF-":
        return "pdf"
    return None


def parse_file(file_path: str) -> pd.DataFrame:
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")

    # Si el contenido real no coincide con la extensión, usar el formato real
    real = _detect_real_format(file_path)
    if real and real != ext:
        ext = real

    if ext in {"xlsx", "xls"}:
        return parse_excel(file_path)
    if ext == "pdf":
        return parse_pdf(file_path)
    if ext == "csv":
        return parse_csv(file_path)
    raise ValueError("Formato no soportado.")


def build_export_filename(summary: dict) -> str:
    base = secure_filename(f"resultado_{summary.get('filename', 'ruta')}")
    if not base.lower().endswith(".html"):
        base = f"{base}.html"
    if not base:
        return "resultado.html"
    return base


def calculate_total_distance(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0

    lat = np.radians(df["lat"].values)
    lon = np.radians(df["lon"].values)

    dlat = lat[1:] - lat[:-1]
    dlon = lon[1:] - lon[:-1]

    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return float(np.sum(6371000 * c)) / 1000.0


def detect_stops(df: pd.DataFrame, radius_meters=STOP_RADIUS_METERS,
                 min_minutes=STOP_MIN_MINUTES, max_outliers=STOP_MAX_OUTLIERS,
                 long_duration_minutes=STOP_LONG_DURATION_MINUTES):
    stops = []
    n = len(df)
    if n == 0:
        return stops

    lats = df["lat"].values
    lons = df["lon"].values
    times = df["fecha_hora"].values

    i = 0
    while i < n:
        group_lats = [lats[i]]
        group_lons = [lons[i]]
        start_time = times[i]

        j = i + 1
        last_in_group = i
        consecutive_outliers = 0

        while j < n:
            centroid_lat = sum(group_lats) / len(group_lats)
            centroid_lon = sum(group_lons) / len(group_lons)
            dist = haversine_meters(centroid_lat, centroid_lon, lats[j], lons[j])

            if dist <= radius_meters:
                group_lats.append(lats[j])
                group_lons.append(lons[j])
                last_in_group = j
                consecutive_outliers = 0
                j += 1
            elif consecutive_outliers < max_outliers:
                consecutive_outliers += 1
                j += 1
            else:
                break

        end_time = times[last_in_group]
        duration_min = (end_time - start_time) / np.timedelta64(1, "m")

        if last_in_group > i and duration_min >= min_minutes:
            stop_lat = sum(group_lats) / len(group_lats)
            stop_lon = sum(group_lons) / len(group_lons)

            st = pd.Timestamp(start_time)
            stop_type = "long" if duration_min >= long_duration_minutes else "normal"
            stops.append({
                "lat": round(float(stop_lat), 6),
                "lon": round(float(stop_lon), 6),
                "start": format_dt(st),
                "end": format_dt(pd.Timestamp(end_time)),
                "duration_min": round(float(duration_min), 1),
                "points": len(group_lats),
                "date": st.strftime("%d/%m/%Y") if not pd.isna(st) else "",
                "type": stop_type,
            })

            i = last_in_group + 1
        else:
            i += 1

    return stops


def build_route_points(df: pd.DataFrame):
    lats = df["lat"].round(6).tolist()
    lons = df["lon"].round(6).tolist()
    dts  = df["fecha_hora"].tolist()

    route = []
    for lat, lon, dt in zip(lats, lons, dts):
        route.append({
            "lat": lat,
            "lon": lon,
            "fecha_hora": format_dt(dt),
            "date": dt.strftime("%d/%m/%Y") if not pd.isna(dt) else "",
        })

    return route


# =========================
# RUTAS
# =========================

@app.route("/", methods=["GET", "POST"])
def index():
    route_points = []
    stops = []
    summary = None
    error_detail = None
    export_id = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            flash("Debes seleccionar un archivo Excel, PDF o CSV.", "danger")
            return redirect(url_for("index"))

        filepath = None
        try:
            original_name = file.filename
            safe_name, filepath = build_safe_upload_path(original_name)

            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
            file.save(filepath)

            if os.path.getsize(filepath) <= 0:
                raise ValueError("El archivo subido está vacío.")

            df = parse_file(filepath)

            route_points = build_route_points(df)
            stops = detect_stops(
                df,
                radius_meters=STOP_RADIUS_METERS,
                min_minutes=STOP_MIN_MINUTES
            )

            summary = {
                "filename": safe_name,
                "total_points": len(df),
                "start_time": format_dt(df["fecha_hora"].min()),
                "end_time": format_dt(df["fecha_hora"].max()),
                "total_distance_km": round(calculate_total_distance(df), 2),
                "total_stops": len(stops),
                "stop_radius_meters": STOP_RADIUS_METERS,
                "stop_min_minutes": STOP_MIN_MINUTES,
            }

            export_id = uuid.uuid4().hex
            RESULT_STORE[export_id] = {
                "route_points": route_points,
                "stops": stops,
                "summary": summary,
            }

            while len(RESULT_STORE) > RESULT_STORE_MAX_ENTRIES:
                oldest_key = next(iter(RESULT_STORE))
                del RESULT_STORE[oldest_key]

        except ValueError as e:
            flash(str(e), "danger")
            error_detail = str(e)
        except Exception as e:
            flash("No se pudo procesar el archivo. Revisa el formato y vuelve a intentarlo.", "danger")
            error_detail = str(e)
        finally:
            try:
                if filepath and os.path.exists(filepath):
                    os.remove(filepath)
            except Exception:
                pass

    return render_template(
        "index.html",
        route_points=route_points,
        stops=stops,
        summary=summary,
        error_detail=error_detail,
        export_id=export_id,
    )


@app.route("/export/<export_id>", methods=["GET"])
def export_result(export_id: str):
    payload = RESULT_STORE.get(export_id)
    if not payload:
        flash("El resultado a exportar ya no está disponible. Vuelve a procesar el archivo.", "warning")
        return redirect(url_for("index"))

    html = render_template(
        "result_export.html",
        route_points=payload.get("route_points", []),
        stops=payload.get("stops", []),
        summary=payload.get("summary"),
    )

    response = make_response(html)
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    response.headers["Content-Disposition"] = f"attachment; filename=\"{build_export_filename(payload.get('summary') or {})}\""
    return response


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    flash(f"El archivo supera el tamaño máximo permitido ({MAX_UPLOAD_MB} MB).", "danger")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)