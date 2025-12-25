from __future__ import annotations

import argparse
import base64
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# Nota:
# - Selenium y pandas se importan de forma lazy (dentro de las funciones/modos que los usan)
#   para evitar fallas al importar este módulo en ambientes donde no estén instalados.

def _repo_root() -> Path:
    # src/etl/dataExtractor.py -> repo root is 2 levels up from src
    return Path(__file__).resolve().parents[2]


def _default_raw_dir() -> Path:
    # Separar por fuente para mantener orden en data/raw
    return _repo_root() / "data" / "raw" / "banrep"


def _default_suameca_dir() -> Path:
    return _default_raw_dir() / "suameca"


def _default_proc_dir() -> Path:
    return _repo_root() / "data" / "proc"


def _safe_filename(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_.") or "file"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for i in range(1, 10_000):
        candidate = path.with_name(f"{stem}__{i}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"No se pudo generar un nombre único para {path}")


def _deep_replace_placeholder(obj: Any, placeholder: str, replacement: Any) -> Any:
    if obj == placeholder:
        return replacement
    if isinstance(obj, list):
        return [_deep_replace_placeholder(x, placeholder, replacement) for x in obj]
    if isinstance(obj, dict):
        return {k: _deep_replace_placeholder(v, placeholder, replacement) for k, v in obj.items()}
    return obj


def _guess_extension(content_type: str | None) -> str:
    ct = (content_type or "").lower()
    if "spreadsheetml" in ct or "application/vnd.openxmlformats-officedocument" in ct:
        return ".xlsx"
    if "ms-excel" in ct:
        return ".xls"
    if "text/csv" in ct or "application/csv" in ct:
        return ".csv"
    if "application/json" in ct:
        return ".json"
    if "application/xml" in ct or "text/xml" in ct:
        return ".xml"
    return ".bin"


@dataclass(frozen=True)
class DownloadResult:
    status: str
    file_path: str | None = None
    error: str | None = None


class BanrepDataExtractor:
    """Extractor (Selenium/requests) para el portal estadístico del Banco de la República."""
    
    def __init__(self, headless: bool = True, output_dir: str | Path | None = None, timeout_s: int = 30):
        """
        Inicializa el navegador
        
        Args:
            headless (bool): Si True, ejecuta el navegador sin interfaz gráfica
        """
        try:
            from selenium.webdriver.chrome.options import Options
        except Exception as e:
            raise RuntimeError("Falta dependencia 'selenium'. Instala requirements.txt") from e

        self.options = Options()
        if headless:
            # Headless “new” es más estable en Chrome recientes
            self.options.add_argument("--headless=new")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--window-size=1920,1080")

        self.download_dir = Path(output_dir) if output_dir else _default_raw_dir()
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        prefs = {
            "download.default_directory": str(self.download_dir.resolve()),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
        }
        self.options.add_experimental_option("prefs", prefs)
        
        self.driver: Any = None
        self.wait: Any = None
        self.base_url = "https://totoro.banrep.gov.co/estadisticas-economicas/"
        self.timeout_s = timeout_s
        
    def iniciar_navegador(self):
        from selenium import webdriver
        from selenium.webdriver.support.ui import WebDriverWait

        self.driver = webdriver.Chrome(options=self.options)
        self.wait = WebDriverWait(self.driver, self.timeout_s)

        # En headless, a veces Chrome ignora prefs de descarga; forzamos vía CDP.
        try:
            self.driver.execute_cdp_cmd(
                "Page.setDownloadBehavior",
                {"behavior": "allow", "downloadPath": str(self.download_dir.resolve())},
            )
        except Exception:
            # No todos los drivers soportan este comando; prefs ya cubre el caso común.
            pass
        
    def cerrar_navegador(self):
        """Cierra el navegador"""
        if self.driver:
            self.driver.quit()
    
    def buscar_serie(self, nombre_serie: str) -> bool:
        """
        Busca una serie específica en el catálogo
        
        Args:
            nombre_serie (str): Nombre de la serie a buscar
        """
        try:
            from selenium.common.exceptions import TimeoutException
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support import expected_conditions as EC

            #print(f"\nBuscando serie: {nombre_serie}")
            
            # Buscar el campo de búsqueda
            search_input = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text']"))
            )
            
            # Limpiar y escribir el término de búsqueda
            search_input.clear()
            search_input.send_keys(nombre_serie)
            time.sleep(2)
            
            # Hacer clic en el primer resultado
            try:
                primer_resultado = self.wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "a.serie-link, .resultado-busqueda"))
                )
                primer_resultado.click()
                time.sleep(2)
                #print(f"Serie encontrada: {nombre_serie}")
                return True
            except TimeoutException:
                #print(f"No se encontró la serie: {nombre_serie}")
                return False
                
        except Exception as e:
            #print(f"Error al buscar {nombre_serie}: {str(e)}")
            return False
    
    def _wait_for_new_download(self, before_files: set[Path], timeout_s: int = 180) -> Path:
        start = time.time()
        while time.time() - start < timeout_s:
            current = set(self.download_dir.glob("*"))
            new_files = [p for p in (current - before_files) if p.is_file()]

            # Ignorar descargas incompletas
            completed = [p for p in new_files if not p.name.endswith(".crdownload")]
            if completed:
                # Si aparecen varios, tomar el más reciente
                return max(completed, key=lambda p: p.stat().st_mtime)

            # Si hay .crdownload, esperar a que termine
            time.sleep(0.5)
        raise TimeoutError("Timeout esperando a que finalice la descarga")

    def descargar_serie_excel(self, serie_key: str) -> DownloadResult:
        """Descarga la serie actual en formato Excel y la deja en data/raw/banrep con nombre estable."""
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support import expected_conditions as EC

            before_files = set(self.download_dir.glob("*"))

            # Buscar botón de descarga/exportar
            btn_exportar = self.wait.until(
                EC.element_to_be_clickable((By.LINK_TEXT, "Exportar"))
            )
            btn_exportar.click()
            time.sleep(1)
            
            # Hacer clic en Excel
            btn_excel = self.wait.until(
                EC.element_to_be_clickable((By.LINK_TEXT, "Excel"))
            )
            btn_excel.click()

            downloaded = self._wait_for_new_download(before_files)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = downloaded.suffix or ".xlsx"
            dest_name = f"banrep__{_safe_filename(serie_key)}__{ts}{ext}"
            dest = _unique_path(self.download_dir / dest_name)

            # Mover/renombrar el archivo a nombre controlado
            downloaded.replace(dest)
            return DownloadResult(status="ok", file_path=str(dest))
            
        except Exception as e:
            return DownloadResult(status="error", error=str(e))
    
    def extraer_datos_completos(self, series_a_extraer: dict[str, str] | None = None) -> dict[str, Any]:
        """
        Extrae:
        1. PIB real trimestral con ajuste estacional
        2. Inflación Total
        3. Tasa de desempleo
        4. Balance fiscal - Gastos
        5. IPC
        """
        
        series_a_extraer = series_a_extraer or {
            "PIB_real_trimestral": "Producto Interno Bruto (PIB)",
            "Inflacion_total": "Inflación y meta",
            "Tasa_laboral": "Mercado laboral y población",
            "IPC": "IPC_Índice de Precios al Consumidor",
            "Balance_fiscal_gastos": "Gobierno Nacional Central",
            "Tasa_de_cambio": "Tasa de cambio del peso colombiano",
        }
        
        resultados: dict[str, Any] = {
            "source": "banrep",
            "base_url": self.base_url,
            "output_dir": str(self.download_dir),
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "series": {},
        }
        
        try:
            self.iniciar_navegador()
            self.driver.get(self.base_url)
            #print("EXTRACCIÓN DE DATOS DEL BANCO DE LA REPÚBLICA")
            
            for clave, nombre_serie in series_a_extraer.items():
                
                # Volver a la página principal entre búsquedas
                if clave != "PIB_real_trimestral":
                    self.driver.get(self.base_url)
                    time.sleep(2)
                
                if self.buscar_serie(nombre_serie):
                    # Intentar descargar
                    dl = self.descargar_serie_excel(clave)
                    resultados["series"][clave] = {
                        "query": nombre_serie,
                        "status": dl.status,
                        "file_path": dl.file_path,
                        "error": dl.error,
                    }
                else:
                    resultados["series"][clave] = {
                        "query": nombre_serie,
                        "status": "not_found",
                        "file_path": None,
                        "error": None,
                    }
                
                time.sleep(2)  # Pausa entre series

            # Guardar un pequeño manifiesto de corrida en raw (útil para trazabilidad)
            manifest_path = _unique_path(
                self.download_dir
                / f"banrep__manifest__{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            manifest_path.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
            resultados["manifest_path"] = str(manifest_path)
            
        except Exception as e:
            resultados["error"] = str(e)
        
        finally:
            self.cerrar_navegador()
        
        return resultados

SDMX_REST_ENDPOINT = "https://totoro.banrep.gov.co/nsi-jax-ws/rest/data"
SDMX_AGENCY_ID = "ESTAT"
SDMX_VERSION = "1.0"

SDMX_TOPIC_TO_FLOW: dict[str, dict[str, str]] = {
    "IBR": {"latest": "DF_IBR_DAILY_LATEST", "hist": "DF_IBR_DAILY_HIST"},
    "DTF": {"latest": "DF_DTF_DAILY_LATEST", "hist": "DF_DTF_DAILY_HIST"},
    "TRM": {"latest": "DF_TRM_DAILY_LATEST", "hist": "DF_TRM_DAILY_HIST"},
    "TPM_DAILY": {"latest": "DF_CBR_DAILY_LATEST", "hist": "DF_CBR_DAILY_HIST"},
    "TPM_MONTHLY": {"latest": "DF_CBR_MONTHLY_LATEST", "hist": "DF_CBR_MONTHLY_HIST"},
    "TIB": {"latest": "DF_IR_DAILY_LATEST", "hist": "DF_IR_DAILY_HIST"},
    "COLCAP_MONTHLY": {"latest": "DF_COLCAP_MONTHLY_LATEST", "hist": "DF_COLCAP_MONTHLY_HIST"},
    "MONETARY_AGG": {"latest": "DF_MONAGG_MONTHLY_LATEST", "hist": "DF_MONAGG_MONTHLY_HIST"},
    "DTF_TRIM_ANT": {"latest": "DF_DTF_TRIM_ANTICIPADO_LATEST", "hist": "DF_DTF_TRIM_ANTICIPADO_HIST"},
    "DTF_MONTHLY": {"latest": "DF_DTF_MONTHLY_LATEST", "hist": "DF_DTF_MONTHLY_HIST"},
    "UVR": {"latest": "DF_UVR_DAILY_LATEST", "hist": "DF_UVR_DAILY_HIST"},
}


def _normalize_flow_id(flow_id: str) -> str:
    # Normaliza los casos que en el PDF salen partidos por saltos de línea.
    compact = "".join(flow_id.split())
    compact = compact.replace("_HIST", "_HIST").replace("_LATEST", "_LATEST")
    return compact


def _sdmx_build_data_url(
    flow_id: str,
    *,
    start_period: str | None = None,
    end_period: str | None = None,
    agency_id: str = SDMX_AGENCY_ID,
    version: str = SDMX_VERSION,
    endpoint: str = SDMX_REST_ENDPOINT,
    key: str = "all",
    provider: str = "ALL",
    detail: str = "full",
    dimension_at_observation: str = "TIME_PERIOD",
) -> str:
    flow_id = _normalize_flow_id(flow_id)
    url = f"{endpoint}/{agency_id},{flow_id},{version}/{key}/{provider}/"
    params = {}
    if start_period:
        params["startPeriod"] = start_period
    if end_period:
        params["endPeriod"] = end_period
    params["dimensionAtObservation"] = dimension_at_observation
    params["detail"] = detail
    if params:
        # construir querystring estable
        from urllib.parse import urlencode

        url = f"{url}?{urlencode(params)}"
    return url


def extraer_con_requests(
    *,
    output_dir: str | Path | None = None,
    category: str = "hist",
    start_period: str | None = None,
    end_period: str | None = None,
    topics: dict[str, str] | None = None,
    http_timeout_s: int = 180,
    verbose: bool = False,
) -> dict[str, Any]:
    """Consulta el servicio SDMX REST (XML) del Banco de la República y guarda los XML en raw."""

    out_dir = Path(output_dir) if output_dir else _default_raw_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    category_norm = category.strip().lower()
    if category_norm not in {"hist", "latest"}:
        raise ValueError("category debe ser 'hist' o 'latest'")

    topics = topics or {
        "PIB_real_trimestral": "PIB",
        "Inflacion_total": "INFLACION",
        "Tasa_laboral": "MERCADO_LABORAL",
        "IPC": "IPC",
        "Balance_fiscal_gastos": "GOBIERNO_NACIONAL_CENTRAL",
        "Tasa_de_cambio": "TRM",
        # Temas que SÍ están disponibles en el SDMX del documento (útiles para empezar):
        "IBR": "IBR",
        "DTF": "DTF",
        "TPM": "TPM_DAILY",
        "TIB": "TIB",
        "COLCAP": "COLCAP_MONTHLY",
        "Agregados_monetarios": "MONETARY_AGG",
        "UVR": "UVR",
    }

    resultados: dict[str, Any] = {
        "source": "banrep_sdmx_rest",
        "endpoint": SDMX_REST_ENDPOINT,
        "output_dir": str(out_dir),
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "category": category_norm,
        "start_period": start_period,
        "end_period": end_period,
        "http_timeout_s": http_timeout_s,
        "topics": {},
    }

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Automated-Time-Series-Forecasting-System/1.0 (+requests)",
            "Accept": "application/xml",
        }
    )

    for pipeline_key, requested in topics.items():
        # Resolver topic -> FLOW_ID
        flow_id: str | None
        requested = (requested or "").strip()
        if requested.upper().startswith("DF_"):
            flow_id = requested
        else:
            flow_id = SDMX_TOPIC_TO_FLOW.get(requested, {}).get(category_norm)

        if not flow_id:
            if verbose:
                print(f"[sdmx] {pipeline_key}: unsupported ({requested})")
            resultados["topics"][pipeline_key] = {
                "requested": requested,
                "status": "unsupported",
                "flow_id": None,
                "url": None,
                "file_path": None,
                "http_status": None,
                "error": "Tema no disponible en el SDMX del documento técnico (revisar anexos 1 y 2)",
            }
            continue

        url = _sdmx_build_data_url(flow_id, start_period=start_period, end_period=end_period)
        try:
            if verbose:
                print(f"[sdmx] {pipeline_key}: {flow_id}")
            resp = session.get(url, timeout=http_timeout_s)
            http_status = resp.status_code
            if http_status != 200:
                resultados["topics"][pipeline_key] = {
                    "requested": requested,
                    "status": "http_error",
                    "flow_id": flow_id,
                    "url": url,
                    "file_path": None,
                    "http_status": http_status,
                    "error": resp.text[:5000],
                }
                continue

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest_name = f"banrep__{_safe_filename(flow_id)}__{ts}.xml"
            dest = _unique_path(out_dir / dest_name)
            dest.write_bytes(resp.content)

            resultados["topics"][pipeline_key] = {
                "requested": requested,
                "status": "ok",
                "flow_id": flow_id,
                "url": url,
                "file_path": str(dest),
                "http_status": http_status,
                "error": None,
            }
        except Exception as e:
            resultados["topics"][pipeline_key] = {
                "requested": requested,
                "status": "error",
                "flow_id": flow_id,
                "url": url,
                "file_path": None,
                "http_status": None,
                "error": str(e),
            }

    manifest_path = _unique_path(out_dir / f"banrep__sdmx_manifest__{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    manifest_path.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    resultados["manifest_path"] = str(manifest_path)
    return resultados


SUAMECA_CONSULTA_DATOS_SERIES_URL = (
    "https://suameca.banrep.gov.co/buscador-de-series/rest/"
    "buscadorSeriesRestService/consultaDatosSeries"
)


SUAMECA_DEFAULT_SERIES: dict[str, dict[str, Any]] = {
    "Inflacion_total": {
        "idSerie": 15270,
        "fechaInicio": 19550731,
        "fechaFin": 20251130,
        "idPeriodicidades": [9],
    },
    "Balance_fiscal_gobierno_nacional": {
        "idSerie": 16723,
        "fechaInicio": 20040131,
        "fechaFin": 20250831,
        "idPeriodicidades": [9, 12, 18],
    },
    "TRM": {
        "idSerie": 1,
        "fechaInicio": 19911127,
        "fechaFin": 20251215,
        "idPeriodicidades": [9, 22],
    },
    "Tasa_desempleo": {
        "idSerie": 15312,
        "fechaInicio": 20010131,
        "fechaFin": 20251031,
        "idPeriodicidades": [9],
    },
    "Tasa_interes_colocacion_total": {
        "idSerie": 15125,
        "fechaInicio": 20020531,
        "fechaFin": 20251130,
        "idPeriodicidades": [22],
    },
    "PIB_real_trimestral_2015_AE": {
        "idSerie": 15152,
        "fechaInicio": 20050331,
        "fechaFin": 20250930,
        "idPeriodicidades": [12, 14],
    },
}


def _build_suameca_payload_from_spec(
    spec: dict[str, Any],
    *,
    start_period: str | None = None,
    end_period: str | None = None,
) -> dict[str, Any]:
    # Basado en lo observado en Network: fechaInicio/fechaFin + series[0].idSerie + series[0].idPeriocidades
    # Nota: el backend parece ser estricto con nombres (idPeriocidades con 'o').
    def _to_int_or_none(v: Any) -> int | None:
        if v is None:
            return None
        if isinstance(v, int):
            return v
        s = str(v).strip()
        if not s:
            return None
        return int(s)

    return {
        "fechaInicio": _to_int_or_none(start_period) or spec.get("fechaInicio"),
        "fechaFin": _to_int_or_none(end_period) or spec.get("fechaFin"),
        "series": [
            {
                "idSerie": spec.get("idSerie"),
                "idPeriodicidades": spec.get("idPeriodicidades", []),
            }
        ],
    }


def extraer_suameca_sin_api(
    *,
    output_dir: str | Path | None = None,
    series: dict[str, dict[str, Any]] | None = None,
    template_json_path: str | Path | None = None,
    start_period: str | None = None,
    end_period: str | None = None,
    http_timeout_s: int = 180,
    verbose: bool = False,
    cookie_header: str | None = None,
    referer: str | None = None,
) -> dict[str, Any]:
    """Descarga series desde el 'Buscador de series' replicando el POST observado en el navegador.

    IMPORTANTE:
    - Este endpoint no está documentado públicamente como API estable.
    - Para que funcione de forma confiable, se recomienda proveer una plantilla JSON EXACTA
      (copiada del Network tab) mediante --template-json.
    - En esa plantilla, coloca el string __SERIE_ID__ donde va el ID de la serie.
      Opcionalmente, __START_PERIOD__ y __END_PERIOD__ si aplica.
    """

    out_dir = Path(output_dir) if output_dir else _default_suameca_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    series = series or SUAMECA_DEFAULT_SERIES

    template: Any | None = None
    if template_json_path:
        template = json.loads(Path(template_json_path).read_text(encoding="utf-8"))

    resultados: dict[str, Any] = {
        "source": "banrep_suameca_post",
        "endpoint": SUAMECA_CONSULTA_DATOS_SERIES_URL,
        "output_dir": str(out_dir),
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "start_period": start_period,
        "end_period": end_period,
        "series": {},
        "template_json_path": str(template_json_path) if template_json_path else None,
    }

    session = requests.Session()
    # Muchos backends esperan un User-Agent "real".
    referer = referer or "https://suameca.banrep.gov.co/descarga-multiple-de-datos/consolidado"
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
            "Origin": "https://suameca.banrep.gov.co",
            "Referer": referer,
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
    )

    if cookie_header:
        # No lo persistimos a disco (sensible); solo se usa en runtime.
        session.headers["Cookie"] = cookie_header

    # Cargar el referer para obtener cookies de ruta / balanceador (si no se proveen)
    try:
        session.get(referer, timeout=30)
    except Exception:
        pass

    for key, spec in series.items():
        serie_id = spec.get("idSerie")
        payload: Any
        if template is None:
            payload = _build_suameca_payload_from_spec(spec, start_period=start_period, end_period=end_period)
        else:
            payload = _deep_replace_placeholder(template, "__SERIE_ID__", serie_id)
            if start_period is not None:
                payload = _deep_replace_placeholder(payload, "__START_PERIOD__", start_period)
            if end_period is not None:
                payload = _deep_replace_placeholder(payload, "__END_PERIOD__", end_period)

        try:
            if verbose:
                print(f"[suameca] {key}: id={serie_id}")
            resp = session.post(SUAMECA_CONSULTA_DATOS_SERIES_URL, json=payload, timeout=http_timeout_s)
            ct = resp.headers.get("content-type", "")

            if resp.status_code != 200:
                resultados["series"][key] = {
                    "id": serie_id,
                    "status": "http_error",
                    "file_path": None,
                    "http_status": resp.status_code,
                    "content_type": ct,
                    "error": resp.text[:5000],
                }
                continue

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Caso 1: el backend devuelve JSON (posible base64)
            if "application/json" in (ct or "").lower():
                data = None
                try:
                    data = resp.json()
                except Exception:
                    data = None

                # Heurística: buscar un campo con base64
                b64_value = None
                filename_hint = None
                if isinstance(data, dict):
                    for k in ("archivo", "file", "contenido", "bytes", "data", "contenidoArchivo"):
                        v = data.get(k)
                        if isinstance(v, str) and len(v) > 100:
                            b64_value = v
                            break
                    for k in ("nombreArchivo", "filename", "name"):
                        v = data.get(k)
                        if isinstance(v, str) and v:
                            filename_hint = v
                            break

                if b64_value:
                    raw = base64.b64decode(b64_value)
                    ext = Path(filename_hint).suffix if filename_hint else ".bin"
                    dest_name = f"banrep__suameca__{_safe_filename(key)}__{serie_id}__{ts}{ext}"
                    dest = _unique_path(out_dir / dest_name)
                    dest.write_bytes(raw)
                    resultados["series"][key] = {
                        "id": serie_id,
                        "status": "ok",
                        "file_path": str(dest),
                        "http_status": resp.status_code,
                        "content_type": ct,
                        "filename_hint": filename_hint,
                        "error": None,
                    }
                else:
                    # Guardar el JSON completo para inspección
                    dest_name = f"banrep__suameca__{_safe_filename(key)}__{serie_id}__{ts}.json"
                    dest = _unique_path(out_dir / dest_name)
                    dest.write_text(json.dumps(data if data is not None else resp.text, ensure_ascii=False, indent=2), encoding="utf-8")
                    resultados["series"][key] = {
                        "id": serie_id,
                        "status": "ok_json",
                        "file_path": str(dest),
                        "http_status": resp.status_code,
                        "content_type": ct,
                        "error": None,
                    }

                continue

            # Caso 2: respuesta binaria/CSV/XML
            ext = _guess_extension(ct)
            dest_name = f"banrep__suameca__{_safe_filename(key)}__{serie_id}__{ts}{ext}"
            dest = _unique_path(out_dir / dest_name)
            dest.write_bytes(resp.content)
            resultados["series"][key] = {
                "id": serie_id,
                "status": "ok",
                "file_path": str(dest),
                "http_status": resp.status_code,
                "content_type": ct,
                "error": None,
            }

        except Exception as e:
            resultados["series"][key] = {
                "id": serie_id,
                "status": "error",
                "file_path": None,
                "http_status": None,
                "error": str(e),
            }

    manifest_path = _unique_path(out_dir / f"banrep__suameca_manifest__{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    manifest_path.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    resultados["manifest_path"] = str(manifest_path)
    return resultados


def consolidar_suameca_json_a_csv_mensual(
    *,
    input_dir: str | Path | None = None,
    output_csv_path: str | Path | None = None,
    proc_dir: str | Path | None = None,
    periodicity_priority: tuple[int, ...] = (22, 9, 12, 18),
    align_start: str = "max",
    align_end: str = "min",
    fill_missing_months: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Consolida archivos suameca (JSON) a un único CSV mensual en data/proc.

    Reglas principales:
    - Convierte todo a frecuencia mensual.
    - Alinea el inicio usando el máximo de los inicios por serie (p.ej. 2005 si PIB empieza allí).
    - Los trimestrales (idPeriodicidad=12) se expanden a 3 meses con el mismo valor.
    - (Opcional) rellena meses faltantes por serie con forward-fill.
    """

    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Falta dependencia 'pandas'. Instala requirements.txt o ejecuta: pip install pandas"
        ) from e

    in_dir = Path(input_dir) if input_dir else _default_suameca_dir()
    if not in_dir.exists():
        raise FileNotFoundError(f"No existe input_dir: {in_dir}")

    json_paths = sorted(p for p in in_dir.glob("*.json") if "manifest" not in p.name.lower())
    if not json_paths:
        raise FileNotFoundError(f"No se encontraron JSON suameca en: {in_dir}")

    def _parse_series_key_from_filename(path: Path) -> str:
        stem = path.stem
        parts = stem.split("__")
        # Esperado: banrep__suameca__<key>__<id>__<timestamp>
        if len(parts) >= 4 and parts[0] == "banrep" and parts[1] == "suameca":
            return parts[2]
        return _safe_filename(stem)

    def _choose_best_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
        with_data = [r for r in records if isinstance(r, dict) and r.get("data")]
        if not with_data:
            return None

        # Intenta escoger por prioridad de periodicidad (idPeriodicidad o idTipoDato)
        for pid in periodicity_priority:
            for r in with_data:
                rid = r.get("idPeriodicidad")
                if rid is None:
                    rid = r.get("idTipoDato")
                if rid == pid:
                    return r
        return with_data[0]

    def _expand_point_to_month_periods(dt: Any, periodicity_id: int | None) -> list[Any]:
        # Normalizamos siempre a Period('M')
        if periodicity_id in (9, 22) or periodicity_id is None:
            return [dt.to_period("M")]

        # Trimestral: expandir al trimestre completo (3 meses)
        if periodicity_id == 12:
            year = int(dt.year)
            quarter_start_month = ((int(dt.month) - 1) // 3) * 3 + 1
            return [pd.Period(year=year, month=quarter_start_month + i, freq="M") for i in range(3)]

        # Anual: expandir a los 12 meses del año
        if periodicity_id == 18:
            year = int(dt.year)
            return [pd.Period(year=year, month=m, freq="M") for m in range(1, 13)]

        # Otros casos: tratar como mensual
        return [dt.to_period("M")]

    series_map: dict[str, Any] = {}
    meta_map: dict[str, Any] = {}

    for path in json_paths:
        key = _parse_series_key_from_filename(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            if verbose:
                print(f"[to_csv] skip {path.name}: no se pudo leer JSON ({e})")
            continue

        records: list[dict[str, Any]]
        if isinstance(payload, list):
            records = [r for r in payload if isinstance(r, dict)]
        elif isinstance(payload, dict):
            records = [payload]
        else:
            records = []

        best = _choose_best_record(records)
        if best is None:
            if verbose:
                print(f"[to_csv] {key}: sin datos")
            continue

        periodicity_id = best.get("idPeriodicidad")
        if periodicity_id is None:
            periodicity_id = best.get("idTipoDato")

        data = best.get("data")
        if not isinstance(data, list) or not data:
            if verbose:
                print(f"[to_csv] {key}: data vacío")
            continue

        month_to_value: dict[Any, float] = {}

        for item in data:
            if not (isinstance(item, list) or isinstance(item, tuple)) or len(item) < 2:
                continue
            ms, val = item[0], item[1]
            try:
                dt = pd.to_datetime(ms, unit="ms", utc=True).tz_convert(None)
            except Exception:
                continue
            try:
                v = float(val) if val is not None else float("nan")
            except Exception:
                v = float("nan")

            for mp in _expand_point_to_month_periods(dt, int(periodicity_id) if periodicity_id is not None else None):
                # Si hay duplicados, conservar el último visto (típicamente el más reciente en el JSON)
                month_to_value[mp] = v

        if not month_to_value:
            if verbose:
                print(f"[to_csv] {key}: no se pudo construir serie mensual")
            continue

        s = pd.Series(month_to_value).sort_index()
        s.name = key
        # Asegurar un solo valor por mes (por si quedó repetido por inputs diarios)
        s = s.groupby(level=0).last()

        series_map[key] = s
        meta_map[key] = {
            "source_file": str(path),
            "id": best.get("id"),
            "nombre": best.get("nombre"),
            "unidad": best.get("unidad"),
            "idPeriodicidad": periodicity_id,
            "descripcionPeriodicidad": best.get("descripcionPeriodicidad"),
            "descripcionTipoDato": best.get("descripcionTipoDato"),
        }

        if verbose:
            print(
                f"[to_csv] {key}: periodicidad={periodicity_id} meses={len(s)} "
                f"({s.index.min()}..{s.index.max()})"
            )

    if not series_map:
        raise RuntimeError("No se pudo construir ninguna serie mensual desde los JSON suameca")

    # Alinear rango temporal
    starts = [s.index.min() for s in series_map.values()]
    ends = [s.index.max() for s in series_map.values()]
    global_start = max(starts) if align_start == "max" else min(starts)
    global_end = min(ends) if align_end == "min" else max(ends)
    if global_start > global_end:
        raise RuntimeError(
            f"Rango inválido al alinear series: start={global_start} end={global_end}. "
            "Revisa periodicidades/series disponibles."
        )

    full_index = pd.period_range(global_start, global_end, freq="M")
    df = pd.DataFrame(index=full_index)
    for key, s in series_map.items():
        col = s.reindex(full_index)
        if fill_missing_months:
            col = col.ffill()
        df[key] = col

    # Índice como timestamp (primer día del mes)
    df.index = df.index.to_timestamp(how="start")
    df.index.name = "date"

    # Determinar path de salida
    proc_dir_path = Path(proc_dir) if proc_dir else _default_proc_dir()
    proc_dir_path.mkdir(parents=True, exist_ok=True)

    if output_csv_path:
        out_path = Path(output_csv_path)
        if not out_path.is_absolute():
            out_path = proc_dir_path / out_path
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = proc_dir_path / f"banrep__suameca__monthly__{ts}.csv"

    out_path = _unique_path(out_path)
    df.to_csv(out_path, index=True)

    return {
        "status": "ok",
        "input_dir": str(in_dir),
        "proc_dir": str(proc_dir_path),
        "output_csv_path": str(out_path),
        "align_start": align_start,
        "align_end": align_end,
        "global_start": str(global_start),
        "global_end": str(global_end),
        "n_months": int(df.shape[0]),
        "series": meta_map,
    }

def main():
    """CLI no-interactiva (apta para correr en scheduler)."""

    parser = argparse.ArgumentParser(description="Extractor de series (Banrep) hacia data/raw")
    parser.add_argument("--method", choices=["selenium", "requests", "suameca", "suameca_to_csv"], default="requests")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--output-dir", default=str(_default_raw_dir()))
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--category", choices=["hist", "latest"], default="hist")
    parser.add_argument("--start-period", default=None)
    parser.add_argument("--end-period", default=None)
    parser.add_argument("--http-timeout", type=int, default=180)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--template-json", default=None, help="Ruta a JSON del POST suameca con placeholders")
    parser.add_argument("--cookie", default=None, help="Header Cookie (copiar desde cURL si el backend lo exige)")
    parser.add_argument("--referer", default=None, help="Referer (por defecto: /descarga-multiple-de-datos/consolidado)")
    parser.add_argument("--proc-dir", default=str(_default_proc_dir()), help="Directorio de salida para data/proc")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Nombre o ruta del CSV consolidado (por defecto se genera en data/proc)",
    )
    args = parser.parse_args()

    if args.method == "requests":
        resultados = extraer_con_requests(
            output_dir=args.output_dir,
            category=args.category,
            start_period=args.start_period,
            end_period=args.end_period,
            http_timeout_s=args.http_timeout,
            verbose=args.verbose,
        )
        print(json.dumps(resultados, ensure_ascii=False, indent=2))
        return

    if args.method == "suameca":
        resultados = extraer_suameca_sin_api(
            output_dir=args.output_dir,
            template_json_path=args.template_json,
            start_period=args.start_period,
            end_period=args.end_period,
            http_timeout_s=args.http_timeout,
            verbose=args.verbose,
            cookie_header=args.cookie,
            referer=args.referer,
        )
        print(json.dumps(resultados, ensure_ascii=False, indent=2))
        return

    if args.method == "suameca_to_csv":
        resultados = consolidar_suameca_json_a_csv_mensual(
            input_dir=args.output_dir,
            proc_dir=args.proc_dir,
            output_csv_path=args.output_csv,
            verbose=args.verbose,
        )
        print(json.dumps(resultados, ensure_ascii=False, indent=2))
        return

    extractor = BanrepDataExtractor(headless=args.headless, output_dir=args.output_dir, timeout_s=args.timeout)
    resultados = extractor.extraer_datos_completos()
    print(json.dumps(resultados, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

