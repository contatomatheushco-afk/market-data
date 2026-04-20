"""
update_market_data.py
─────────────────────────────────────────────────────────────
Script de atualização diária de dados de mercado brasileiro.

Fontes de dados:
  • BCB / SGS  → CDI, SELIC, IPCA, IGPM
  • CVM        → Cotas de fundos de investimento
  • Yahoo Finance → Ações, ETFs e FIIs listados na B3
  • Tesouro Nacional → Preços e taxas do Tesouro Direto

Arquivos gerados (pasta /data):
  indices.csv, fundos.csv, acoes_etfs.csv,
  tesouro.csv, ultima_atualizacao.json
─────────────────────────────────────────────────────────────
"""

import os
import json
import time
import logging
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
import holidays

# ── Configuração de log ───────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Diretório de saída ────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# UTILITÁRIO: verificação de dia útil
# ─────────────────────────────────────────────────────────────
def is_business_day(d: date = None) -> bool:
    if d is None:
        d = date.today()
    br_holidays = holidays.Brazil()
    return d.weekday() < 5 and d not in br_holidays


# ═════════════════════════════════════════════════════════════
# 1. ÍNDICES MACRO — BCB / SGS
#    CDI, SELIC, IPCA, IGPM
# ═════════════════════════════════════════════════════════════

SERIES_BCB = {
    "CDI_diario":  12,
    "SELIC_meta":  432,
    "IPCA_mensal": 433,
    "IGPM_mensal": 189,
}


def fetch_bcb_series(codigo: int, nome: str, dias: int = 90) -> pd.DataFrame:
    """Baixa os últimos `dias` registros de uma série do BCB SGS."""
    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}"
        f"/dados/ultimos/{dias}?formato=json"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        dados = r.json()
        df = pd.DataFrame(dados)
        if df.empty:
            return pd.DataFrame()
        df["data"] = pd.to_datetime(df["data"], dayfirst=True).dt.date
        df["valor"] = pd.to_numeric(
            df["valor"].astype(str).str.replace(",", "."), errors="coerce"
        )
        df["serie"] = nome
        return df[["data", "serie", "valor"]]
    except Exception as e:
        log.warning(f"BCB {nome} ({codigo}): {e}")
        return pd.DataFrame()


def update_indices():
    """Atualiza indices.csv com CDI, SELIC, IPCA e IGPM."""
    log.info("Atualizando índices macro (BCB)...")
    arquivo = DATA_DIR / "indices.csv"

    dfs = []
    for nome, codigo in SERIES_BCB.items():
        df = fetch_bcb_series(codigo, nome, dias=90)
        if not df.empty:
            dfs.append(df)
            log.info(f"  {nome}: {len(df)} registros obtidos")
        time.sleep(0.5)  # respeita rate limit da BCB

    if not dfs:
        log.error("Nenhum dado do BCB obtido.")
        return

    novo = pd.concat(dfs, ignore_index=True)

    if arquivo.exists():
        existente = pd.read_csv(arquivo)
        existente["data"] = pd.to_datetime(existente["data"]).dt.date
        combinado = pd.concat([existente, novo], ignore_index=True)
        combinado.drop_duplicates(subset=["data", "serie"], keep="last", inplace=True)
    else:
        combinado = novo

    combinado.sort_values(["serie", "data"], inplace=True)
    combinado.to_csv(arquivo, index=False)
    log.info(f"indices.csv salvo — {len(combinado)} linhas no total")


# ═════════════════════════════════════════════════════════════
# 2. COTAS DE FUNDOS — CVM
# ═════════════════════════════════════════════════════════════

def load_cnpjs() -> list:
    """Lê os CNPJs do arquivo cnpjs_monitorados.txt."""
    arq = Path(__file__).parent / "cnpjs_monitorados.txt"
    if not arq.exists():
        log.warning("cnpjs_monitorados.txt não encontrado. Pulando fundos.")
        return []
    linhas = arq.read_text(encoding="utf-8").splitlines()
    cnpjs_raw = [l.strip() for l in linhas if l.strip() and not l.startswith("#")]
    # Normaliza: remove pontuação, mantém só dígitos
    cnpjs = ["".join(filter(str.isdigit, c)) for c in cnpjs_raw]
    cnpjs = [c for c in cnpjs if len(c) == 14]
    log.info(f"CNPJs carregados: {len(cnpjs)}")
    return cnpjs


def fetch_cotas_cvm(ano: int, mes: int, cnpjs: list) -> pd.DataFrame:
    """Baixa o arquivo mensal de cotas da CVM e filtra os CNPJs desejados."""
    url = (
        f"https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/"
        f"inf_diario_fi_{ano}{mes:02d}.csv"
    )
    try:
        df = pd.read_csv(
            url, sep=";", encoding="latin1", dtype={"CNPJ_FUNDO": str}, low_memory=False
        )
        df.columns = df.columns.str.strip()

        # Normaliza CNPJ para comparação
        df["CNPJ_CLEAN"] = df["CNPJ_FUNDO"].str.replace(r"\D", "", regex=True)
        df = df[df["CNPJ_CLEAN"].isin(cnpjs)].copy()

        if df.empty:
            return pd.DataFrame()

        rename_map = {
            "CNPJ_FUNDO":       "cnpj",
            "DT_COMPTC":        "data",
            "VL_QUOTA":         "cota",
            "VL_PATRIM_LIQ":    "patrimonio_liquido",
            "NR_COTST":         "num_cotistas",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        colunas = ["cnpj", "data", "cota", "patrimonio_liquido", "num_cotistas"]
        colunas_existentes = [c for c in colunas if c in df.columns]
        df = df[colunas_existentes].copy()

        df["data"] = pd.to_datetime(df["data"], errors="coerce").dt.date
        df["cota"] = pd.to_numeric(
            df["cota"].astype(str).str.replace(",", "."), errors="coerce"
        )
        return df

    except Exception as e:
        log.warning(f"CVM {ano}/{mes:02d}: {e}")
        return pd.DataFrame()


def update_fundos():
    """Atualiza fundos.csv com os 2 últimos meses da CVM."""
    cnpjs = load_cnpjs()
    if not cnpjs:
        return

    log.info(f"Atualizando cotas de {len(cnpjs)} fundo(s) (CVM)...")
    arquivo = DATA_DIR / "fundos.csv"

    hoje = date.today()
    meses = [(hoje.year, hoje.month)]
    # Inclui mês anterior (arquivo do mês corrente pode estar incompleto no início)
    mes_ant = hoje.month - 1 if hoje.month > 1 else 12
    ano_ant = hoje.year if hoje.month > 1 else hoje.year - 1
    meses.append((ano_ant, mes_ant))

    dfs = []
    for ano, mes in meses:
        df = fetch_cotas_cvm(ano, mes, cnpjs)
        if not df.empty:
            dfs.append(df)
            log.info(f"  CVM {ano}/{mes:02d}: {len(df)} registros")

    if not dfs:
        log.warning("Nenhuma cota de fundo obtida.")
        return

    novo = pd.concat(dfs, ignore_index=True)

    if arquivo.exists():
        existente = pd.read_csv(arquivo)
        existente["data"] = pd.to_datetime(existente["data"]).dt.date
        combinado = pd.concat([existente, novo], ignore_index=True)
        combinado.drop_duplicates(subset=["cnpj", "data"], keep="last", inplace=True)
    else:
        combinado = novo

    combinado.sort_values(["cnpj", "data"], inplace=True)
    combinado.to_csv(arquivo, index=False)
    log.info(f"fundos.csv salvo — {len(combinado)} linhas no total")


# ═════════════════════════════════════════════════════════════
# 3. AÇÕES, ETFs E FIIs — Yahoo Finance
# ═════════════════════════════════════════════════════════════

def load_tickers() -> list:
    """Lê os tickers do arquivo tickers_monitorados.txt."""
    arq = Path(__file__).parent / "tickers_monitorados.txt"
    if not arq.exists():
        log.warning("tickers_monitorados.txt não encontrado. Pulando ações/ETFs.")
        return []
    linhas = arq.read_text(encoding="utf-8").splitlines()
    tickers = [l.strip().upper() for l in linhas if l.strip() and not l.startswith("#")]
    log.info(f"Tickers carregados: {len(tickers)}")
    return tickers


def update_acoes():
    """Atualiza acoes_etfs.csv com os últimos 60 dias de cotações."""
    tickers = load_tickers()
    if not tickers:
        return

    log.info(f"Atualizando {len(tickers)} ticker(s) (Yahoo Finance)...")
    arquivo = DATA_DIR / "acoes_etfs.csv"

    hoje = date.today()
    inicio = (hoje - timedelta(days=60)).strftime("%Y-%m-%d")

    dfs = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(start=inicio, auto_adjust=True)
            if hist.empty:
                log.warning(f"  {ticker}: sem dados retornados")
                continue
            hist = hist.reset_index()
            hist["ticker"] = ticker
            hist["data"] = hist["Date"].dt.date
            hist = hist.rename(columns={"Close": "fechamento", "Volume": "volume"})
            hist = hist[["ticker", "data", "fechamento", "volume"]]
            dfs.append(hist)
            log.info(f"  {ticker}: {len(hist)} registros")
        except Exception as e:
            log.warning(f"  {ticker}: {e}")

    if not dfs:
        log.warning("Nenhum dado de ações/ETFs obtido.")
        return

    novo = pd.concat(dfs, ignore_index=True)

    if arquivo.exists():
        existente = pd.read_csv(arquivo)
        existente["data"] = pd.to_datetime(existente["data"]).dt.date
        combinado = pd.concat([existente, novo], ignore_index=True)
        combinado.drop_duplicates(subset=["ticker", "data"], keep="last", inplace=True)
    else:
        combinado = novo

    combinado.sort_values(["ticker", "data"], inplace=True)
    combinado.to_csv(arquivo, index=False)
    log.info(f"acoes_etfs.csv salvo — {len(combinado)} linhas no total")


# ═════════════════════════════════════════════════════════════
# 4. TESOURO DIRETO — Tesouro Nacional
# ═════════════════════════════════════════════════════════════

TESOURO_URL = (
    "https://www.tesourotransparente.gov.br/ckan/dataset/"
    "df56aa42-484a-4a59-8184-7676580c81e3/resource/"
    "796d2059-14e9-44e3-80a7-2dff9149b7fc/download/PrecoTaxaTesouroDireto.csv"
)


def update_tesouro():
    """Atualiza tesouro.csv com preços e taxas do Tesouro Direto (últimos 365 dias)."""
    log.info("Atualizando Tesouro Direto...")
    arquivo = DATA_DIR / "tesouro.csv"

    try:
        df = pd.read_csv(TESOURO_URL, sep=";", encoding="latin1", decimal=",")
        df.columns = df.columns.str.strip()

        rename_map = {
            "Tipo Titulo":        "titulo",
            "Data Vencimento":    "vencimento",
            "Data Base":          "data",
            "Taxa Compra Manha":  "taxa_compra",
            "Taxa Venda Manha":   "taxa_venda",
            "PU Compra Manha":    "pu_compra",
            "PU Venda Manha":     "pu_venda",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        if "data" in df.columns:
            df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce").dt.date
            cutoff = date.today() - timedelta(days=365)
            df = df[df["data"] >= cutoff]

        df.sort_values(["titulo", "data"], inplace=True)
        df.to_csv(arquivo, index=False)
        log.info(f"tesouro.csv salvo — {len(df)} linhas no total")

    except Exception as e:
        log.error(f"Tesouro Direto: {e}")


# ═════════════════════════════════════════════════════════════
# 5. METADADOS
# ═════════════════════════════════════════════════════════════

def save_metadata():
    """Registra data/hora da última atualização bem-sucedida."""
    meta = {
        "ultima_atualizacao": datetime.now().isoformat(),
        "data": date.today().isoformat(),
        "fontes": ["BCB/SGS", "CVM", "Yahoo Finance", "Tesouro Nacional"],
    }
    (DATA_DIR / "ultima_atualizacao.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Metadados salvos em ultima_atualizacao.json")


# ═════════════════════════════════════════════════════════════
# EXECUÇÃO PRINCIPAL
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("ATUALIZAÇÃO DIÁRIA DE DADOS DE MERCADO")
    log.info(f"Data: {date.today()} | Dia útil: {is_business_day()}")
    log.info("=" * 60)

    if not is_business_day():
        log.info("Hoje é feriado nacional. Nenhuma atualização necessária.")
        raise SystemExit(0)

    update_indices()
    update_fundos()
    update_acoes()
    update_tesouro()
    save_metadata()

    log.info("=" * 60)
    log.info("ATUALIZAÇÃO CONCLUÍDA COM SUCESSO")
    log.info("=" * 60)
