import joblib
import xgboost as xgb
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, r2_score
from streamlit_option_menu import option_menu
import requests
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

TJRIBA_LAT = 35.76
TJRIBA_LON = -5.80

_P_STC    = 545
_G_STC    = 1000
_GAMMA_P  = -0.0034
_T_STC    = 25.0
_N_PANELS = 3166
_CO2_KWHE = 0.604

@st.cache_data(ttl=3600)
def fetch_day_meteo_openmeteo(date):
    """
    Récupère les 24h de T° (°C) et irradiance (W/m²) pour une date donnée via Open-Meteo.
    - Dates récentes (≤7 j) ou futures : API Forecast avec past_days (pas de délai).
    - Dates anciennes (>7 j)           : API Archive (ERA5, lag ~5-7 j).
    Retourne DataFrame(heure, T_moy, Irr_moy) ou None si échec.
    """
    date_str  = pd.Timestamp(date).strftime("%Y-%m-%d")
    today     = datetime.today().date()
    _date_obj = pd.Timestamp(date).date()
    _days_ago = (today - _date_obj).days if _date_obj <= today else -1

    if _days_ago < 0:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": TJRIBA_LAT, "longitude": TJRIBA_LON,
            "hourly": "temperature_2m,shortwave_radiation",
            "timezone": "Africa/Casablanca",
            "start_date": date_str, "end_date": date_str,
        }
    elif _days_ago <= 7:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": TJRIBA_LAT, "longitude": TJRIBA_LON,
            "hourly": "temperature_2m,shortwave_radiation",
            "timezone": "Africa/Casablanca",
            "past_days": _days_ago + 1,
            "forecast_days": 1,
        }
    else:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": TJRIBA_LAT, "longitude": TJRIBA_LON,
            "hourly": "temperature_2m,shortwave_radiation",
            "timezone": "Africa/Casablanca",
            "start_date": date_str, "end_date": date_str,
        }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data  = resp.json()
        times = data["hourly"]["time"]
        temps = data["hourly"]["temperature_2m"]
        irrs  = data["hourly"]["shortwave_radiation"]
        rows  = []
        for i, t_str in enumerate(times):
            if not t_str.startswith(date_str):
                continue
            h = int(t_str.split("T")[1].split(":")[0])
            rows.append({
                "heure":   h,
                "T_moy":   float(temps[i]) if temps[i] is not None else 25.0,
                "Irr_moy": max(0.0, float(irrs[i]) if irrs[i] is not None else 0.0),
            })
        if not rows:
            return None
        return pd.DataFrame(rows)
    except Exception:
        return None

def fetch_meteo_openmeteo(date, hour):
    """Retourne (temp °C, irradiance W/m²) pour une date+heure via Open-Meteo (avec cache)."""
    df_day = fetch_day_meteo_openmeteo(date)
    if df_day is None:
        return None, None
    row = df_day[df_day["heure"] == hour]
    if len(row) == 0:
        return None, None
    return float(row["T_moy"].values[0]), float(row["Irr_moy"].values[0])

st.set_page_config(layout="wide", page_title="SolarVision PV · Tjriba", page_icon="☀️")

def check_alerts(df_current, df_historical):
    """
    Alertes statistiques :
    - Production : μ − 2σ  par heure × saison (données historiques)
    - Température : P95 historique en heures de production
    - Écrêtage   : μ + 2σ  par saison
    """
    alerts = []
    if len(df_current) == 0 or len(df_historical) == 0:
        return alerts

    _excl_myt = (df_historical["Date"].dt.year == 2025) & (df_historical["Date"].dt.month == 8)
    _ETE_MONTHS  = [4, 5, 6, 7, 8, 9]
    _HIVER_MONTHS = [10, 11, 12, 1, 2, 3]

    def _is_ete(month):
        return month in _ETE_MONTHS

    prod_alerts = []
    for _, row in df_current.iterrows():
        h = row["Date"].hour
        m = row["Date"].month
        season_months = _ETE_MONTHS if _is_ete(m) else _HIVER_MONTHS
        mask_hist = (
            (df_historical["Date"].dt.hour == h) &
            (df_historical["Date"].dt.month.isin(season_months)) &
            (df_historical["Irradiance_Wpm2"] > 20) &
            ~_excl_myt
        )
        hist_prod = df_historical[mask_hist]["Production apres limitation [kWh]"].dropna()
        if len(hist_prod) < 5:
            continue
        mu    = hist_prod.mean()
        sigma = hist_prod.std()
        seuil = mu - 2.0 * sigma
        val   = float(row.get("Production apres limitation [kWh]", 0) or 0)
        if mu > 0.05 and val < seuil:
            prod_alerts.append({
                "heure": h, "valeur": val,
                "mu": mu, "sigma": sigma, "seuil": seuil,
                "ecart_pct": (val - mu) / mu * 100
            })
    if prod_alerts:
        worst = min(prod_alerts, key=lambda x: x["ecart_pct"])
        alerts.append({
            "type": "production", "severity": "high", "emoji": "📉",
            "title": "Production Anormalement Faible",
            "message": (
                f"Production à {worst['heure']:02d}h : **{worst['valeur']:.1f} kWh** "
                f"({worst['ecart_pct']:.1f}% vs réf. historique {worst['mu']:.1f} kWh) "
                f"— Seuil μ−2σ : {worst['seuil']:.1f} kWh "
                f"(μ={worst['mu']:.1f}, σ={worst['sigma']:.1f})"
            )
        })

    mask_temp = (df_historical["Irradiance_Wpm2"] > 20) & ~_excl_myt
    hist_temps = df_historical[mask_temp]["Temperature_C"].dropna()
    if len(hist_temps) >= 10:
        p95_temp = float(np.percentile(hist_temps, 95))
        max_temp_cur = float(df_current["Temperature_C"].max())
        if max_temp_cur > p95_temp:
            alerts.append({
                "type": "temperature", "severity": "high", "emoji": "🔥",
                "title": "Température Cellule Anormale",
                "message": (
                    f"Température max : **{max_temp_cur:.1f}°C** "
                    f"(seuil P95 historique : {p95_temp:.1f}°C) "
                    f"— Risque de dégradation des performances et du matériel"
                )
            })
    else:
        max_temp_cur = float(df_current["Temperature_C"].max())
        if max_temp_cur > 70:
            alerts.append({
                "type": "temperature", "severity": "high", "emoji": "🔥",
                "title": "Température Élevée",
                "message": f"Température cellule : {max_temp_cur:.1f}°C (seuil fixe : 70°C)"
            })

    total_avant  = df_current["Production avant limitation Kwh"].sum()
    total_ecrete = df_current["Gap_kWh"].sum()
    curtail_pct  = (total_ecrete / total_avant * 100) if total_avant > 0 else 0.0

    if len(df_current) > 0:
        dom_month    = int(df_current["Date"].dt.month.mode().iloc[0])
        season_months = _ETE_MONTHS if _is_ete(dom_month) else _HIVER_MONTHS
        saison_label  = "Été" if _is_ete(dom_month) else "Hiver"

        mask_s = (
            df_historical["Date"].dt.month.isin(season_months) &
            (df_historical["Production avant limitation Kwh"] > 0) &
            ~_excl_myt
        )
        df_hist_s = df_historical[mask_s].copy()
        if len(df_hist_s) >= 10:
            df_hist_s["_rate"] = np.where(
                df_hist_s["Production avant limitation Kwh"] > 0,
                df_hist_s["Gap_kWh"] / df_hist_s["Production avant limitation Kwh"] * 100,
                0.0
            )
            mu_ecr    = df_hist_s["_rate"].mean()
            sigma_ecr = df_hist_s["_rate"].std()
            seuil_ecr = mu_ecr + 2.0 * sigma_ecr
            if curtail_pct > seuil_ecr:
                alerts.append({
                    "type": "curtailment", "severity": "medium", "emoji": "⛔",
                    "title": "Écrêtage Anormalement Élevé",
                    "message": (
                        f"Taux d'écrêtage : **{curtail_pct:.1f}%** "
                        f"(seuil μ+2σ {saison_label} : {seuil_ecr:.1f}% ; "
                        f"μ={mu_ecr:.1f}%, σ={sigma_ecr:.1f}%) "
                        f"— {total_ecrete:.1f} kWh perdus · Voir page Scénarios"
                    )
                })

    return alerts

def display_alerts(alerts, theme=None):
    """Affiche les alertes avec des cartes HTML stylisées."""
    if theme is None:
        theme = {
            "card_bg": "#fff", "text_primary": "#1a1a1a",
            "success": "#43A047", "warning": "#FB8C00", "error": "#E53935",
        }
    c_ok  = theme.get("success", "#43A047")
    c_err = theme.get("error",   "#E53935")
    c_wrn = theme.get("warning", "#FB8C00")
    txt   = theme.get("text_primary", "#1a1a1a")

    if not alerts:
        st.markdown(f"""
        <div style="
            background:{c_ok}15;
            border-left:5px solid {c_ok};
            border-radius:10px;
            padding:14px 20px;
            display:flex;align-items:center;gap:14px;
            margin:6px 0;
        ">
            <span style="font-size:26px;">✅</span>
            <div>
                <div style="font-weight:700;font-size:15px;color:{c_ok};">
                    Système Opérationnel
                </div>
                <div style="font-size:13px;color:{txt};opacity:0.8;margin-top:3px;">
                    Aucune anomalie détectée — Fonctionnement normal
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
        return

    cards = ""
    for alert in alerts:
        color  = c_err if alert.get("severity") == "high" else c_wrn
        badge  = "CRITIQUE" if alert.get("severity") == "high" else "ATTENTION"
        cards += f"""
        <div style="
            background:{color}12;
            border-left:5px solid {color};
            border-radius:10px;
            padding:14px 20px;
            margin:10px 0;
            box-shadow:0 2px 10px {color}20;
        ">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:7px;">
                <span style="font-size:24px;">{alert['emoji']}</span>
                <span style="font-weight:700;font-size:15px;color:{color};">
                    {alert['title']}
                </span>
                <span style="
                    background:{color};color:#fff;
                    font-size:10px;font-weight:700;
                    padding:2px 9px;border-radius:20px;
                    letter-spacing:0.6px;
                ">{badge}</span>
            </div>
            <div style="font-size:13px;color:{txt};opacity:0.85;padding-left:34px;line-height:1.5;">
                {alert['message']}
            </div>
        </div>"""

    st.markdown(cards, unsafe_allow_html=True)

def create_download_button(df_export, filename, label="📥 Télécharger les données (CSV)", key=None):
    """Bouton de téléchargement CSV placé au-dessus d'un graphique"""
    csv = df_export.to_csv(index=False).encode('utf-8')
    st.download_button(label=label, data=csv, file_name=filename, mime='text/csv', key=key)

def generate_html_report(df, df_period, pred_date, temp_input, irrad_input, data_exists):
    """Génère un rapport HTML téléchargeable et imprimable en PDF."""
    total_prod = df_period["Production apres limitation [kWh]"].sum() if len(df_period) > 0 else 0
    total_theo = df_period["PV_theoretical_kWh"].sum() if len(df_period) > 0 else 0
    rendement = (total_prod / total_theo * 100) if total_theo > 0 else 0
    total_ecrete = df_period["Gap_kWh"].sum() if len(df_period) > 0 else 0
    temp_moy = df_period["Temperature_C"].mean() if len(df_period) > 0 else temp_input
    irr_moy = df_period["Irradiance_Wpm2"].mean() if len(df_period) > 0 else irrad_input
    total_self = df_period["SelfCons_kWh"].sum() if len(df_period) > 0 else 0
    co2_evite = total_prod * _CO2_KWHE
    rows_html = ""
    if len(df_period) > 0:
        for _, row in df_period.iterrows():
            rows_html += f"<tr><td>{row['Date'].strftime('%d/%m/%Y %H:%M')}</td><td>{row['Production apres limitation [kWh]']:.3f}</td><td>{row['Irradiance_Wpm2']:.0f}</td><td>{row['Temperature_C']:.1f}</td></tr>"
    html = f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<title>Rapport PV - {pred_date.strftime('%d/%m/%Y')}</title>
<style>
  body{{font-family:Arial,sans-serif;margin:30px auto;max-width:900px;color:#333;}}
  h1{{color:#E65100;border-bottom:3px solid #FF9800;padding-bottom:10px;}}
  h2{{color:#333;border-bottom:1px solid #FF9800;padding-bottom:6px;margin-top:28px;}}
  .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:16px 0;}}
  .card{{background:#fff3e0;border:2px solid #FFB74D;border-radius:8px;padding:14px;text-align:center;}}
  .card-value{{font-size:20px;font-weight:bold;color:#E65100;}}
  .card-label{{font-size:11px;color:#666;margin-top:4px;text-transform:uppercase;}}
  table{{width:100%;border-collapse:collapse;margin:14px 0;font-size:13px;}}
  th{{background:#FF9800;color:white;padding:9px;text-align:left;}}
  td{{padding:7px 9px;border-bottom:1px solid #eee;}}
  tr:nth-child(even){{background:#FFF8F0;}}
  .footer{{margin-top:36px;font-size:11px;color:#aaa;border-top:1px solid #eee;padding-top:12px;}}
  @media print{{body{{margin:10px;}}}}
</style></head><body>
<h1>☀️ Rapport Système Photovoltaïque</h1>
<p><b>Généré le:</b> {datetime.now().strftime('%d/%m/%Y à %H:%M')} | <b>Période:</b> 7 jours avant le {pred_date.strftime('%d/%m/%Y')}</p>
<h2>📊 Résumé de Production (7 jours)</h2>
<div class="grid">
  <div class="card"><div class="card-value">{total_prod:.1f} kWh</div><div class="card-label">Production Réelle</div></div>
  <div class="card"><div class="card-value">{total_theo:.1f} kWh</div><div class="card-label">Production Théorique</div></div>
  <div class="card"><div class="card-value">{rendement:.1f}%</div><div class="card-label">Rendement Global</div></div>
  <div class="card"><div class="card-value">{total_ecrete:.1f} kWh</div><div class="card-label">Énergie Écrêtée</div></div>
  <div class="card"><div class="card-value">{total_self:.1f} kWh</div><div class="card-label">Autoconsommation</div></div>
  <div class="card"><div class="card-value">{co2_evite:.1f} kg</div><div class="card-label">CO₂ Évité</div></div>
</div>
<h2>☀️ Conditions Météo</h2>
<table>
  <tr><th>Paramètre</th><th>Valeur actuelle</th><th>Moyenne 7 jours</th></tr>
  <tr><td>Température</td><td>{temp_input:.1f} °C</td><td>{temp_moy:.1f} °C</td></tr>
  <tr><td>Irradiance POA</td><td>{irrad_input:.1f} W/m²</td><td>{irr_moy:.1f} W/m²</td></tr>
  <tr><td>Statut données</td><td colspan="2">{"Données réelles historiques" if data_exists else "Prédictions ML"}</td></tr>
</table>
<h2>🏭 Informations Système</h2>
<table>
  <tr><th>Paramètre</th><th>Valeur</th></tr>
  <tr><td>Nombre de panneaux</td><td>{_N_PANELS:,} panneaux</td></tr>
  <tr><td>Puissance unitaire</td><td>{_P_STC} Wc</td></tr>
  <tr><td>Puissance crête totale</td><td>{_N_PANELS * _P_STC / 1000:.1f} kWc</td></tr>
  <tr><td>Plage de données</td><td>{df["Date"].min().strftime('%d/%m/%Y')} — {df["Date"].max().strftime('%d/%m/%Y')}</td></tr>
</table>
<div class="footer">
  <p>Rapport généré — Tableau de bord SCADA PV | Pour imprimer en PDF : Fichier &gt; Imprimer &gt; Enregistrer au format PDF</p>
</div>
</body></html>"""
    return html.encode("utf-8")

GRANDEURS_INFO = {
    "Temperature_C": {
        "type": "MESURÉE",
        "unité": "°C",
        "source": "Capteur de température",
        "description": "Température de la cellule photovoltaïque"
    },
    "Irradiance_Wpm2": {
        "type": "MESURÉE",
        "unité": "W/m²",
        "source": "Pyranomètre POA",
        "description": "Irradiance plane incidée perpendiculairement aux panneaux"
    },
    "Production avant limitation": {
        "type": "CALCULÉE",
        "unité": "kWh",
        "source": "Calcul temps réel",
        "description": "Énergie produite AVANT écrêtage (limiteur)"
    },
    "Production apres limitation": {
        "type": "MESURÉE",
        "unité": "kWh",
        "source": "Compteur",
        "description": "Énergie effectivement livrée APRÈS écrêtage"
    },
    "Production théorique": {
        "type": "CALCULÉE",
        "unité": "kWh",
        "source": "Modèle physique",
        "description": "Production théorique selon conditions physiques"
    },
    "Énergie Écrêtée": {
        "type": "CALCULÉE",
        "unité": "kWh",
        "source": "Différence (Production avant - Production après)",
        "description": "Énergie perdue à cause de l'écrêtage"
    },
}

KPI_INFO = {
    "Production Totale (kWh)": "Σ Production après limitation sur la journée sélectionnée",
    "Production Moyenne (kWh)": "Moyenne horaire de la production après limitation sur la journée",
    "Production Max (kWh)": "Pic horaire de production après limitation sur la journée",
    "Autoconsommation Totale (kWh)": "Σ min(Production APRÈS, Consommation) — énergie PV consommée sur site",
    "Taux Autoconsommation (%)": "Autoconsommation / Production APRÈS × 100",
    "Consommation Réseau Totale (kWh)": "Σ max(0, Consommation − Production APRÈS) — énergie achetée à l'ONEE",
    "Consommation Totale (kWh)": "Σ Consommation générale de l'usine sur la journée",
    "Irradiance Moyenne (W/m²)": "Irradiance POA moyenne sur les heures de la journée",
    "Température Moyenne (°C)": "Température cellule PV moyenne sur la journée",
    "Énergie Écrêtée Totale (kWh)": "Σ (Production AVANT − Production APRÈS) — énergie perdue par écrêtage",
    "Rendement Global (%)": "Production AVANT / Production théorique × 100",
    "Taux Couverture Solaire (%)": "Production APRÈS / Consommation totale × 100",
    "Autonomie Énergétique (%)": "Autoconsommation / Consommation totale × 100",
}

def get_theme_colors(is_dark_mode):
    if is_dark_mode:
        return {
            'bg_primary': 'linear-gradient(135deg, #0F0F12 0%, #16161C 50%, #111318 100%)',
            'text_primary': '#FFE0A3',
            'accent': '#FFB74D',
            'accent_bright': '#FFC107',
            'border': '#FF9800',
            'card_bg': 'rgba(26, 21, 32, 0.6)',
            'card_border': '#FF9800',
            'success': '#66BB6A',
            'warning': '#FFA726',
            'error': '#EF5350',
        }
    else:
        return {
            'bg_primary': 'linear-gradient(135deg, #fffbf0 0%, #fff3e0 50%, #ffeaa7 100%)',
            'text_primary': '#1a1a1a',
            'accent': '#FF9800',
            'accent_bright': '#1A1A1A',
            'border': '#FF9800',
            'card_bg': '#FFFFFF',
            'card_border': '#FF9800',
            'success': '#43A047',
            'warning': '#FB8C00',
            'error': '#E53935',
        }

def get_plot_bg(is_dark):
    if is_dark:
        return {}
    return {
        "paper_bgcolor": "#FFF8F0",
        "plot_bgcolor": "#FFF8F0",
        "font_color": "#1A1A1A",
        "legend_bgcolor": "rgba(255,255,255,0.92)",
        "legend_bordercolor": "rgba(0,0,0,0.12)",
        "legend_borderwidth": 1,
        "legend_font_color": "#1A1A1A",
    }

def get_curve_colors(is_dark):
    if is_dark:
        return dict(
            reel="#FFC107", theorique="#8D6E63", xgb="#42A5F5", lstm="#66BB6A",
            avant="#FFB74D", ecrete="#FF5722", conso="#009688", conso_pred="#AB47BC",
            reseau="#2196F3", autoconso="#9C27B0", irr="#FF9800", temp="#F44336",
        )
    return dict(
        reel="#E65100", theorique="#8E24AA", xgb="#1E88E5", lstm="#43A047",
        avant="#E64A19", ecrete="#F4511E", conso="#00838F", conso_pred="#5E35B1",
        reseau="#1565C0", autoconso="#7B1FA2", irr="#E65100", temp="#E53935",
    )

def show_documentation():
    st.markdown("# 📚 Documentation - Grandeurs, KPI & Scénarios")
    st.markdown("---")

    st.markdown("""
<style>
div[data-testid="stTextInput"] input {
    font-size: 17px !important;
    padding: 14px 20px !important;
    height: 54px !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)
    _q = st.text_input(
        "🔍",
        placeholder="Entrer...",
        label_visibility="collapsed",
        key="doc_search"
    ).strip().lower()

    def _match(*texts):
        return any(_q in str(t).lower() for t in texts)

    _show_all = (_q == "")
    _found = False

    _ve_eqs = [
        ("1", "Énergie requise",
         "E_requise = N_VE × C_bat_VE",
         "Énergie totale nécessaire pour charger tous les VE (kWh). "
         "N_VE : nombre de véhicules | C_bat_VE : capacité batterie d'un VE (kWh)"),
        ("2", "Puissance de charge disponible",
         "P_charge = min(P_borne, E_dispo)  si E_dispo > 0\n"
         "P_charge = P_borne                 sinon (tout réseau)",
         "P_borne : puissance nominale d'une borne (kW) | "
         "E_dispo = Énergie écrêtée disponible à l'heure sélectionnée (kWh ≈ kW sur 1h)"),
        ("3", "Temps de charge",
         "t = (C_bat_VE × (SOC_f − SOC_i) / 100) / (P_charge × η)",
         "Durée (h) pour charger 1 VE de SOC_i% à SOC_f%. "
         "η : rendement de charge (configurable 80–99%, défaut 92%) — pertes Joule + conversion AC/DC"),
        ("4", "Couverture PV (%)",
         "Couverture = min(E_dispo, E_requise) / E_requise × 100",
         "Part de la demande couverte par l'énergie PV écrêtée disponible"),
        ("5", "VE chargés à 100% via PV",
         "N_VE_chargés = floor(E_dispo / C_bat_VE)",
         "Nombre maximum de VE qui peuvent être chargés à 100% avec l'écrêtée disponible"),
        ("6", "Puissance totale bornes",
         "P_total = N_VE × P_borne",
         "Puissance totale de l'installation de recharge (kW)"),
        ("7", "Économie sans PV",
         "Coût_sans = E_requise × Tarif(h, mois)",
         "Coût de recharge si tout vient du réseau ONEE (MAD)"),
        ("8", "Économie avec PV",
         "Coût_avec = E_réseau × Tarif(h, mois)  avec  E_réseau = max(0, E_requise − E_dispo)",
         "Coût résiduel avec la PV écrêtée. E_réseau : énergie encore prélevée sur réseau (kWh)"),
        ("9", "CO₂ sans PV",
         "CO₂_sans = E_requise × 0,604",
         "Émissions si tout réseau (kg CO₂e) — facteur 0,604 kg CO₂e/kWh (ONEE Maroc)"),
        ("10", "CO₂ avec PV",
         "CO₂_avec = E_réseau × 0,604",
         "Émissions résiduelles avec valorisation PV écrêtée (kg CO₂e)"),
        ("11", "CO₂ évité",
         "CO₂_évité = CO₂_sans − CO₂_avec = E_PV × 0,604",
         "Réduction effective des émissions grâce à la recharge PV (kg CO₂e)"),
    ]

    _sf_eqs = [
        ("1", "Besoin frigorifique nocturne",
         "E_froid_nuit = N_actif × P_clim × 12  (kWh)",
         "N_actif : nb de climatiseurs actifs la nuit | P_clim : puissance frigorifique unitaire (kW) | "
         "12h : plage nocturne 19h–7h. "
         "N_actif = N_clim_fours + N_clim_autres (été) ou N_clim_fours seul (hiver)"),
        ("2", "Énergie PV écrêtée disponible le jour",
         "E_écrêtée_jour = Σ (Production_avant[h] − Production_après[h])  pour h = 0..23",
         "Énergie totale perdue dans l'écrêteur pendant la journée (kWh)"),
        ("3", "Énergie froide produite et stockée",
         "E_stock_froid = E_écrêtée_jour × COP",
         "COP (Coefficient of Performance) : énergie froide produite / énergie électrique consommée. "
         "Typique splits : 2,5–3,5 | Chiller : 3,5–5. Se lit dans la fiche technique (rubrique COP ou EER)"),
        ("4", "Énergie froide couverte par le stock",
         "E_couvert = min(E_stock_froid, E_froid_nuit)",
         "Part du besoin nocturne que le stock peut couvrir (kWh). "
         "Si E_stock_froid > E_froid_nuit → tout le besoin est couvert"),
        ("5", "Taux de couverture",
         "Taux = E_couvert / E_froid_nuit × 100  (%)",
         "Pourcentage du besoin nocturne couvert par l'énergie PV écrêtée valorisée"),
        ("6", "Volume du ballon d'eau froide",
         "V = E_couvert × 3 600 / (c_eau × ΔT)  (litres)\n"
         "avec c_eau = 4,186 kJ/kg·°C  et  ΔT = 8°C  →  V ≈ E_couvert × 107,5 L/kWh",
         "Formule thermodynamique : Q = m × c × ΔT. "
         "ΔT = 8°C (eau à 7°C stockée, déchargée à 15°C). "
         "3 600 : conversion kWh → kJ"),
        ("7", "Économies nocturnes (réseau évité)",
         "Économies = E_couvert / COP × Tarif_nuit  (MAD/jour)",
         "Énergie électrique nuit évitée = E_couvert / COP. "
         "Tarif_nuit = tarif Heures Creuses ONEE (0,73294 MAD/kWh par défaut)"),
        ("8", "CO₂ évité",
         "CO₂_évité = (E_couvert / COP) × 0,604  (kg CO₂e/jour)",
         "Facteur 0,604 kg CO₂e/kWh (ONEE Maroc). "
         "Seule l'énergie électrique nuit réellement évitée est comptabilisée"),
    ]

    _pac_eqs = [
        ("1", "Débit massique boucle",
         "ṁ = Q_m³h × 1 000 × ρ / 3 600  [kg/s]\n"
         "avec ρ = 0,965 kg/L",
         "Conversion du débit volumique (m³/h) en débit massique (kg/s). "
         "ρ = 0,965 kg/L : densité du fluide boucle Jacob Delafon (eau traitée). "
         "Exemple : 21 m³/h → ṁ = 5,637 kg/s"),
        ("2", "Puissance thermique de la boucle",
         "E_boucle = ṁ × Cp × ΔT_boucle  [kWh/h]\n"
         "ΔT_boucle = T_finale − T_retour",
         "Énergie thermique totale que la boucle doit recevoir chaque heure pour passer "
         "de T_retour à T_finale. Cp = 4,186 kJ/(kg·K). "
         "Exemple : ṁ=5,637 kg/s · ΔT=25°C → E_boucle ≈ 147 kWh/h"),
        ("3", "Énergie thermique produite par la PAC",
         "E_therm = E_écrêtée × COP",
         "L'écrêtée (kWh élec) est transformée en chaleur par la PAC avec le COP. "
         "COP typique PAC haute température eau/eau (55–85°C) : 2,5–3,5. "
         "Exemple : E_écrêtée=50 kWh · COP=3 → E_therm=150 kWh th"),
        ("4", "Élévation de température apportée par la PAC",
         "ΔT_PAC = E_therm / (ṁ × Cp)  [°C]\n"
         "(pour un intervalle d'1 heure : kWh / (kg/s · kJ/kg·K) = °C)",
         "Hausse de température de l'eau que la PAC seule peut produire sur l'intervalle horaire. "
         "Si ΔT_PAC > ΔT_boucle, la PAC suffit à chauffer toute la boucle sans appoint."),
        ("5", "Température intermédiaire après PAC",
         "T_inter = min(T_retour + ΔT_PAC,  T_finale)  [°C]",
         "Température de l'eau après passage dans la PAC, avant la chaudière. "
         "Plafonnée à T_finale car la PAC ne peut pas dépasser la consigne. "
         "La chaudière assure uniquement le complément T_inter → T_finale."),
        ("6", "Fraction de la chaudière remplacée",
         "Frac = min(E_therm / E_boucle × 100,  100)  [%]",
         "Part du besoin thermique de la boucle couverte par la PAC. "
         "100% = la PAC suffit à chauffer toute la boucle sans appoint chaudière."),
        ("7", "Énergie d'appoint chaudière",
         "E_appoint = max(0,  E_boucle − E_therm)  [kWh th]",
         "Complément thermique que la chaudière gaz doit encore fournir après le préchauffage PAC. "
         "Si E_therm ≥ E_boucle → appoint = 0 (chaudière éteinte sur cet intervalle)."),
        ("8", "Consommation de gaz (sans et avec PAC)",
         "Q_gaz_sans = E_boucle  / η_chaud  [kWh gaz]\n"
         "Q_gaz_avec = E_appoint / η_chaud  [kWh gaz]",
         "η_chaud : rendement de la chaudière gaz industrielle (typique 85–92%). "
         "Q_gaz_sans : gaz consommé si la chaudière assure tout. "
         "Q_gaz_avec : gaz résiduel après préchauffage PAC."),
        ("9", "Économie de gaz",
         "Éco_gaz = (Q_gaz_sans − Q_gaz_avec) × Tarif_gaz  [MAD]\n"
         "       = (E_therm / η_chaud) × Tarif_gaz",
         "Économie directe sur la facture gaz industriel. "
         "Tarif gaz naturel industriel Maroc : 0,10–0,20 MAD/kWh th. "
         "Note : les économies sont en MAD gaz (pas en MAD électricité réseau)."),
        ("10", "CO₂ évité (gaz naturel)",
         "CO₂_évité = (Q_gaz_sans − Q_gaz_avec) × 0,234  [kg CO₂]\n"
         "          = (E_therm / η_chaud) × 0,234",
         "Facteur d'émission gaz naturel : 0,234 kg CO₂/kWh gaz (source IPCC/ADEME). "
         "Différent du facteur réseau électrique (0,604 kg CO₂/kWh élec ONEE Maroc). "
         "Le bilan CO₂ est favorable car 1 kWh PV écrêté × COP remplace davantage de gaz."),
        ("11", "Volume d'eau équivalent préchauffé",
         "V = E_therm × 3 600 / (Cp × ΔT_boucle × ρ)  [litres]",
         "Volume d'eau boucle que la PAC a l'équivalent de chauffer sur l'intervalle. "
         "3 600 : conversion kWh → kJ. ΔT_boucle = T_finale − T_retour. "
         "Indicateur physique permettant de vérifier la cohérence avec le débit réel de la boucle."),
    ]

    _gr_items = [(g, i) for g, i in GRANDEURS_INFO.items()
                 if _show_all or _match(g, i["type"], i["unité"], i["source"], i["description"])]
    if _gr_items:
        _found = True
        st.markdown("## Grandeurs Mesurées et Calculées")
        for grandeur, info in _gr_items:
            st.markdown(f"""
        <div style="background: rgba(255,152,0,0.1); border-left: 4px solid #FF9800;
        padding: 15px; border-radius: 12px; margin: 10px 0;">
            <h4 style="margin: 0 0 8px 0;">{grandeur}</h4>
            <p style="margin: 5px 0;"><b>Type:</b> {info['type']} | <b>Unité:</b> {info['unité']}</p>
            <p style="margin: 5px 0;"><b>Source:</b> {info['source']}</p>
            <p style="margin: 5px 0;"><b>Description:</b> {info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

    _kpi_items = [(k, d) for k, d in KPI_INFO.items()
                  if _show_all or _match(k, d)]
    if _kpi_items:
        _found = True
        st.markdown("## Indicateurs Clés de Performance (KPI)")
        for kpi, desc in _kpi_items:
            st.markdown(f"""
        <div style="background: rgba(66,165,245,0.1); border-left: 4px solid #42A5F5;
        padding: 15px; border-radius: 12px; margin: 10px 0;">
            <h4 style="margin: 0 0 8px 0;">{kpi}</h4>
            <p style="margin: 5px 0;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

    _tarif_corpus = ("tarif amendis tarification heure pointe creuse pleine mad kwh "
                     "hiver été hpt hp hc 0,73 0,92 1,22 onee mt tfz co2 0,604 heures réseau")
    if _show_all or _match(_tarif_corpus):
        _found = True
        st.markdown("## Tarification Amendis")
        st.markdown("""
    <p style="margin-bottom:10px;">
    Tarifs appliqués dans les scénarios VE et Stockage Froid pour le calcul des économies réseau.
    <b>Facteur d'émission CO₂ réseau : 0,604 kg CO₂e/kWh</b> (source ONEE Maroc).
    </p>
    """, unsafe_allow_html=True)
        st.markdown("""
<table style="width:100%;border-collapse:collapse;font-size:14px;margin-bottom:16px;">
  <thead>
    <tr style="background:#FF9800;color:white;">
      <th style="padding:10px;text-align:left;">Tranche</th>
      <th style="padding:10px;text-align:center;">Hiver (Oct–Mar)</th>
      <th style="padding:10px;text-align:center;">Été (Avr–Sep)</th>
      <th style="padding:10px;text-align:center;">Tarif (MAD/kWh)</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#FFEBEE;">
      <td style="padding:9px 10px;font-weight:700;color:#E53935;">Heures de Pointe (HPt)</td>
      <td style="padding:9px 10px;text-align:center;">17h–21h</td>
      <td style="padding:9px 10px;text-align:center;">18h–22h</td>
      <td style="padding:9px 10px;text-align:center;font-family:monospace;font-weight:700;color:#E53935;">1,22005</td>
    </tr>
    <tr style="background:#FFF3E0;">
      <td style="padding:9px 10px;font-weight:700;color:#FB8C00;">Heures Pleines (HP)</td>
      <td style="padding:9px 10px;text-align:center;">7h–17h / 21h–22h</td>
      <td style="padding:9px 10px;text-align:center;">7h–18h / 22h–23h</td>
      <td style="padding:9px 10px;text-align:center;font-family:monospace;font-weight:700;color:#FB8C00;">0,92488</td>
    </tr>
    <tr style="background:#E8F5E9;">
      <td style="padding:9px 10px;font-weight:700;color:#43A047;">Heures Creuses (HC)</td>
      <td style="padding:9px 10px;text-align:center;">22h–7h</td>
      <td style="padding:9px 10px;text-align:center;">23h–7h</td>
      <td style="padding:9px 10px;text-align:center;font-family:monospace;font-weight:700;color:#43A047;">0,73294</td>
    </tr>
  </tbody>
</table>
""", unsafe_allow_html=True)
        st.markdown("---")

    _ve_filtered = [eq for eq in _ve_eqs if _show_all or _match(*eq)]
    if _ve_filtered:
        _found = True
        st.markdown("## 🚗 Équations — Scénario Recharge Véhicules Électriques")
        for eq_num, nom, formule, description in _ve_filtered:
            formule_html = formule.replace("\n", "<br>")
            st.markdown(f"""
        <div style="background: rgba(46,204,113,0.08); border-left: 4px solid #2ECC71;
        padding: 14px 18px; border-radius: 10px; margin: 8px 0;">
            <h4 style="margin: 0 0 6px 0; color:#1B5E20;">Éq {eq_num} — {nom}</h4>
            <p style="margin: 4px 0; font-family: monospace; font-size: 14px;
               background:#f0fff4; padding:6px 10px; border-radius:6px;"><b>{formule_html}</b></p>
            <p style="margin: 6px 0 0 0; color:#555; font-size:13px;">{description}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

    _sf_filtered = [eq for eq in _sf_eqs if _show_all or _match(*eq)]
    if _sf_filtered:
        _found = True
        st.markdown("## ❄️ Équations — Scénario Stockage Froid (Climatisation nocturne)")
        st.markdown("""
    <div style="background:rgba(0,151,167,0.08);border-left:4px solid #0097A7;
    padding:12px 16px;border-radius:10px;margin-bottom:16px;font-size:13px;">
    <b>Principe :</b> L'énergie PV écrêtée (perdue le jour) est utilisée pour produire du froid via un groupe
    frigorifique (COP). Ce froid est stocké dans un ballon d'eau froide et restitué la nuit (19h–7h) pour
    climatiser les bureaux adjacents aux fours et, en été, les autres bureaux.
    <br><br>
    <b>Saison active :</b> Été (avr–sep) → fours + autres bureaux | Hiver (oct–mar) → fours uniquement.
    </div>
    """, unsafe_allow_html=True)
        for eq_num, nom, formule, description in _sf_filtered:
            formule_html = formule.replace("\n", "<br>")
            st.markdown(f"""
        <div style="background: rgba(0,151,167,0.08); border-left: 4px solid #0097A7;
        padding: 14px 18px; border-radius: 10px; margin: 8px 0;">
            <h4 style="margin: 0 0 6px 0; color:#004D40;">Éq {eq_num} — {nom}</h4>
            <p style="margin: 4px 0; font-family: monospace; font-size: 14px;
               background:#e0f7fa; padding:6px 10px; border-radius:6px;"><b>{formule_html}</b></p>
            <p style="margin: 6px 0 0 0; color:#555; font-size:13px;">{description}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

    _pac_filtered = [eq for eq in _pac_eqs if _show_all or _match(*eq)]
    if _pac_filtered:
        _found = True
        st.markdown("## 🌡️ Équations — Scénario PAC Préchauffage Boucle Industrielle (Jacob Delafon)")
        st.markdown("""
    <div style="background:rgba(230,74,25,0.08);border-left:4px solid #E64A19;
    padding:12px 16px;border-radius:10px;margin-bottom:16px;font-size:13px;">
    <b>Principe :</b> L'énergie PV écrêtée (perdue le jour) alimente une <b>PAC haute température</b>
    qui préchauffe l'eau de retour de la boucle industrielle <em>avant</em> la chaudière gaz.
    La chaudière n'assure plus que le complément (appoint). Les économies sont exprimées en <b>gaz</b>,
    pas en électricité réseau.
    <br><br>
    <b>Constantes physiques boucle Jacob Delafon :</b>
    ρ = 0,965 kg/L · Cp = 4,186 kJ/(kg·K) · CO₂ gaz naturel = 0,234 kg CO₂/kWh gaz.
    </div>
    """, unsafe_allow_html=True)
        for eq_num, nom, formule, description in _pac_filtered:
            formule_html = formule.replace("\n", "<br>")
            st.markdown(f"""
        <div style="background: rgba(230,74,25,0.08); border-left: 4px solid #E64A19;
        padding: 14px 18px; border-radius: 10px; margin: 8px 0;">
            <h4 style="margin: 0 0 6px 0; color:#BF360C;">Éq {eq_num} — {nom}</h4>
            <p style="margin: 4px 0; font-family: monospace; font-size: 14px;
               background:#fbe9e7; padding:6px 10px; border-radius:6px;"><b>{formule_html}</b></p>
            <p style="margin: 6px 0 0 0; color:#555; font-size:13px;">{description}</p>
        </div>
        """, unsafe_allow_html=True)

    if not _found and not _show_all:
        st.info(f"Aucun résultat pour « {_q} ». Essayez un terme différent (ex : irradiance, COP, rendement, autoconsommation).")

def calculate_kpis(df_filtered, selected_kpis):
    kpis_with_units = {}
    
    if "Production Totale (kWh)" in selected_kpis:
        val = df_filtered["Production apres limitation [kWh]"].sum()
        kpis_with_units["Production Totale (kWh)"] = val
    if "Production Moyenne (kWh)" in selected_kpis:
        val = df_filtered["Production apres limitation [kWh]"].mean()
        kpis_with_units["Production Moyenne (kWh)"] = val
    if "Production Max (kWh)" in selected_kpis:
        val = df_filtered["Production apres limitation [kWh]"].max()
        kpis_with_units["Production Max (kWh)"] = val
    if "Autoconsommation Totale (kWh)" in selected_kpis:
        val = df_filtered["SelfCons_kWh"].sum()
        kpis_with_units["Autoconsommation Totale (kWh)"] = val
    if "Taux Autoconsommation (%)" in selected_kpis:
        val = df_filtered["SelfConsumption_Rate"].mean()
        kpis_with_units["Taux Autoconsommation (%)"] = val
    if "Consommation Réseau Totale (kWh)" in selected_kpis:
        val = df_filtered["Consommation réseau [kWh]"].sum()
        kpis_with_units["Consommation Réseau Totale (kWh)"] = val
    if "Consommation Totale (kWh)" in selected_kpis:
        val = df_filtered["TotalLoad_kWh"].sum()
        kpis_with_units["Consommation Totale (kWh)"] = val
    if "Irradiance Moyenne (W/m²)" in selected_kpis:
        val = df_filtered["Irradiance_Wpm2"].mean()
        kpis_with_units["Irradiance Moyenne (W/m²)"] = val
    if "Température Moyenne (°C)" in selected_kpis:
        val = df_filtered["Temperature_C"].mean()
        kpis_with_units["Température Moyenne (°C)"] = val
    if "Énergie Écrêtée Totale (kWh)" in selected_kpis:
        val = df_filtered["Gap_kWh"].sum()
        kpis_with_units["Énergie Écrêtée Totale (kWh)"] = val
    if "Rendement Global (%)" in selected_kpis:
        total_theo = df_filtered["PV_theoretical_kWh"].sum()
        total_real = df_filtered["Production apres limitation [kWh]"].sum()
        val = (total_real / total_theo * 100) if total_theo > 0 else 0
        kpis_with_units["Rendement Global (%)"] = val
    if "Taux Couverture Solaire (%)" in selected_kpis:
        total_load = df_filtered["TotalLoad_kWh"].sum()
        total_prod = df_filtered["Production apres limitation [kWh]"].sum()
        val = (total_prod / total_load * 100) if total_load > 0 else 0
        kpis_with_units["Taux Couverture Solaire (%)"] = val
    if "Autonomie Énergétique (%)" in selected_kpis:
        total_load = df_filtered["TotalLoad_kWh"].sum()
        total_self = df_filtered["SelfCons_kWh"].sum()
        val = (total_self / total_load * 100) if total_load > 0 else 0
        kpis_with_units["Autonomie Énergétique (%)"] = val
    
    return kpis_with_units

DATA_PATH = "Base de données-VERSION-FINALE.xlsx"

def calculate_theoretical_energy(irradiance, T_cell, hours=1):
    eta_T = 1 + _GAMMA_P * (T_cell - _T_STC)
    P_DC  = _P_STC * (irradiance / _G_STC) * eta_T * _N_PANELS / 1000
    return max(0.0, P_DC * hours)

@st.cache_data
def load_data():
    try:
        df = pd.read_excel(DATA_PATH, sheet_name="Base de données ", engine="openpyxl")
        df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", case=False)]

        df = df.rename(columns={
            "Temp [°C]": "Temperature_C",
            "Irradiance POA (capteur) [W/m²]": "Irradiance_Wpm2",
            "Autoconsommation [kWh]": "SelfCons_kWh",
            "Consommation générale [kWh]": "TotalLoad_kWh",
            "Production avant limitation  Kwh": "Production avant limitation Kwh"
        })

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

        numeric_cols = ["Temperature_C", "Irradiance_Wpm2", "SelfCons_kWh", "TotalLoad_kWh", 
                       "Production apres limitation [kWh]", "Production avant limitation Kwh",
                       "Consommation réseau [kWh]"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["SelfConsumption_Rate"] = np.where(
            df["Production apres limitation [kWh]"] > 0,
            (df["SelfCons_kWh"] / df["Production apres limitation [kWh]"]) * 100, 0)

        df["PV_theoretical_kWh"] = df.apply(
            lambda r: max(0.0, _P_STC * (r["Irradiance_Wpm2"] / _G_STC)
                         * (1 + _GAMMA_P * (r["Temperature_C"] - _T_STC))
                         * _N_PANELS / 1000),
            axis=1
        )
        df["Gap_kWh"] = df["Production avant limitation Kwh"] - df["Production apres limitation [kWh]"]

        return df
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

try:
    xgb_avant = joblib.load("modele_xgb_avant_limitation.pkl")
except:
    xgb_avant = None

try:
    xgb_apres = joblib.load("modele_xgb_apres_limitation.pkl")
except:
    xgb_apres = None

try:
    xgb_conso = joblib.load("modele_xgb_conso_generale.pkl")
except:
    xgb_conso = None

def _build_lstm_model():
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    inputs = tf.keras.Input(shape=(24, 11))
    x = LSTM(32, activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

_lstm_load_error = None
try:
    _d_avant = np.load("weights_lstm_avant.npz")
    _w_avant = [_d_avant[f"w{i}"] for i in range(len(_d_avant.files))]
    lstm_model_avant = _build_lstm_model()
    lstm_model_avant.set_weights(_w_avant)
    scaler_X_lstm_avant = joblib.load("scaler_X_lstm_avant.pkl")
    scaler_y_lstm_avant = joblib.load("scaler_y_lstm_avant.pkl")

    _d_apres = np.load("weights_lstm_apres.npz")
    _w_apres = [_d_apres[f"w{i}"] for i in range(len(_d_apres.files))]
    lstm_model_apres = _build_lstm_model()
    lstm_model_apres.set_weights(_w_apres)
    scaler_X_lstm_apres = joblib.load("scaler_X_lstm_apres.pkl")
    scaler_y_lstm_apres = joblib.load("scaler_y_lstm_apres.pkl")
except Exception as _e:
    _lstm_load_error = str(_e)
    lstm_model_avant = None
    lstm_model_apres = None
    scaler_X_lstm_avant = None
    scaler_X_lstm_apres = None
    scaler_y_lstm_avant = None
    scaler_y_lstm_apres = None

@st.cache_data(show_spinner=False)
def _eval_xgb_perf_full(_model_av, _model_ap, _model_co, df):
    """Évalue les 3 modèles XGBoost sur tout le dataset — résultat mis en cache."""
    out = {}
    _df = df.dropna(subset=["Temperature_C", "Irradiance_Wpm2"]).copy().reset_index(drop=True)
    if len(_df) == 0:
        return out
    _hp  = _df["Date"].dt.hour.values.astype(float)
    _mp  = _df["Date"].dt.month.values.astype(float)
    _jp  = _df["Date"].dt.dayofyear.values.astype(float)
    _tp  = _df["Temperature_C"].values
    _irp = _df["Irradiance_Wpm2"].values
    _sjp = np.sin(2*np.pi*_jp/365.25); _cjp = np.cos(2*np.pi*_jp/365.25)
    _shp = np.sin(2*np.pi*_hp/24);     _chp = np.cos(2*np.pi*_hp/24)
    _smp = np.sin(2*np.pi*_mp/12);     _cmp = np.cos(2*np.pi*_mp/12)
    _Xp = pd.DataFrame({
        "sin_jour": _sjp, "cos_jour": _cjp, "sin_heure": _shp, "cos_heure": _chp,
        "sin_mois": _smp, "cos_mois": _cmp, "temp_c": _tp, "irradiance_poa": _irp,
        "irr_temp": _irp*_tp, "is_day": (_irp > 20).astype(int),
        "irradiance_poa_sq": _irp**2, "temp_c_sq": _tp**2,
        "irr_sin_heure": _irp*_shp, "irr_sin_jour": _irp*_sjp,
        "irr_cos_jour": _irp*_cjp, "irr_sin_mois": _irp*_smp,
        "temp_sin_jour": _tp*_sjp, "irr_cube": _irp**3,
    })
    if _model_ap is not None and "Production apres limitation [kWh]" in _df.columns:
        _pp = np.maximum(0.0, _model_ap.predict(_Xp)); _pp[_irp < 20] = 0.0
        _df["_PredAp"] = _pp
        _s = _df.dropna(subset=["Production apres limitation [kWh]", "_PredAp"])
        if len(_s) > 1:
            _mae = mean_absolute_error(_s["Production apres limitation [kWh]"], _s["_PredAp"])
            _r2  = r2_score(_s["Production apres limitation [kWh]"], _s["_PredAp"])
            out["ap"] = {"r2": _r2, "mae": _mae, "moy": _s["Production apres limitation [kWh]"].mean()}
    if _model_av is not None and "Production avant limitation Kwh" in _df.columns:
        _pp = np.maximum(0.0, _model_av.predict(_Xp)); _pp[_irp < 20] = 0.0
        _df["_PredAv"] = _pp
        _s = _df.dropna(subset=["Production avant limitation Kwh", "_PredAv"])
        if len(_s) > 1:
            _mae = mean_absolute_error(_s["Production avant limitation Kwh"], _s["_PredAv"])
            _r2  = r2_score(_s["Production avant limitation Kwh"], _s["_PredAv"])
            out["av"] = {"r2": _r2, "mae": _mae, "moy": _s["Production avant limitation Kwh"].mean()}
    if _model_co is not None and "TotalLoad_kWh" in _df.columns:
        _df_co = df.dropna(subset=["Temperature_C", "TotalLoad_kWh"]).copy().reset_index(drop=True)
        _df_co["_PredCo"] = _df_co.apply(
            lambda r: max(0.0, float(_model_co.predict(pd.DataFrame([{
                "sin_jour": np.sin(2*np.pi*r["Date"].timetuple().tm_yday/365.25),
                "cos_jour": np.cos(2*np.pi*r["Date"].timetuple().tm_yday/365.25),
                "sin_heure": np.sin(2*np.pi*r["Date"].hour/24),
                "cos_heure": np.cos(2*np.pi*r["Date"].hour/24),
                "sin_mois": np.sin(2*np.pi*r["Date"].month/12),
                "cos_mois": np.cos(2*np.pi*r["Date"].month/12),
                "temp_c": r["Temperature_C"], "temp_c_sq": r["Temperature_C"]**2,
                "temp_c_cu": r["Temperature_C"]**3,
                "temp_sin_jour": r["Temperature_C"]*np.sin(2*np.pi*r["Date"].timetuple().tm_yday/365.25),
                "temp_cos_mois": r["Temperature_C"]*np.cos(2*np.pi*r["Date"].month/12),
                "temp_sin_heure": r["Temperature_C"]*np.sin(2*np.pi*r["Date"].hour/24),
                "heure_int": r["Date"].hour,
            }]))[0])), axis=1)
        _s = _df_co.dropna(subset=["TotalLoad_kWh", "_PredCo"])
        if len(_s) > 1:
            _mae = mean_absolute_error(_s["TotalLoad_kWh"], _s["_PredCo"])
            _r2  = r2_score(_s["TotalLoad_kWh"], _s["_PredCo"])
            out["co"] = {"r2": _r2, "mae": _mae, "moy": _s["TotalLoad_kWh"].mean()}
    return out

def _build_features(date_input, heure_input, temp_c, irradiance_poa):
    jour_annee = date_input.timetuple().tm_yday
    mois = date_input.month
    sin_jour = np.sin(2 * np.pi * jour_annee / 365.25)
    cos_jour = np.cos(2 * np.pi * jour_annee / 365.25)
    sin_heure = np.sin(2 * np.pi * heure_input / 24)
    cos_heure = np.cos(2 * np.pi * heure_input / 24)
    sin_mois = np.sin(2 * np.pi * mois / 12)
    cos_mois = np.cos(2 * np.pi * mois / 12)
    
    input_dict = {
        "sin_jour": sin_jour, "cos_jour": cos_jour,
        "sin_heure": sin_heure, "cos_heure": cos_heure,
        "sin_mois": sin_mois, "cos_mois": cos_mois,
        "temp_c": temp_c, "irradiance_poa": irradiance_poa,
        "irr_temp": irradiance_poa * temp_c, "is_day": 1 if irradiance_poa > 20 else 0,
        "irradiance_poa_sq": irradiance_poa ** 2, "temp_c_sq": temp_c ** 2,
        "irr_sin_heure": irradiance_poa * sin_heure,
        "irr_sin_jour": irradiance_poa * sin_jour, "irr_cos_jour": irradiance_poa * cos_jour,
        "irr_sin_mois": irradiance_poa * sin_mois, "temp_sin_jour": temp_c * sin_jour,
        "irr_cube": irradiance_poa ** 3
    }
    return pd.DataFrame([input_dict])

def predict_lstm(date_input, heure_input, temp_c, irradiance_poa, model, scaler_X, scaler_y):
    """Prédiction LSTM — séquence 24h réelle ou moyenne historique mois+heure."""
    if model is None or scaler_X is None or scaler_y is None:
        return None

    try:
        target_dt  = datetime.combine(date_input, datetime.min.time()).replace(hour=heure_input)
        start_dt   = target_dt - pd.Timedelta(hours=24)
        mois       = date_input.month

        _excl_myt = (df["Date"].dt.year == 2025) & (df["Date"].dt.month == 8)
        def _hist_avg(h):
            mask = (df["Date"].dt.month == mois) & (df["Date"].dt.hour == h) & ~_excl_myt
            rows = df[mask]
            if len(rows) >= 3:
                return rows["Irradiance_Wpm2"].mean(), rows["Temperature_C"].mean()
            mois_voisins = [(mois - 2) % 12 + 1, (mois - 1) % 12 + 1,
                            mois, mois % 12 + 1, (mois + 1) % 12 + 1]
            mask2 = df["Date"].dt.month.isin(mois_voisins) & (df["Date"].dt.hour == h) & ~_excl_myt
            rows2 = df[mask2]
            if len(rows2) >= 3:
                return rows2["Irradiance_Wpm2"].mean(), rows2["Temperature_C"].mean()
            return irradiance_poa, temp_c

        hist_data = df[(df["Date"] >= start_dt) & (df["Date"] < target_dt)].copy()

        X_sequence = []
        for i in range(24):
            seq_dt = start_dt + pd.Timedelta(hours=i)
            h = seq_dt.hour
            j = seq_dt.timetuple().tm_yday
            m = seq_dt.month

            real = hist_data[hist_data["Date"] == seq_dt]
            if not real.empty:
                irr  = real.iloc[0]["Irradiance_Wpm2"]
                temp = real.iloc[0]["Temperature_C"]
            else:
                irr, temp = _hist_avg(h)

            sin_jour  = np.sin(2 * np.pi * j / 365.25)
            cos_jour  = np.cos(2 * np.pi * j / 365.25)
            sin_h     = np.sin(2 * np.pi * h / 24)
            cos_h     = np.cos(2 * np.pi * h / 24)
            sin_mois  = np.sin(2 * np.pi * m / 12)
            cos_mois  = np.cos(2 * np.pi * m / 12)

            X_sequence.append([
                irr, temp,
                sin_jour, cos_jour, sin_h, cos_h, sin_mois, cos_mois,
                irr * temp, irr * sin_h, irr * sin_jour
            ])

        X_input = np.array([X_sequence])
        X_scaled = scaler_X.transform(X_input.reshape(-1, 11)).reshape(1, 24, 11)

        y_pred_scaled = model.predict(X_scaled, verbose=0)[0][0]
        y_pred = scaler_y.inverse_transform([[y_pred_scaled]])[0][0]

        if irradiance_poa < 20:
            return 0.0

        return max(0.0, float(y_pred))

    except:
        return None

def predict_xgb(date_input, heure_input, temp_c, irradiance_poa, model):
    if model is None:
        return None
    X_input = _build_features(date_input, heure_input, temp_c, irradiance_poa)
    pred = model.predict(X_input)[0]
    if irradiance_poa < 20:
        return 0.0
    return max(0.0, float(pred))

def _build_features_conso(date_input, heure_input, temp_c):
    """14 features — production continue 24h/7j, arrêt annuel août (vrai arrêt 2024)."""
    jour_annee = date_input.timetuple().tm_yday
    mois       = date_input.month
    sin_jour   = np.sin(2 * np.pi * jour_annee / 365.25)
    cos_jour   = np.cos(2 * np.pi * jour_annee / 365.25)
    sin_heure  = np.sin(2 * np.pi * heure_input / 24)
    cos_heure  = np.cos(2 * np.pi * heure_input / 24)
    sin_mois   = np.sin(2 * np.pi * mois / 12)
    cos_mois   = np.cos(2 * np.pi * mois / 12)
    return pd.DataFrame([{
        "sin_jour": sin_jour, "cos_jour": cos_jour,
        "sin_heure": sin_heure, "cos_heure": cos_heure,
        "sin_mois": sin_mois, "cos_mois": cos_mois,
        "temp_c": temp_c, "temp_c_sq": temp_c ** 2, "temp_c_cu": temp_c ** 3,
        "temp_sin_jour": temp_c * sin_jour,
        "temp_cos_mois": temp_c * cos_mois,
        "temp_sin_heure": temp_c * sin_heure,
        "heure_int": heure_input,
    }])

def predict_xgb_conso(date_input, heure_input, temp_c, model):
    """Prédit la consommation générale (kWh) sans irradiance."""
    if model is None:
        return None
    X = _build_features_conso(date_input, heure_input, temp_c)
    return max(0.0, float(model.predict(X)[0]))

@st.cache_data(ttl=3600, show_spinner=False)
def build_predicted_nday_df(pred_date, n_days, df_hist, _xgb_avant, _xgb_apres, _xgb_conso):
    """DataFrame simulé par XGBoost sur n_days jours (modèles préfixés _ exclus du hash Streamlit)."""
    rows = []
    for day_offset in range(-n_days, 0):
        target_day = pred_date + pd.Timedelta(days=day_offset + 1)
        profil = get_day_profile(target_day, df_hist)
        for h in range(24):
            row_p = profil[profil["heure"] == h]
            T   = float(row_p["T_moy"].values[0])   if len(row_p) > 0 else 25.0
            Irr = float(row_p["Irr_moy"].values[0]) if len(row_p) > 0 else 0.0
            av    = predict_xgb(target_day, h, T, Irr, _xgb_avant) or 0.0
            ap    = predict_xgb(target_day, h, T, Irr, _xgb_apres) or 0.0
            conso = predict_xgb_conso(target_day, h, T, _xgb_conso) or 0.0
            autoconso    = min(ap, conso)
            conso_reseau = max(0.0, conso - autoconso)
            gap          = max(0.0, av - ap)
            selfcons_r   = (autoconso / conso * 100) if conso > 0 else 0.0
            rows.append({
                "Date": datetime.combine(target_day, datetime.min.time()).replace(hour=h),
                "Temperature_C": T, "Irradiance_Wpm2": Irr,
                "Production avant limitation Kwh": av,
                "Production apres limitation [kWh]": ap,
                "Gap_kWh": gap,
                "SelfCons_kWh": autoconso,
                "TotalLoad_kWh": conso,
                "Consommation réseau [kWh]": conso_reseau,
                "PV_theoretical_kWh": calculate_theoretical_energy(Irr, T),
                "SelfConsumption_Rate": selfcons_r,
            })
    return pd.DataFrame(rows)

def build_predicted_7day_df(pred_date, df_hist, _xgb_avant, _xgb_apres, _xgb_conso):
    return build_predicted_nday_df(pred_date, 7, df_hist, _xgb_avant, _xgb_apres, _xgb_conso)

def build_predicted_1day_df(pred_date, df_hist, _xgb_avant, _xgb_apres, _xgb_conso):
    return build_predicted_nday_df(pred_date, 1, df_hist, _xgb_avant, _xgb_apres, _xgb_conso)

def get_day_profile(target_date, df_hist):
    """Retourne DataFrame(heure, T_moy, Irr_moy) pour une date.
    Priorité : Open-Meteo → moyenne mensuelle historique (hors août 2025 MYT)."""
    profile = fetch_day_meteo_openmeteo(target_date)
    if profile is not None:
        return profile
    _excl = (df_hist["Date"].dt.year == 2025) & (df_hist["Date"].dt.month == 8)
    df_m  = df_hist[(df_hist["Date"].dt.month == target_date.month) & ~_excl]
    if len(df_m) == 0:
        df_m = df_hist
    return df_m.groupby(df_m["Date"].dt.hour).agg(
        T_moy=("Temperature_C", "mean"),
        Irr_moy=("Irradiance_Wpm2", "mean")
    ).reset_index().rename(columns={"Date": "heure"})

_st_plotly = st.plotly_chart

def render_chart(fig, use_container_width=True, **kw):
    if not theme_toggle:
        fig.update_xaxes(color="black", gridcolor="rgba(0,0,0,0.07)",
                         linecolor="rgba(0,0,0,0.2)",
                         tickfont=dict(color="black"), title_font=dict(color="black"))
        fig.update_yaxes(color="black", gridcolor="rgba(0,0,0,0.07)",
                         linecolor="rgba(0,0,0,0.2)",
                         tickfont=dict(color="black"), title_font=dict(color="black"))
    _st_plotly(fig, use_container_width=use_container_width, **kw)

with st.sidebar:
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] > div:first-child { padding-top: 0px !important; }
    section[data-testid="stSidebar"] .block-container { padding-top: 0px !important; }
    section[data-testid="stSidebar"] [data-testid="stImage"] { margin-top: -2rem !important; }
    </style>
    """, unsafe_allow_html=True)
    st.image("logo-JDM.png", use_container_width=True)

    with st.expander("☰", expanded=False):
        theme_toggle = st.toggle("🌙 Mode Sombre", value=False)
    theme = get_theme_colors(theme_toggle)
    cc = get_curve_colors(theme_toggle)

    if not theme_toggle:
        st.markdown("""
        <style>
        /* Métriques */
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricLabel"] {
            color: #1A1A1A !important;
            font-weight: 600 !important;
        }
        [data-testid="stMetricValue"] {
            color: #1A1A1A !important;
        }
        /* Sidebar — palette soleil */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #FF9800 0%, #FFB74D 18%, #FFE0B2 45%, #FFF8F0 100%) !important;
        }
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
            color: #1A1A1A !important;
        }
        /* Containers / expanders — supprimer fonds noirs */
        [data-testid="stExpander"],
        [data-testid="stExpander"] details,
        [data-testid="stExpanderDetails"],
        [data-testid="stExpanderContent"],
        .streamlit-expanderContent {
            background-color: #FFF8F0 !important;
            border-color: #FFD199 !important;
            color: #1A1A1A !important;
        }
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span {
            color: #1A1A1A !important;
            background-color: #FFF3E0 !important;
        }
        /* Blocs verticaux / containers généraux */
        [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"],
        [data-testid="element-container"],
        .block-container {
            background-color: transparent !important;
        }
        /* Inputs, selects, radio, sliders */
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stDateInput"] input,
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stMultiSelect"] > div > div {
            background-color: #FFF8F0 !important;
            color: #1A1A1A !important;
            border-color: #FFB74D !important;
        }
        /* Texte général */
        p, span, label, div, h1, h2, h3, h4 {
            color: #1A1A1A;
        }
        /* Option menu container */
        .nav-link, .nav-link span {
            color: #1A1A1A !important;
        }
        /* Alertes / success / info boxes */
        [data-testid="stAlert"] {
            background-color: #FFF8F0 !important;
            border-color: #FFB74D !important;
            color: #1A1A1A !important;
        }
        /* Tables */
        [data-testid="stDataFrame"],
        .stDataFrame {
            background-color: #FFF8F0 !important;
        }
        </style>
        """, unsafe_allow_html=True)

    _ac = theme["accent"]
    st.markdown(f"""
    <style>
    /* H2 — titres de sections */
    div[data-testid="stMarkdownContainer"] h2 {{
        font-size: 1.22rem !important;
        font-weight: 700 !important;
        color: {_ac} !important;
        padding: 9px 16px !important;
        border-left: 5px solid {_ac} !important;
        background: {_ac}18 !important;
        border-radius: 0 8px 8px 0 !important;
        margin: 20px 0 10px 0 !important;
        letter-spacing: 0.2px !important;
    }}
    /* H3 — sous-titres */
    div[data-testid="stMarkdownContainer"] h3 {{
        font-size: 1.03rem !important;
        font-weight: 600 !important;
        color: {_ac} !important;
        padding: 5px 12px !important;
        border-left: 3px solid {_ac} !important;
        margin: 14px 0 8px 0 !important;
    }}
    /* Métriques — label uppercase discret */
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] span {{
        font-size: 0.74rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.6px !important;
        opacity: 0.7 !important;
    }}
    /* Métriques — valeur en gras */
    [data-testid="stMetricValue"] {{
        font-size: 1.55rem !important;
        font-weight: 800 !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("---")
    pred_date = st.date_input("📅 Date", value=datetime.today().date())
    heures_options = [f"{h:02d}:00" for h in range(24)]
    heure_selectionnee = st.selectbox("🕐 Heure", heures_options, index=12)
    pred_heure = int(heure_selectionnee.split(":")[0])

    st.markdown("---")

    if "num_temp" not in st.session_state:
        st.session_state["num_temp"] = 25.0
    if "num_irr" not in st.session_state:
        st.session_state["num_irr"] = 600.0
    if "meteo_source" not in st.session_state:
        st.session_state.meteo_source = ""

    if st.button("🌤️ Météo automatique", use_container_width=True, key="btn_meteo"):
        with st.spinner("Connexion Open-Meteo..."):
            _t, _irr = fetch_meteo_openmeteo(pred_date, pred_heure)
        if _t is not None:
            st.session_state["num_temp"] = round(_t, 1)
            st.session_state["num_irr"]  = max(0.0, round(_irr or 0.0, 1))
            st.session_state.meteo_source = f"Open-Meteo · {pred_date.strftime('%d/%m/%Y')} {pred_heure:02d}h"
            st.rerun()
        else:
            st.error("Connexion impossible — vérifiez internet")

    if st.session_state.meteo_source:
        st.caption(f"Source : {st.session_state.meteo_source}")

    temp_input = st.number_input(
        "🌡️ Température (°C)", step=0.5, key="num_temp"
    )
    irrad_input = st.number_input(
        "☀️ Irradiance (W/m²)", step=10.0, key="num_irr"
    )

selected_datetime = datetime.combine(pred_date, datetime.min.time()).replace(hour=pred_heure)
row = None
data_exists = False

min_date = df["Date"].min()
max_date = df["Date"].max()
date_in_range = (selected_datetime >= min_date) and (selected_datetime <= max_date)

matching_rows = df[df["Date"] == selected_datetime]
if not matching_rows.empty:
    row = matching_rows.iloc[0]
    data_exists = True
    temp_input = float(row["Temperature_C"])
    irrad_input = float(row["Irradiance_Wpm2"])

css_template = f"""
<style>
html, body, .stApp {{
    background: {theme['bg_primary']} !important;
    color: {theme['text_primary']} !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}}

section[data-testid="stSidebar"] {{
    background: {'linear-gradient(180deg, #FF9800 0%, #FFB74D 18%, #FFE0B2 45%, #FFF8F0 100%)' if not theme_toggle else 'linear-gradient(180deg, #B35900 0%, #E65100 15%, #1A1200 50%, #0F0F12 100%)'} !important;
}}

[data-testid="stMetric"] {{
    background: {theme['card_bg']} !important;
    border: 2px solid {theme['card_border']} !important;
    border-radius: 12px !important;
    padding: 20px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
}}

[data-testid="stMetricValue"] {{
    font-size: 26px !important;
    font-weight: 700 !important;
    color: {theme['accent_bright']} !important;
}}

[data-testid="stMetricLabel"] {{
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    color: #555555 !important;
}}

.stButton > button {{
    background: linear-gradient(135deg, {theme['accent']} 0%, {theme['accent_bright']} 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    transition: all 0.3s ease !important;
}}

.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(0,0,0,0.2) !important;
}}

hr {{
    border: none !important;
    border-top: 1px solid {theme['border']} !important;
    margin: 20px 0 !important;
}}

/* ── Barre de navigation figée (comme Excel figer la première ligne) ── */
[data-testid="stCustomComponentV1"] {{
    position: sticky !important;
    top: 0 !important;
    z-index: 999 !important;
    background: {theme['bg_primary']} !important;
    padding-bottom: 4px !important;
}}
section[data-testid="stMain"] > div {{
    overflow: visible !important;
}}

/* ── Cacher la barre du haut Streamlit + remonter le contenu ── */
[data-testid="stHeader"] {{
    display: none !important;
    height: 0 !important;
}}
[data-testid="stMainBlockContainer"],
.main > .block-container {{
    padding-top: 0.5rem !important;
}}
</style>
"""
st.markdown(css_template, unsafe_allow_html=True)

time_str = f"{pred_heure:02d}:00"
date_str = pred_date.strftime("%d/%m/%Y")

if data_exists:
    status_badge = "📊 DONNÉES RÉELLES"
    status_color = theme['success']
else:
    status_badge = "🤖 PRÉDICTIONS"
    status_color = theme['warning']

date_range_info = f"✅ ({min_date.strftime('%d/%m/%Y')} à {max_date.strftime('%d/%m/%Y')})" if date_in_range else f"❌ RÉEL : Date hors plage de données ({min_date.strftime('%d/%m/%Y')} à {max_date.strftime('%d/%m/%Y')})"

st.markdown(f"""
<div style="background: {theme['card_bg']}; border: 2px solid {theme['card_border']}; 
border-radius: 16px; padding: 24px; margin-bottom: 24px; backdrop-filter: blur(10px);">
    <div style="text-align: center; margin-bottom: 12px;">
        <h1 style="font-size: 32px; font-weight: 800; margin: 0; color: {theme['accent_bright']};">
            ☀️ Système Photovoltaïque
        </h1>
        <p style="font-size: 11px; color: #888; margin: 8px 0 0 0; letter-spacing: 1px; font-weight: 600;">
            ANALYSE & PRÉVISION D'ÉNERGIE
        </p>
    </div>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 16px;
    padding-top: 16px; border-top: 1px solid {theme['border']};">
        <div>
            <p style="font-size: 10px; color: #888; margin: 0 0 4px 0; font-weight: 600;">DATE & HEURE</p>
            <p style="font-size: 14px; font-weight: 700; margin: 0;">{date_str} • {time_str}</p>
        </div>
        <div>
            <p style="font-size: 10px; color: #888; margin: 0 0 4px 0; font-weight: 600;">TEMPÉRATURE</p>
            <p style="font-size: 14px; font-weight: 700; margin: 0;">{temp_input:.1f}°C</p>
        </div>
        <div>
            <p style="font-size: 10px; color: #888; margin: 0 0 4px 0; font-weight: 600;">IRRADIANCE</p>
            <p style="font-size: 14px; font-weight: 700; margin: 0;">{irrad_input:.1f} W/m²</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

_accent = "#FF9800" if not theme_toggle else "#FFB74D"
_nav_bg = "rgba(26,21,32,0.85)" if theme_toggle else "rgba(255,248,240,0.9)"
_nav_sel = "#FF9800"
_nav_hover = "rgba(255,152,0,0.15)"

page = option_menu(
    menu_title=None,
    options=["Accueil", "Modèle Théorique", "Machine Learning", "Graphiques",
             "Scénarios", "Multi-Dates", "Documentation"],
    icons=["", "", "", "", "", "", ""],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "6px 0px",
            "background-color": _nav_bg,
            "border-radius": "12px",
            "margin-bottom": "18px",
            "border": f"1px solid {_accent}33",
        },
        "icon": {"color": _accent, "font-size": "15px"},
        "nav-link": {
            "font-size": "12px",
            "font-weight": "600",
            "text-align": "center",
            "padding": "8px 6px",
            "--hover-color": _nav_hover,
            "border-radius": "8px",
            "color": "#1A1A1A" if not theme_toggle else "#E0E0E0",
        },
        "nav-link-selected": {
            "background-color": _nav_sel,
            "color": "white",
            "font-weight": "700",
        },
    }
)

if page == "Accueil":
    end_date = datetime.combine(pred_date, datetime.min.time()).replace(hour=pred_heure)
    start_date = end_date - pd.Timedelta(days=7)
    mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)
    df_period = df[mask]
    
    alerts = check_alerts(df_period, df)

    st.markdown("## Production")
    
    xgb_avant_pred = predict_xgb(pred_date, pred_heure, temp_input, irrad_input, xgb_avant)
    xgb_apres_pred = predict_xgb(pred_date, pred_heure, temp_input, irrad_input, xgb_apres)
    lstm_apres_pred = predict_lstm(pred_date, pred_heure, temp_input, irrad_input, lstm_model_apres, scaler_X_lstm_apres, scaler_y_lstm_apres)
    lstm_avant_pred = predict_lstm(pred_date, pred_heure, temp_input, irrad_input, lstm_model_avant, scaler_X_lstm_avant, scaler_y_lstm_avant)

    if data_exists and date_in_range:
        real_avant = row.get("Production avant limitation Kwh", 0)
        real_apres = row.get("Production apres limitation [kWh]", 0)
        
        st.markdown("""
        <div style="background: rgba(102, 187, 106, 0.1); border-left: 4px solid #66BB6A; 
        padding: 16px; border-radius: 12px; margin: 16px 0;">
            <p style="margin: 0; font-weight: 600; font-size: 12px; color: #66BB6A; letter-spacing: 1px;">
                ✅ DONNÉES RÉELLES
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AVANT", f"{real_avant:.3f} kWh")
        c2.metric("APRÈS", f"{real_apres:.3f} kWh")
        c3.metric("ÉCRÊTÉE", f"{max(0, real_avant - real_apres):.3f} kWh")
        c4.metric("% ÉCRÊTAGE", f"{(max(0, real_avant - real_apres) / real_avant * 100) if real_avant > 0 else 0:.1f}%")
    
    st.markdown("""
    <div style="background: rgba(66, 165, 245, 0.1); border-left: 4px solid #42A5F5; 
    padding: 16px; border-radius: 12px; margin: 16px 0;">
        <p style="margin: 0; font-weight: 600; font-size: 12px; color: #42A5F5; letter-spacing: 1px;">
            PRÉDICTIONS XGBoost
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("AVANT", f"{xgb_avant_pred:.3f} kWh")
    c6.metric("APRÈS", f"{xgb_apres_pred:.3f} kWh")
    c7.metric("ÉCRÊTÉE", f"{max(0, xgb_avant_pred - xgb_apres_pred):.3f} kWh")
    c8.metric("% ÉCRÊTAGE", f"{(max(0, xgb_avant_pred - xgb_apres_pred) / xgb_avant_pred * 100) if xgb_avant_pred > 0 else 0:.1f}%")
    
    if lstm_model_apres is not None:
        st.markdown("""
        <div style="background: rgba(102, 187, 106, 0.1); border-left: 4px solid #66BB6A; 
        padding: 16px; border-radius: 12px; margin: 16px 0;">
            <p style="margin: 0; font-weight: 600; font-size: 12px; color: #66BB6A; letter-spacing: 1px;">
                PRÉDICTIONS LSTM
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        c9, c10, c11, c12 = st.columns(4)
        c9.metric("AVANT", f"{lstm_avant_pred:.3f} kWh" if lstm_avant_pred is not None else "N/A")
        c10.metric("APRÈS", f"{lstm_apres_pred:.3f} kWh" if lstm_apres_pred is not None else "N/A")
        c11.metric("ÉCRÊTÉE", f"{max(0, lstm_avant_pred - lstm_apres_pred):.3f} kWh" if lstm_avant_pred is not None else "N/A")
        c12.metric("% ÉCRÊTAGE", f"{(max(0, lstm_avant_pred - lstm_apres_pred) / lstm_avant_pred * 100) if lstm_avant_pred and lstm_avant_pred > 0 else 0:.1f}%")

    st.markdown("---")
    st.markdown("##  Consommation Générale")

    xgb_conso_pred = predict_xgb_conso(pred_date, pred_heure, temp_input, xgb_conso)

    _xgb_ap        = xgb_apres_pred or 0.0
    _xgb_conso_v   = xgb_conso_pred or 0.0
    _xgb_autoconso = min(_xgb_ap, _xgb_conso_v)
    _xgb_reseau    = max(0.0, _xgb_conso_v - _xgb_autoconso)
    _xgb_autosuff  = (_xgb_autoconso / _xgb_conso_v * 100) if _xgb_conso_v > 0 else 0.0

    _has_real = False
    _real_conso = _r_auto = _r_res = _r_suff = None
    if data_exists and date_in_range:
        _real_conso    = row.get("TotalLoad_kWh", None)
        _real_selfcons = row.get("SelfCons_kWh", None)
        _real_reseau_v = row.get("Consommation réseau [kWh]", None)
        if _real_conso is not None and not pd.isna(_real_conso):
            _has_real = True
            _r_auto = _real_selfcons if (_real_selfcons is not None and not pd.isna(_real_selfcons)) else None
            _r_res  = _real_reseau_v if (_real_reseau_v is not None and not pd.isna(_real_reseau_v)) else None
            _r_suff = (_r_auto / _real_conso * 100) if (_r_auto is not None and _real_conso > 0) else None

    if xgb_conso_pred is None:
        st.info("Modèle de consommation non chargé (modele_xgb_conso_generale.pkl introuvable).")
    elif _has_real:
        _δ_conso_pct = (_xgb_conso_v - _real_conso) / _real_conso * 100 if _real_conso else None
        _δ_auto_pct  = (_xgb_autoconso - _r_auto) / _r_auto * 100 if _r_auto else None
        _δ_res_pct   = (_xgb_reseau - _r_res) / _r_res * 100 if _r_res else None
        _δ_suff      = _xgb_autosuff - _r_suff if _r_suff is not None else None
        _hdr_l, _hdr_r = st.columns(2)
        _hdr_l.markdown("""
        <div style="background:rgba(102,187,106,0.1);border-left:4px solid #66BB6A;
        padding:10px 16px;border-radius:10px;margin-bottom:8px;">
            <span style="font-weight:700;font-size:12px;color:#66BB6A;letter-spacing:1px;">
                ✅ DONNÉES RÉELLES
            </span>
        </div>""", unsafe_allow_html=True)
        _hdr_r.markdown("""
        <div style="background:rgba(66,165,245,0.1);border-left:4px solid #42A5F5;
        padding:10px 16px;border-radius:10px;margin-bottom:8px;">
            <span style="font-weight:700;font-size:12px;color:#42A5F5;letter-spacing:1px;">
                🤖 PRÉDICTION XGBoost
            </span>
        </div>""", unsafe_allow_html=True)
        _r1c1, _r1c2, _r1c3, _r1c4 = st.columns(4)
        _r1c1.metric("Consommation (réelle)", f"{_real_conso:.1f} kWh")
        _r1c2.metric("Autoconsommée réelle", f"{_r_auto:.1f} kWh" if _r_auto is not None else "N/A")
        _r1c3.metric("Consommation", f"{_xgb_conso_v:.1f} kWh",
                     help="Précision modèle ≈ ±3-5%")
        _r1c4.metric("Autoconsommée (prédite)", f"{_xgb_autoconso:.1f} kWh",
                     help="Production après limitation XGBoost")
        _r2c1, _r2c2, _r2c3, _r2c4 = st.columns(4)
        _r2c1.metric("Réseau (réel)", f"{_r_res:.1f} kWh" if _r_res is not None else "N/A")
        _r2c2.metric("Autonomie Énergétique réelle", f"{_r_suff:.1f}%" if _r_suff is not None else "N/A")
        _r2c3.metric("Réseau", f"{_xgb_reseau:.1f} kWh")
        _r2c4.metric("Autonomie Énergétique", f"{_xgb_autosuff:.1f}%")
    else:
        st.markdown("""
        <div style="background: rgba(66,165,245,0.1); border-left:4px solid #42A5F5;
        padding:12px 16px; border-radius:10px; margin:12px 0;">
            <span style="font-weight:700; font-size:12px; color:#42A5F5; letter-spacing:1px;">
                🤖 PRÉDICTION XGBoost
            </span>
        </div>""", unsafe_allow_html=True)
        _cca, _ccb, _ccc, _ccd = st.columns(4)
        _cca.metric("Consommation", f"{_xgb_conso_v:.1f} kWh")
        _ccb.metric("Autoconsommée (PV)", f"{_xgb_autoconso:.1f} kWh",
                    help="Production après limitation XGBoost")
        _ccc.metric("Réseau", f"{_xgb_reseau:.1f} kWh",
                    help="Consommation générale − Production après limitation")
        _ccd.metric("Autonomie Énergétique", f"{_xgb_autosuff:.1f}%",
                    help="Part de la consommation couverte par le PV")

    st.markdown("---")
    st.markdown("## Énergie Écrêtée Journalière")

    _profil_acc = get_day_profile(pred_date, df)

    _df_day_acc = df[df["Date"].dt.date == pred_date]
    _heures_acc, _ecr_acc, _av_acc, _ap_acc = [], [], [], []
    _n_real_h = 0

    for h_acc in range(24):
        _row_h = _df_day_acc[_df_day_acc["Date"].dt.hour == h_acc]
        if not _row_h.empty:
            _av = float(_row_h["Production avant limitation Kwh"].iloc[0] or 0.0)
            _ap = float(_row_h["Production apres limitation [kWh]"].iloc[0] or 0.0)
            if pd.isna(_av): _av = 0.0
            if pd.isna(_ap): _ap = 0.0
            _ec = max(0.0, _av - _ap)
            _n_real_h += 1
        else:
            _r = _profil_acc[_profil_acc["heure"] == h_acc]
            _t = float(_r["T_moy"].values[0]) if len(_r) > 0 else 25.0
            _i = float(_r["Irr_moy"].values[0]) if len(_r) > 0 else 0.0
            _av = predict_xgb(pred_date, h_acc, _t, _i, xgb_avant) or 0.0
            _ap = predict_xgb(pred_date, h_acc, _t, _i, xgb_apres) or 0.0
            _ec = max(0.0, _av - _ap)

        _heures_acc.append(h_acc)
        _av_acc.append(_av)
        _ap_acc.append(_ap)
        _ecr_acc.append(_ec)

    _E_ecr_jour  = sum(_ecr_acc)
    _E_prod_jour = sum(_ap_acc)

    if _n_real_h == 24:
        _src_acc   = "données réelles"
        _prod_lbl  = "Production réelle journée"
        _chart_ttl = f"Profil journalier — Données réelles · {pred_date.strftime('%d/%m/%Y')}"
    elif _n_real_h > 0:
        _src_acc   = f"réelles ({_n_real_h}h) + XGBoost ({24 - _n_real_h}h)"
        _prod_lbl  = "Production journée"
        _chart_ttl = f"Profil journalier — Réelles + XGBoost · {pred_date.strftime('%d/%m/%Y')}"
    else:
        _src_acc   = "XGBoost · Open-Meteo Tanger"
        _prod_lbl  = "Production estimée journée"
        _chart_ttl = f"Profil journalier XGBoost — {pred_date.strftime('%d/%m/%Y')} · Open-Meteo Tanger"

    _ca, _cb, _cc = st.columns(3)
    _ca.metric("Énergie écrêtée journée", f"{_E_ecr_jour:.3f} kWh",
               help=f"Source : {_src_acc}")
    _cb.metric(_prod_lbl, f"{_E_prod_jour:.3f} kWh",
               help=f"Source : {_src_acc}")
    _cc.metric("Taux d'écrêtage journée",
               f"{(_E_ecr_jour / (_E_ecr_jour + _E_prod_jour) * 100) if (_E_ecr_jour + _E_prod_jour) > 0 else 0:.1f}%")

    _fig_ecr_acc = go.Figure()
    _fig_ecr_acc.add_trace(go.Bar(
        x=_heures_acc, y=_ap_acc, name="Production après limitation",
        marker_color=cc["xgb"],
        hovertemplate="<b>%{x}h</b><br>Après : <b>%{y:.4f} kWh</b><extra></extra>"
    ))
    _fig_ecr_acc.add_trace(go.Bar(
        x=_heures_acc, y=_ecr_acc, name="Énergie écrêtée",
        marker_color=cc["ecrete"],
        hovertemplate="<b>%{x}h</b><br>Écrêtée : <b>%{y:.4f} kWh</b><extra></extra>"
    ))
    _fig_ecr_acc.update_layout(
        barmode="stack", height=300,
        title=_chart_ttl,
        xaxis=dict(title="Heure", dtick=1),
        yaxis=dict(title="Énergie (kWh)", gridcolor="rgba(128,128,128,0.2)"),
        legend=dict(orientation="h", y=1.12, x=0.78, xanchor="center", yanchor="bottom"),
        template="plotly_dark" if theme_toggle else "plotly",
        margin=dict(t=80, b=30),
        **get_plot_bg(theme_toggle)
    )
    render_chart(_fig_ecr_acc, use_container_width=True)

    st.markdown("---")
    st.markdown(f"## Indicateurs Clés — {pred_date.strftime('%d/%m/%Y')}")

    _available_kpis = [
        "Production Totale (kWh)", "Production Moyenne (kWh)", "Production Max (kWh)",
        "Autoconsommation Totale (kWh)", "Taux Autoconsommation (%)",
        "Consommation Réseau Totale (kWh)", "Consommation Totale (kWh)",
        "Irradiance Moyenne (W/m²)", "Température Moyenne (°C)",
        "Énergie Écrêtée Totale (kWh)", "Rendement Global (%)",
        "Taux Couverture Solaire (%)", "Autonomie Énergétique (%)"
    ]
    with st.expander("Personnaliser les indicateurs", expanded=False):
        selected_kpis = st.multiselect(
            "Indicateurs à afficher",
            _available_kpis,
            default=_available_kpis,
            label_visibility="collapsed"
        )
    if not selected_kpis:
        selected_kpis = _available_kpis

    _df_day_kpi = df[df["Date"].dt.date == pred_date]
    _df_kpi_source = _df_day_kpi
    _kpi_source_label = "données réelles"
    if len(_df_day_kpi) == 0 and xgb_conso is not None:
        _df_kpi_source = build_predicted_1day_df(pred_date, df, xgb_avant, xgb_apres, xgb_conso)
        _kpi_source_label = "prédiction XGBoost"

    if len(_df_kpi_source) > 0:
        kpis = calculate_kpis(_df_kpi_source, selected_kpis)
        cols = st.columns(3)
        for idx, (kpi_name, kpi_value) in enumerate(kpis.items()):
            with cols[idx % 3]:
                st.metric(
                    kpi_name,
                    f"{kpi_value:.2f}",
                    help=f"Source : {_kpi_source_label}" if _kpi_source_label != "données réelles" else None
                )
    else:
        st.warning("Aucune donnée disponible pour cette période — KPI indisponibles.")
        cols = st.columns(3)
        for idx, kpi_name in enumerate(selected_kpis):
            with cols[idx % 3]:
                st.metric(kpi_name, "N/A")
    
    st.markdown("---")
    st.markdown("## Système d'Alertes")
    display_alerts(alerts, theme=theme)

    st.markdown("---")
    _html_report = generate_html_report(df, df_period, pred_date, temp_input, irrad_input, data_exists)
    st.download_button(
        "📄 Télécharger Rapport HTML — imprimable en PDF",
        data=_html_report,
        file_name=f"rapport_pv_{pred_date.strftime('%Y%m%d')}.html",
        mime="text/html",
        use_container_width=True,
        key="dl_rapport_html"
    )

elif page == "Modèle Théorique":

    if theme_toggle:
        color_prod = "#D7CCC8"
        color_irr = "#FFCC80"
        color_temp = "#FF8A80"
        title_color = "white"
        template_mode = "plotly_dark"
    else:
        color_prod = cc["theorique"]
        color_irr = cc["irr"]
        color_temp = cc["temp"]
        title_color = "#1A1A1A"
        template_mode = "plotly"

    st.markdown("## Modèle Théorique")

    heures = np.arange(0, 24)
    heures_dt = [datetime.combine(pred_date, datetime.min.time()).replace(hour=int(h)) for h in heures]

    _profil_theo = get_day_profile(pred_date, df)
    irr_jour = [
        float(_profil_theo[_profil_theo["heure"] == h]["Irr_moy"].values[0])
        if len(_profil_theo[_profil_theo["heure"] == h]) > 0 else 0.0
        for h in range(24)
    ]
    temp_jour = [
        float(_profil_theo[_profil_theo["heure"] == h]["T_moy"].values[0])
        if len(_profil_theo[_profil_theo["heure"] == h]) > 0 else 25.0
        for h in range(24)
    ]

    theo_jour = [calculate_theoretical_energy(irr, temp, 1) for irr, temp in zip(irr_jour, temp_jour)]
    total_theorique = np.sum(theo_jour)
    theo_selected = calculate_theoretical_energy(irrad_input, temp_input, 1)

    _pr_day_start = datetime.combine(pred_date, datetime.min.time())
    _pr_day_end   = _pr_day_start + pd.Timedelta(days=1)
    _pr_df_real   = df[(df["Date"] >= _pr_day_start) & (df["Date"] < _pr_day_end)].sort_values("Date")

    if len(_pr_df_real) > 0 and "Production avant limitation Kwh" in _pr_df_real.columns:
        _av_day_page = _pr_df_real["Production avant limitation Kwh"].sum()
        _ap_day_page = (_pr_df_real["Production apres limitation [kWh]"].sum()
                        if "Production apres limitation [kWh]" in _pr_df_real.columns
                        else _av_day_page)
        _pr_src_day = "réelle"
    elif xgb_avant is not None:
        _av_day_page = sum(predict_xgb(pred_date, h, temp_jour[h], irr_jour[h], xgb_avant) or 0.0 for h in range(24))
        _ap_day_page = (sum(predict_xgb(pred_date, h, temp_jour[h], irr_jour[h], xgb_apres) or 0.0 for h in range(24))
                        if xgb_apres is not None else _av_day_page)
        _pr_src_day = "XGBoost"
    else:
        _av_day_page = _ap_day_page = None
        _pr_src_day = None

    _pr_hr_row = _pr_df_real[_pr_df_real["Date"].dt.hour == pred_heure] if len(_pr_df_real) > 0 else pd.DataFrame()
    if len(_pr_hr_row) > 0 and "Production avant limitation Kwh" in _pr_hr_row.columns:
        _av_hour_page = float(_pr_hr_row["Production avant limitation Kwh"].mean())
        _ap_hour_page = (float(_pr_hr_row["Production apres limitation [kWh]"].mean())
                         if "Production apres limitation [kWh]" in _pr_hr_row.columns else _av_hour_page)
        _pr_src_hour = "réelle"
    elif xgb_avant is not None:
        _av_hour_page = predict_xgb(pred_date, pred_heure, temp_input, irrad_input, xgb_avant) or 0.0
        _ap_hour_page = (predict_xgb(pred_date, pred_heure, temp_input, irrad_input, xgb_apres) or 0.0
                         if xgb_apres is not None else _av_hour_page)
        _pr_src_hour = "XGBoost"
    else:
        _av_hour_page = _ap_hour_page = None
        _pr_src_hour = None

    _mc1, _mc2 = st.columns(2)
    _mc1.metric(
        f"Production théorique {pred_heure:02d}h",
        f"{theo_selected:.3f} kWh",
        help=f"T°={temp_input:.1f}°C · Irr={irrad_input:.1f} W/m²"
    )
    _mc2.metric(
        "Production théorique journalière",
        f"{total_theorique:.2f} kWh",
        help=f"Somme 24h · Open-Meteo Tanger · {pred_date.strftime('%d/%m/%Y')}"
    )

    if _pr_src_hour is not None and _pr_src_day is not None:
        _mpr1, _mpr2 = st.columns(2)
        _pr_val_h = (_av_hour_page / theo_selected * 100) if theo_selected > 0 else 0.0
        _pr_val_d = (_av_day_page / total_theorique * 100) if total_theorique > 0 else 0.0
        _mpr1.metric(
            f"PR {pred_heure:02d}h",
            f"{_pr_val_h:.1f}%",
            help=f"Production AVANT ({_pr_src_hour}) / Théorique × 100"
        )
        _mpr2.metric(
            "PR journalier",
            f"{_pr_val_d:.1f}%",
            help=f"Production AVANT ({_pr_src_day}) / Théorique × 100"
        )

        _mps1, _mps2 = st.columns(2)
        _mps1.metric(
            f"Pertes système {pred_heure:02d}h (kWh)",
            f"{(theo_selected - _av_hour_page):.3f}",
            help="Théorique − Production AVANT (pertes DC/AC/câbles)"
        )
        _mps2.metric(
            "Pertes système journalières (kWh)",
            f"{(total_theorique - _av_day_page):.2f}",
            help="Théorique − Production AVANT (pertes DC/AC/câbles)"
        )

        _mpt1, _mpt2 = st.columns(2)
        _mpt1.metric(
            f"Pertes totales {pred_heure:02d}h (kWh)",
            f"{(theo_selected - _ap_hour_page):.3f}",
            help="Théorique − Production APRÈS (inclut écrêtage)"
        )
        _mpt2.metric(
            "Pertes totales journalières (kWh)",
            f"{(total_theorique - _ap_day_page):.2f}",
            help="Théorique − Production APRÈS (inclut écrêtage)"
        )

    fig_prod = go.Figure()
    fig_prod.add_trace(go.Scatter(
        x=heures_dt, y=theo_jour,
        mode="lines+markers",
        name="Production théorique",
        line=dict(color=color_prod, width=3),
        marker=dict(size=6),
        hovertemplate="<b>%{x|%H:%M}</b><br>Production : %{y:.3f} kWh<extra></extra>"
    ))
    fig_prod.update_layout(
        height=380, hovermode="x unified", template=template_mode,
        xaxis=dict(title="Heure"),
        yaxis=dict(title="Production (kWh)"),
        **get_plot_bg(theme_toggle)
    )
    render_chart(fig_prod, use_container_width=True)

    _exp_theo = pd.DataFrame({
        "Heure": [h.strftime("%H:%M") for h in heures_dt],
        "Production théorique (kWh)": theo_jour,
        "Irradiance (W/m²)": irr_jour,
        "Température (°C)": temp_jour
    })
    create_download_button(_exp_theo, "modele_theorique.csv", "📥 Exporter (CSV)", key="dl_theo")

    st.markdown("---")
    _day_start = datetime.combine(pred_date, datetime.min.time())
    _day_end   = _day_start + pd.Timedelta(days=1)
    _df_day_real = df[(df["Date"] >= _day_start) & (df["Date"] < _day_end)].sort_values("Date")

    if len(_df_day_real) > 0 and "Production avant limitation Kwh" in _df_day_real.columns:
        st.markdown(f"### Théorique vs Réelle — {pred_date.strftime('%d/%m/%Y')}")

        _theo_day = [calculate_theoretical_energy(
            float(_df_day_real[_df_day_real["Date"].dt.hour == h]["Irradiance_Wpm2"].mean()) if len(_df_day_real[_df_day_real["Date"].dt.hour == h]) > 0 else 0.0,
            float(_df_day_real[_df_day_real["Date"].dt.hour == h]["Temperature_C"].mean()) if len(_df_day_real[_df_day_real["Date"].dt.hour == h]) > 0 else 25.0,
            1
        ) for h in _df_day_real["Date"].dt.hour.unique()]
        _hours_real = sorted(_df_day_real["Date"].dt.hour.unique())
        _theo_day   = [calculate_theoretical_energy(
            float(_df_day_real[_df_day_real["Date"].dt.hour == h]["Irradiance_Wpm2"].mean()) if len(_df_day_real[_df_day_real["Date"].dt.hour == h]) > 0 else 0.0,
            float(_df_day_real[_df_day_real["Date"].dt.hour == h]["Temperature_C"].mean()) if len(_df_day_real[_df_day_real["Date"].dt.hour == h]) > 0 else 25.0,
            1
        ) for h in _hours_real]
        _real_avant_day = [
            float(_df_day_real[_df_day_real["Date"].dt.hour == h]["Production avant limitation Kwh"].mean())
            for h in _hours_real
        ]
        _dates_real = [_day_start.replace(hour=h) for h in _hours_real]

        _total_theo = sum(_theo_day)
        _total_real = sum(_real_avant_day)
        _rendement  = (_total_real / _total_theo * 100) if _total_theo > 0 else 0.0
        _ecart      = _total_real - _total_theo

        _cm1, _cm2, _cm3 = st.columns(3)
        _cm1.metric("Théorique total (kWh)", f"{_total_theo:.2f}")
        _cm2.metric("Réelle AVANT total (kWh)", f"{_total_real:.2f}")
        _cm3.metric("Rendement réel / théorique", f"{_rendement:.1f}%")

        fig_comp_real = go.Figure()
        fig_comp_real.add_trace(go.Scatter(
            x=_dates_real, y=_theo_day,
            mode="lines+markers", name="Théorique (Modèle Physique)",
            line=dict(color=color_prod, width=3, dash="dash"),
            marker=dict(size=6),
            hovertemplate="<b>Théorique</b><br>Heure: %{x|%H:%M}<br>Production: %{y:.3f} kWh<extra></extra>"
        ))
        fig_comp_real.add_trace(go.Scatter(
            x=_dates_real, y=_real_avant_day,
            mode="lines+markers", name="Réelle AVANT limitation",
            line=dict(color=cc["reel"], width=3),
            marker=dict(size=6),
            hovertemplate="<b>Réelle AVANT</b><br>Heure: %{x|%H:%M}<br>Production: %{y:.3f} kWh<extra></extra>"
        ))
        fig_comp_real.update_layout(
            height=420,
            hovermode="x unified",
            template=template_mode,
            title=dict(
                text=f"Théorique vs Réelle AVANT limitation — {pred_date.strftime('%d/%m/%Y')}",
                font=dict(color=title_color, size=17)
            ),
            xaxis=dict(title="Heure du jour"),
            yaxis=dict(title="Production (kWh)"),
            legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top",
                        bgcolor="rgba(255,255,255,0.7)", bordercolor="rgba(0,0,0,0.1)", borderwidth=1),
            **get_plot_bg(theme_toggle)
        )
        render_chart(fig_comp_real, use_container_width=True)
    else:
        st.info(f"ℹ️ Pas de données réelles pour le {pred_date.strftime('%d/%m/%Y')} — comparaison indisponible.")

elif page == "Machine Learning":
    st.markdown("## 1. Production — XGBoost & LSTM")

    xgb_avant_pred = predict_xgb(pred_date, pred_heure, temp_input, irrad_input, xgb_avant)
    xgb_apres_pred = predict_xgb(pred_date, pred_heure, temp_input, irrad_input, xgb_apres)
    lstm_apres_pred = predict_lstm(pred_date, pred_heure, temp_input, irrad_input, lstm_model_apres, scaler_X_lstm_apres, scaler_y_lstm_apres)
    lstm_avant_pred = predict_lstm(pred_date, pred_heure, temp_input, irrad_input, lstm_model_avant, scaler_X_lstm_avant, scaler_y_lstm_avant)

    xgb_conso_pred_ml = predict_xgb_conso(pred_date, pred_heure, temp_input, xgb_conso)

    st.markdown("### XGBoost")
    c1, c2, c3 = st.columns(3)
    c1.metric("AVANT", f"{xgb_avant_pred:.3f} kWh")
    c2.metric("APRÈS", f"{xgb_apres_pred:.3f} kWh")
    c3.metric("ÉCRÊTÉE", f"{max(0, xgb_avant_pred - xgb_apres_pred):.3f} kWh")
    
    if lstm_model_apres is not None:
        st.markdown("### LSTM")
        c4, c5, c6 = st.columns(3)
        c4.metric("AVANT", f"{lstm_avant_pred:.3f} kWh" if lstm_avant_pred is not None else "N/A")
        c5.metric("APRÈS", f"{lstm_apres_pred:.3f} kWh" if lstm_apres_pred is not None else "N/A")
        c6.metric("ÉCRÊTÉE", f"{max(0, lstm_avant_pred - lstm_apres_pred):.3f} kWh" if lstm_avant_pred is not None else "N/A")
    
    if xgb_avant is not None or xgb_apres is not None:
        st.markdown("---")
        st.markdown("### Performance XGBoost")
        _perf_pv = _eval_xgb_perf_full(xgb_avant, xgb_apres, xgb_conso, df)

        if "ap" in _perf_pv:
            st.markdown("**Production APRÈS Limitation**")
            _ca1, _ca2, _ca3 = st.columns(3)
            _ca1.metric("R²", f"{_perf_pv['ap']['r2']:.4f}")
            _ca2.metric("MAE", f"{_perf_pv['ap']['mae']:.4f} kWh")
            _moy_ap = _perf_pv["ap"]["moy"]
            _ca3.metric("MAE %", f"{_perf_pv['ap']['mae']/_moy_ap*100:.1f}%" if _moy_ap > 0 else "N/A")

        if "av" in _perf_pv:
            st.markdown("**Production AVANT Limitation**")
            _cb1, _cb2, _cb3 = st.columns(3)
            _cb1.metric("R²", f"{_perf_pv['av']['r2']:.4f}")
            _cb2.metric("MAE", f"{_perf_pv['av']['mae']:.4f} kWh")
            _moy_av = _perf_pv["av"]["moy"]
            _cb3.metric("MAE %", f"{_perf_pv['av']['mae']/_moy_av*100:.1f}%" if _moy_av > 0 else "N/A")

    if xgb_avant is not None and xgb_apres is not None:
        st.markdown("---")
        st.markdown("## Analyse XGBoost — Importance & Qualité du Modèle")

        _feat_ref = list(_build_features(pred_date, 12, 25.0, 500.0).columns)

        def _get_feat_imp(model, feat_names):
            try:
                sc = model.get_booster().get_score(importance_type="gain")
                total = sum(sc.values()) or 1.0
                return {k: round(v / total * 100, 2) for k, v in sc.items()}
            except Exception:
                fi = model.feature_importances_
                return {feat_names[i]: round(float(fi[i]) * 100, 2) for i in range(len(feat_names))}

        _imp_av = _get_feat_imp(xgb_avant, _feat_ref)
        _imp_ap = _get_feat_imp(xgb_apres, _feat_ref)
        _srt_av = sorted(_imp_av.items(), key=lambda x: x[1], reverse=True)
        _srt_ap = sorted(_imp_ap.items(), key=lambda x: x[1], reverse=True)

        _fig_imp = make_subplots(
            rows=1, cols=2,
            subplot_titles=["AVANT Limitation", "APRÈS Limitation"],
            horizontal_spacing=0.16
        )
        _fig_imp.add_trace(go.Bar(
            x=[v for _, v in _srt_av], y=[k for k, _ in _srt_av],
            orientation="h", marker_color=cc["avant"],
            text=[f"{v:.1f}%" for _, v in _srt_av], textposition="outside",
            hovertemplate="<b>%{y}</b> : %{x:.2f}%<extra></extra>",
        ), row=1, col=1)
        _fig_imp.add_trace(go.Bar(
            x=[v for _, v in _srt_ap], y=[k for k, _ in _srt_ap],
            orientation="h", marker_color=cc["xgb"],
            text=[f"{v:.1f}%" for _, v in _srt_ap], textposition="outside",
            hovertemplate="<b>%{y}</b> : %{x:.2f}%<extra></extra>",
        ), row=1, col=2)
        _fig_imp.update_layout(
            height=540, showlegend=False,
            template="plotly_dark" if theme_toggle else "plotly",
            margin=dict(l=170, r=90, t=60, b=30),
            **get_plot_bg(theme_toggle)
        )
        _fig_imp.update_xaxes(title_text="Importance (%)", ticksuffix="%")
        render_chart(_fig_imp, use_container_width=True)

    if lstm_model_apres is not None:
        st.markdown("---")
        _lstm_end   = datetime.combine(pred_date, datetime.min.time()).replace(hour=pred_heure)
        _lstm_start = _lstm_end - pd.Timedelta(days=7)
        _df_lstm_ref = df[(df["Date"] >= _lstm_start) & (df["Date"] <= _lstm_end)].copy()
        if len(_df_lstm_ref) == 0:
            _last_end   = df["Date"].max()
            _df_lstm_ref = df[(df["Date"] >= _last_end - pd.Timedelta(days=7)) & (df["Date"] <= _last_end)].copy()

        if len(_df_lstm_ref) > 0:
            _df_lstm_ref["LSTM_Pred_Ap"] = _df_lstm_ref.apply(
                lambda r: predict_lstm(r["Date"].date(), r["Date"].hour, r["Temperature_C"], r["Irradiance_Wpm2"],
                                       lstm_model_apres, scaler_X_lstm_apres, scaler_y_lstm_apres), axis=1)
            if lstm_model_avant is not None and scaler_X_lstm_avant is not None:
                _df_lstm_ref["LSTM_Pred_Av"] = _df_lstm_ref.apply(
                    lambda r: predict_lstm(r["Date"].date(), r["Date"].hour, r["Temperature_C"], r["Irradiance_Wpm2"],
                                           lstm_model_avant, scaler_X_lstm_avant, scaler_y_lstm_avant), axis=1)

        st.markdown("### Performance LSTM")
        if len(_df_lstm_ref) > 0:
            st.markdown("**Production APRÈS Limitation**")
            _sl_ap = _df_lstm_ref.dropna(subset=["Production apres limitation [kWh]", "LSTM_Pred_Ap"])
            if len(_sl_ap) > 1:
                _mae_l_ap = mean_absolute_error(_sl_ap["Production apres limitation [kWh]"], _sl_ap["LSTM_Pred_Ap"])
                _r2_l_ap  = r2_score(_sl_ap["Production apres limitation [kWh]"], _sl_ap["LSTM_Pred_Ap"])
                _moy_l_ap = _sl_ap["Production apres limitation [kWh]"].mean()
                _lc1, _lc2, _lc3 = st.columns(3)
                _lc1.metric("R²",    f"{_r2_l_ap:.4f}")
                _lc2.metric("MAE",   f"{_mae_l_ap:.4f} kWh")
                _lc3.metric("MAE %", f"{_mae_l_ap/_moy_l_ap*100:.1f}%" if _moy_l_ap > 0 else "N/A")
            if "LSTM_Pred_Av" in _df_lstm_ref.columns and "Production avant limitation Kwh" in _df_lstm_ref.columns:
                st.markdown("**Production AVANT Limitation**")
                _sl_av = _df_lstm_ref.dropna(subset=["Production avant limitation Kwh", "LSTM_Pred_Av"])
                if len(_sl_av) > 1:
                    _mae_l_av = mean_absolute_error(_sl_av["Production avant limitation Kwh"], _sl_av["LSTM_Pred_Av"])
                    _r2_l_av  = r2_score(_sl_av["Production avant limitation Kwh"], _sl_av["LSTM_Pred_Av"])
                    _moy_l_av = _sl_av["Production avant limitation Kwh"].mean()
                    _ld1, _ld2, _ld3 = st.columns(3)
                    _ld1.metric("R²",    f"{_r2_l_av:.4f}")
                    _ld2.metric("MAE",   f"{_mae_l_av:.4f} kWh")
                    _ld3.metric("MAE %", f"{_mae_l_av/_moy_l_av*100:.1f}%" if _moy_l_av > 0 else "N/A")

    if xgb_conso is not None:
        st.markdown("---")
        st.markdown("## 2. Consommation — XGBoost")

        if xgb_conso_pred_ml is not None:
            _cs2_ap      = xgb_apres_pred or 0.0
            _cs2_reseau  = max(0.0, xgb_conso_pred_ml - _cs2_ap)
            _cs2_autono  = (_cs2_ap / xgb_conso_pred_ml * 100) if xgb_conso_pred_ml > 0 else 0.0
            _cv1, _cv2, _cv3 = st.columns(3)
            _cv1.metric(
                "Consommation générale",
                f"{xgb_conso_pred_ml:.1f} kWh",
                help="Prédiction XGBoost consommation"
            )
            _cv2.metric(
                "Réseau",
                f"{_cs2_reseau:.1f} kWh",
                help="Consommation générale − Production après limitation"
            )
            _cv3.metric(
                "Autonomie",
                f"{_cs2_autono:.1f}%",
                help="Production après limitation / Consommation générale × 100"
            )

        _perf_co = _eval_xgb_perf_full(xgb_avant, xgb_apres, xgb_conso, df)
        st.markdown("### Performance XGBoost")
        if "co" in _perf_co:
            _moy_cp = _perf_co["co"]["moy"]
            _cp1, _cp2, _cp3 = st.columns(3)
            _cp1.metric("R²",    f"{_perf_co['co']['r2']:.4f}")
            _cp2.metric("MAE",   f"{_perf_co['co']['mae']:.1f} kWh")
            _cp3.metric("MAE %", f"{_perf_co['co']['mae']/_moy_cp*100:.1f}%" if _moy_cp > 0 else "N/A")

        st.markdown("### Importance des variables — XGBoost Consommation")
        _feat_conso = [
            "sin_jour", "cos_jour", "sin_heure", "cos_heure",
            "sin_mois", "cos_mois",
            "temp_c", "temp_c_sq", "temp_c_cu",
            "temp_sin_jour", "temp_cos_mois", "temp_sin_heure",
            "heure_int",
        ]
        try:
            _sc_conso = xgb_conso.get_booster().get_score(importance_type="gain")
            _tot_conso = sum(_sc_conso.values()) or 1.0
            _imp_conso_raw = {k: round(v / _tot_conso * 100, 2) for k, v in _sc_conso.items()}
        except Exception:
            _fi_conso = xgb_conso.feature_importances_
            _imp_conso_raw = {_feat_conso[i]: round(float(_fi_conso[i]) * 100, 2) for i in range(len(_feat_conso))}
        _srt_conso = sorted(_imp_conso_raw.items(), key=lambda x: x[1], reverse=True)

        _fig_imp_conso = go.Figure()
        _fig_imp_conso.add_trace(go.Bar(
            x=[v for _, v in _srt_conso], y=[k for k, _ in _srt_conso],
            orientation="h", marker_color=cc["conso"],
            text=[f"{v:.1f}%" for _, v in _srt_conso], textposition="outside",
            hovertemplate="<b>%{y}</b> : %{x:.2f}%<extra></extra>",
        ))
        _fig_imp_conso.update_layout(
            height=480, showlegend=False,
            title="Importance des variables — Modèle XGBoost Consommation",
            template="plotly_dark" if theme_toggle else "plotly",
            margin=dict(l=180, r=90, t=60, b=30),
            xaxis=dict(title="Importance (%)", ticksuffix="%"),
            **get_plot_bg(theme_toggle)
        )
        render_chart(_fig_imp_conso, use_container_width=True)

    if xgb_avant is not None or xgb_apres is not None or xgb_conso is not None:
        st.markdown("---")
        st.markdown("## Comparaison Réelle vs XGBoost — Journée sélectionnée")

        _df_day = df[df["Date"].dt.date == pred_date].copy()
        _has_day = len(_df_day) > 0

        if not _has_day:
            st.info("Sélectionnez une date historique pour afficher les courbes réelles vs prédites.")
        else:
            _df_day["Pred_Avant"] = _df_day.apply(
                lambda r: predict_xgb(r["Date"].date(), r["Date"].hour,
                                      r["Temperature_C"], r["Irradiance_Wpm2"], xgb_avant) or 0.0, axis=1)
            _df_day["Pred_Apres"] = _df_day.apply(
                lambda r: predict_xgb(r["Date"].date(), r["Date"].hour,
                                      r["Temperature_C"], r["Irradiance_Wpm2"], xgb_apres) or 0.0, axis=1)
            _df_day["Pred_Conso"] = _df_day.apply(
                lambda r: predict_xgb_conso(r["Date"].date(), r["Date"].hour,
                                            r["Temperature_C"], xgb_conso) or 0.0, axis=1)

            _lyt_rp = dict(hovermode="x unified", height=360,
                           xaxis_title="Heure", margin=dict(t=60),
                           legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", yanchor="bottom"),
                           template="plotly_dark" if theme_toggle else "plotly",
                           **get_plot_bg(theme_toggle))

            if xgb_avant is not None and "Production avant limitation Kwh" in _df_day.columns:
                st.markdown("### Production AVANT Limitation — Réelle vs XGBoost")
                _fig_rp1 = go.Figure()
                _fig_rp1.add_trace(go.Scatter(
                    x=_df_day["Date"], y=_df_day["Production avant limitation Kwh"],
                    mode="lines", name="Réelle", line=dict(color=cc["reel"], width=2),
                    hovertemplate="<b>Réelle</b><br>%{x|%Hh}<br>%{y:.3f} kWh<extra></extra>"))
                _fig_rp1.add_trace(go.Scatter(
                    x=_df_day["Date"], y=_df_day["Pred_Avant"],
                    mode="lines", name="XGBoost", line=dict(color=cc["xgb"], width=2, dash="dash"),
                    hovertemplate="<b>XGBoost</b><br>%{x|%Hh}<br>%{y:.3f} kWh<extra></extra>"))
                _fig_rp1.update_layout(yaxis_title="Production AVANT limitation (kWh)", **_lyt_rp)
                render_chart(_fig_rp1, use_container_width=True)

            if xgb_apres is not None and "Production apres limitation [kWh]" in _df_day.columns:
                st.markdown("### Production APRÈS Limitation — Réelle vs XGBoost")
                _fig_rp2 = go.Figure()
                _fig_rp2.add_trace(go.Scatter(
                    x=_df_day["Date"], y=_df_day["Production apres limitation [kWh]"],
                    mode="lines", name="Réelle", line=dict(color=cc["reel"], width=2),
                    hovertemplate="<b>Réelle</b><br>%{x|%Hh}<br>%{y:.3f} kWh<extra></extra>"))
                _fig_rp2.add_trace(go.Scatter(
                    x=_df_day["Date"], y=_df_day["Pred_Apres"],
                    mode="lines", name="XGBoost", line=dict(color=cc["xgb"], width=2, dash="dash"),
                    hovertemplate="<b>XGBoost</b><br>%{x|%Hh}<br>%{y:.3f} kWh<extra></extra>"))
                _fig_rp2.update_layout(yaxis_title="Production APRÈS limitation (kWh)", **_lyt_rp)
                render_chart(_fig_rp2, use_container_width=True)

            if xgb_conso is not None and "TotalLoad_kWh" in _df_day.columns:
                st.markdown("### Consommation Générale — Réelle vs XGBoost")
                _fig_rp3 = go.Figure()
                _fig_rp3.add_trace(go.Scatter(
                    x=_df_day["Date"], y=_df_day["TotalLoad_kWh"],
                    mode="lines", name="Réelle", line=dict(color=cc["conso"], width=2),
                    hovertemplate="<b>Réelle</b><br>%{x|%Hh}<br>%{y:.1f} kWh<extra></extra>"))
                _fig_rp3.add_trace(go.Scatter(
                    x=_df_day["Date"], y=_df_day["Pred_Conso"],
                    mode="lines", name="XGBoost", line=dict(color=cc["conso_pred"], width=2, dash="dash"),
                    hovertemplate="<b>XGBoost</b><br>%{x|%Hh}<br>%{y:.1f} kWh<extra></extra>"))
                _fig_rp3.update_layout(yaxis_title="Consommation (kWh)", **_lyt_rp)
                render_chart(_fig_rp3, use_container_width=True)

            st.markdown("### Métriques — 3 Modèles XGBoost")
            _met_rp = []
            if xgb_avant is not None and "Production avant limitation Kwh" in _df_day.columns:
                _s = _df_day.dropna(subset=["Production avant limitation Kwh", "Pred_Avant"])
                if len(_s) > 1:
                    _mae = mean_absolute_error(_s["Production avant limitation Kwh"], _s["Pred_Avant"])
                    _r2  = r2_score(_s["Production avant limitation Kwh"], _s["Pred_Avant"])
                    _moy = _s["Production avant limitation Kwh"].mean()
                    _met_rp.append({"Modèle": "Production AVANT limitation", "R²": f"{_r2:.4f}",
                        "MAE (kWh)": f"{_mae:.4f}", "MAE (%)": f"{_mae/_moy*100:.1f}%" if _moy > 0 else "—"})
            if xgb_apres is not None and "Production apres limitation [kWh]" in _df_day.columns:
                _s = _df_day.dropna(subset=["Production apres limitation [kWh]", "Pred_Apres"])
                if len(_s) > 1:
                    _mae = mean_absolute_error(_s["Production apres limitation [kWh]"], _s["Pred_Apres"])
                    _r2  = r2_score(_s["Production apres limitation [kWh]"], _s["Pred_Apres"])
                    _moy = _s["Production apres limitation [kWh]"].mean()
                    _met_rp.append({"Modèle": "Production APRÈS limitation", "R²": f"{_r2:.4f}",
                        "MAE (kWh)": f"{_mae:.4f}", "MAE (%)": f"{_mae/_moy*100:.1f}%" if _moy > 0 else "—"})
            if xgb_conso is not None and "TotalLoad_kWh" in _df_day.columns:
                _s = _df_day.dropna(subset=["TotalLoad_kWh", "Pred_Conso"])
                if len(_s) > 1:
                    _mae = mean_absolute_error(_s["TotalLoad_kWh"], _s["Pred_Conso"])
                    _r2  = r2_score(_s["TotalLoad_kWh"], _s["Pred_Conso"])
                    _moy = _s["TotalLoad_kWh"].mean()
                    _met_rp.append({"Modèle": "Consommation Générale", "R²": f"{_r2:.4f}",
                        "MAE (kWh)": f"{_mae:.1f}", "MAE (%)": f"{_mae/_moy*100:.1f}%" if _moy > 0 else "—"})
            if _met_rp:
                st.dataframe(pd.DataFrame(_met_rp), use_container_width=True, hide_index=True)

elif page == "Graphiques":
    _df_day_g  = df[df["Date"].dt.date == pred_date].copy()
    _has_day_g = len(_df_day_g) > 0

    _heures_g = list(range(24))
    _dates_g  = [datetime.combine(pred_date, datetime.min.time()).replace(hour=h) for h in _heures_g]

    _profil_g = get_day_profile(pred_date, df)
    _temps_g = []; _irrs_g = []
    for _hg in _heures_g:
        _row_g = _profil_g[_profil_g["heure"] == _hg]
        _temps_g.append(float(_row_g["T_moy"].values[0])   if len(_row_g) > 0 else 25.0)
        _irrs_g.append(float(_row_g["Irr_moy"].values[0])  if len(_row_g) > 0 else 0.0)

    _xgb_av_g    = [predict_xgb(pred_date, h, _temps_g[h], _irrs_g[h], xgb_avant)  or 0.0 for h in _heures_g]
    _xgb_ap_g    = [predict_xgb(pred_date, h, _temps_g[h], _irrs_g[h], xgb_apres) or 0.0 for h in _heures_g]
    _xgb_conso_g = ([predict_xgb_conso(pred_date, h, _temps_g[h], xgb_conso) or 0.0 for h in _heures_g]
                    if xgb_conso is not None else None)
    _autoconso_g = ([min(_xgb_ap_g[h], _xgb_conso_g[h]) for h in _heures_g]
                    if _xgb_conso_g is not None else _xgb_ap_g)
    _reseau_g    = ([max(0.0, _xgb_conso_g[h] - _xgb_ap_g[h]) for h in _heures_g]
                    if _xgb_conso_g is not None else [0.0] * 24)

    st.markdown(f"### Bilan de la Journée — {pred_date.strftime('%d/%m/%Y')}")

    fig_grp = go.Figure()

    _C = {
        "prod_av"   : "#FF8F00",
        "prod_ap"   : "#7E57C2",
        "autoconso" : "#4CAF50",
        "reseau"    : "#FF7043",
        "reel_ap"   : "#EC407A",
    }

    if _has_day_g and "Production apres limitation [kWh]" in _df_day_g.columns:
        fig_grp.add_trace(go.Scatter(
            x=_df_day_g["Date"], y=_df_day_g["Production apres limitation [kWh]"],
            mode="lines", name="Production réelle (après)",
            line=dict(color=_C["reel_ap"], width=2, dash="dot"),
            hovertemplate="<b>Réelle APRÈS</b><br>%{x|%Hh} → %{y:.3f} kWh<extra></extra>"))

    fig_grp.add_trace(go.Scatter(
        x=_dates_g, y=_xgb_av_g, mode="lines",
        name="Production avant limitation (XGB)",
        line=dict(color=_C["prod_av"], width=2.5),
        hovertemplate="<b>Production AVANT (XGB)</b><br>%{x|%Hh} → %{y:.3f} kWh<extra></extra>"))
    fig_grp.add_trace(go.Scatter(
        x=_dates_g, y=_xgb_ap_g, mode="lines",
        name="Production après limitation (XGB)",
        line=dict(color=_C["prod_ap"], width=2.5),
        fill="tonexty", fillcolor="rgba(255,143,0,0.18)",
        hovertemplate="<b>Production APRÈS (XGB)</b><br>%{x|%Hh} → %{y:.3f} kWh<extra></extra>"))
    fig_grp.add_trace(go.Scatter(
        x=_dates_g, y=_autoconso_g, mode="lines",
        name="Autoconsommation",
        line=dict(color=_C["autoconso"], width=2.5),
        fill="tozeroy", fillcolor="rgba(76,175,80,0.12)",
        hovertemplate="<b>Autoconsommation</b><br>%{x|%Hh} → %{y:.3f} kWh<extra></extra>"))
    if _xgb_conso_g is not None:
        fig_grp.add_trace(go.Scatter(
            x=_dates_g, y=_reseau_g, mode="lines",
            name="Consommation Réseau",
            line=dict(color=_C["reseau"], width=2.5),
            fill="tozeroy", fillcolor="rgba(255,112,67,0.10)",
            hovertemplate="<b>Consommation Réseau</b><br>%{x|%Hh} → %{y:.3f} kWh<extra></extra>"))

    fig_grp.update_layout(
        height=480, hovermode="x unified",
        xaxis_title="Heure", yaxis_title="Énergie (kWh)",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", yanchor="bottom"),
        margin=dict(t=80),
        template="plotly_dark" if theme_toggle else "plotly",
        **get_plot_bg(theme_toggle))
    render_chart(fig_grp, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Production (après limitation)")
        fig1 = go.Figure()
        if _has_day_g and "Production apres limitation [kWh]" in _df_day_g.columns:
            fig1.add_trace(go.Scatter(
                x=_df_day_g["Date"], y=_df_day_g["Production apres limitation [kWh]"],
                mode="lines", name="Réelle",
                line=dict(color=cc["reel"], width=2),
                hovertemplate="<b>Réelle</b><br>%{x|%Hh} → %{y:.3f} kWh<extra></extra>"))
        fig1.add_trace(go.Scatter(
            x=_dates_g, y=_xgb_ap_g, mode="lines", name="XGBoost",
            line=dict(color=cc["xgb"], width=2, dash="dash"),
            hovertemplate="<b>XGBoost</b><br>%{x|%Hh} → %{y:.3f} kWh<extra></extra>"))
        fig1.update_xaxes(title_text="Heure")
        fig1.update_yaxes(title_text="Production (kWh)")
        fig1.update_layout(
            height=350, hovermode="x unified",
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", yanchor="bottom"),
            margin=dict(t=60),
            template="plotly_dark" if theme_toggle else "plotly", **get_plot_bg(theme_toggle))
        _exp_g1 = pd.DataFrame({
            "Heure": _dates_g,
            "Production XGB (kWh)": _xgb_ap_g,
        })
        if _has_day_g and "Production apres limitation [kWh]" in _df_day_g.columns:
            _exp_g1["Production réelle (kWh)"] = _df_day_g.set_index(_df_day_g["Date"].dt.hour).reindex(range(24))["Production apres limitation [kWh]"].values
        create_download_button(_exp_g1, "graphique_production.csv", "📥 Exporter (CSV)", key="dl_g1")
        render_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("#### Conditions Météo (Open-Meteo)")
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        _meteo_dates = _dates_g
        fig2.add_trace(go.Scatter(
            x=_meteo_dates, y=_irrs_g, mode="lines", name="Irradiance",
            line=dict(color=cc["irr"], width=2),
            hovertemplate="<b>Irradiance</b><br>%{x|%Hh} → %{y:.1f} W/m²<extra></extra>"),
            secondary_y=False)
        fig2.add_trace(go.Scatter(
            x=_meteo_dates, y=_temps_g, mode="lines", name="Température",
            line=dict(color="#1976D2", width=2),
            hovertemplate="<b>Température</b><br>%{x|%Hh} → %{y:.1f} °C<extra></extra>"),
            secondary_y=True)
        fig2.update_xaxes(title_text="Heure")
        fig2.update_yaxes(title_text="Irradiance (W/m²)", secondary_y=False)
        fig2.update_yaxes(title_text="Température (°C)", secondary_y=True)
        fig2.update_layout(
            height=350, hovermode="x unified",
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", yanchor="bottom"),
            margin=dict(t=60),
            template="plotly_dark" if theme_toggle else "plotly", **get_plot_bg(theme_toggle))
        _exp_g2 = pd.DataFrame({"Heure": _meteo_dates, "Irradiance (W/m²)": _irrs_g, "Température (°C)": _temps_g})
        create_download_button(_exp_g2, "graphique_meteo.csv", "📥 Exporter (CSV)", key="dl_g2")
        render_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Autoconsommation")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=_dates_g, y=_autoconso_g, mode="lines", name="Autoconsommation",
            line=dict(color=cc["autoconso"], width=2.5), fill="tozeroy",
            hovertemplate="<b>Autoconsommation</b><br>%{x|%Hh} → %{y:.3f} kWh<extra></extra>"))
        if _has_day_g and "SelfCons_kWh" in _df_day_g.columns:
            fig3.add_trace(go.Scatter(
                x=_df_day_g["Date"], y=_df_day_g["SelfCons_kWh"], mode="lines",
                name="Réelle", line=dict(color=cc["reel"], width=2, dash="dot"),
                hovertemplate="<b>Réelle</b><br>%{x|%Hh} → %{y:.3f} kWh<extra></extra>"))
        fig3.update_xaxes(title_text="Heure")
        fig3.update_yaxes(title_text="Autoconsommation (kWh)")
        fig3.update_layout(
            height=350, hovermode="x unified",
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", yanchor="bottom"),
            margin=dict(t=60),
            template="plotly_dark" if theme_toggle else "plotly", **get_plot_bg(theme_toggle))
        _exp_g3 = pd.DataFrame({"Heure": _dates_g, "Autoconsommation XGB (kWh)": _autoconso_g})
        create_download_button(_exp_g3, "graphique_autoconso.csv", "📥 Exporter (CSV)", key="dl_g3")
        render_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("#### Consommation Réseau")
        fig4 = go.Figure()
        if _xgb_conso_g is not None:
            fig4.add_trace(go.Scatter(
                x=_dates_g, y=_xgb_conso_g, mode="lines", name="Conso. Générale (XGB)",
                line=dict(color=cc["conso"], width=2),
                hovertemplate="<b>Conso. Générale</b><br>%{x|%Hh} → %{y:.3f} kWh<extra></extra>"))
            fig4.add_trace(go.Scatter(
                x=_dates_g, y=_reseau_g, mode="lines", name="Réseau (XGB)",
                line=dict(color=cc["reseau"], width=2.5), fill="tozeroy",
                hovertemplate="<b>Réseau</b><br>%{x|%Hh} → %{y:.3f} kWh<extra></extra>"))
        elif _has_day_g and "Consommation réseau [kWh]" in _df_day_g.columns:
            fig4.add_trace(go.Scatter(
                x=_df_day_g["Date"], y=_df_day_g["Consommation réseau [kWh]"],
                mode="lines", name="Réseau (réel)",
                line=dict(color=cc["reseau"], width=2.5), fill="tozeroy",
                hovertemplate="<b>Réseau</b><br>%{x|%Hh} → %{y:.3f} kWh<extra></extra>"))
        fig4.update_xaxes(title_text="Heure")
        fig4.update_yaxes(title_text="Consommation (kWh)")
        fig4.update_layout(
            height=350, hovermode="x unified",
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", yanchor="bottom"),
            margin=dict(t=60),
            template="plotly_dark" if theme_toggle else "plotly", **get_plot_bg(theme_toggle))
        if _xgb_conso_g is not None:
            _exp_g4 = pd.DataFrame({"Heure": _dates_g, "Conso. Générale XGB (kWh)": _xgb_conso_g, "Réseau XGB (kWh)": _reseau_g})
        elif _has_day_g and "Consommation réseau [kWh]" in _df_day_g.columns:
            _exp_g4 = _df_day_g[["Date", "Consommation réseau [kWh]"]].rename(columns={"Consommation réseau [kWh]": "Réseau (kWh)"})
        else:
            _exp_g4 = pd.DataFrame()
        if not _exp_g4.empty:
            create_download_button(_exp_g4, "graphique_consommation.csv", "📥 Exporter (CSV)", key="dl_g4")
        render_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown("### Répartition de la Consommation")

    if _xgb_conso_g is not None:
        _tot_conso_pie   = sum(_xgb_conso_g)
        _tot_autoconso_pie = sum(_autoconso_g)
        _tot_reseau_pie  = sum(_reseau_g)
    elif _has_day_g and "TotalLoad_kWh" in _df_day_g.columns:
        _tot_conso_pie     = _df_day_g["TotalLoad_kWh"].sum()
        _tot_autoconso_pie = _df_day_g["SelfCons_kWh"].sum() if "SelfCons_kWh" in _df_day_g.columns else 0.0
        _tot_reseau_pie    = _df_day_g["Consommation réseau [kWh]"].sum() if "Consommation réseau [kWh]" in _df_day_g.columns else 0.0
    else:
        _tot_conso_pie = _tot_autoconso_pie = _tot_reseau_pie = 0.0

    if _tot_conso_pie > 0:
        _pie_colors = ["#4CAF50", "#FF9800"] if theme_toggle else ["#2E7D32", "#E65100"]
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Autoconsommée (PV)", "Réseau"],
            values=[_tot_autoconso_pie, _tot_reseau_pie],
            hole=0.4, marker_colors=_pie_colors,
            textinfo="label+percent", textposition="inside")])
        fig_pie.update_layout(
            height=400, template="plotly_dark" if theme_toggle else "plotly",
            title=f"Répartition de la consommation — {pred_date.strftime('%d/%m/%Y')}",
            **get_plot_bg(theme_toggle))
        render_chart(fig_pie, use_container_width=True)

        colp1, colp2, colp3 = st.columns(3)
        colp1.metric("Consommation journalière", f"{_tot_conso_pie:.1f} kWh")
        colp2.metric("Part autoconsommée", f"{_tot_autoconso_pie:.1f} kWh ({_tot_autoconso_pie/_tot_conso_pie*100:.1f}%)")
        colp3.metric("Part réseau", f"{_tot_reseau_pie:.1f} kWh ({_tot_reseau_pie/_tot_conso_pie*100:.1f}%)")
    else:
        st.info("Modèle de consommation non disponible — répartition non calculable.")

    st.markdown("---")
    st.markdown("### Analyse Saisonnière")
    st.markdown("Comparaison de la production, rendement et conditions par saison.")

    if len(df) == 0:
        st.warning("Aucune donnée disponible")
    else:
        def _get_season(m):
            if m in [12, 1, 2]: return "Hiver"
            if m in [3, 4, 5]: return "Printemps"
            if m in [6, 7, 8]: return "Été"
            return "Automne"

        df_s = df.copy()
        df_s["saison"] = df_s["Date"].dt.month.apply(_get_season)
        df_s["rendement_s"] = np.where(
            df_s["PV_theoretical_kWh"] > 0.01,
            df_s["Production apres limitation [kWh]"] / df_s["PV_theoretical_kWh"] * 100, np.nan)

        s_stats = df_s.groupby("saison").agg(
            prod_tot=("Production apres limitation [kWh]", "sum"),
            ecrete_tot=("Gap_kWh", "sum"),
            irr_moy=("Irradiance_Wpm2", "mean"),
            temp_moy=("Temperature_C", "mean"),
            rend_moy=("rendement_s", "mean")
        ).reset_index()

        s_order = ["Hiver", "Printemps", "Été", "Automne"]
        s_colors_map = (
            {"Hiver": "#42A5F5", "Printemps": "#66BB6A", "Été": "#FFC107", "Automne": "#FF7043"}
            if theme_toggle else
            {"Hiver": "#1565C0", "Printemps": "#2E7D32", "Été": "#E65100", "Automne": "#BF360C"}
        )
        s_stats["saison"] = pd.Categorical(s_stats["saison"], categories=s_order, ordered=True)
        s_stats = s_stats.sort_values("saison")
        clrs_s = [s_colors_map.get(str(s), "#FFC107") for s in s_stats["saison"]]

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fig_s1 = go.Figure(go.Bar(x=s_stats["saison"], y=s_stats["prod_tot"],
                                       marker_color=clrs_s,
                                       text=[f"{v:.0f}" for v in s_stats["prod_tot"]],
                                       textposition="outside"))
            fig_s1.update_layout(height=350, yaxis_title="Production totale (kWh)",
                                  title="Production par saison",
                                  template="plotly_dark" if theme_toggle else "plotly",
                                  **get_plot_bg(theme_toggle))
            render_chart(fig_s1, use_container_width=True)

        with col_s2:
            fig_s2 = go.Figure(go.Bar(x=s_stats["saison"], y=s_stats["rend_moy"],
                                       marker_color=clrs_s,
                                       text=[f"{v:.1f}%" for v in s_stats["rend_moy"]],
                                       textposition="outside"))
            fig_s2.add_hline(y=85, line_dash="dash", line_color="orange",
                              annotation_text="Seuil alerte 85%")
            fig_s2.update_layout(height=350, yaxis_title="Rendement moyen (%)",
                                  title="Rendement par saison",
                                  template="plotly_dark" if theme_toggle else "plotly",
                                  **get_plot_bg(theme_toggle))
            render_chart(fig_s2, use_container_width=True)

        col_s3, col_s4 = st.columns(2)
        with col_s3:
            fig_s3 = go.Figure(go.Bar(x=s_stats["saison"], y=s_stats["irr_moy"],
                                       marker_color=clrs_s,
                                       text=[f"{v:.0f}" for v in s_stats["irr_moy"]],
                                       textposition="outside"))
            fig_s3.update_layout(height=300, yaxis_title="Irradiance moy. (W/m²)",
                                  title="Irradiance par saison",
                                  template="plotly_dark" if theme_toggle else "plotly",
                                  **get_plot_bg(theme_toggle))
            render_chart(fig_s3, use_container_width=True)

        with col_s4:
            fig_s4 = go.Figure(go.Bar(x=s_stats["saison"], y=s_stats["temp_moy"],
                                       marker_color=clrs_s,
                                       text=[f"{v:.1f}°C" for v in s_stats["temp_moy"]],
                                       textposition="outside"))
            fig_s4.update_layout(height=300, yaxis_title="Température moy. (°C)",
                                  title="Température par saison",
                                  template="plotly_dark" if theme_toggle else "plotly",
                                  **get_plot_bg(theme_toggle))
            render_chart(fig_s4, use_container_width=True)

        st.markdown("---")
        df_s_exp = s_stats.rename(columns={
            "saison": "Saison", "prod_tot": "Production totale (kWh)", "ecrete_tot": "Écrêtage total (kWh)",
            "irr_moy": "Irradiance moy. (W/m²)", "temp_moy": "Temp. moy. (°C)", "rend_moy": "Rendement moy. (%)"
        }).round(2)
        st.dataframe(df_s_exp, use_container_width=True)
        create_download_button(df_s_exp, "analyse_saisonniere.csv",
                               "📥 Exporter analyse saisonnière (CSV)", key="dl_saison")

elif page == "Scénarios":
    st.markdown("## Scénarios d'Optimisation de l'Énergie Écrêtée")

    st.markdown("""
<style>
/* Label du radio : caché (on utilise notre propre titre) */
div[data-testid="stRadio"] > label { display: none !important; }

/* Conteneur des boutons radio */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
}

/* Chaque option : style "pill" */
div[data-testid="stRadio"] > div[role="radiogroup"] > label {
    display: flex;
    align-items: center;
    gap: 8px;
    background: white;
    border: 2px solid #FF9800;
    border-radius: 30px;
    padding: 10px 26px;
    font-size: 15px;
    font-weight: 600;
    color: #E65100;
    cursor: pointer;
    transition: background 0.2s, color 0.2s;
    box-shadow: 0 2px 6px rgba(255,152,0,0.15);
}

/* Option sélectionnée */
div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(135deg, #FF9800, #F57C00);
    color: white !important;
    border-color: #F57C00;
    box-shadow: 0 4px 12px rgba(255,152,0,0.35);
}

/* Masquer le bouton radio natif */
div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
    display: none;
}
</style>
""", unsafe_allow_html=True)

    st.markdown(
        '<p style="font-size:20px;font-weight:700;color:#E65100;margin:0 0 10px 0;">'
        '🔀 Sélectionner un scénario</p>',
        unsafe_allow_html=True
    )
    scenario_choix = st.radio(
        "Sélectionner un scénario :",
        ["🚗 Recharge Véhicules Électriques",
         "❄️ Stockage Froid — Climatisation nocturne",
         "🌡️ PAC Eau Chaude — Chauffage & Process"],
        horizontal=True,
        key="scenario_choix",
        label_visibility="collapsed"
    )
    st.markdown("---")

    CO2_RESEAU  = 0.604
    _is_ete_sc  = pred_date.month in [4, 5, 6, 7, 8, 9]
    saison_str  = "Été" if _is_ete_sc else "Hiver"

    _avant_h = predict_xgb(pred_date, pred_heure, temp_input, irrad_input, xgb_avant) or 0.0
    _apres_h = predict_xgb(pred_date, pred_heure, temp_input, irrad_input, xgb_apres) or 0.0
    E_dispo_heure = max(0.0, _avant_h - _apres_h)

    if scenario_choix == "🚗 Recharge Véhicules Électriques":

        st.markdown(f"""
<div style="background:linear-gradient(90deg,#F57C00,#FFB74D);padding:14px 20px;border-radius:10px;margin-bottom:16px;">
  <span style="font-size:18px;font-weight:700;color:white;letter-spacing:0.5px;">
    🚗 Recharge Véhicules Électriques · {pred_date.strftime('%d/%m/%Y')}
  </span>
</div>
""", unsafe_allow_html=True)

        def _get_tarif_mt(month, hour):
            is_hiver = month in [10, 11, 12, 1, 2, 3]
            if is_hiver:
                if 17 <= hour < 21:
                    return 1.22005, "Heures de Pointe (HPt)", "#E53935"
                elif hour >= 22 or hour < 7:
                    return 0.73294, "Heures Creuses (HC)", "#43A047"
                else:
                    return 0.92488, "Heures Pleines (HP)", "#FB8C00"
            else:
                if 18 <= hour < 22:
                    return 1.22005, "Heures de Pointe (HPt)", "#E53935"
                elif hour >= 23 or hour < 7:
                    return 0.73294, "Heures Creuses (HC)", "#43A047"
                else:
                    return 0.92488, "Heures Pleines (HP)", "#FB8C00"

        tarif_kwh, tarif_nom, tarif_color = _get_tarif_mt(pred_date.month, pred_heure)
        _tarif_abr = tarif_nom.split("(")[-1].rstrip(")")

        st.markdown("#### Paramètres VE")
        col1, col2, col3 = st.columns(3)
        with col1:
            N_VE = st.number_input(
                "**N_VE** : Nombre de véhicules",
                min_value=1, max_value=100, value=5, step=1,
                help="Nombre total de VE à recharger", key="ve_nve"
            )
        with col2:
            P_borne = st.number_input(
                "**P_borne** : Puissance borne (kW)",
                min_value=3.0, max_value=150.0, value=7.0, step=0.5,
                help="Puissance nominale d'une borne de recharge", key="ve_pborne"
            )
        with col3:
            C_bat_VE = st.number_input(
                "**E_batt** : Capacité batterie VE (kWh)",
                min_value=20.0, max_value=200.0, value=60.0, step=5.0,
                help="Capacité totale de la batterie du VE", key="ve_cbat"
            )

        col4, col5, col6 = st.columns(3)
        with col4:
            SOC_i = st.number_input(
                "**SOC_i** : État de charge initial (%)",
                min_value=0, max_value=99, value=20, step=5,
                help="Niveau de charge initial de la batterie (%)", key="ve_soci"
            )
        with col5:
            SOC_f = st.number_input(
                "**SOC_f** : État de charge final souhaité (%)",
                min_value=1, max_value=100, value=80, step=5,
                help="Niveau de charge cible (%)", key="ve_socf"
            )
        with col6:
            eta_charge_pct = st.number_input(
                "**η** : Rendement de charge (%)",
                min_value=80, max_value=99, value=92, step=1,
                help="Rendement du chargeur AC. Typique : 88–92% pour borne 7 kW.",
                key="ve_eta"
            )
            eta_charge = eta_charge_pct / 100.0

        if SOC_f <= SOC_i:
            st.warning("⚠️ SOC_f doit être supérieur à SOC_i.")

        col_inf1, col_inf2, col_inf3 = st.columns(3)
        col_inf1.metric("Énergie disponible PV", f"{E_dispo_heure:.3f} kWh")
        col_inf2.metric("Saison / Tranche", f"{saison_str} — {_tarif_abr}")
        col_inf3.metric("Tarif réseau", f"{tarif_kwh:.5f} MAD/kWh")

        st.markdown("---")

        st.markdown(f"""
<div style="background:linear-gradient(90deg,#FF9800,#FFB74D);padding:14px 20px;border-radius:10px;margin-bottom:8px;">
  <span style="font-size:18px;font-weight:700;color:white;letter-spacing:0.5px;">
    ÉTUDE 1 — Énergie écrêtée à {pred_heure:02d}h (heure sélectionnée)
  </span>
</div>
""", unsafe_allow_html=True)

        E_dispo = E_dispo_heure
        E_requise = N_VE * C_bat_VE
        P_total = N_VE * P_borne

        if E_dispo > 0:
            P_charge = min(P_borne, E_dispo)
            source_pcharge = f"PV écrêté (limité à min(P_borne, E_dispo) = {P_charge:.2f} kW)"
        else:
            P_charge = P_borne
            source_pcharge = f"Réseau (P_borne = {P_charge:.2f} kW)"

        delta_SOC = (SOC_f - SOC_i) / 100.0
        denom_t   = P_charge * eta_charge
        t_charge  = (C_bat_VE * delta_SOC) / denom_t if denom_t > 0 else 0

        cout_sans_pv    = E_requise * tarif_kwh
        co2_sans_pv     = E_requise * CO2_RESEAU
        E_PV            = min(E_dispo, E_requise)
        E_reseau_avec_pv = max(0.0, E_requise - E_dispo)
        cout_avec_pv    = E_reseau_avec_pv * tarif_kwh
        co2_avec_pv     = E_reseau_avec_pv * CO2_RESEAU
        couverture_pct  = (E_PV / E_requise * 100) if E_requise > 0 else 0
        N_VE_charges    = int(E_dispo / C_bat_VE) if C_bat_VE > 0 else 0
        economie_mad    = cout_sans_pv - cout_avec_pv
        co2_evite_kg    = co2_sans_pv - co2_avec_pv

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("⚡ E_requise",      f"{E_requise:.2f} kWh")
        k2.metric("☀️ Couverture PV",  f"{couverture_pct:.1f}%")
        k3.metric("🕐 t_charge",       f"{t_charge:.3f} h")
        k4.metric("🚗 VE couverts PV", f"{N_VE_charges}")

        st.markdown("#### Bilan — Réseau seul vs PV + Réseau")
        st.markdown(f"""
<table style="width:100%;border-collapse:collapse;font-size:14px;margin:10px 0;">
  <thead>
    <tr style="background:#F57C00;color:white;">
      <th style="padding:10px;text-align:left;">Paramètre</th>
      <th style="padding:10px;text-align:center;">🔌 Sans PV (Réseau seul)</th>
      <th style="padding:10px;text-align:center;">☀️+🔌 Avec PV écrêté</th>
      <th style="padding:10px;text-align:center;">Gain PV</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:rgba(245,124,0,0.06);">
      <td style="padding:9px;">Part PV utilisée (kWh)</td>
      <td style="padding:9px;text-align:center;">0.000</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">{E_PV:.3f}</td>
      <td style="padding:9px;text-align:center;">—</td>
    </tr>
    <tr>
      <td style="padding:9px;">Part réseau (kWh)</td>
      <td style="padding:9px;text-align:center;color:#E53935;font-weight:bold;">{E_requise:.3f}</td>
      <td style="padding:9px;text-align:center;">{E_reseau_avec_pv:.3f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;">↓ {E_PV:.3f} kWh</td>
    </tr>
    <tr style="background:rgba(245,124,0,0.06);">
      <td style="padding:9px;">Tranche tarifaire</td>
      <td style="padding:9px;text-align:center;font-style:italic;" colspan="2">{tarif_nom} — {tarif_kwh:.5f} MAD/kWh</td>
      <td style="padding:9px;text-align:center;">—</td>
    </tr>
    <tr>
      <td style="padding:9px;font-weight:bold;">Coût réseau (MAD)</td>
      <td style="padding:9px;text-align:center;color:#E53935;font-weight:bold;">{cout_sans_pv:.4f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">{cout_avec_pv:.4f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">💰 {economie_mad:.4f} MAD</td>
    </tr>
    <tr style="background:rgba(245,124,0,0.06);">
      <td style="padding:9px;font-weight:bold;">Émissions CO₂ (kg)</td>
      <td style="padding:9px;text-align:center;color:#E53935;font-weight:bold;">{co2_sans_pv:.4f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">{co2_avec_pv:.4f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">🌍 {co2_evite_kg:.4f} kg</td>
    </tr>
  </tbody>
</table>
""", unsafe_allow_html=True)

        st.markdown("#### Visualisation — Étude 1")
        _y_max_cout = max(cout_sans_pv, cout_avec_pv, 1e-9) * 1.8
        _y_max_co2  = max(co2_sans_pv,  co2_avec_pv,  1e-9) * 1.8
        fig_ve2 = make_subplots(
            rows=1, cols=2,
            subplot_titles=["💰 Coût réseau (MAD)", "🌍 Émissions CO₂ (kg)"],
            horizontal_spacing=0.18
        )
        fig_ve2.add_trace(go.Bar(
            x=["Sans PV", "Avec PV"],
            y=[cout_sans_pv, cout_avec_pv],
            marker=dict(color=["#E53935", "#43A047"],
                        line=dict(width=1.5, color="rgba(0,0,0,0.15)")),
            text=[f"{cout_sans_pv:.4f}", f"{cout_avec_pv:.4f}"],
            textposition="outside", textfont=dict(size=12),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Coût : <b>%{y:.5f} MAD</b><extra></extra>"
        ), row=1, col=1)
        fig_ve2.add_trace(go.Bar(
            x=["Sans PV", "Avec PV"],
            y=[co2_sans_pv, co2_avec_pv],
            marker=dict(color=["#FF7043", "#66BB6A"],
                        line=dict(width=1.5, color="rgba(0,0,0,0.15)")),
            text=[f"{co2_sans_pv:.4f}", f"{co2_avec_pv:.4f}"],
            textposition="outside", textfont=dict(size=12),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>CO₂ : <b>%{y:.5f} kg</b><extra></extra>"
        ), row=1, col=2)
        if economie_mad > 0:
            fig_ve2.add_annotation(
                x="Avec PV", y=_y_max_cout * 0.22,
                text=f"💰 Économie<br><b>−{economie_mad:.4f} MAD</b>",
                showarrow=False, xanchor="center",
                font=dict(color="#43A047", size=11),
                bgcolor="rgba(67,160,71,0.13)", bordercolor="#43A047",
                borderwidth=1, borderpad=5, xref="x", yref="y"
            )
        if co2_evite_kg > 0:
            fig_ve2.add_annotation(
                x="Avec PV", y=_y_max_co2 * 0.22,
                text=f"🌿 CO₂ évité<br><b>−{co2_evite_kg:.4f} kg</b>",
                showarrow=False, xanchor="center",
                font=dict(color="#66BB6A", size=11),
                bgcolor="rgba(102,187,106,0.13)", bordercolor="#66BB6A",
                borderwidth=1, borderpad=5, xref="x2", yref="y2"
            )
        fig_ve2.update_yaxes(title_text="MAD", row=1, col=1,
                             range=[0, _y_max_cout],
                             gridcolor="rgba(128,128,128,0.2)", zeroline=False)
        fig_ve2.update_yaxes(title_text="kg CO₂", row=1, col=2,
                             range=[0, _y_max_co2],
                             gridcolor="rgba(128,128,128,0.2)", zeroline=False)
        fig_ve2.update_layout(
            height=400,
            title=dict(
                text=f"Analyse coût & CO₂ à {pred_heure:02d}h · {_tarif_abr} · {tarif_kwh:.5f} MAD/kWh",
                font=dict(size=12)
            ),
            template="plotly_dark" if theme_toggle else "plotly",
            bargap=0.42, margin=dict(t=90, b=40, l=65, r=65),
            **get_plot_bg(theme_toggle)
        )
        render_chart(fig_ve2, use_container_width=True)

        st.markdown("---")

        st.markdown(f"""
<div style="background:linear-gradient(90deg,#5C6BC0,#7986CB);padding:14px 20px;border-radius:10px;margin-bottom:8px;">
  <span style="font-size:18px;font-weight:700;color:white;letter-spacing:0.5px;">
    ÉTUDE 2 — Énergie écrêtée totale de la journée &nbsp;·&nbsp; {pred_date.strftime('%d/%m/%Y')} (0h → 23h)
  </span>
</div>
""", unsafe_allow_html=True)

        _df_real_sc   = df[df["Date"].dt.date == pred_date].copy()
        _has_real_sc  = len(_df_real_sc) > 0
        _om_profil_sc = fetch_day_meteo_openmeteo(pred_date)
        if _om_profil_sc is None and not _has_real_sc:
            _excl_myt_sc = (df["Date"].dt.year == 2025) & (df["Date"].dt.month == 8)
            _df_mois_sc  = df[(df["Date"].dt.month == pred_date.month) & ~_excl_myt_sc]
            if len(_df_mois_sc) > 0:
                _om_profil_sc = (_df_mois_sc.groupby(_df_mois_sc["Date"].dt.hour)
                                 .agg(T_moy=("Temperature_C", "mean"),
                                      Irr_moy=("Irradiance_Wpm2", "mean"))
                                 .reset_index().rename(columns={"Date": "heure"}))
            else:
                _om_profil_sc = pd.DataFrame({"heure": range(24), "T_moy": [25.0]*24,
                                              "Irr_moy": [0.0]*24})
        _src_label_sc = ("données réelles" if _has_real_sc
                         else ("Open-Meteo" if _om_profil_sc is not None else "moy. mensuelle"))

        def _get_cond_sc(h):
            if _has_real_sc:
                r = _df_real_sc[_df_real_sc["Date"].dt.hour == h]
                if not r.empty:
                    return float(r.iloc[0]["Temperature_C"]), float(r.iloc[0]["Irradiance_Wpm2"])
            if _om_profil_sc is not None:
                row = _om_profil_sc[_om_profil_sc["heure"] == h]
                if len(row) > 0:
                    return float(row["T_moy"].values[0]), float(row["Irr_moy"].values[0])
            return 25.0, 0.0

        profil_list_j = []
        for h_j in range(24):
            t_h, irr_h = _get_cond_sc(h_j)
            av_h  = predict_xgb(pred_date, h_j, t_h, irr_h, xgb_avant) or 0.0
            ap_h  = predict_xgb(pred_date, h_j, t_h, irr_h, xgb_apres) or 0.0
            gap_h = max(0.0, av_h - ap_h)
            profil_list_j.append({
                "Heure": h_j,
                "T (°C)": round(t_h, 1),
                "Irr. (W/m²)": round(irr_h, 1),
                "Avant XGB (kWh)": round(av_h, 4),
                "Après XGB (kWh)": round(ap_h, 4),
                "Énergie écrêtée (kWh)": round(gap_h, 4)
            })
        df_profil_j  = pd.DataFrame(profil_list_j)
        E_dispo_jour = df_profil_j["Énergie écrêtée (kWh)"].sum()

        _ej1, _ej2, _ej3 = st.columns(3)
        _ej1.metric("Énergie disponible (journée)", f"{E_dispo_jour:.3f} kWh")
        _ej2.metric("Production AVANT (journée)",
                    f"{df_profil_j['Avant XGB (kWh)'].sum():.2f} kWh")
        _ej3.metric("Énergie écrêtée / AVANT",
                    f"{E_dispo_jour / df_profil_j['Avant XGB (kWh)'].sum() * 100:.1f}%"
                    if df_profil_j["Avant XGB (kWh)"].sum() > 0 else "—")

        fig_profil_j = go.Figure()
        fig_profil_j.add_trace(go.Bar(
            x=df_profil_j["Heure"], y=df_profil_j["Après XGB (kWh)"],
            marker_color=cc["xgb"], name="Production après (XGB)",
            hovertemplate="<b>%{x}h</b><br>Après : <b>%{y:.4f} kWh</b><extra></extra>"
        ))
        fig_profil_j.add_trace(go.Bar(
            x=df_profil_j["Heure"], y=df_profil_j["Énergie écrêtée (kWh)"],
            marker_color=cc["ecrete"], name="Énergie écrêtée → Recharge VE",
            hovertemplate="<b>%{x}h</b><br>Écrêtée : <b>%{y:.4f} kWh</b><extra></extra>"
        ))
        fig_profil_j.update_layout(
            barmode="stack", height=300,
            title=f"Profil écrêtage → Recharge VE · {pred_date.strftime('%d/%m/%Y')} · {_src_label_sc}",
            xaxis=dict(title="Heure", dtick=1),
            yaxis=dict(title="Énergie (kWh)", gridcolor="rgba(128,128,128,0.2)"),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", yanchor="bottom"),
            template="plotly_dark" if theme_toggle else "plotly",
            margin=dict(t=70, b=40),
            **get_plot_bg(theme_toggle)
        )
        render_chart(fig_profil_j, use_container_width=True)

        with st.expander("Voir le profil horaire détaillé"):
            st.dataframe(df_profil_j, use_container_width=True)

        E_dispo_j        = E_dispo_jour
        E_requise_j      = N_VE * C_bat_VE
        P_charge_j       = min(P_borne, E_dispo_j) if E_dispo_j > 0 else P_borne
        delta_SOC_j      = (SOC_f - SOC_i) / 100.0
        t_charge_j       = (C_bat_VE * delta_SOC_j) / (P_charge_j * eta_charge) if P_charge_j > 0 else 0
        E_PV_j           = min(E_dispo_j, E_requise_j)
        E_reseau_j       = max(0.0, E_requise_j - E_dispo_j)
        cout_sans_pv_j   = E_requise_j * tarif_kwh
        cout_avec_pv_j   = E_reseau_j  * tarif_kwh
        co2_sans_pv_j    = E_requise_j * CO2_RESEAU
        co2_avec_pv_j    = E_reseau_j  * CO2_RESEAU
        couverture_pct_j = (E_PV_j / E_requise_j * 100) if E_requise_j > 0 else 0
        N_VE_charges_j   = int(E_dispo_j / C_bat_VE) if C_bat_VE > 0 else 0
        economie_j       = cout_sans_pv_j - cout_avec_pv_j
        co2_evite_j      = co2_sans_pv_j  - co2_avec_pv_j

        kj1, kj2, kj3, kj4 = st.columns(4)
        kj1.metric("⚡ E_requise",      f"{E_requise_j:.2f} kWh")
        kj2.metric("☀️ Couverture PV",  f"{couverture_pct_j:.1f}%")
        kj3.metric("🕐 t_charge",       f"{t_charge_j:.3f} h")
        kj4.metric("🚗 VE couverts PV", f"{N_VE_charges_j}")

        st.markdown("#### Visualisation — Journée Complète")
        _y_max_cout_j = max(cout_sans_pv_j, cout_avec_pv_j, 1e-9) * 1.8
        _y_max_co2_j  = max(co2_sans_pv_j,  co2_avec_pv_j,  1e-9) * 1.8
        fig_vej2 = make_subplots(rows=1, cols=2,
                                 subplot_titles=["💰 Coût réseau (MAD)", "🌍 Émissions CO₂ (kg)"],
                                 horizontal_spacing=0.18)
        fig_vej2.add_trace(go.Bar(
            x=["Sans PV", "Avec PV"], y=[cout_sans_pv_j, cout_avec_pv_j],
            marker=dict(color=["#E53935", "#43A047"],
                        line=dict(width=1.5, color="rgba(0,0,0,0.15)")),
            text=[f"{cout_sans_pv_j:.3f}", f"{cout_avec_pv_j:.3f}"],
            textposition="outside", textfont=dict(size=12),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Coût : <b>%{y:.4f} MAD</b><extra></extra>"
        ), row=1, col=1)
        fig_vej2.add_trace(go.Bar(
            x=["Sans PV", "Avec PV"], y=[co2_sans_pv_j, co2_avec_pv_j],
            marker=dict(color=["#FF7043", "#66BB6A"],
                        line=dict(width=1.5, color="rgba(0,0,0,0.15)")),
            text=[f"{co2_sans_pv_j:.3f}", f"{co2_avec_pv_j:.3f}"],
            textposition="outside", textfont=dict(size=12),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>CO₂ : <b>%{y:.4f} kg</b><extra></extra>"
        ), row=1, col=2)
        if economie_j > 0:
            fig_vej2.add_annotation(
                x="Avec PV", y=_y_max_cout_j * 0.22,
                text=f"💰 Économie<br><b>−{economie_j:.3f} MAD</b>",
                showarrow=False, xanchor="center",
                font=dict(color="#43A047", size=11),
                bgcolor="rgba(67,160,71,0.13)", bordercolor="#43A047",
                borderwidth=1, borderpad=5, xref="x", yref="y"
            )
        if co2_evite_j > 0:
            fig_vej2.add_annotation(
                x="Avec PV", y=_y_max_co2_j * 0.22,
                text=f"🌿 CO₂ évité<br><b>−{co2_evite_j:.3f} kg</b>",
                showarrow=False, xanchor="center",
                font=dict(color="#66BB6A", size=11),
                bgcolor="rgba(102,187,106,0.13)", bordercolor="#66BB6A",
                borderwidth=1, borderpad=5, xref="x2", yref="y2"
            )
        fig_vej2.update_yaxes(title_text="MAD", row=1, col=1,
                              range=[0, _y_max_cout_j],
                              gridcolor="rgba(128,128,128,0.2)", zeroline=False)
        fig_vej2.update_yaxes(title_text="kg CO₂", row=1, col=2,
                              range=[0, _y_max_co2_j],
                              gridcolor="rgba(128,128,128,0.2)", zeroline=False)
        fig_vej2.update_layout(
            height=420,
            title=dict(
                text=f"Analyse coût & CO₂ journalière — {pred_date.strftime('%d/%m/%Y')} · {_tarif_abr} · {tarif_kwh:.5f} MAD/kWh",
                font=dict(size=12)
            ),
            template="plotly_dark" if theme_toggle else "plotly",
            bargap=0.42, margin=dict(t=90, b=40, l=65, r=65),
            **get_plot_bg(theme_toggle)
        )
        render_chart(fig_vej2, use_container_width=True)

        st.markdown(f"""
<table style="width:100%;border-collapse:collapse;font-size:14px;margin:10px 0;">
  <thead>
    <tr style="background:#F57C00;color:white;">
      <th style="padding:10px;text-align:left;">Paramètre</th>
      <th style="padding:10px;text-align:center;">🔌 Sans PV</th>
      <th style="padding:10px;text-align:center;">☀️+🔌 Avec PV écrêté</th>
      <th style="padding:10px;text-align:center;">Gain journalier</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:rgba(245,124,0,0.06);">
      <td style="padding:9px;">Part PV utilisée (kWh)</td>
      <td style="padding:9px;text-align:center;">0.000</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">{E_PV_j:.3f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;">+{E_PV_j:.3f} kWh</td>
    </tr>
    <tr>
      <td style="padding:9px;">Part réseau (kWh)</td>
      <td style="padding:9px;text-align:center;color:#E53935;font-weight:bold;">{E_requise_j:.3f}</td>
      <td style="padding:9px;text-align:center;">{E_reseau_j:.3f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;">↓ {E_PV_j:.3f} kWh</td>
    </tr>
    <tr style="background:rgba(245,124,0,0.06);">
      <td style="padding:9px;font-weight:bold;">Coût réseau (MAD)</td>
      <td style="padding:9px;text-align:center;color:#E53935;font-weight:bold;">{cout_sans_pv_j:.3f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">{cout_avec_pv_j:.3f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">💰 {economie_j:.3f} MAD</td>
    </tr>
    <tr>
      <td style="padding:9px;font-weight:bold;">Émissions CO₂ (kg)</td>
      <td style="padding:9px;text-align:center;color:#E53935;font-weight:bold;">{co2_sans_pv_j:.3f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">{co2_avec_pv_j:.3f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">🌍 {co2_evite_j:.3f} kg</td>
    </tr>
  </tbody>
</table>
""", unsafe_allow_html=True)

        st.caption(
            f"Hypothèses : {N_VE} VE · P_borne {P_borne} kW · E_batt {C_bat_VE} kWh · "
            f"SOC {SOC_i}%→{SOC_f}% · η {eta_charge_pct}% · "
            f"Tranche {_tarif_abr} {tarif_kwh:.5f} MAD/kWh · CO₂ réseau {CO2_RESEAU} kg/kWh · "
            f"{pred_date.strftime('%B %Y')} — Source T/G : {_src_label_sc}"
        )

        st.markdown("---")
        export_df_ve = pd.DataFrame({
            "Paramètre": [
                "Date", "Heure sélectionnée",
                "N_VE", "P_borne (kW)", "E_batt (kWh)", "SOC_i (%)", "SOC_f (%)",
                "η charge", "Saison", "Tranche tarifaire", "Tarif (MAD/kWh)",
                "— ÉTUDE 1 (heure) —",
                "E_requise (kWh)", "E_PV utilisée (kWh)", "E_réseau (kWh)",
                "Couverture PV (%)", "t_charge (h)", "VE couverts PV",
                "Coût Sans PV (MAD)", "Coût Avec PV (MAD)", "Économie (MAD)",
                "CO₂ Sans PV (kg)", "CO₂ Avec PV (kg)", "CO₂ Évité (kg)",
                "— ÉTUDE 2 (journée) —",
                "E_dispo PV journée (kWh)", "E_requise journée (kWh)",
                "E_PV utilisée journée (kWh)", "E_réseau journée (kWh)",
                "Couverture PV journée (%)", "VE couverts PV journée",
                "Coût Sans PV (MAD)", "Coût Avec PV (MAD)", "Économie journalière (MAD)",
                "CO₂ évité journalier (kg)",
            ],
            "Valeur": [
                pred_date.strftime("%d/%m/%Y"), f"{pred_heure:02d}h",
                N_VE, P_borne, C_bat_VE, SOC_i, SOC_f,
                eta_charge, saison_str, tarif_nom, tarif_kwh,
                "—",
                round(E_requise, 3), round(E_PV, 4), round(E_reseau_avec_pv, 4),
                round(couverture_pct, 2), round(t_charge, 4), N_VE_charges,
                round(cout_sans_pv, 4), round(cout_avec_pv, 4), round(economie_mad, 4),
                round(co2_sans_pv, 4), round(co2_avec_pv, 4), round(co2_evite_kg, 4),
                "—",
                round(E_dispo_jour, 4), round(E_requise_j, 3),
                round(E_PV_j, 4), round(E_reseau_j, 4),
                round(couverture_pct_j, 2), N_VE_charges_j,
                round(cout_sans_pv_j, 3), round(cout_avec_pv_j, 3), round(economie_j, 3),
                round(co2_evite_j, 3),
            ]
        })
        st.download_button(
            label="📥 Télécharger CSV — Recharge Véhicules Électriques",
            data=export_df_ve.to_csv(index=False),
            file_name=f"scenario_ve_{pred_date.strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    elif scenario_choix == "❄️ Stockage Froid — Climatisation nocturne":

        st.markdown(f"""
<div style="background:linear-gradient(90deg,#0097A7,#26C6DA);padding:14px 20px;border-radius:10px;margin-bottom:16px;">
  <span style="font-size:18px;font-weight:700;color:white;letter-spacing:0.5px;">
    ❄️ Stockage Froid · {pred_date.strftime('%d/%m/%Y')}
    &nbsp;·&nbsp; Saison : {saison_str}
  </span>
</div>
""", unsafe_allow_html=True)

        st.markdown("#### Paramètres — À confirmer avec les fiches techniques")

        _sf_c1, _sf_c2, _sf_c3 = st.columns(3)
        with _sf_c1:
            sf_n_clim_fours = st.number_input(
                "Nb climatiseurs côté **fours**",
                min_value=1, max_value=50, value=5, step=1,
                help="Zones adjacentes aux fours — actives toute l'année la nuit",
                key="sf_n_fours"
            )
        with _sf_c2:
            sf_n_clim_autres = st.number_input(
                "Nb climatiseurs **autres zones**",
                min_value=0, max_value=100, value=5, step=1,
                help="Zones éloignées des fours — actives seulement en été (avr–sep)",
                key="sf_n_autres"
            )
        with _sf_c3:
            sf_p_clim_kw = st.number_input(
                "Puissance frigorifique / clim (kW)",
                min_value=1.0, max_value=20.0, value=3.5, step=0.5,
                help="Sur fiche technique : 'Cooling capacity' ou 'Puissance frigorifique' en kW",
                key="sf_p_clim"
            )

        _sf_c4, _sf_c5 = st.columns(2)
        with _sf_c4:
            sf_cop = st.number_input(
                "COP du système",
                min_value=1.5, max_value=6.0, value=3.0, step=0.1,
                help="Sur fiche technique : 'COP' ou 'EER'. Splits muraux ≈ 2.5–3.5 · Groupe froid ≈ 3.5–5",
                key="sf_cop"
            )
        with _sf_c5:
            sf_tarif_nuit = st.number_input(
                "Tarif réseau nuit (MAD/kWh)",
                min_value=0.40, max_value=2.00, value=0.73294, step=0.01,
                help="Heures Creuses ONEE MT TFZ : 0.73294 MAD/kWh",
                key="sf_tarif_nuit"
            )

        st.markdown("---")

        _SF_DUREE_NUIT  = 12

        sf_n_actif = sf_n_clim_fours + (sf_n_clim_autres if _is_ete_sc else 0)

        sf_p_froid_kw   = sf_n_actif * sf_p_clim_kw
        sf_e_froid_nuit = sf_p_froid_kw * _SF_DUREE_NUIT

        _sf_df_real   = df[df["Date"].dt.date == pred_date].copy()
        _sf_has_real  = len(_sf_df_real) > 0
        _sf_om        = fetch_day_meteo_openmeteo(pred_date)
        _sf_om_is_api = _sf_om is not None
        if _sf_om is None and not _sf_has_real:
            _excl_myt_sf = (df["Date"].dt.year == 2025) & (df["Date"].dt.month == 8)
            _df_mois_sf  = df[(df["Date"].dt.month == pred_date.month) & ~_excl_myt_sf]
            if len(_df_mois_sf) > 0:
                _sf_om = (_df_mois_sf.groupby(_df_mois_sf["Date"].dt.hour)
                          .agg(T_moy=("Temperature_C","mean"), Irr_moy=("Irradiance_Wpm2","mean"))
                          .reset_index().rename(columns={"Date":"heure"}))
            else:
                _sf_om = pd.DataFrame({"heure": range(24), "T_moy":[25.0]*24, "Irr_moy":[0.0]*24})
        _sf_src = ("données réelles" if _sf_has_real
                   else ("Open-Meteo" if _sf_om_is_api else "moy. mensuelle"))

        def _sf_get_cond(h):
            if _sf_has_real:
                r = _sf_df_real[_sf_df_real["Date"].dt.hour == h]
                if not r.empty:
                    return float(r.iloc[0]["Temperature_C"]), float(r.iloc[0]["Irradiance_Wpm2"])
            if _sf_om is not None:
                row = _sf_om[_sf_om["heure"] == h]
                if len(row) > 0:
                    return float(row["T_moy"].values[0]), float(row["Irr_moy"].values[0])
            return 25.0, 0.0

        sf_e_ecrete_jour = 0.0
        sf_profil = []
        for _h_sf in range(24):
            _t_sf, _irr_sf = _sf_get_cond(_h_sf)
            _av_sf = predict_xgb(pred_date, _h_sf, _t_sf, _irr_sf, xgb_avant) or 0.0
            _ap_sf = predict_xgb(pred_date, _h_sf, _t_sf, _irr_sf, xgb_apres) or 0.0
            _ec_sf = max(0.0, _av_sf - _ap_sf)
            sf_e_ecrete_jour += _ec_sf
            sf_profil.append({
                "h": _h_sf,
                "ecrete": _ec_sf,
                "besoin_h": sf_p_froid_kw if (_h_sf >= 19 or _h_sf < 7) else 0.0
            })

        sf_e_stock_froid = sf_e_ecrete_jour * sf_cop
        sf_e_couvert     = min(sf_e_stock_froid, sf_e_froid_nuit)
        sf_taux_couvert  = (sf_e_couvert / sf_e_froid_nuit * 100) if sf_e_froid_nuit > 0 else 0.0
        sf_e_elec_eco    = sf_e_couvert / sf_cop if sf_cop > 0 else 0.0
        sf_economie_mad  = sf_e_elec_eco * sf_tarif_nuit
        sf_co2_evite     = sf_e_elec_eco * CO2_RESEAU
        sf_v_ballon_L    = sf_e_couvert * 3600.0 / (4.186 * 8.0)

        _sf_k1, _sf_k2, _sf_k3 = st.columns(3)
        _sf_k1.metric("Écrêtée disponible (jour)",
                      f"{sf_e_ecrete_jour:.2f} kWh")
        _sf_k2.metric("Froid stockable",
                      f"{sf_e_stock_froid:.2f} kWh froid")
        _sf_k3.metric("Besoin froid nuit (12h)",
                      f"{sf_e_froid_nuit:.2f} kWh froid")

        _sf_k4, _sf_k5, _sf_k6 = st.columns(3)
        _sf_k4.metric("Couverture nuit",
                      f"{sf_taux_couvert:.1f}%")
        _sf_k5.metric("Économie réseau",
                      f"{sf_economie_mad:.3f} MAD")
        _sf_k6.metric("CO₂ évité · Ballon",
                      f"{sf_co2_evite:.3f} kg CO₂")

        _sf_e_reste_nuit = max(0.0, sf_e_froid_nuit - sf_e_couvert)
        _sf_e_res_sans   = sf_e_froid_nuit / sf_cop if sf_cop > 0 else 0.0
        _sf_e_res_avec   = _sf_e_reste_nuit / sf_cop if sf_cop > 0 else 0.0

        fig_sf = make_subplots(
            rows=1, cols=2,
            column_widths=[0.45, 0.55],
            subplot_titles=[
                f"☀️ Jour — Charge du ballon<br><sup>Énergie écrêtée disponible</sup>",
                f"🌙 Nuit (19h→7h) — Besoin de refroidissement<br><sup>Couvert ballon vs complément réseau</sup>"
            ],
            horizontal_spacing=0.12
        )

        fig_sf.add_trace(go.Bar(
            x=["Écrêtée PV"],
            y=[sf_e_ecrete_jour],
            marker_color="#FF9800",
            text=[f"{sf_e_ecrete_jour:.2f} kWh"],
            textposition="outside", textfont=dict(size=13, color="#FF9800"),
            name="Écrêtée → ballon",
            hovertemplate="Énergie écrêtée : <b>%{y:.2f} kWh</b><extra></extra>"
        ), row=1, col=1)
        fig_sf.add_trace(go.Bar(
            x=["Froid stocké"],
            y=[sf_e_stock_froid],
            marker_color="#26C6DA",
            text=[f"{sf_e_stock_froid:.2f} kWh froid"],
            textposition="outside", textfont=dict(size=13, color="#0097A7"),
            name="Froid stocké (ballon)",
            hovertemplate="Froid stocké : <b>%{y:.2f} kWh froid</b><br>(écrêtée × COP {sf_cop})<extra></extra>"
        ), row=1, col=1)

        fig_sf.add_trace(go.Bar(
            x=["Besoin nuit"],
            y=[sf_e_couvert],
            marker_color="#26C6DA",
            text=[f"{sf_e_couvert:.2f} kWh"],
            textposition="inside", textfont=dict(color="white", size=12),
            name="Couvert par ballon PV",
            hovertemplate="Couvert (ballon) : <b>%{y:.2f} kWh froid</b><extra></extra>"
        ), row=1, col=2)
        fig_sf.add_trace(go.Bar(
            x=["Besoin nuit"],
            y=[_sf_e_reste_nuit],
            marker_color="#EF5350",
            text=[f"{_sf_e_reste_nuit:.2f} kWh" if _sf_e_reste_nuit > 0 else "✅ 100%"],
            textposition="inside", textfont=dict(color="white", size=12),
            name="Reste → réseau électrique",
            hovertemplate="Reste réseau : <b>%{y:.2f} kWh froid</b><extra></extra>"
        ), row=1, col=2)

        fig_sf.add_annotation(
            x="Besoin nuit", y=sf_e_froid_nuit * 1.12,
            text=f"<b>{sf_taux_couvert:.1f}% couvert</b>",
            showarrow=False, xanchor="center",
            font=dict(size=13, color="#0097A7"), row=1, col=2
        )

        fig_sf.update_layout(
            barmode="stack",
            height=380,
            title=dict(
                text=f"Bilan stockage froid — {pred_date.strftime('%d/%m/%Y')} · {saison_str} · {sf_n_actif} clims actives",
                font=dict(size=13)
            ),
            yaxis=dict(title="kWh froid", gridcolor="rgba(128,128,128,0.2)", zeroline=False),
            yaxis2=dict(title="kWh froid", gridcolor="rgba(128,128,128,0.2)", zeroline=False),
            legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", yanchor="top"),
            template="plotly_dark" if theme_toggle else "plotly",
            bargap=0.45, margin=dict(t=90, b=80, l=60, r=40),
            **get_plot_bg(theme_toggle)
        )
        render_chart(fig_sf, use_container_width=True)

        st.markdown(f"""
<table style="width:100%;border-collapse:collapse;font-size:14px;margin:10px 0;">
  <thead>
    <tr style="background:#0097A7;color:white;">
      <th style="padding:10px;text-align:left;">Paramètre</th>
      <th style="padding:10px;text-align:center;">🔌 Sans stockage PV</th>
      <th style="padding:10px;text-align:center;">☀️❄️ Avec stockage PV</th>
      <th style="padding:10px;text-align:center;">Gain</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:rgba(0,151,167,0.06);">
      <td style="padding:9px;">Clims actives la nuit</td>
      <td style="padding:9px;text-align:center;" colspan="2">{sf_n_actif} climatiseurs ({saison_str})</td>
      <td style="padding:9px;text-align:center;">—</td>
    </tr>
    <tr>
      <td style="padding:9px;">Besoin froid nuit (12h)</td>
      <td style="padding:9px;text-align:center;" colspan="2">{sf_e_froid_nuit:.2f} kWh froid</td>
      <td style="padding:9px;text-align:center;">—</td>
    </tr>
    <tr style="background:rgba(0,151,167,0.06);">
      <td style="padding:9px;">kWh élec réseau (clims nuit)</td>
      <td style="padding:9px;text-align:center;color:#E53935;font-weight:bold;">{_sf_e_res_sans:.2f} kWh</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">{_sf_e_res_avec:.2f} kWh</td>
      <td style="padding:9px;text-align:center;color:#43A047;">↓ {sf_e_elec_eco:.2f} kWh</td>
    </tr>
    <tr>
      <td style="padding:9px;font-weight:bold;">Couverture nuit par PV</td>
      <td style="padding:9px;text-align:center;">0.0%</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">{sf_taux_couvert:.1f}%</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">+{sf_taux_couvert:.1f}%</td>
    </tr>
    <tr style="background:rgba(0,151,167,0.06);">
      <td style="padding:9px;font-weight:bold;">Coût réseau nuit (MAD)</td>
      <td style="padding:9px;text-align:center;color:#E53935;font-weight:bold;">{_sf_e_res_sans*sf_tarif_nuit:.4f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">{_sf_e_res_avec*sf_tarif_nuit:.4f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">💰 {sf_economie_mad:.4f} MAD</td>
    </tr>
    <tr>
      <td style="padding:9px;font-weight:bold;">Émissions CO₂ (kg)</td>
      <td style="padding:9px;text-align:center;color:#E53935;font-weight:bold;">{_sf_e_res_sans*CO2_RESEAU:.4f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">{_sf_e_res_avec*CO2_RESEAU:.4f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">🌍 {sf_co2_evite:.4f} kg</td>
    </tr>
    <tr style="background:rgba(0,151,167,0.06);">
      <td style="padding:9px;font-weight:bold;">Volume ballon recommandé</td>
      <td style="padding:9px;text-align:center;" colspan="2">
        <b>{sf_v_ballon_L:.0f} litres</b> — eau froide, ΔT = 8°C
      </td>
      <td style="padding:9px;text-align:center;">—</td>
    </tr>
  </tbody>
</table>
""", unsafe_allow_html=True)
        st.caption(
            f"Hypothèses : {sf_n_actif} clims × {sf_p_clim_kw} kW · COP {sf_cop} · "
            f"Plage nuit 12h (19h→7h) · Ballon eau froide ΔT=8°C · "
            f"Tarif HC {sf_tarif_nuit:.5f} MAD/kWh · CO₂ réseau {CO2_RESEAU} kg/kWh · "
            f"Saison : {saison_str} ({pred_date.strftime('%B %Y')})"
        )

    elif scenario_choix == "🌡️ PAC Eau Chaude — Chauffage & Process":

        st.markdown(f"""
<div style="background:linear-gradient(90deg,#E64A19,#FF7043);padding:14px 20px;border-radius:10px;margin-bottom:16px;">
  <span style="font-size:18px;font-weight:700;color:white;letter-spacing:0.5px;">
    🌡️ PAC Préchauffage — Boucle Industrielle Jacob Delafon · {pred_date.strftime('%d/%m/%Y')}
  </span>
</div>
""", unsafe_allow_html=True)

        st.markdown("#### Paramètres — Boucle industrielle & PAC")

        _pac_c1, _pac_c2, _pac_c3 = st.columns(3)
        with _pac_c1:
            pac_cop = st.number_input(
                "COP PAC haute température",
                min_value=2.0, max_value=4.0, value=3.0, step=0.1,
                help="PAC haute température (eau/eau) : COP typique 2.5–3.5 à 85°C.",
                key="pac_cop"
            )
        with _pac_c2:
            pac_t_retour = st.number_input(
                "T retour boucle (°C)",
                min_value=40, max_value=80, value=60, step=1,
                help="Température eau en retour de boucle. PowerStudio Jacob Delafon : 60–70°C.",
                key="pac_t_retour"
            )
        with _pac_c3:
            pac_t_finale = st.number_input(
                "T finale chaudière (°C)",
                min_value=70, max_value=100, value=85, step=1,
                help="Température de départ boucle après chaudière. Jacob Delafon : 85–90°C.",
                key="pac_t_finale"
            )

        _pac_c4, _pac_c5, _pac_c6 = st.columns(3)
        with _pac_c4:
            pac_eta_pct = st.number_input(
                "Rendement chaudière η (%)",
                min_value=60, max_value=99, value=90, step=1,
                help="Rendement chaudière gaz industrielle : 85–92%.",
                key="pac_eta"
            )
            pac_eta = pac_eta_pct / 100.0
        with _pac_c5:
            pac_tarif_gaz = st.number_input(
                "Tarif gaz (MAD/kWh th)",
                min_value=0.05, max_value=1.00, value=0.15, step=0.01,
                help="Tarif gaz naturel industriel Maroc : 0.10–0.20 MAD/kWh.",
                key="pac_tarif_gaz"
            )
        with _pac_c6:
            pac_debit_m3h = st.number_input(
                "Débit boucle (m³/h)",
                min_value=1.0, max_value=200.0, value=21.0, step=0.5,
                help="Débit total boucle industrielle. PowerStudio Jacob Delafon : 21 m³/h (3 circuits).",
                key="pac_debit"
            )

        _PAC_RHO      = 0.965
        _PAC_CP       = 4.186
        _PAC_CO2_GAZ  = 0.234

        pac_debit_kg_s    = pac_debit_m3h * 1000.0 * _PAC_RHO / 3600.0
        pac_dt_boucle     = max(1.0, float(pac_t_finale) - float(pac_t_retour))
        pac_e_boucle_h    = pac_debit_kg_s * _PAC_CP * pac_dt_boucle

        st.markdown("---")

        st.markdown(f"""
<div style="background:linear-gradient(90deg,#FF9800,#FFB74D);padding:14px 20px;border-radius:10px;margin-bottom:8px;">
  <span style="font-size:18px;font-weight:700;color:white;letter-spacing:0.5px;">
    ÉTUDE 1 — Énergie écrêtée à {pred_heure:02d}h (heure sélectionnée)
  </span>
</div>
""", unsafe_allow_html=True)

        pac_e_elec_h      = E_dispo_heure
        pac_e_therm_h     = pac_e_elec_h * pac_cop
        pac_dt_pac_h      = pac_e_therm_h / (pac_debit_kg_s * _PAC_CP) if pac_debit_kg_s > 0 else 0.0
        pac_T_inter_h     = min(float(pac_t_retour) + pac_dt_pac_h, float(pac_t_finale))
        pac_frac_h        = min(100.0, pac_e_therm_h / pac_e_boucle_h * 100) if pac_e_boucle_h > 0 else 0.0
        pac_e_appoint_h   = max(0.0, pac_e_boucle_h - pac_e_therm_h)
        pac_gaz_sans_h    = pac_e_boucle_h / pac_eta
        pac_gaz_avec_h    = pac_e_appoint_h / pac_eta
        pac_eco_gaz_h     = (pac_gaz_sans_h - pac_gaz_avec_h) * pac_tarif_gaz
        pac_co2_sans_h    = pac_gaz_sans_h * _PAC_CO2_GAZ
        pac_co2_avec_h    = pac_gaz_avec_h * _PAC_CO2_GAZ
        pac_co2_evite_h   = pac_co2_sans_h - pac_co2_avec_h
        pac_v_h           = (pac_e_therm_h * 3600.0 / (_PAC_CP * pac_dt_boucle * _PAC_RHO)
                             if pac_dt_boucle > 0 else 0.0)

        _pe1, _pe2, _pe3 = st.columns(3)
        _pe1.metric("⚡ Énergie écrêtée PV",              f"{pac_e_elec_h:.3f} kWh")
        _pe2.metric("🌡️ Énergie thermique PAC",           f"{pac_e_therm_h:.3f} kWh th")
        _pe3.metric("🌡 T° intermédiaire après PAC",      f"{pac_T_inter_h:.1f} °C")

        _pe4, _pe5, _pe6 = st.columns(3)
        _pe4.metric("📊 Fraction chaudière remplacée",    f"{pac_frac_h:.1f} %")
        _pe5.metric("💰 Économie gaz",                    f"{pac_eco_gaz_h:.4f} MAD")
        _pe6.metric("🌿 CO₂ évité (gaz)",                 f"{pac_co2_evite_h:.4f} kg CO₂")

        st.markdown("#### Bilan — Sans PAC vs Avec PAC (préchauffage)")
        st.markdown(f"""
<table style="width:100%;border-collapse:collapse;font-size:14px;margin:10px 0;">
  <thead>
    <tr style="background:#E64A19;color:white;">
      <th style="padding:10px;text-align:left;">Paramètre</th>
      <th style="padding:10px;text-align:center;">🔥 Sans PAC (chaudière seule)</th>
      <th style="padding:10px;text-align:center;">☀️🌡️ Avec PAC PV (préchauffage)</th>
      <th style="padding:10px;text-align:center;">Gain</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:rgba(230,74,25,0.06);">
      <td style="padding:9px;">Température eau</td>
      <td style="padding:9px;text-align:center;">{pac_t_retour}°C → {pac_t_finale}°C</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">{pac_t_retour}°C → {pac_T_inter_h:.1f}°C → {pac_t_finale}°C</td>
      <td style="padding:9px;text-align:center;color:#43A047;">ΔT PAC = {pac_dt_pac_h:.1f}°C</td>
    </tr>
    <tr>
      <td style="padding:9px;">Énergie thermique boucle (kWh th/h)</td>
      <td style="padding:9px;text-align:center;" colspan="2">
        <b>{pac_e_boucle_h:.1f} kWh/h</b> &nbsp;({pac_debit_m3h:.0f} m³/h · ΔT {pac_dt_boucle:.0f}°C · ρ={_PAC_RHO} kg/L)
      </td>
      <td style="padding:9px;text-align:center;">—</td>
    </tr>
    <tr style="background:rgba(230,74,25,0.06);">
      <td style="padding:9px;">Contribution PAC (kWh th)</td>
      <td style="padding:9px;text-align:center;color:#9E9E9E;">0.000</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">{pac_e_therm_h:.3f} kWh th</td>
      <td style="padding:9px;text-align:center;color:#43A047;">{pac_frac_h:.1f}% de la boucle</td>
    </tr>
    <tr>
      <td style="padding:9px;">Appoint chaudière (kWh th)</td>
      <td style="padding:9px;text-align:center;font-weight:bold;">{pac_e_boucle_h:.1f}</td>
      <td style="padding:9px;text-align:center;font-weight:bold;">{pac_e_appoint_h:.3f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;">↓ {pac_e_therm_h:.3f} kWh</td>
    </tr>
    <tr style="background:rgba(230,74,25,0.06);">
      <td style="padding:9px;font-weight:bold;">Consommation gaz (kWh gaz)</td>
      <td style="padding:9px;text-align:center;color:#E53935;font-weight:bold;">{pac_gaz_sans_h:.2f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">{pac_gaz_avec_h:.2f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">↓ {(pac_gaz_sans_h - pac_gaz_avec_h):.2f} kWh gaz</td>
    </tr>
    <tr>
      <td style="padding:9px;font-weight:bold;">Coût gaz (MAD)</td>
      <td style="padding:9px;text-align:center;color:#E53935;font-weight:bold;">{pac_gaz_sans_h * pac_tarif_gaz:.4f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">{pac_gaz_avec_h * pac_tarif_gaz:.4f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">💰 {pac_eco_gaz_h:.4f} MAD</td>
    </tr>
    <tr style="background:rgba(230,74,25,0.06);">
      <td style="padding:9px;font-weight:bold;">Émissions CO₂ gaz (kg)</td>
      <td style="padding:9px;text-align:center;color:#E53935;font-weight:bold;">{pac_co2_sans_h:.4f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">{pac_co2_avec_h:.4f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">🌿 {pac_co2_evite_h:.4f} kg</td>
    </tr>
    <tr>
      <td style="padding:9px;">Volume eau équivalent préchauffé (L)</td>
      <td style="padding:9px;text-align:center;" colspan="2">
        <b>{pac_v_h:.0f} L</b> &nbsp;(ΔT boucle={pac_dt_boucle:.0f}°C · ρ={_PAC_RHO} kg/L)
      </td>
      <td style="padding:9px;text-align:center;">—</td>
    </tr>
  </tbody>
</table>
""", unsafe_allow_html=True)

        st.markdown("#### Visualisation — Étude 1")
        _p1_ymax_cout = max(pac_gaz_sans_h * pac_tarif_gaz, 1e-9) * 1.8
        _p1_ymax_co2  = max(pac_co2_sans_h, 1e-9) * 1.8
        fig_pac1 = make_subplots(
            rows=1, cols=2,
            subplot_titles=["💰 Coût gaz (MAD)", "🌿 CO₂ gaz (kg)"],
            horizontal_spacing=0.18
        )
        fig_pac1.add_trace(go.Bar(
            x=["Sans PAC PV", "Avec PAC PV"],
            y=[pac_gaz_sans_h * pac_tarif_gaz, pac_gaz_avec_h * pac_tarif_gaz],
            marker=dict(color=["#E53935", "#43A047"],
                        line=dict(width=1.5, color="rgba(0,0,0,0.15)")),
            text=[f"{pac_gaz_sans_h * pac_tarif_gaz:.4f}",
                  f"{pac_gaz_avec_h * pac_tarif_gaz:.4f}"],
            textposition="outside", textfont=dict(size=12),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Coût gaz : <b>%{y:.5f} MAD</b><extra></extra>"
        ), row=1, col=1)
        fig_pac1.add_trace(go.Bar(
            x=["Sans PAC PV", "Avec PAC PV"],
            y=[pac_co2_sans_h, pac_co2_avec_h],
            marker=dict(color=["#FF7043", "#66BB6A"],
                        line=dict(width=1.5, color="rgba(0,0,0,0.15)")),
            text=[f"{pac_co2_sans_h:.4f}", f"{pac_co2_avec_h:.4f}"],
            textposition="outside", textfont=dict(size=12),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>CO₂ : <b>%{y:.5f} kg</b><extra></extra>"
        ), row=1, col=2)
        if pac_eco_gaz_h > 0:
            fig_pac1.add_annotation(
                x="Avec PAC PV", y=_p1_ymax_cout * 0.22,
                text=f"💰 Économie<br><b>−{pac_eco_gaz_h:.4f} MAD</b>",
                showarrow=False, xanchor="center",
                font=dict(color="#43A047", size=11),
                bgcolor="rgba(67,160,71,0.13)", bordercolor="#43A047",
                borderwidth=1, borderpad=5, xref="x", yref="y"
            )
        if pac_co2_evite_h > 0:
            fig_pac1.add_annotation(
                x="Avec PAC PV", y=_p1_ymax_co2 * 0.22,
                text=f"🌿 CO₂ évité<br><b>−{pac_co2_evite_h:.4f} kg</b>",
                showarrow=False, xanchor="center",
                font=dict(color="#66BB6A", size=11),
                bgcolor="rgba(102,187,106,0.13)", bordercolor="#66BB6A",
                borderwidth=1, borderpad=5, xref="x2", yref="y2"
            )
        fig_pac1.update_yaxes(title_text="MAD", row=1, col=1,
                              range=[0, _p1_ymax_cout],
                              gridcolor="rgba(128,128,128,0.2)", zeroline=False)
        fig_pac1.update_yaxes(title_text="kg CO₂", row=1, col=2,
                              range=[0, _p1_ymax_co2],
                              gridcolor="rgba(128,128,128,0.2)", zeroline=False)
        fig_pac1.update_layout(
            height=400,
            title=dict(
                text=f"Analyse coût gaz & CO₂ à {pred_heure:02d}h · tarif gaz {pac_tarif_gaz:.2f} MAD/kWh",
                font=dict(size=12)
            ),
            template="plotly_dark" if theme_toggle else "plotly",
            bargap=0.42, margin=dict(t=90, b=40, l=65, r=65),
            **get_plot_bg(theme_toggle)
        )
        render_chart(fig_pac1, use_container_width=True)

        st.markdown("---")

        st.markdown(f"""
<div style="background:linear-gradient(90deg,#5C6BC0,#7986CB);padding:14px 20px;border-radius:10px;margin-bottom:8px;">
  <span style="font-size:18px;font-weight:700;color:white;letter-spacing:0.5px;">
    ÉTUDE 2 — Énergie écrêtée totale de la journée &nbsp;·&nbsp; {pred_date.strftime('%d/%m/%Y')} (0h → 23h)
  </span>
</div>
""", unsafe_allow_html=True)

        _pac_df_real   = df[df["Date"].dt.date == pred_date].copy()
        _pac_has_real  = len(_pac_df_real) > 0
        _pac_om        = fetch_day_meteo_openmeteo(pred_date)
        _pac_om_is_api = _pac_om is not None
        if _pac_om is None and not _pac_has_real:
            _excl_myt_pac = (df["Date"].dt.year == 2025) & (df["Date"].dt.month == 8)
            _df_mois_pac  = df[(df["Date"].dt.month == pred_date.month) & ~_excl_myt_pac]
            if len(_df_mois_pac) > 0:
                _pac_om = (_df_mois_pac.groupby(_df_mois_pac["Date"].dt.hour)
                           .agg(T_moy=("Temperature_C", "mean"), Irr_moy=("Irradiance_Wpm2", "mean"))
                           .reset_index().rename(columns={"Date": "heure"}))
            else:
                _pac_om = pd.DataFrame({"heure": range(24), "T_moy": [25.0]*24, "Irr_moy": [0.0]*24})
        _pac_src = ("données réelles" if _pac_has_real
                    else ("Open-Meteo" if _pac_om_is_api else "moy. mensuelle"))

        def _pac_get_cond(h):
            if _pac_has_real:
                r = _pac_df_real[_pac_df_real["Date"].dt.hour == h]
                if not r.empty:
                    return float(r.iloc[0]["Temperature_C"]), float(r.iloc[0]["Irradiance_Wpm2"])
            if _pac_om is not None:
                row = _pac_om[_pac_om["heure"] == h]
                if len(row) > 0:
                    return float(row["T_moy"].values[0]), float(row["Irr_moy"].values[0])
            return 25.0, 0.0

        pac_profil_j  = []
        pac_e_ecr_j   = 0.0
        for _h_pac in range(24):
            _t_pac, _irr_pac = _pac_get_cond(_h_pac)
            _av_pac  = predict_xgb(pred_date, _h_pac, _t_pac, _irr_pac, xgb_avant) or 0.0
            _ap_pac  = predict_xgb(pred_date, _h_pac, _t_pac, _irr_pac, xgb_apres) or 0.0
            _ec_pac  = max(0.0, _av_pac - _ap_pac)
            _et_pac  = _ec_pac * pac_cop
            _dT_pac  = _et_pac / (pac_debit_kg_s * _PAC_CP) if pac_debit_kg_s > 0 else 0.0
            _Ti_pac  = min(float(pac_t_retour) + _dT_pac, float(pac_t_finale))
            pac_e_ecr_j += _ec_pac
            pac_profil_j.append({
                "Heure": _h_pac,
                "T (°C)": round(_t_pac, 1),
                "Irr. (W/m²)": round(_irr_pac, 1),
                "Avant XGB (kWh)": round(_av_pac, 4),
                "Après XGB (kWh)": round(_ap_pac, 4),
                "Écrêtée (kWh)": round(_ec_pac, 4),
                "E therm PAC (kWh th)": round(_et_pac, 4),
                "T intermédiaire (°C)": round(_Ti_pac, 1),
            })
        df_profil_pac = pd.DataFrame(pac_profil_j)

        pac_e_therm_j   = pac_e_ecr_j * pac_cop
        pac_e_boucle_j  = pac_e_boucle_h * 24.0
        pac_e_appoint_j = max(0.0, pac_e_boucle_j - pac_e_therm_j)
        pac_frac_j      = min(100.0, pac_e_therm_j / pac_e_boucle_j * 100) if pac_e_boucle_j > 0 else 0.0
        pac_gaz_sans_j  = pac_e_boucle_j / pac_eta
        pac_gaz_avec_j  = pac_e_appoint_j / pac_eta
        pac_eco_gaz_j   = (pac_gaz_sans_j - pac_gaz_avec_j) * pac_tarif_gaz
        pac_co2_sans_j  = pac_gaz_sans_j * _PAC_CO2_GAZ
        pac_co2_avec_j  = pac_gaz_avec_j * _PAC_CO2_GAZ
        pac_co2_evite_j = pac_co2_sans_j - pac_co2_avec_j
        pac_v_j         = (pac_e_therm_j * 3600.0 / (_PAC_CP * pac_dt_boucle * _PAC_RHO)
                           if pac_dt_boucle > 0 else 0.0)

        _pj1, _pj2, _pj3 = st.columns(3)
        _pj1.metric("⚡ Écrêtée PV totale",               f"{pac_e_ecr_j:.3f} kWh")
        _pj2.metric("🌡️ Énergie thermique PAC",           f"{pac_e_therm_j:.3f} kWh th")
        _pj3.metric("📊 Fraction chaudière remplacée",     f"{pac_frac_j:.1f} %")

        _pj4, _pj5 = st.columns(2)
        _pj4.metric("💰 Économie gaz journalière",        f"{pac_eco_gaz_j:.3f} MAD")
        _pj5.metric("🌿 CO₂ évité journalier",             f"{pac_co2_evite_j:.3f} kg CO₂")

        fig_pac_prof = go.Figure()
        fig_pac_prof.add_trace(go.Bar(
            x=df_profil_pac["Heure"], y=df_profil_pac["Après XGB (kWh)"],
            marker_color=cc["xgb"], name="Production après (XGB)",
            hovertemplate="<b>%{x}h</b><br>Après : <b>%{y:.4f} kWh</b><extra></extra>"
        ))
        fig_pac_prof.add_trace(go.Bar(
            x=df_profil_pac["Heure"], y=df_profil_pac["Écrêtée (kWh)"],
            marker_color="#FF5722", name="Écrêtée → PAC eau chaude",
            hovertemplate="<b>%{x}h</b><br>Écrêtée : <b>%{y:.4f} kWh</b><extra></extra>"
        ))
        fig_pac_prof.update_layout(
            barmode="stack", height=300,
            title=f"Profil écrêtage → PAC · {pred_date.strftime('%d/%m/%Y')} · {_pac_src}",
            xaxis=dict(title="Heure", dtick=1),
            yaxis=dict(title="Énergie (kWh)", gridcolor="rgba(128,128,128,0.2)"),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", yanchor="bottom"),
            template="plotly_dark" if theme_toggle else "plotly",
            margin=dict(t=70, b=40),
            **get_plot_bg(theme_toggle)
        )
        render_chart(fig_pac_prof, use_container_width=True)

        with st.expander("Voir le profil horaire détaillé"):
            st.dataframe(df_profil_pac, use_container_width=True)

        st.markdown("#### Visualisation — Journée Complète")

        _p2_ymax_cout = max(pac_gaz_sans_j * pac_tarif_gaz, 1e-9) * 1.8
        _p2_ymax_co2  = max(pac_co2_sans_j, 1e-9) * 1.8

        fig_pac2 = make_subplots(
            rows=1, cols=2,
            subplot_titles=["💰 Coût gaz journalier (MAD)", "🌿 CO₂ gaz journalier (kg)"],
            horizontal_spacing=0.18
        )
        fig_pac2.add_trace(go.Bar(
            x=["Sans PAC PV", "Avec PAC PV"],
            y=[pac_gaz_sans_j * pac_tarif_gaz, pac_gaz_avec_j * pac_tarif_gaz],
            marker=dict(color=["#E53935", "#43A047"],
                        line=dict(width=1.5, color="rgba(0,0,0,0.15)")),
            text=[f"{pac_gaz_sans_j * pac_tarif_gaz:.2f}",
                  f"{pac_gaz_avec_j * pac_tarif_gaz:.2f}"],
            textposition="outside", textfont=dict(size=12),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Coût gaz : <b>%{y:.3f} MAD</b><extra></extra>"
        ), row=1, col=1)
        fig_pac2.add_trace(go.Bar(
            x=["Sans PAC PV", "Avec PAC PV"],
            y=[pac_co2_sans_j, pac_co2_avec_j],
            marker=dict(color=["#FF7043", "#66BB6A"],
                        line=dict(width=1.5, color="rgba(0,0,0,0.15)")),
            text=[f"{pac_co2_sans_j:.2f}", f"{pac_co2_avec_j:.2f}"],
            textposition="outside", textfont=dict(size=12),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>CO₂ : <b>%{y:.3f} kg</b><extra></extra>"
        ), row=1, col=2)
        if pac_eco_gaz_j > 0:
            fig_pac2.add_annotation(
                x="Avec PAC PV", y=_p2_ymax_cout * 0.22,
                text=f"💰 Économie<br><b>−{pac_eco_gaz_j:.2f} MAD</b>",
                showarrow=False, xanchor="center",
                font=dict(color="#43A047", size=11),
                bgcolor="rgba(67,160,71,0.13)", bordercolor="#43A047",
                borderwidth=1, borderpad=5, xref="x", yref="y"
            )
        if pac_co2_evite_j > 0:
            fig_pac2.add_annotation(
                x="Avec PAC PV", y=_p2_ymax_co2 * 0.22,
                text=f"🌿 CO₂ évité<br><b>−{pac_co2_evite_j:.2f} kg</b>",
                showarrow=False, xanchor="center",
                font=dict(color="#66BB6A", size=11),
                bgcolor="rgba(102,187,106,0.13)", bordercolor="#66BB6A",
                borderwidth=1, borderpad=5, xref="x2", yref="y2"
            )
        fig_pac2.update_yaxes(title_text="MAD", row=1, col=1,
                              range=[0, _p2_ymax_cout],
                              gridcolor="rgba(128,128,128,0.2)", zeroline=False)
        fig_pac2.update_yaxes(title_text="kg CO₂", row=1, col=2,
                              range=[0, _p2_ymax_co2],
                              gridcolor="rgba(128,128,128,0.2)", zeroline=False)
        fig_pac2.update_layout(
            height=420,
            title=dict(
                text=f"Analyse coût gaz & CO₂ journalière — {pred_date.strftime('%d/%m/%Y')} · tarif gaz {pac_tarif_gaz:.2f} MAD/kWh",
                font=dict(size=12)
            ),
            template="plotly_dark" if theme_toggle else "plotly",
            bargap=0.42, margin=dict(t=90, b=40, l=65, r=65),
            **get_plot_bg(theme_toggle)
        )
        render_chart(fig_pac2, use_container_width=True)

        st.markdown(f"""
<table style="width:100%;border-collapse:collapse;font-size:14px;margin:10px 0;">
  <thead>
    <tr style="background:#E64A19;color:white;">
      <th style="padding:10px;text-align:left;">Paramètre</th>
      <th style="padding:10px;text-align:center;">🔥 Sans PAC PV</th>
      <th style="padding:10px;text-align:center;">☀️🌡️ Avec PAC PV</th>
      <th style="padding:10px;text-align:center;">Gain journalier</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:rgba(230,74,25,0.06);">
      <td style="padding:9px;">Énergie PV écrêtée valorisée (kWh)</td>
      <td style="padding:9px;text-align:center;">0.000</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">{pac_e_ecr_j:.3f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;">+{pac_e_ecr_j:.3f} kWh</td>
    </tr>
    <tr>
      <td style="padding:9px;">Énergie thermique PAC (kWh th)</td>
      <td style="padding:9px;text-align:center;">—</td>
      <td style="padding:9px;text-align:center;font-weight:bold;">{pac_e_therm_j:.3f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;">{pac_frac_j:.1f}% de la boucle 24h</td>
    </tr>
    <tr style="background:rgba(230,74,25,0.06);">
      <td style="padding:9px;">Énergie boucle totale 24h (kWh th)</td>
      <td style="padding:9px;text-align:center;" colspan="2">
        <b>{pac_e_boucle_j:.1f} kWh/jour</b> &nbsp;({pac_debit_m3h:.0f} m³/h · ΔT {pac_dt_boucle:.0f}°C · 24h)
      </td>
      <td style="padding:9px;text-align:center;">—</td>
    </tr>
    <tr>
      <td style="padding:9px;font-weight:bold;">Consommation gaz (kWh gaz)</td>
      <td style="padding:9px;text-align:center;color:#E53935;font-weight:bold;">{pac_gaz_sans_j:.1f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">{pac_gaz_avec_j:.1f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">↓ {(pac_gaz_sans_j - pac_gaz_avec_j):.1f} kWh gaz</td>
    </tr>
    <tr style="background:rgba(230,74,25,0.06);">
      <td style="padding:9px;font-weight:bold;">Coût gaz (MAD)</td>
      <td style="padding:9px;text-align:center;color:#E53935;font-weight:bold;">{pac_gaz_sans_j * pac_tarif_gaz:.2f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">{pac_gaz_avec_j * pac_tarif_gaz:.2f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">💰 {pac_eco_gaz_j:.2f} MAD</td>
    </tr>
    <tr>
      <td style="padding:9px;font-weight:bold;">Émissions CO₂ gaz (kg)</td>
      <td style="padding:9px;text-align:center;color:#E53935;font-weight:bold;">{pac_co2_sans_j:.2f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">{pac_co2_avec_j:.2f}</td>
      <td style="padding:9px;text-align:center;color:#43A047;font-weight:bold;">🌿 {pac_co2_evite_j:.2f} kg</td>
    </tr>
    <tr style="background:rgba(230,74,25,0.06);">
      <td style="padding:9px;">Volume eau équivalent préchauffé (L)</td>
      <td style="padding:9px;text-align:center;" colspan="2">
        <b>{pac_v_j:.0f} L</b> &nbsp;(ΔT boucle={pac_dt_boucle:.0f}°C · ρ={_PAC_RHO} kg/L)
      </td>
      <td style="padding:9px;text-align:center;">—</td>
    </tr>
  </tbody>
</table>
""", unsafe_allow_html=True)

        st.caption(
            f"Hypothèses : COP {pac_cop} · T_retour {pac_t_retour}°C → T_inter {pac_T_inter_h:.1f}°C → T_finale {pac_t_finale}°C · "
            f"η_chaud {pac_eta_pct}% · Débit {pac_debit_m3h} m³/h · ρ={_PAC_RHO} kg/L · "
            f"Tarif gaz {pac_tarif_gaz:.2f} MAD/kWh · CO₂ gaz {_PAC_CO2_GAZ} kg/kWh · "
            f"{pred_date.strftime('%B %Y')} — Source T/G : {_pac_src}"
        )

        st.markdown("---")
        _export_pac = pd.DataFrame({
            "Paramètre": [
                "Date", "Heure sélectionnée",
                "COP PAC", "T retour (°C)", "T finale (°C)",
                "Rendement chaudière (%)", "Tarif gaz (MAD/kWh)", "Débit (m³/h)",
                "Énergie boucle horaire (kWh th/h)",
                "— ÉTUDE 1 (heure) —",
                "Écrêtée PV (kWh)", "Énergie thermique PAC (kWh th)",
                "T intermédiaire (°C)", "Fraction chaudière remplacée (%)",
                "Gaz sans PAC (kWh)", "Gaz avec PAC (kWh)",
                "Économie gaz (MAD)", "CO₂ évité (kg CO₂)",
                "Volume eau équivalent préchauffé (L)",
                "— ÉTUDE 2 (journée) —",
                "Écrêtée PV totale (kWh)", "Énergie thermique PAC (kWh th)",
                "Fraction chaudière remplacée (%)",
                "Gaz sans PAC (kWh)", "Gaz avec PAC (kWh)",
                "Économie gaz journalière (MAD)", "CO₂ évité journalier (kg CO₂)",
                "Volume eau équivalent préchauffé journée (L)",
            ],
            "Valeur": [
                pred_date.strftime("%d/%m/%Y"), f"{pred_heure:02d}h",
                pac_cop, pac_t_retour, pac_t_finale,
                pac_eta_pct, pac_tarif_gaz, pac_debit_m3h,
                round(pac_e_boucle_h, 2),
                "—",
                round(pac_e_elec_h, 4), round(pac_e_therm_h, 4),
                round(pac_T_inter_h, 1), round(pac_frac_h, 2),
                round(pac_gaz_sans_h, 4), round(pac_gaz_avec_h, 4),
                round(pac_eco_gaz_h, 4), round(pac_co2_evite_h, 4),
                round(pac_v_h, 0),
                "—",
                round(pac_e_ecr_j, 4), round(pac_e_therm_j, 4),
                round(pac_frac_j, 2),
                round(pac_gaz_sans_j, 2), round(pac_gaz_avec_j, 2),
                round(pac_eco_gaz_j, 3), round(pac_co2_evite_j, 3),
                round(pac_v_j, 0),
            ]
        })
        st.download_button(
            label="📥 Télécharger CSV — PAC Préchauffage Boucle Industrielle",
            data=_export_pac.to_csv(index=False),
            file_name=f"scenario_pac_prechauffage_{pred_date.strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

elif page == "Multi-Dates":
    st.markdown("## Comparaison Multi-Dates")
    st.markdown("Données réelles si disponibles, sinon XGBoost + Open-Meteo — toute date acceptée.")

    _max_dv = df["Date"].max().date()
    _min_dv = df["Date"].min().date()

    col1, col2, col3 = st.columns(3)
    with col1:
        date_a = st.date_input("Date 1", value=_max_dv, key="md_a")
    with col2:
        date_b = st.date_input("Date 2", value=_max_dv - pd.Timedelta(days=60), key="md_b")
    with col3:
        date_c = st.date_input("Date 3", value=_max_dv - pd.Timedelta(days=120), key="md_c")

    use_c    = st.checkbox("Inclure la date 3", value=False)
    selected_dates_md = [date_a, date_b] + ([date_c] if use_c else [])

    _MD_METRICS = {
        "Production après limitation (kWh)" : "ap",
        "Production avant limitation (kWh)" : "av",
        "Énergie écrêtée (kWh)"             : "ecrete",
        "Autoconsommation (kWh)"            : "autoconso",
        "Consommation générale (kWh)"       : "conso",
        "Consommation réseau (kWh)"         : "reseau",
        "Irradiance (W/m²)"                 : "irr",
        "Température (°C)"                  : "temp",
    }
    metric_label_md = st.selectbox("Grandeur à comparer", list(_MD_METRICS.keys()), key="md_metric")
    metric_col_md   = _MD_METRICS[metric_label_md]

    colors_md = (
        ["#FFC107", "#42A5F5", "#66BB6A"] if theme_toggle
        else ["#E65100", "#1565C0", "#2E7D32"]
    )

    def _build_day_md(d):
        df_real  = df[df["Date"].dt.date == d].copy()
        has_real = len(df_real) > 0
        om_prof  = fetch_day_meteo_openmeteo(d)

        _excl_myt = (df["Date"].dt.year == 2025) & (df["Date"].dt.month == 8)
        def _tg(h):
            if has_real:
                r = df_real[df_real["Date"].dt.hour == h]
                if not r.empty:
                    return float(r.iloc[0]["Temperature_C"]), float(r.iloc[0]["Irradiance_Wpm2"])
            if om_prof is not None:
                row = om_prof[om_prof["heure"] == h]
                if len(row) > 0:
                    return float(row["T_moy"].values[0]), float(row["Irr_moy"].values[0])
            _dm = df[(df["Date"].dt.month == d.month) & ~_excl_myt]
            _dh = _dm[_dm["Date"].dt.hour == h]
            _iv = _dh["Irradiance_Wpm2"].mean() if len(_dh) > 0 else 0.0
            return (float(_dh["Temperature_C"].mean()) if len(_dh) > 0 else 25.0,
                    0.0 if (isinstance(_iv, float) and np.isnan(_iv)) else float(_iv))

        rows = []
        for h in range(24):
            t, irr = _tg(h)
            av_p  = predict_xgb(d, h, t, irr, xgb_avant)  or 0.0
            ap_p  = predict_xgb(d, h, t, irr, xgb_apres) or 0.0
            co_p  = (predict_xgb_conso(d, h, t, xgb_conso) or 0.0) if xgb_conso else 0.0
            if has_real:
                r = df_real[df_real["Date"].dt.hour == h]
                if not r.empty:
                    ri = r.iloc[0]
                    av_p  = float(ri.get("Production avant limitation Kwh",     av_p))
                    ap_p  = float(ri.get("Production apres limitation [kWh]",   ap_p))
                    co_p  = float(ri.get("TotalLoad_kWh",                       co_p))
            ec_p   = max(0.0, av_p - ap_p)
            ac_p   = min(ap_p, co_p) if co_p > 0 else ap_p
            re_p   = max(0.0, co_p - ap_p) if co_p > 0 else 0.0
            rows.append({"heure": h, "temp": t, "irr": irr,
                         "av": av_p, "ap": ap_p, "ecrete": ec_p,
                         "conso": co_p, "autoconso": ac_p, "reseau": re_p})
        src = "réel" if has_real else ("XGB+Open-Meteo" if om_prof is not None else "XGB+moy.mois")
        return pd.DataFrame(rows), src

    fig_md   = go.Figure()
    summary_md = []
    _day_dfs   = {}

    for i_md, d_md in enumerate(selected_dates_md):
        day_df, src_lbl = _build_day_md(d_md)
        _day_dfs[d_md] = (day_df, src_lbl)
        lbl = f"{d_md.strftime('%d/%m/%Y')} ({src_lbl})"
        fig_md.add_trace(go.Scatter(
            x=day_df["heure"], y=day_df[metric_col_md],
            mode="lines+markers", name=lbl,
            line=dict(color=colors_md[i_md], width=2.5),
            hovertemplate=f"<b>{lbl}</b><br>%{{x}}h → %{{y:.3f}}<extra></extra>"
        ))
        _ap_tot  = day_df["ap"].sum()
        _av_tot  = day_df["av"].sum()
        _ec_tot  = day_df["ecrete"].sum()
        _co_tot  = day_df["conso"].sum()
        _ac_tot  = day_df["autoconso"].sum()
        _re_tot  = day_df["reseau"].sum()
        summary_md.append({
            "Date"                    : d_md.strftime("%d/%m/%Y"),
            "Source"                  : src_lbl,
            "Prod. après (kWh)"       : round(_ap_tot, 2),
            "Prod. avant (kWh)"       : round(_av_tot, 2),
            "Écrêtage (kWh)"          : round(_ec_tot, 2),
            "Écrêtage (%)"            : f"{_ec_tot / _av_tot * 100:.1f}%" if _av_tot > 0 else "—",
            "Conso. totale (kWh)"     : round(_co_tot, 1),
            "Autoconsommation (kWh)"  : round(_ac_tot, 2),
            "Conso. réseau (kWh)"     : round(_re_tot, 1),
            "Autonomie (%)"           : f"{_ac_tot / _co_tot * 100:.1f}%" if _co_tot > 0 else "—",
            "Irr. moy. (W/m²)"        : round(day_df["irr"].mean(), 1),
            "Temp. moy. (°C)"         : round(day_df["temp"].mean(), 1),
        })

    _yz0 = st.checkbox("Axe Y depuis 0 (décocher pour zoomer sur les différences)", value=True, key="md_yzero")

    fig_md.update_layout(
        height=480,
        xaxis_title="Heure du jour (h)", yaxis_title=metric_label_md,
        yaxis=dict(rangemode="tozero" if _yz0 else "normal"),
        title=f"Comparaison — {metric_label_md}",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", yanchor="bottom"),
        margin=dict(t=80),
        template="plotly_dark" if theme_toggle else "plotly",
        **get_plot_bg(theme_toggle))
    render_chart(fig_md, use_container_width=True)

    st.markdown("---")
    st.markdown("### Conditions météo utilisées (T° & Irradiance)")
    st.caption("Si les courbes d'irradiance sont identiques → fallback moyenne mensuelle actif → production identique attendue.")

    _cma, _cmb = st.columns(2)
    with _cma:
        st.markdown("##### Irradiance (W/m²)")
        _fig_irr = go.Figure()
        for i_md, d_md in enumerate(selected_dates_md):
            day_df, src_lbl = _day_dfs[d_md]
            _fig_irr.add_trace(go.Scatter(
                x=day_df["heure"], y=day_df["irr"],
                mode="lines+markers", marker=dict(size=4),
                name=f"{d_md.strftime('%d/%m/%Y')} ({src_lbl})",
                line=dict(color=colors_md[i_md], width=2),
                hovertemplate=f"<b>{d_md.strftime('%d/%m/%Y')}</b><br>%{{x}}h → %{{y:.1f}} W/m²<extra></extra>"
            ))
        _fig_irr.update_layout(
            height=300, hovermode="x unified",
            xaxis_title="Heure", yaxis_title="W/m²",
            yaxis=dict(rangemode="tozero"),
            legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center", yanchor="bottom"),
            margin=dict(t=60, b=30),
            template="plotly_dark" if theme_toggle else "plotly",
            **get_plot_bg(theme_toggle))
        render_chart(_fig_irr, use_container_width=True)

    with _cmb:
        st.markdown("##### Température (°C)")
        _fig_tmp = go.Figure()
        for i_md, d_md in enumerate(selected_dates_md):
            day_df, src_lbl = _day_dfs[d_md]
            _fig_tmp.add_trace(go.Scatter(
                x=day_df["heure"], y=day_df["temp"],
                mode="lines+markers", marker=dict(size=4),
                name=f"{d_md.strftime('%d/%m/%Y')} ({src_lbl})",
                line=dict(color=colors_md[i_md], width=2),
                hovertemplate=f"<b>{d_md.strftime('%d/%m/%Y')}</b><br>%{{x}}h → %{{y:.1f}} °C<extra></extra>"
            ))
        _fig_tmp.update_layout(
            height=300, hovermode="x unified",
            xaxis_title="Heure", yaxis_title="°C",
            yaxis=dict(rangemode="normal"),
            legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center", yanchor="bottom"),
            margin=dict(t=60, b=30),
            template="plotly_dark" if theme_toggle else "plotly",
            **get_plot_bg(theme_toggle))
        render_chart(_fig_tmp, use_container_width=True)

    st.markdown("---")
    st.markdown("### Bilan journalier — Totaux")

    _bar_cols = [
        ("ap",        "Prod. après (kWh)",      cc["xgb"]),
        ("ecrete",    "Écrêtage (kWh)",          cc["ecrete"]),
        ("conso",     "Conso. générale (kWh)",   "#26A69A"),
        ("autoconso", "Autoconsommation (kWh)",  "#4CAF50"),
        ("reseau",    "Conso. réseau (kWh)",      "#FF7043"),
    ]
    _bar_labels = [d_md.strftime("%d/%m/%Y") for d_md in selected_dates_md]

    _fig_bar = go.Figure()
    for _bc, _bl, _bclr in _bar_cols:
        _vals = [_day_dfs[d_md][0][_bc].sum() for d_md in selected_dates_md]
        _fig_bar.add_trace(go.Bar(
            name=_bl, x=_bar_labels, y=_vals,
            marker_color=_bclr,
            text=[f"{v:.1f}" for v in _vals],
            textposition="outside",
            hovertemplate=f"<b>%{{x}}</b><br>{_bl} : <b>%{{y:.2f}} kWh</b><extra></extra>"
        ))
    _fig_bar.update_layout(
        barmode="group", height=400,
        xaxis_title="Date", yaxis_title="Énergie (kWh)",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", yanchor="bottom"),
        margin=dict(t=80, b=40),
        template="plotly_dark" if theme_toggle else "plotly",
        **get_plot_bg(theme_toggle))
    render_chart(_fig_bar, use_container_width=True)

    if summary_md:
        st.markdown("### Tableau comparatif")
        _df_sum = pd.DataFrame(summary_md)

        if len(summary_md) >= 2:
            _num_cols = [c for c in _df_sum.columns
                         if _df_sum[c].dtype in [float, int] or
                         str(_df_sum[c].dtype).startswith("float") or
                         str(_df_sum[c].dtype).startswith("int")]
            _ecart = {"Date": "Écart (1→2)", "Source": "—"}
            for c in _df_sum.columns:
                if c in ("Date", "Source", "Écrêtage (%)", "Autonomie (%)"):
                    continue
                try:
                    v1 = float(_df_sum.iloc[0][c])
                    v2 = float(_df_sum.iloc[1][c])
                    _ecart[c] = round(v2 - v1, 2)
                except Exception:
                    _ecart[c] = "—"
            _df_sum = pd.concat([_df_sum, pd.DataFrame([_ecart])], ignore_index=True)

        st.dataframe(_df_sum, use_container_width=True, hide_index=True)
        create_download_button(pd.DataFrame(summary_md), "comparaison_multi_dates.csv",
                               "📥 Exporter (CSV)", key="dl_md")

elif page == "Documentation":
    show_documentation() 
    