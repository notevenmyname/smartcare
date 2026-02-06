#!/usr/bin/env python3
"""
Dashboard Streamlit - AED & Prédictions (RandomForest + SARIMA)
Hôpital Pitié-Salpêtrière 2012-2017
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

st.set_page_config(
    page_title="SmartCare Dashboard",
    page_icon="🏥",
    layout="wide"
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "urgences_2012_2016_mensuel.csv"

# ============================================================
# CHARGEMENT DONNÉES
# ============================================================

@st.cache_data
def load_data():
    """Charge et prépare le DataFrame."""
    df = pd.read_csv(DATA_PATH, sep=';', decimal='.')
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df = df.sort_values('date').reset_index(drop=True)

    # Features temporelles
    df['Année'] = df['date'].dt.year
    df['Mois'] = df['date'].dt.month
    df['Trimestre'] = df['date'].dt.quarter
    df['Mois_Nom'] = df['date'].dt.strftime('%B')
    df['Taux_Occupation'] = df['Passages_Urgences'] / df['Lits_Theoriques']

    return df

df = load_data()


# ============================================================================
# FONCTIONS DE SIMULATION CRISE SANITAIRE
# ============================================================================

def simulate_crisis_scenario(pred_passages_sarima, 
                             crisis_intensity=0.5,  # 0-1 scale
                             baseline_staff=10657,
                             baseline_beds=2229,
                             baseline_avg_passage_2016=10639.83):
    """
    Simule un scénario de crise sanitaire basé sur la prévision SARIMA.
    
    Args:
        pred_passages_sarima: array des prédictions SARIMA 2017
        crisis_intensity: 0-1 (0=normal, 1=crise max 70%)
        baseline_staff: nombre de professionnels baseline
        baseline_beds: nombre de lits baseline
        baseline_avg_passage_2016: passages moyens par mois 2016
    
    Returns:
        dict avec tous les indicateurs de crise
    """
    
    # Paramètres de crise (évolutifs avec intensité)
    passage_increase = 0.30 + (crisis_intensity * 0.40)  # 30-70%
    staff_reduction = 0.10 + (crisis_intensity * 0.20)   # 10-30%
    los_increase = 1.0 + (crisis_intensity * 0.50)       # durée séjour x1.0 à x1.5
    dasri_multiplier = 1.0 + (crisis_intensity * 2.0)    # 1x à 3x
    
    # Prévisions de crise
    crisis_passages = pred_passages_sarima * (1 + passage_increase)
    crisis_staff = baseline_staff * (1 - staff_reduction)
    
    # Indicateurs de tension
    normal_occupation_rate = pred_passages_sarima / baseline_beds
    crisis_occupation_rate = crisis_passages / baseline_beds
    
    normal_ratio_staff = baseline_staff / pred_passages_sarima
    crisis_ratio_staff = crisis_staff / crisis_passages
    
    # Indice de tension synthétique (0-6 scale)
    normal_tension = np.minimum(6.0, (pred_passages_sarima / baseline_avg_passage_2016) * 4.88)
    crisis_tension = np.minimum(6.0, (crisis_passages / baseline_avg_passage_2016) * 4.88 * (1 + crisis_intensity * 0.5))
    
    return {
        'passages_normal': pred_passages_sarima,
        'passages_crisis': crisis_passages,
        'staff_baseline': baseline_staff,
        'staff_crisis': crisis_staff,
        'beds_available': baseline_beds,
        'occupation_normal': normal_occupation_rate,
        'occupation_crisis': crisis_occupation_rate,
        'ratio_staff_normal': normal_ratio_staff,
        'ratio_staff_crisis': crisis_ratio_staff,
        'tension_normal': normal_tension,
        'tension_crisis': crisis_tension,
        'los_multiplier': los_increase,
        'dasri_multiplier': dasri_multiplier,
        'passage_increase_pct': passage_increase * 100,
        'staff_reduction_pct': staff_reduction * 100
    }

# ============================================================================
# PAGES
# ============================================================================

pages = {
    "📊 Analyse Exploratoire": "eda",
    "🔮 Prédictions 2017": "predictions",
    "🔥 Scénario Crise Sanitaire": "crisis"
}

page = st.sidebar.radio("Navigation", list(pages.keys()))

# ============================================================================
# PAGE 1: ANALYSE EXPLORATOIRE
# ============================================================================

if pages[page] == "eda":
    st.title("📊 Analyse Exploratoire des Données (AED)")
    st.markdown("### Urgences Hôpital Pitié-Salpêtrière 2012-2016")
    
    # SECTION 0: CHIFFRES CLÉS DE L'HÔPITAL
    st.markdown("---")
    st.subheader("🏥 Chiffres Clés de l'Établissement")
    
    # Données clés
    lits_2012 = 2132
    lits_2016 = 2229
    lits_evolution = ((lits_2016 - lits_2012) / lits_2012) * 100
    
    staff_2012 = 8335
    staff_2016 = 10657
    staff_evolution = ((staff_2016 - staff_2012) / staff_2012) * 100
    
    passages_2012 = 85993
    passages_2016 = 127678
    passages_evolution = ((passages_2016 - passages_2012) / passages_2012) * 100
    
    # Afficher en 3 colonnes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;'>
            <h3 style='margin: 0; font-size: 14px; opacity: 0.9;'>NOMBRE DE LITS</h3>
            <p style='margin: 10px 0; font-size: 24px; font-weight: bold;'>{lits_2012:,} → {lits_2016:,}</p>
            <p style='margin: 0; font-size: 14px; font-weight: bold; color: #90EE90;'>+{lits_evolution:.1f}%</p>
            <p style='margin: 5px 0 0 0; font-size: 12px; opacity: 0.8;'>2012 → 2016</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;'>
            <h3 style='margin: 0; font-size: 14px; opacity: 0.9;'>PROFESSIONNELS DE SANTÉ</h3>
            <p style='margin: 10px 0; font-size: 24px; font-weight: bold;'>{staff_2012:,} → {staff_2016:,}</p>
            <p style='margin: 0; font-size: 14px; font-weight: bold; color: #90EE90;'>+{staff_evolution:.1f}%</p>
            <p style='margin: 5px 0 0 0; font-size: 12px; opacity: 0.8;'>2012 → 2016</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;'>
            <h3 style='margin: 0; font-size: 14px; opacity: 0.9;'>PASSAGES AUX URGENCES</h3>
            <p style='margin: 10px 0; font-size: 24px; font-weight: bold;'>{passages_2012:,} → {passages_2016:,}</p>
            <p style='margin: 0; font-size: 14px; font-weight: bold; color: #90EE90;'>+{passages_evolution:.1f}%</p>
            <p style='margin: 5px 0 0 0; font-size: 12px; opacity: 0.8;'>2012 → 2016</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Métriques globales
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Durée", f"{(df['date'].max() - df['date'].min()).days} j")
    #with col2:
    #    st.metric("Observations", len(df))
    with col3:
        st.metric("Passages Total", f"{df['Passages_Urgences'].sum():,.0f}")
    with col4:
        st.metric("Moyenne/Mois", f"{df['Passages_Urgences'].mean():.0f}")
    with col5:
        st.metric("Tension Moy", f"{df['Indice_Tension'].mean():.2f}")
    
    # SECTION 1: Tendances annuelles
    st.markdown("---")
    st.subheader("1️⃣ Tendances par Année")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Total passages par année
        yearly_total = df.groupby('Année')['Passages_Urgences'].sum()
        yearly_mean = df.groupby('Année')['Passages_Urgences'].mean()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=yearly_total.index,
            y=yearly_total.values,
            name='Total annuel',
            marker_color='#1f77b4',
            text=yearly_total.values.astype(int),
            textposition='auto'
        ))
        fig.update_layout(
            title="Total Passages par Année",
            xaxis_title="Année",
            yaxis_title="Passages",
            height=350,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Indice de tension par année
        tension_stats = df.groupby('Année')['Indice_Tension'].agg(['mean', 'std'])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=tension_stats.index,
            y=tension_stats['mean'],
            error_y=dict(type='data', array=tension_stats['std']),
            marker_color='#d62728',
            name='Tension'
        ))
        fig.add_hline(y=4, line_dash="dash", line_color="orange", 
                     annotation_text="Seuil Tendu", annotation_position="right")
        fig.update_layout(
            title="Indice de Tension par Année",
            xaxis_title="Année",
            yaxis_title="Tension",
            height=350,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution par année
        data_by_year = [df[df['Année'] == y]['Passages_Urgences'].values for y in sorted(df['Année'].unique())]
        
        fig = go.Figure()
        for i, year in enumerate(sorted(df['Année'].unique())):
            fig.add_trace(go.Box(
                y=data_by_year[i],
                name=str(year),
                boxmean='sd'
            ))
        fig.update_layout(
            title="Distribution Passages par Année",
            yaxis_title="Passages",
            height=350,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Taux occupation par année
        occupation_stats = df.groupby('Année')['Taux_Occupation'].agg(['mean', 'std'])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=occupation_stats.index,
            y=occupation_stats['mean'],
            error_y=dict(type='data', array=occupation_stats['std']),
            marker_color='#ff7f0e',
            name='Taux'
        ))
        fig.update_layout(
            title="Taux d'Occupation par Année",
            xaxis_title="Année",
            yaxis_title="Taux",
            height=350,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # SECTION 2: Saisonnalité
    st.markdown("---")
    st.subheader("2️⃣ Saisonnalité Mensuelle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Passages mensuels moyens
        months_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
        monthly_stats = df.groupby('Mois')['Passages_Urgences'].agg(['mean', 'std'])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[months_names[int(m)-1] for m in monthly_stats.index],
            y=monthly_stats['mean'],
            error_y=dict(type='data', array=monthly_stats['std']),
            marker_color='#2ca02c',
            name='Passages'
        ))
        fig.add_hline(y=monthly_stats['mean'].mean(), line_dash="dash", 
                     line_color="red", annotation_text="Moyenne annuelle")
        fig.update_layout(
            title="Saisonnalité: Passages Mensuels",
            xaxis_title="Mois",
            yaxis_title="Passages",
            height=350,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Heatmap année vs mois
        pivot_data = df.pivot_table(values='Passages_Urgences', index='Année', columns='Mois', aggfunc='mean')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=[months_names[int(m)-1] for m in pivot_data.columns],
            y=pivot_data.index,
            colorscale='RdYlGn_r'
        ))
        fig.update_layout(
            title="Heatmap: Passages par Année-Mois",
            xaxis_title="Mois",
            yaxis_title="Année",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Série temporelle avec MA
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['Passages_Urgences'],
            mode='lines', name='Données réelles',
            line=dict(color='#1f77b4', width=2)
        ))
        ma3 = df['Passages_Urgences'].rolling(3).mean()
        fig.add_trace(go.Scatter(
            x=df['date'], y=ma3,
            mode='lines', name='MA(3)',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        fig.update_layout(
            title="Série Temporelle 2012-2016",
            xaxis_title="Date",
            yaxis_title="Passages",
            height=350,
            template='plotly_white',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['Passages_Urgences'],
            nbinsx=20,
            marker_color='#1f77b4',
            name='Distribution'
        ))
        fig.add_vline(x=df['Passages_Urgences'].mean(), 
                     line_dash="dash", line_color="red",
                     annotation_text=f"Moyenne: {df['Passages_Urgences'].mean():.0f}")
        fig.update_layout(
            title="Distribution Passages",
            xaxis_title="Passages",
            yaxis_title="Fréquence",
            height=350,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    

# ============================================================================
# PAGE 2: PRÉDICTIONS 2017
# ============================================================================

elif pages[page] == "predictions":
    st.title("🔮 Prédictions 2017")
    st.markdown("### Comparaison RandomForest vs SARIMA")
    
    st.markdown("---")
    
    # Préparation des données pour les modèles
    df_train = df.copy()
    
    # Features temporelles manquantes (vérifier qu'elles existent)
    if 'Mois' not in df_train.columns:
        df_train['Mois'] = df_train['date'].dt.month
    if 'Année' not in df_train.columns:
        df_train['Année'] = df_train['date'].dt.year
    
    # Features pour RandomForest
    df_train['time_idx'] = np.arange(len(df_train))
    X_train = df_train[['time_idx', 'Mois', 'Année']].values
    y_passages = df_train['Passages_Urgences'].values.astype(float)
    y_tension = df_train['Indice_Tension'].values.astype(float)
    
    # Entraîner les modèles
    with st.spinner('Entraînement des modèles...'):
        # RandomForest
        rf_passages = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf_passages.fit(X_train, y_passages)
        
        rf_tension = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf_tension.fit(X_train, y_tension)
        
        # SARIMA
        model_sarima_passages = SARIMAX(y_passages, order=(1,1,1), 
                                       seasonal_order=(1,1,1,12),
                                       enforce_stationarity=False,
                                       enforce_invertibility=False)
        result_sarima_passages = model_sarima_passages.fit(disp=False)
        
        model_sarima_tension = SARIMAX(y_tension, order=(1,1,1),
                                      seasonal_order=(1,1,1,12),
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
        result_sarima_tension = model_sarima_tension.fit(disp=False)
    
    st.success("✅ Modèles entraînés!")
    
    # Prédictions 2017
    months_2017 = np.arange(1, 13)
    years_2017 = np.full(12, 2017)
    time_idx_2017 = np.arange(len(df_train), len(df_train) + 12)
    
    X_2017 = np.column_stack([time_idx_2017, months_2017, years_2017])
    
    # Prédictions RandomForest
    pred_rf_passages = rf_passages.predict(X_2017)
    pred_rf_tension = rf_tension.predict(X_2017)
    
    # Prédictions SARIMA
    forecast_sarima_passages = result_sarima_passages.get_forecast(steps=12)
    pred_sarima_passages = np.asarray(forecast_sarima_passages.predicted_mean)
    ci_sarima_passages = forecast_sarima_passages.conf_int()
    
    forecast_sarima_tension = result_sarima_tension.get_forecast(steps=12)
    pred_sarima_tension = np.asarray(forecast_sarima_tension.predicted_mean)
    ci_sarima_tension = forecast_sarima_tension.conf_int()
    
    # Dates 2017
    dates_2017 = pd.date_range(start='2017-01-01', periods=12, freq='MS')
    months_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
    
    # Dates 2017
    dates_2017 = pd.date_range(start='2017-01-01', periods=12, freq='MS')
    months_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
    
    # ========== PASSAGES URGENCES ==========
    st.subheader("📊 Prédictions: Passages Urgences")
    
    col1, col2 = st.columns(2)
    
    # Graphe 1: RandomForest seul
    with col1:
        fig_rf = go.Figure()
        
        # Historique
        fig_rf.add_trace(go.Scatter(
            x=df['date'], y=df['Passages_Urgences'],
            mode='lines', name='Historique 2012-2016',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # RF Prédiction
        fig_rf.add_trace(go.Scatter(
            x=dates_2017, y=pred_rf_passages,
            mode='lines+markers', name='RandomForest 2017',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=10, symbol='circle')
        ))
        
        fig_rf.update_layout(
            title="RandomForest - Prédictions Passages 2017",
            xaxis_title="Date",
            yaxis_title="Passages",
            height=450,
            template='plotly_white',
            hovermode='x unified'
        )
        st.plotly_chart(fig_rf, use_container_width=True)
    
    # Graphe 2: SARIMA seul
    with col2:
        fig_sarima = go.Figure()
        
        # Historique
        fig_sarima.add_trace(go.Scatter(
            x=df['date'], y=df['Passages_Urgences'],
            mode='lines', name='Historique 2012-2016',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # SARIMA Prédiction
        fig_sarima.add_trace(go.Scatter(
            x=dates_2017, y=pred_sarima_passages,
            mode='lines+markers', name='SARIMA 2017',
            line=dict(color='#d62728', width=3),
            marker=dict(size=10, symbol='square')
        ))
        
        # IC SARIMA (intervalle de confiance) - sans les afficher visuellement
        # (juste garder les valeurs pour reference)
        
        fig_sarima.update_layout(
            title="SARIMA - Prédictions Passages 2017",
            xaxis_title="Date",
            yaxis_title="Passages",
            height=450,
            template='plotly_white',
            hovermode='x unified',
            yaxis=dict(
                range=[df['Passages_Urgences'].min() - 500, df['Passages_Urgences'].max() + 500]
            )
        )
        st.plotly_chart(fig_sarima, use_container_width=True)
    
    # ========== INDICE DE TENSION ==========
    st.subheader("📊 Prédictions: Indice de Tension")
    
    col1, col2 = st.columns(2)
    
    # Graphe 3: RandomForest Tension
    with col1:
        fig_rf_tension = go.Figure()
        
        # Historique
        fig_rf_tension.add_trace(go.Scatter(
            x=df['date'], y=df['Indice_Tension'],
            mode='lines', name='Historique 2012-2016',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # RF Tension
        fig_rf_tension.add_trace(go.Scatter(
            x=dates_2017, y=pred_rf_tension,
            mode='lines+markers', name='RandomForest 2017',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=10, symbol='circle')
        ))
        
        # Seuils
        fig_rf_tension.add_hline(y=3, line_dash="dash", line_color="orange", 
                         annotation_text="Normal (3.0)", annotation_position="right")
        fig_rf_tension.add_hline(y=4, line_dash="dash", line_color="red",
                         annotation_text="Tendu (4.0)", annotation_position="right")
        
        fig_rf_tension.update_layout(
            title="RandomForest - Prédictions Tension 2017",
            xaxis_title="Date",
            yaxis_title="Indice de Tension",
            height=450,
            template='plotly_white',
            hovermode='x unified'
        )
        st.plotly_chart(fig_rf_tension, use_container_width=True)
    
    # Graphe 4: SARIMA Tension
    with col2:
        fig_sarima_tension = go.Figure()
        
        # Historique
        fig_sarima_tension.add_trace(go.Scatter(
            x=df['date'], y=df['Indice_Tension'],
            mode='lines', name='Historique 2012-2016',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # SARIMA Tension
        fig_sarima_tension.add_trace(go.Scatter(
            x=dates_2017, y=pred_sarima_tension,
            mode='lines+markers', name='SARIMA 2017',
            line=dict(color='#d62728', width=3),
            marker=dict(size=10, symbol='square')
        ))
        
        # Seuils
        fig_sarima_tension.add_hline(y=3, line_dash="dash", line_color="orange", 
                             annotation_text="Normal (3.0)", annotation_position="right")
        fig_sarima_tension.add_hline(y=4, line_dash="dash", line_color="red",
                             annotation_text="Tendu (4.0)", annotation_position="right")
        
        fig_sarima_tension.update_layout(
            title="SARIMA - Prédictions Tension 2017",
            xaxis_title="Date",
            yaxis_title="Indice de Tension",
            height=450,
            template='plotly_white',
            hovermode='x unified',
            yaxis=dict(
                range=[df['Indice_Tension'].min() - 0.5, df['Indice_Tension'].max() + 0.5]
            )
        )
        st.plotly_chart(fig_sarima_tension, use_container_width=True)
    
    # Tableau de prédictions
    st.markdown("---")
    st.subheader("Tableau Détaillé des Prédictions 2017")
    
    pred_df = pd.DataFrame({
        'Mois': months_names,
        'RF Passages': pred_rf_passages.astype(int).tolist(),
        'SARIMA Passages': pred_sarima_passages.astype(int).tolist(),
        'SARIMA IC Lower': ci_sarima_passages[:, 0].astype(int).tolist(),
        'SARIMA IC Upper': ci_sarima_passages[:, 1].astype(int).tolist(),
        'RF Tension': pred_rf_tension.round(2).tolist(),
        'SARIMA Tension': pred_sarima_tension.round(2).tolist()
    })
    
    st.dataframe(pred_df, use_container_width=True)
    
    # Résumé
    st.markdown("---")
    st.subheader("Résumé des Prédictions 2017")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total RF", f"{pred_rf_passages.sum():,.0f}")
    with col2:
        st.metric("Total SARIMA", f"{pred_sarima_passages.sum():,.0f}")
    with col3:
        diff = pred_sarima_passages.sum() - pred_rf_passages.sum()
        st.metric("Différence", f"{diff:+,.0f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Tension Moyenne 2017**")
        st.write(f"• RandomForest: {pred_rf_tension.mean():.2f}")
        st.write(f"• SARIMA: {pred_sarima_tension.mean():.2f}")
    
    with col2:
        st.write("**Mois Critiques (Tension > 4)**")
        critical_rf = sum(1 for t in pred_rf_tension if t >= 4)
        critical_sarima = sum(1 for t in pred_sarima_tension if t >= 4)
        st.write(f"• RandomForest: {critical_rf} mois")
        st.write(f"• SARIMA: {critical_sarima} mois")

# ============================================================================
# PAGE 3: SCÉNARIO CRISE SANITAIRE
# ============================================================================

elif pages[page] == "crisis":
    st.title("🔥 Prévision 2017: Normal vs Crise Sanitaire")
    st.markdown("### Scénario contrefactuel - Stress test hospitalier")
    
    st.markdown("---")
    
    # Paramètres de crise
    st.sidebar.markdown("## ⚠️ Paramètres Crise")
    crisis_intensity = st.sidebar.slider(
        "Intensité de crise (0 = normal, 1 = max)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Paramètre de stress test: 0 = scénario normal, 1 = crise extrême"
    )
    
    # ========== RECALCULER LES PRÉDICTIONS ==========
    with st.spinner("⏳ Calcul des scénarios normal et crise..."):
        # Préparation données
        df_sorted = df.sort_values('date').reset_index(drop=True)
        X = np.arange(len(df_sorted)).reshape(-1, 1)
        y_passages = df_sorted['Passages_Urgences'].values
        y_tension = df_sorted['Indice_Tension'].values
        
        # SARIMA
        model_sarima_p = SARIMAX(
            y_passages,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results_sarima_p = model_sarima_p.fit(disp=False)
        pred_sarima_passages = np.asarray(results_sarima_p.forecast(steps=12)).astype(float)
        pred_sarima_passages = np.maximum(pred_sarima_passages, 0)
        
        # Dates pour 2017
        dates_2017 = pd.date_range(start='2017-01-01', periods=12, freq='MS')
        months_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
    
    # Simulations (utiliser les prédictions SARIMA)
    crisis_data = simulate_crisis_scenario(
        pred_sarima_passages,
        crisis_intensity=crisis_intensity
    )
    
    # ========== CARTES KPI ==========
    st.subheader("📊 Indicateurs Clés de Comparaison")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_normal = crisis_data['passages_normal'].mean()
        avg_crisis = crisis_data['passages_crisis'].mean()
        st.metric(
            "Passages/mois NORMAL",
            f"{avg_normal:,.0f}",
            f"Baseline"
        )
    
    with col2:
        st.metric(
            "Passages/mois CRISE",
            f"{avg_crisis:,.0f}",
            f"+{crisis_data['passage_increase_pct']:.0f}%"
        )
    
    with col3:
        st.metric(
            "Personnel NORMAL",
            f"{crisis_data['staff_baseline']:,.0f}",
            "Baseline"
        )
    
    with col4:
        st.metric(
            "Personnel CRISE",
            f"{crisis_data['staff_crisis']:,.0f}",
            f"-{crisis_data['staff_reduction_pct']:.0f}%"
        )
    
    # ========== GRAPHIQUES PASSAGES ==========
    st.markdown("---")
    st.subheader("📈 Passages aux Urgences: Normal vs Crise")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphe détaillé
        fig_passages = go.Figure()
        
        # Historique
        fig_passages.add_trace(go.Scatter(
            x=df['date'], y=df['Passages_Urgences'],
            mode='lines', name='Historique 2012-2016',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Scénario normal
        fig_passages.add_trace(go.Scatter(
            x=dates_2017, y=crisis_data['passages_normal'],
            mode='lines+markers', name='Scénario Normal (SARIMA)',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=8)
        ))
        
        # Scénario crise
        fig_passages.add_trace(go.Scatter(
            x=dates_2017, y=crisis_data['passages_crisis'],
            mode='lines+markers', name='Scénario Crise Sanitaire',
            line=dict(color='#d62728', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ))
        
        # Zone de saturation (seuil critique)
        saturation_threshold = crisis_data['beds_available'] * 0.9
        fig_passages.add_hline(
            y=saturation_threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Seuil Critique ({saturation_threshold:.0f})",
            annotation_position="right"
        )
        
        fig_passages.update_layout(
            title="Passages aux Urgences: Trajectoires Comparées",
            xaxis_title="Date",
            yaxis_title="Passages/mois",
            height=450,
            template='plotly_white',
            hovermode='x unified'
        )
        st.plotly_chart(fig_passages, use_container_width=True)
    
    with col2:
        # Comparaison mensuelle
        diff_passages = crisis_data['passages_crisis'] - crisis_data['passages_normal']
        
        fig_diff = go.Figure()
        
        colors = ['#d62728' if x > 0 else '#2ca02c' for x in diff_passages]
        
        fig_diff.add_trace(go.Bar(
            x=months_names,
            y=diff_passages,
            marker_color=colors,
            name='Différence',
            hovertemplate='<b>%{x}</b><br>Surcharge: %{y:.0f} passages<extra></extra>'
        ))
        
        fig_diff.add_hline(y=0, line_color="black", line_width=1)
        
        fig_diff.update_layout(
            title="Surcharge Mensuelle en Crise",
            xaxis_title="Mois",
            yaxis_title="Passages supplémentaires",
            height=450,
            template='plotly_white',
            showlegend=False
        )
        st.plotly_chart(fig_diff, use_container_width=True)
    
    # ========== TENSION HOSPITALIÈRE ==========
    st.markdown("---")
    st.subheader("⚠️ Indice de Tension Hospitalière")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_tension = go.Figure()
        
        # Normal
        fig_tension.add_trace(go.Scatter(
            x=dates_2017, y=crisis_data['tension_normal'],
            mode='lines+markers', name='Tension (Normal)',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=8)
        ))
        
        # Crise
        fig_tension.add_trace(go.Scatter(
            x=dates_2017, y=crisis_data['tension_crisis'],
            mode='lines+markers', name='Tension (Crise)',
            line=dict(color='#d62728', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ))
        
        # Seuils
        fig_tension.add_hline(y=3.0, line_dash="dash", line_color="orange", annotation_text="Normal (3.0)")
        fig_tension.add_hline(y=4.0, line_dash="dash", line_color="red", annotation_text="Critique (4.0)")
        fig_tension.add_hline(y=6.0, line_dash="dash", line_color="darkred", annotation_text="Saturation (6.0)")
        
        fig_tension.update_layout(
            title="Indice de Tension: Normal vs Crise",
            xaxis_title="Mois",
            yaxis_title="Indice (0-6)",
            height=400,
            template='plotly_white',
            yaxis=dict(range=[0, 6.5])
        )
        st.plotly_chart(fig_tension, use_container_width=True)
    
    with col2:
        fig_occu = go.Figure()
        
        # Normal
        fig_occu.add_trace(go.Scatter(
            x=dates_2017, y=crisis_data['occupation_normal'],
            mode='lines+markers', name='Occupation (Normal)',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=8)
        ))
        
        # Crise
        fig_occu.add_trace(go.Scatter(
            x=dates_2017, y=crisis_data['occupation_crisis'],
            mode='lines+markers', name='Occupation (Crise)',
            line=dict(color='#d62728', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ))
        
        # Seuil de saturation
        fig_occu.add_hline(y=0.85, line_dash="dash", line_color="orange", annotation_text="Alerte (85%)")
        fig_occu.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Saturation (100%)")
        
        fig_occu.update_layout(
            title="Taux d'Occupation des Lits",
            xaxis_title="Mois",
            yaxis_title="Taux occupation",
            height=400,
            template='plotly_white',
            yaxis=dict(tickformat='.0%')
        )
        st.plotly_chart(fig_occu, use_container_width=True)
    
    # ========== RATIO PERSONNEL/PATIENTS ==========
    st.markdown("---")
    st.subheader("👨‍⚕️ Ratio Personnel / Patients")
    
    fig_ratio = go.Figure()
    
    fig_ratio.add_trace(go.Scatter(
        x=dates_2017, y=crisis_data['ratio_staff_normal'],
        mode='lines+markers', name='Ratio Normal',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{x|%b}</b><br>Personnel/Patient: %{y:.3f}<extra></extra>'
    ))
    
    fig_ratio.add_trace(go.Scatter(
        x=dates_2017, y=crisis_data['ratio_staff_crisis'],
        mode='lines+markers', name='Ratio Crise (absentéisme)',
        line=dict(color='#d62728', width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond'),
        hovertemplate='<b>%{x|%b}</b><br>Personnel/Patient: %{y:.3f}<extra></extra>'
    ))
    
    fig_ratio.add_hline(y=0.001, line_dash="dash", line_color="orange", annotation_text="Alerte: 1 staff/1000 patients")
    
    fig_ratio.update_layout(
        title="Charge par Professionnel: Impact de la Crise",
        xaxis_title="Mois",
        yaxis_title="Ratio personnel/patients",
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )
    st.plotly_chart(fig_ratio, use_container_width=True)
    
    # ========== MESSAGES INTERPRÉTATIFS ==========
    st.markdown("---")
    st.subheader("💡 Interprétation et Recommandations")
    
    # Déterminer les mois critiques
    months_over_capacity_normal = np.sum(crisis_data['tension_normal'] >= 4)
    months_over_capacity_crisis = np.sum(crisis_data['tension_crisis'] >= 4)
    
    max_tension_normal = crisis_data['tension_normal'].max()
    max_tension_crisis = crisis_data['tension_crisis'].max()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🟢 Scénario Normal")
        
        if months_over_capacity_normal == 0:
            st.success(f"""
            ✅ **Fonctionnement stable**
            
            • Aucun mois en tension critique
            • Indice max: {max_tension_normal:.2f}/6.0
            • Capacité hospitalière suffisante
            • Flux d'urgences gérable
            """)
        else:
            st.warning(f"""
            ⚠️ **Tensions saisonnières détectées**
            
            • {months_over_capacity_normal} mois(s) en tension
            • Indice max: {max_tension_normal:.2f}/6.0
            • Pics prévisibles, planification possible
            """)
    
    with col2:
        st.markdown("### 🔴 Scénario Crise (+{:.0f}% passages)".format(crisis_data['passage_increase_pct']))
        
        if crisis_intensity < 0.3:
            st.info(f"""
            🟡 **Perturbation Modérée**
            
            • {months_over_capacity_crisis} mois(s) en tension critique
            • Indice max: {max_tension_crisis:.2f}/6.0
            • Absentéisme: -{crisis_data['staff_reduction_pct']:.0f}%
            • Actions court-terme suffisantes
            """)
        elif crisis_intensity < 0.7:
            st.warning(f"""
            🔴 **CRISE MAJEURE**
            
            • {months_over_capacity_crisis} mois(s) en tension extrême
            • Indice max: {max_tension_crisis:.2f}/6.0
            • Personnel réduit de {crisis_data['staff_reduction_pct']:.0f}%
            • **Actions immédiates requises**
            """)
        else:
            st.error(f"""
            🔴 **CRISE EXTRÊME**
            
            • {months_over_capacity_crisis} mois(s) saturés
            • Indice max: {max_tension_crisis:.2f}/6.0
            • Personnel réduit de {crisis_data['staff_reduction_pct']:.0f}%
            • **Capacités dépassées - État d'urgence**
            """)
    
    # Recommandations opérationnelles
    st.markdown("---")
    st.subheader("📋 Recommandations Opérationnelles")
    
    with st.expander("🎯 Actions Immédiates (Renforcement Préventif)"):
        st.markdown(f"""
        **En cas de crise d'intensité {crisis_intensity:.0%}:**
        
        1. **Renfort en Personnel** (-{crisis_data['staff_reduction_pct']:.0f}% = {int(crisis_data['staff_baseline'] - crisis_data['staff_crisis'])} agents attendus manquants)
           - Recruter contractuels d'urgence
           - Activer plans de continuité
           - Préparer appels aux retraités
        
        2. **Gestion de Capacité**
           - Déprogrammer chirurgies non-urgentes
           - Augmenter capacité: {int(crisis_data['passages_crisis'].mean() - crisis_data['passages_normal'].mean()):.0f} passages/mois en moyenne
           - Activer structures alternatives (hôtel hospitalier, etc.)
        
        3. **DASRI et Risques Infectieux**
           - Flux DASRI ×{crisis_data['dasri_multiplier']:.1f}
           - Augmenter stocks de matériel (masques, gels, etc.)
           - Renforcer circuits d'évacuation
        
        4. **Durée Moyenne de Séjour**
           - Ratio séjour ×{crisis_data['los_multiplier']:.1f}
           - Réduire délais d'orientation (lits) / sortie (services)
           - Augmenter lits de stabilisation/court-séjour
        """)
    
    with st.expander("📊 Monitoring et Indicateurs d'Alerte"):
        st.markdown(f"""
        **Seuils de basculement en action d'escalade:**
        
        - **Passages > {crisis_data['passages_normal'].max() * 1.3:.0f}/jour**: activation plan blanc
        - **Indice tension > 4.5**: renfort personnel 2ème niveau
        - **Indice tension > 5.5**: décision direction/ARS
        - **Taux occupation > 90%**: limitation urgences non-urgentes
        - **Ratio personnel < 1/1500 patients**: restructuration urgente
        """)
    
    with st.expander("💰 Évaluation Budgétaire"):
        # Estimation budgétaire simple
        surcharge_passages_year = (crisis_data['passages_crisis'].sum() - crisis_data['passages_normal'].sum())
        cost_per_passage = 120  # € estimé
        cost_surcharge = surcharge_passages_year * cost_per_passage
        
        missing_staff_count = int(crisis_data['staff_baseline'] - crisis_data['staff_crisis'])
        cost_per_agent = 45000  # € annuel
        cost_staff = missing_staff_count * cost_per_agent
        
        total_cost = cost_surcharge + cost_staff
        
        st.markdown(f"""
        **Coûts supplémentaires estimés en crise (intensité {crisis_intensity:.0%}):**
        
        | Poste | Montant |
        |------|---------|
        | Surcharge passages ({surcharge_passages_year:.0f} passages) | €{cost_surcharge:,.0f} |
        | Renfort personnel ({missing_staff_count} agents) | €{cost_staff:,.0f} |
        | **TOTAL ANNUEL** | **€{total_cost:,.0f}** |
        
        ⚠️ *Estimation basée sur paramètres simplifiés - Demander préchiffrage réel à finance*
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; font-size: 11px;'>
    SmartCare Dashboard | Analyse Temporelle & Prédictions 2017
    <br>
    Hôpital Pitié-Salpêtrière | © 2024
</div>
""", unsafe_allow_html=True)
