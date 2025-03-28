# === DASHBOARD CONSEILLER STREAMLIT ===
# Objectif : afficher le score d'éligibilité d'un client et les visuels SHAP globaux

import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Chargement du modèle entraîné
model = joblib.load("model.pkl")

# Interface Streamlit
st.set_page_config(page_title="Scoring Crédit Client", layout="centered")
st.title("🔍 Dashboard Éligibilité Crédit - Néo-Banque")
st.markdown("""
Ce tableau de bord vous permet de :
- Charger un fichier client (extrait de application_test.csv)
- Obtenir un score d'éligibilité
- Visualiser les facteurs influents globaux selon SHAP
""")

# Upload fichier client
uploaded_file = st.file_uploader("📂 Charger un fichier CSV client (1 ligne)", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)

    # Vérification que le fichier contient bien 1 seule ligne
    if len(input_df) != 1:
        st.error("Le fichier doit contenir les données d'un seul client.")
    else:
        # Nettoyage + transformation des données comme à l'entraînement
        input_df = pd.get_dummies(input_df, drop_first=True)

        # Compléter les colonnes manquantes avec zéro (comme modèle entraîné)
        model_features = model.named_steps['scaler'].get_feature_names_out()
        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_features]

        # Prédiction du score
        score = model.predict_proba(input_df)[0][1]
        st.success(f"Score de probabilité de défaut : **{score:.2%}**")

        if score < 0.4:
            st.markdown("🟢 **Client plutôt éligible** (score bas)")
        elif score < 0.7:
            st.markdown("🟠 **Client à risque modéré**")
        else:
            st.markdown("🔴 **Client potentiellement à risque élevé**")

# Affichage des visuels SHAP globaux
st.markdown("---")
st.subheader("📊 Analyse Globale des Variables (SHAP)")

col1, col2 = st.columns(2)

with col1:
    st.image("shap_Figure_1.png", caption="Distribution des effets SHAP par variable", use_column_width=True)

with col2:
    st.image("shap_Figure_2.png", caption="Importance moyenne des variables", use_column_width=True)

st.markdown("---")
st.caption("© Projet ISCODE Bloc 2 — C16 à C21. Dashboard démo local.")