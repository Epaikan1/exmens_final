# === DASHBOARD STREAMLIT CONNECTÉ À API (UX FINAL + STYLE CENTERED) ===
# Objectif : dashboard UX élégant, filtrable, centré et harmonisé

import streamlit as st
import pandas as pd
import requests
from PIL import Image
import json

# === Configuration de l'API ===
API_URL = "http://127.0.0.1:8000"
API_KEY = "123secure-key"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# === Page config ===
st.set_page_config(page_title="Scoring Crédit IA", layout="wide")

with st.container():
    st.markdown("<h1 style='text-align: center;'>🏦 Scoring Crédit IA — Interface Conseiller</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center;'>
    Ce tableau de bord permet aux conseillers de :<br>
    ✅ Évaluer le risque d’un client à l’aide d’un modèle IA<br>
    ✅ Comprendre l’origine du score via des explications simples<br>
    ✅ Visualiser ses données principales et filtrer les clients
    </div>
    """, unsafe_allow_html=True)

# === Blocs pédagogiques et fichiers exemples ===
st.info("ℹ️ Pour des résultats fiables, utilisez un fichier complet avec toutes les colonnes d'entraînement. Un fichier simplifié donne un score moins précis.")

col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
with col_btn1:
    st.download_button("⬇️ Faible risque (complet)", open("client_risk_min_complet.csv", "rb"), file_name="client_risk_min_complet.csv")
with col_btn2:
    st.download_button("⬇️ Risque élevé (complet)", open("client_risk_max_complet.csv", "rb"), file_name="client_risk_max_complet.csv")
with col_btn3:
    st.download_button("⬇️ Client simplifié", open("client_risk_moyen_simplifie.csv", "rb"), file_name="client_risk_moyen_simplifie.csv")
with col_btn4:
    st.download_button("⬇️ Multi-clients", open("multi_clients_sample.csv", "rb"), file_name="multi_clients_sample.csv")

# === Upload d'un fichier CSV ===
with st.expander("📁 Uploader un fichier CSV client (1 ou plusieurs lignes)"):
    st.markdown("Le fichier doit être pré-formaté comme ceux ci-dessus, avec des colonnes encodées.")
    uploaded_file = st.file_uploader("Uploader le fichier ici", type=["csv"])

if uploaded_file:
    df_all = pd.read_csv(uploaded_file)

    if df_all.empty:
        st.error("Le fichier est vide.")
    else:
        st.success(f"✅ {len(df_all)} client(s) chargé(s)")

        selected_index = st.selectbox("👤 Sélectionnez un client à analyser :", df_all.index, format_func=lambda i: f"Client #{i+1}")
        input_df = df_all.loc[[selected_index]]
        client_dict = input_df.to_dict(orient="records")[0]

        with st.container():
            st.markdown("<h3 style='text-align: center;'>👤 Informations du client sélectionné</h3>", unsafe_allow_html=True)
            st.dataframe(input_df.T, use_container_width=True)

        with st.spinner("🔍 Analyse du risque en cours..."):
            predict_response = requests.post(f"{API_URL}/predict", headers=HEADERS, data=json.dumps({"features": client_dict}))

        if predict_response.status_code == 200:
            result = predict_response.json()
            score = result['score']
            decision = result['decision']

            st.markdown("<h3 style='text-align: center;'>🎯 Résultat du Score IA</h3>", unsafe_allow_html=True)
            st.metric(label="🧮 Probabilité de défaut", value=f"{score:.2%}", help="Plus le score est proche de 1, plus le risque est élevé.")

            if score < 0.4:
                st.success("🟢 Client éligible au crédit")
            elif score < 0.7:
                st.warning("🟠 Client à risque modéré")
            else:
                st.error("🔴 Client potentiellement à risque élevé")

            with st.spinner("📊 Analyse locale SHAP en cours..."):
                explain_response = requests.post(f"{API_URL}/explain", headers=HEADERS, data=json.dumps({"features": client_dict}))

            if explain_response.status_code == 200:
                contribs = explain_response.json()['top_contributions']
                shap_df = pd.DataFrame(contribs)
                st.markdown("<h4 style='text-align: center;'>🔍 Variables les plus influentes</h4>", unsafe_allow_html=True)
                st.dataframe(shap_df, use_container_width=True)
            else:
                st.warning("⚠️ L'explication SHAP n'a pas pu être générée pour ce client.")
        else:
            st.error(f"Erreur API /predict : {predict_response.status_code}")

# === Visuels SHAP globaux (explicatifs) ===
with st.expander("📊 Analyse Globale du Modèle (SHAP) - Cliquez pour voir"):
    st.subheader("📊 Analyse Globale du Modèle (SHAP)")
    st.markdown("Ces graphiques permettent de comprendre quelles variables influencent le plus souvent les décisions du modèle, tous clients confondus.")
    st.markdown("💡 **Astuce :** ces analyses SHAP sont générées sur l'ensemble du jeu de données.")
    col1, col2 = st.columns(2)
    with col1:
        st.image("shap_Figure_1.png", caption="Distribution des effets SHAP", use_column_width=True)
    with col2:
        st.image("shap_Figure_2.png", caption="Importance moyenne des variables", use_column_width=True)

st.markdown("---")
st.caption("Projet ISCODE | Bloc 2 IA — API, Dashboard, UX, Sécurité et Interprétabilité")
