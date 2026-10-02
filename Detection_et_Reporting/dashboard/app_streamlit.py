import contextlib
import importlib.util
import io
import os
import time

import altair as alt
import pandas as pd
import streamlit as st

RACINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DOSSIER_DATA = os.path.join(RACINE, "data")
DOSSIER_RAPPORTS = os.path.join(RACINE, "rapports")
DOSSIER_SCRIPTS = os.path.join(RACINE, "scripts")

st.set_page_config(page_title="Suivi production TAQA (simulation)",
                   layout="wide")

VARIABLES_AFFICHABLES = {
    "Puissance (MW)": "puissance_MW",
    "Rendement thermique (%)": "rendement_pct",
    "Vibrations turbine (mm/s)": "vibrations_mm_s",
    "Temperature vapeur (degres C)": "temp_vapeur_C",
    "Pression vapeur (bar)": "pression_bar",
    "Consommation charbon (t/h)": "conso_charbon_t_h",
    "Debit eau alimentaire (t/h)": "debit_eau_t_h",
}


def importer(nom_fichier, alias):
    """Nos modules commencent par un chiffre (02_, 03_...), donc un import
    classique ne marche pas : on passe par importlib."""
    chemin = os.path.join(DOSSIER_SCRIPTS, nom_fichier)
    spec = importlib.util.spec_from_file_location(alias, chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@st.cache_data
def charger_donnees():
    df = pd.read_csv(os.path.join(DOSSIER_DATA, "detection_horaire.csv"),
                     parse_dates=["date"])
    episodes = pd.read_csv(os.path.join(DOSSIER_DATA, "anomalies_detectees.csv"),
                           parse_dates=["date_debut", "date_fin"])
    maintenance = pd.read_csv(os.path.join(DOSSIER_DATA, "historique_maintenance.csv"))
    return df, episodes, maintenance


def lancer_detection():
    """Rejoue l'agent 1 sur le dataset et reecrit les CSV de resultats."""
    detection = importer("02_agent_detection.py", "agent_detection")
    df = pd.read_csv(os.path.join(DOSSIER_DATA, "dataset_taqa_synthetique.csv"),
                     parse_dates=["date"])
    df, episodes = detection.executer_agent(df)
    df.to_csv(os.path.join(DOSSIER_DATA, "detection_horaire.csv"), index=False)
    episodes.to_csv(os.path.join(DOSSIER_DATA, "anomalies_detectees.csv"),
                    index=False)
    return len(df), len(episodes)


def lancer_reporting(debut, fin):
    """Appelle l'agent 2 sur la periode demandee (Gemini ou template)."""
    reporting = importer("03_agent_reporting.py", "agent_reporting")
    return reporting.generer_rapport(str(debut), str(fin))


# ------------------------------------------------------------------
# Navigation
# ------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Page", ["Consultation", "Pilotage du pipeline"],
                        label_visibility="collapsed")
st.sidebar.divider()


# ==================================================================
# PAGE 2 : pilotage du pipeline
# ==================================================================
if page == "Pilotage du pipeline":
    st.title("Pilotage du pipeline")
    st.caption("Relancer les agents et generer un rapport sans passer par la "
               "ligne de commande.")

    dataset = os.path.join(DOSSIER_DATA, "dataset_taqa_synthetique.csv")
    if not os.path.exists(dataset):
        st.error("Dataset introuvable dans le dossier data.")
        st.stop()

    st.subheader("Etat des fichiers de resultats")
    lignes_etat = []
    for nom in ["detection_horaire.csv", "anomalies_detectees.csv"]:
        chemin = os.path.join(DOSSIER_DATA, nom)
        if os.path.exists(chemin):
            horodatage = time.strftime("%d/%m/%Y %H:%M",
                                       time.localtime(os.path.getmtime(chemin)))
            lignes_etat.append({"Fichier": nom, "Etat": "present",
                                "Derniere mise a jour": horodatage})
        else:
            lignes_etat.append({"Fichier": nom, "Etat": "absent",
                                "Derniere mise a jour": "-"})
    st.dataframe(pd.DataFrame(lignes_etat), use_container_width=True,
                 hide_index=True)

    st.divider()

    st.subheader("1. Agent de detection")
    st.write("Rejoue la couche statistique et la couche Isolation Forest sur "
             "les 18 mois du dataset, puis reecrit les CSV de resultats.")

    if st.button("Lancer la detection", type="primary"):
        journal = io.StringIO()
        with st.spinner("Detection en cours (environ 10 a 20 secondes)..."):
            try:
                with contextlib.redirect_stdout(journal):
                    nb_lignes, nb_episodes = lancer_detection()
                st.cache_data.clear()
                st.success(f"Detection terminee : {nb_lignes} heures analysees, "
                           f"{nb_episodes} episodes d'alerte.")
            except Exception as e:
                st.error(f"Echec de la detection : {e}")
        sortie = journal.getvalue().strip()
        if sortie:
            with st.expander("Journal d'execution"):
                st.code(sortie)

    st.divider()

    st.subheader("2. Agent de reporting")

    if not os.path.exists(os.path.join(DOSSIER_DATA, "detection_horaire.csv")):
        st.info("Lancer d'abord la detection pour pouvoir generer un rapport.")
    else:
        reference = pd.read_csv(os.path.join(DOSSIER_DATA, "detection_horaire.csv"),
                                usecols=["date"], parse_dates=["date"])
        date_min = reference["date"].min().date()
        date_max = reference["date"].max().date()
        defaut_debut = max(date_min, date_max - pd.Timedelta(days=6))

        col_a, col_b = st.columns(2)
        debut = col_a.date_input("Debut de periode", value=defaut_debut,
                                 min_value=date_min, max_value=date_max)
        fin = col_b.date_input("Fin de periode", value=date_max,
                               min_value=date_min, max_value=date_max)

        cle_presente = os.path.exists(os.path.join(RACINE, ".env"))
        st.caption("Mode : API Gemini si une cle est configuree dans le fichier "
                   ".env, sinon bascule automatique sur le generateur template "
                   "local. " + ("Fichier .env detecte." if cle_presente
                                else "Aucun fichier .env detecte."))

        if st.button("Generer le rapport", type="primary"):
            if debut > fin:
                st.error("La date de debut doit preceder la date de fin.")
            else:
                journal = io.StringIO()
                with st.spinner("Redaction du rapport en cours..."):
                    try:
                        with contextlib.redirect_stdout(journal):
                            rapport, kpis = lancer_reporting(debut, fin)
                        st.cache_data.clear()
                        st.success(f"Rapport genere pour la periode "
                                   f"{kpis['periode']}.")
                        st.text_area("Apercu", rapport, height=350)
                    except Exception as e:
                        st.error(f"Echec de la generation : {e}")
                sortie = journal.getvalue().strip()
                if sortie:
                    with st.expander("Journal d'execution"):
                        st.code(sortie)

    st.stop()


# ==================================================================
# PAGE 1 : consultation
# ==================================================================
if not os.path.exists(os.path.join(DOSSIER_DATA, "detection_horaire.csv")):
    st.error("Resultats introuvables. Lancer la detection depuis la page "
             "Pilotage du pipeline (menu de gauche).")
    st.stop()

df, episodes, maintenance = charger_donnees()

st.title("Suivi de production : centrale thermique (donnees simulees)")
st.caption("Projet de stage TAQA Morocco / DPST. Dataset synthetique, "
           "aucune donnee reelle utilisee.")

# ------------------------------------------------------------------
# Barre laterale : filtres
# ------------------------------------------------------------------
st.sidebar.header("Filtres")

date_min = df["date"].min().date()
date_max = df["date"].max().date()
plage = st.sidebar.date_input(
    "Periode affichee",
    value=(pd.Timestamp("2026-05-01").date(), date_max),
    min_value=date_min, max_value=date_max)
if len(plage) != 2:
    st.stop()
debut, fin = [pd.Timestamp(d) for d in plage]

nom_variable = st.sidebar.selectbox("Variable a tracer",
                                    list(VARIABLES_AFFICHABLES.keys()))
variable = VARIABLES_AFFICHABLES[nom_variable]

seulement_hautes = st.sidebar.checkbox("Alertes de severite haute uniquement")

periode = df[(df["date"] >= debut) & (df["date"] <= fin + pd.Timedelta(hours=23))]
marche = periode[periode["en_marche"] == 1]
ep_periode = episodes[(episodes["date_debut"] <= fin + pd.Timedelta(hours=23))
                      & (episodes["date_fin"] >= debut)]
if seulement_hautes:
    ep_periode = ep_periode[ep_periode["severite"] == "haute"]

# ------------------------------------------------------------------
# Ligne de KPIs
# ------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Production", f"{periode['puissance_MW'].sum() / 1000:.1f} GWh")
c2.metric("Puissance moyenne",
          f"{marche['puissance_MW'].mean():.0f} MW" if len(marche) else "arret")
c3.metric("Disponibilite", f"{100 * periode['en_marche'].mean():.1f} %")
c4.metric("Rendement moyen",
          f"{marche['rendement_pct'].mean():.2f} %" if len(marche) else "arret")
c5.metric("Episodes d'alerte", len(ep_periode))

# ------------------------------------------------------------------
# Graphique principal : variable + anomalies en rouge
# ------------------------------------------------------------------
st.subheader(f"{nom_variable} : evolution et anomalies")

base = alt.Chart(periode).encode(x=alt.X("date:T", title=""))
courbe = base.mark_line(strokeWidth=1).encode(
    y=alt.Y(f"{variable}:Q", title=nom_variable,
            scale=alt.Scale(zero=False)))
points_anomalies = alt.Chart(periode[periode["anomalie"] == True]).mark_circle(
    color="red", size=25).encode(
    x="date:T", y=f"{variable}:Q",
    tooltip=["date:T", f"{variable}:Q", "severite:N", "diagnostic:N"])

st.altair_chart((courbe + points_anomalies).properties(height=350),
                use_container_width=True)
st.caption("Points rouges : heures marquees anormales par l'agent "
           "(couche statistique et/ou ML). Survoler un point pour le detail.")

# ------------------------------------------------------------------
# Tableau des alertes
# ------------------------------------------------------------------
st.subheader("Alertes sur la periode")
if ep_periode.empty:
    st.info("Aucune alerte sur la periode selectionnee.")
else:
    tableau = ep_periode[["date_debut", "date_fin", "duree_h", "variables",
                          "methode", "severite", "diagnostic"]].copy()
    tableau = tableau.sort_values("date_debut", ascending=False)
    st.dataframe(tableau, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------
# Rapports generes
# ------------------------------------------------------------------
st.subheader("Rapports generes")
if os.path.exists(DOSSIER_RAPPORTS):
    fichiers = sorted([f for f in os.listdir(DOSSIER_RAPPORTS)
                       if f.endswith(".txt")], reverse=True)
else:
    fichiers = []

if not fichiers:
    st.info("Aucun rapport disponible. En generer un depuis la page "
            "Pilotage du pipeline.")
else:
    choix = st.selectbox("Choisir un rapport", fichiers)
    with open(os.path.join(DOSSIER_RAPPORTS, choix), encoding="utf-8") as f:
        contenu = f.read()
    st.text_area("Contenu du rapport", contenu, height=400)
    col_txt, col_pdf = st.columns(2)
    col_txt.download_button("Telecharger (TXT)", contenu, file_name=choix)
    chemin_pdf = os.path.join(DOSSIER_RAPPORTS, choix.replace(".txt", ".pdf"))
    if os.path.exists(chemin_pdf):
        with open(chemin_pdf, "rb") as f:
            col_pdf.download_button("Telecharger (PDF)", f.read(),
                                    file_name=os.path.basename(chemin_pdf),
                                    mime="application/pdf")

# ------------------------------------------------------------------
# Maintenance (pour le contexte)
# ------------------------------------------------------------------
with st.expander("Historique de maintenance"):
    st.dataframe(maintenance.sort_values("date", ascending=False),
                 use_container_width=True, hide_index=True)
