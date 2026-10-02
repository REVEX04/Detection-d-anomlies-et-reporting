

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Variables surveillees par la couche statistique
VARIABLES_ZSCORE = ["puissance_MW", "rendement_pct", "conso_charbon_t_h",
                    "temp_vapeur_C", "pression_bar", "vibrations_mm_s"]

# Variables combinees pour la couche ML
FEATURES_ML = ["puissance_MW", "rendement_pct", "conso_charbon_t_h",
               "temp_vapeur_C", "pression_bar", "vibrations_mm_s", "debit_eau_t_h"]

SEUIL_ZSCORE = 3.5
FENETRE = 24           # heures
SEUIL_VIB_ALARME = 4.5  # mm/s, seuil absolu type ISO 10816

# Couche ML : le score Isolation Forest est lisse sur 12h et on ne garde que
# le 4% le plus anormal (1% manquait les anomalies multivariees, les derives fortes monopolisant le quantile). Sans lissage, le modele flaggue des heures isolees
# un peu partout (bruit), alors que les vraies anomalies multivariees durent
# plusieurs heures d'affilee.
FENETRE_LISSAGE_ML = 12
QUANTILE_ML = 0.04

# ------------------------------------------------------------------
# Preparation : quelles heures sont analysables ?
# ------------------------------------------------------------------
def marquer_regime_stable(df, heures_rampe=6):
    """Exclut les heures d'arret et les rampes de redemarrage."""
    stable = df["en_marche"].values.astype(bool).copy()
    redemarrages = np.where((df["en_marche"].values[1:] == 1)
                            & (df["en_marche"].values[:-1] == 0))[0] + 1
    for r in redemarrages:
        stable[r:r + heures_rampe] = False
    df["regime_stable"] = stable
    return df


# ------------------------------------------------------------------
# Couche 1 : statistique
# ------------------------------------------------------------------
def detecter_zscore(df, colonne, fenetre=FENETRE, seuil=SEUIL_ZSCORE):
    """Z-score par rapport a la moyenne mobile. Une valeur est anormale si elle
    s'ecarte de plus de `seuil` ecarts-types mobiles."""
    serie = df[colonne].where(df["regime_stable"])
    moyenne_mobile = serie.rolling(fenetre, min_periods=12).mean()
    ecart_mobile = serie.rolling(fenetre, min_periods=12).std()
    z = (serie - moyenne_mobile) / ecart_mobile
    df["z_" + colonne] = z.round(2)
    df["anom_stat_" + colonne] = (z.abs() > seuil) & df["regime_stable"]
    return df


def detecter_seuil_vibrations(df, seuil=SEUIL_VIB_ALARME):
    """Seuil absolu sur les vibrations : indispensable pour les derives lentes,
    que le z-score glissant finit par integrer dans sa moyenne."""
    df["anom_stat_vibrations_mm_s"] = df["anom_stat_vibrations_mm_s"] | (
        (df["vibrations_mm_s"] > seuil) & df["regime_stable"])
    return df


def couche_statistique(df):
    for col in VARIABLES_ZSCORE:
        df = detecter_zscore(df, col)
    df = detecter_seuil_vibrations(df)
    cols_stat = ["anom_stat_" + c for c in VARIABLES_ZSCORE]
    df["anomalie_stat"] = df[cols_stat].any(axis=1)
    return df


# ------------------------------------------------------------------
# Couche 2 : Machine Learning (Isolation Forest)
# ------------------------------------------------------------------
def couche_ml(df, fenetre_lissage=FENETRE_LISSAGE_ML, quantile=QUANTILE_ML):
    X = df.loc[df["regime_stable"], FEATURES_ML].dropna()
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    modele = IsolationForest(n_estimators=200, contamination="auto",
                             random_state=42)
    modele.fit(X_std)
    score = modele.decision_function(X_std)  # plus c'est bas, plus c'est anormal

    df["score_ml"] = np.nan
    df.loc[X.index, "score_ml"] = np.round(score, 3)
    score_lisse = df["score_ml"].rolling(fenetre_lissage, min_periods=8).mean()
    df["score_ml_lisse"] = score_lisse.round(3)

    seuil_ml = score_lisse.quantile(quantile)
    df["anomalie_ml"] = (score_lisse < seuil_ml) & df["regime_stable"]
    return df


# ------------------------------------------------------------------
# Diagnostic et severite
# ------------------------------------------------------------------
def diagnostiquer(row):
    if row["anom_stat_vibrations_mm_s"]:
        return "Montee des vibrations turbine : possible balourd ou usure palier, risque d'arret force"
    if row["anom_stat_temp_vapeur_C"]:
        return "Ecart temperature vapeur : verifier la regulation de desurchauffe"
    if row["anom_stat_pression_bar"]:
        return "Ecart de pression vapeur : controler regulation et soupapes"
    if row["anom_stat_rendement_pct"]:
        return "Baisse de rendement : possible encrassement chaudiere ou derive combustion"
    if row["anom_stat_puissance_MW"]:
        return "Ecart de puissance significatif : possible limitation ou defaut auxiliaire (broyeur, pompe)"
    if row["anom_stat_conso_charbon_t_h"]:
        return "Consommation charbon anormale : verifier qualite combustible et dosage"
    if row["anomalie_ml"]:
        return "Combinaison inhabituelle de parametres (anomalie multivariee) : inspection recommandee"
    return ""


def evaluer_severite(row):
    z_cols = ["z_" + c for c in VARIABLES_ZSCORE]
    z_max = max((abs(row[c]) for c in z_cols if pd.notna(row[c])), default=0)
    if row.get("vibrations_mm_s", 0) and row["vibrations_mm_s"] > SEUIL_VIB_ALARME:
        return "haute"
    if z_max >= 5:
        return "haute"
    if row["anomalie_stat"] and row["anomalie_ml"]:
        return "haute"
    if row["anomalie_stat"]:
        return "moyenne"
    return "faible"  # detection ML seule : signal subtil


# ------------------------------------------------------------------
# Regroupement en episodes (une alerte par evenement)
# ------------------------------------------------------------------
def construire_episodes(df, ecart_max_h=3):
    """Regroupe les heures anormales consecutives (tolerance de `ecart_max_h`
    heures de trou) en episodes : c'est plus lisible qu'une alerte par heure."""
    anom = df[df["anomalie"]].copy()
    if anom.empty:
        return pd.DataFrame()

    anom["groupe"] = (anom["date"].diff() > pd.Timedelta(hours=ecart_max_h)).cumsum()
    episodes = []
    ordre_sev = {"faible": 0, "moyenne": 1, "haute": 2}
    for _, g in anom.groupby("groupe"):
        variables = [c for c in VARIABLES_ZSCORE
                     if g["anom_stat_" + c].any()]
        methodes = []
        if g["anomalie_stat"].any():
            methodes.append("statistique")
        if g["anomalie_ml"].any():
            methodes.append("ML")
        diagnostics = g.loc[g["diagnostic"] != "", "diagnostic"]
        episodes.append({
            "date_debut": g["date"].iloc[0],
            "date_fin": g["date"].iloc[-1],
            "duree_h": len(g),
            "variables": "|".join(variables) if variables else "multivariable",
            "methode": "+".join(methodes),
            "severite": g["severite"].map(ordre_sev).max(),
            "diagnostic": diagnostics.mode().iloc[0] if len(diagnostics) else
                          "Combinaison inhabituelle de parametres (anomalie multivariee) : inspection recommandee",
            "score_ml_min": g["score_ml"].min(),
        })
    ep = pd.DataFrame(episodes)
    ep["severite"] = ep["severite"].map({0: "faible", 1: "moyenne", 2: "haute"})
    return ep


# ------------------------------------------------------------------
# Evaluation contre la verite terrain
# ------------------------------------------------------------------
def evaluer(episodes, chemin_journal, marge_h=6):
    """Une anomalie injectee est consideree comme detectee si au moins un
    episode chevauche sa fenetre (avec une petite marge)."""
    journal = pd.read_csv(chemin_journal)
    # formats de dates legerement differents selon les anomalies, on force la conversion
    journal["date_debut"] = pd.to_datetime(journal["date_debut"], format="mixed")
    journal["date_fin"] = pd.to_datetime(journal["date_fin"], format="mixed")
    marge = pd.Timedelta(hours=marge_h)
    resultats = []
    episodes_utiles = set()
    for _, a in journal.iterrows():
        chevauche = episodes[(episodes["date_debut"] <= a["date_fin"] + marge)
                             & (episodes["date_fin"] >= a["date_debut"] - marge)]
        detectee = len(chevauche) > 0
        episodes_utiles.update(chevauche.index)
        resultats.append({
            "id": a["id"], "type_anomalie": a["type_anomalie"],
            "variables": a["variables_concernees"],
            "date_debut": a["date_debut"],
            "detectee": "oui" if detectee else "non",
            "methode": "+".join(sorted(set(
                m for mm in chevauche["methode"] for m in mm.split("+")))) if detectee else "",
        })
    evaluation = pd.DataFrame(resultats)
    fausses_alertes = len(episodes) - len(episodes_utiles)
    return evaluation, fausses_alertes


# ------------------------------------------------------------------
# Agent complet
# ------------------------------------------------------------------
def executer_agent(df):
    """Enchaine les deux couches, le diagnostic et le regroupement."""
    df = marquer_regime_stable(df)
    df = couche_statistique(df)
    df = couche_ml(df)
    df["anomalie"] = df["anomalie_stat"] | df["anomalie_ml"]
    df["diagnostic"] = df.apply(diagnostiquer, axis=1)
    df["severite"] = df.apply(evaluer_severite, axis=1)
    df.loc[~df["anomalie"], "severite"] = ""
    episodes = construire_episodes(df)
    return df, episodes


if __name__ == "__main__":
    dossier = os.path.join(os.path.dirname(__file__), "..", "data")
    df = pd.read_csv(os.path.join(dossier, "dataset_taqa_synthetique.csv"),
                     parse_dates=["date"])

    df, episodes = executer_agent(df)

    print(f"Heures analysees (regime stable) : {df['regime_stable'].sum()}")
    print(f"Heures anormales : {df['anomalie'].sum()} "
          f"(stat : {df['anomalie_stat'].sum()}, ML : {df['anomalie_ml'].sum()})")
    print(f"Episodes d'alerte : {len(episodes)}")

    evaluation, fausses_alertes = evaluer(
        episodes, os.path.join(dossier, "anomalies_injectees.csv"))
    nb_ok = (evaluation["detectee"] == "oui").sum()
    print(f"\nTaux de detection : {nb_ok}/{len(evaluation)} anomalies injectees retrouvees")
    print(f"Episodes hors verite terrain (a trier) : {fausses_alertes}")
    print("\n" + evaluation.to_string(index=False))

    # exports pour le dashboard et le reporting
    df.to_csv(os.path.join(dossier, "detection_horaire.csv"), index=False)
    episodes.to_csv(os.path.join(dossier, "anomalies_detectees.csv"), index=False)
    evaluation.to_csv(os.path.join(dossier, "evaluation_detection.csv"), index=False)
    print("\nFichiers ecrits dans data/ : detection_horaire.csv, "
          "anomalies_detectees.csv, evaluation_detection.csv")
