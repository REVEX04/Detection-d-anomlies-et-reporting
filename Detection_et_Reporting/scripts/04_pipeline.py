
import argparse
import importlib.util
import os

import pandas as pd

DOSSIER_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.join(DOSSIER_SCRIPTS, "..")


def importer(nom_fichier, alias):
    """Nos modules commencent par un chiffre (01_, 03_...), donc un import
    classique ne marche pas : on passe par importlib."""
    chemin = os.path.join(DOSSIER_SCRIPTS, nom_fichier)
    spec = importlib.util.spec_from_file_location(alias, chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pipeline_complet(debut=None, fin=None):
    detection = importer("02_agent_detection.py", "agent_detection")
    reporting = importer("03_agent_reporting.py", "agent_reporting")
    dossier_data = os.path.join(RACINE, "data")

    # 1) chargement des donnees
    df = pd.read_csv(os.path.join(dossier_data, "dataset_taqa_synthetique.csv"),
                     parse_dates=["date"])
    print(f"[1/3] Donnees chargees : {len(df)} lignes")

    # 2) agent de detection
    df, episodes = detection.executer_agent(df)
    df.to_csv(os.path.join(dossier_data, "detection_horaire.csv"), index=False)
    episodes.to_csv(os.path.join(dossier_data, "anomalies_detectees.csv"), index=False)
    print(f"[2/3] Detection terminee : {len(episodes)} episodes d'alerte")

    # 3) agent de reporting
    rapport, kpis = reporting.generer_rapport(debut, fin)
    print(f"[3/3] Rapport genere pour la periode {kpis['periode']}")
    return df, episodes, rapport


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="Pipeline complet TAQA")
    parseur.add_argument("--debut", help="date de debut du rapport (AAAA-MM-JJ)")
    parseur.add_argument("--fin", help="date de fin du rapport (AAAA-MM-JJ)")
    args = parseur.parse_args()

    pipeline_complet(args.debut, args.fin)
    print("\nPipeline termine. Lancer le dashboard avec :")
    print("  python -m streamlit run dashboard\\app_streamlit.py")
