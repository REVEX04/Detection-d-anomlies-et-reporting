# Agent de détection d'anomalies et de reporting automatique

Projet de stage d'étude réalisé au sein de TAQA Morocco (département DPST).
École des Sciences de l'Information (ESI), 2ACI ICSD.

## Objectif

Automatiser la surveillance des indicateurs de production d'une centrale
thermique et la rédaction des rapports de suivi, afin de détecter plus tôt
les dérives et de réduire la charge de reporting manuel.

Le système combine deux agents :

1. **Agent de détection d'anomalies** : surveillance des indicateurs horaires
   en deux couches complémentaires, une couche statistique (z-score sur
   moyenne mobile + seuil absolu vibrations) et une couche Machine Learning
   (Isolation Forest sur 7 variables combinées, score lissé sur 12 h).
2. **Agent de reporting** : calcul des KPIs de la période et rédaction
   automatique d'un rapport en langage naturel via l'API Gemini (avec un
   mode template hors ligne en secours).

Les résultats sont restitués sur un tableau de bord Streamlit.

## Note sur les données

Les données réelles de TAQA Morocco étant confidentielles, l'étude repose sur
un **dataset synthétique généré à l'aide de l'IA** : 18 mois de données
horaires (13 104 lignes, 15 variables) reproduisant le comportement réaliste
d'une tranche charbon d'environ 350 MW (production, paramètres techniques,
maintenance, contexte). Le dataset est fourni avec le journal des 11 anomalies
qu'il contient (`data/anomalies_injectees.csv`), utilisé comme vérité terrain
pour évaluer l'agent de détection. Aucune donnée réelle n'est utilisée.

## Structure du projet

```
Detection_et_Reporting/
├── data/                          # dataset + résultats des agents (CSV)
├── notebooks/
│   └── 01_analyse_exploratoire.ipynb   # EDA complète
├── scripts/
│   ├── 02_agent_detection.py      # Agent 1 : détection (stat + ML)
│   ├── 03_agent_reporting.py      # Agent 2 : reporting (Gemini / template)
│   └── 04_pipeline.py             # pipeline complet
├── dashboard/
│   └── app_streamlit.py           # tableau de bord
├── figures/                       # graphiques exportés pour le rapport
├── rapports/                      # rapports générés (TXT + PDF)
└── requirements.txt
```

Le tableau de bord Power BI se trouve dans le dossier voisin `../Dashboard PowerBI/` :

- [Tableau de bord Power BI](../Dashboard%20PowerBI/DashBoard%20PowerBI.pbix)
- [Guide Power BI](../Dashboard%20PowerBI/guide_power_bi.md)
- [Thème Power BI](../Dashboard%20PowerBI/theme_taqa.json)

## Installation

Ouvrir un terminal dans le dossier `code source/Detection_et_Reporting` après extraction du ZIP.
Toutes les commandes ci-dessous se lancent depuis ce dossier, qui contient `requirements.txt`.

Sur le PC de l'auteur, le dossier se trouve ici :

```powershell
Set-Location -LiteralPath "C:\Users\PC\Desktop\rapport de stage+code source\code source\Detection_et_Reporting"
```

Sur un autre ordinateur, adapter ce chemin à l'emplacement du dossier extrait.

```
python -m pip install -r requirements.txt
```

Pour le reporting via LLM : créer une clé gratuite sur aistudio.google.com,
puis créer dans `Detection_et_Reporting/` un fichier `.env` contenant la ligne
`GEMINI_API_KEY=votre_cle`. Sans clé, l'agent bascule automatiquement en mode
template (hors ligne).

## Utilisation

```
# pipeline complet : détection + rapport sur les 7 derniers jours
python scripts/04_pipeline.py

# rapport sur une période précise
python scripts/03_agent_reporting.py --debut 2026-05-25 --fin 2026-05-31

# tableau de bord (2 pages : Consultation et Pilotage du pipeline)
python -m streamlit run dashboard\app_streamlit.py
```

La page **Pilotage du pipeline** du tableau de bord permet de relancer l'agent
de détection et de générer un rapport sur une période choisie directement
depuis l'interface, sans passer par la ligne de commande.

## Analyse exploratoire

Ouvrir `notebooks/01_analyse_exploratoire.ipynb` dans un outil compatible Jupyter.
Le dossier de travail du notebook doit être `Detection_et_Reporting/notebooks/`.
Ses chemins `../data/` et `../figures/` désignent les données et les figures du projet.

## Résultats

- **11/11 anomalies** de la vérité terrain retrouvées par l'agent, pour une
  seule fausse alerte sur 18 mois (24 épisodes d'alerte au total).
- Les 2 anomalies **multivariées** (combinaisons anormales sous les seuils
  univariés) ne sont détectées que par la couche Isolation Forest, ce qui
  démontre sa valeur ajoutée par rapport aux seuils statistiques seuls.
- Rapport hebdomadaire généré en quelques secondes, structuré en résumé,
  indicateurs, anomalies et recommandations, disponible en texte et en PDF.
