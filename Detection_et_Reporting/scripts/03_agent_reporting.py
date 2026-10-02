
import argparse
import json
import os
import urllib.request

import pandas as pd

RACINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODELE_GEMINI = "gemini-2.5-flash"


# ------------------------------------------------------------------
# Lecture de la cle API (fichier .env a la racine du projet)
# ------------------------------------------------------------------
def charger_cle_api():
    """Petit lecteur de .env maison pour eviter une dependance de plus.
    Retourne None si pas de cle : on basculera en mode template."""
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    chemin = os.path.join(RACINE, ".env")
    if os.path.exists(chemin):
        with open(chemin, encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if ligne.startswith("GEMINI_API_KEY=") and len(ligne) > 15:
                    return ligne.split("=", 1)[1].strip()
    return None


# ------------------------------------------------------------------
# Calcul des KPIs de la periode
# ------------------------------------------------------------------
def calculer_kpis(df, episodes, debut, fin):
    """Resume chiffre de la periode : production, disponibilite, rendement,
    anomalies. C'est ce qui alimente le prompt (ou le template)."""
    periode = df[(df["date"] >= debut) & (df["date"] <= fin + " 23:59")]
    if periode.empty:
        raise ValueError(f"Aucune donnee entre {debut} et {fin}")
    marche = periode[periode["en_marche"] == 1]

    ep = episodes[(episodes["date_debut"] <= fin + " 23:59")
                  & (episodes["date_fin"] >= debut)].copy()

    kpis = {
        "periode": f"{debut} au {fin}",
        "heures_totales": len(periode),
        "production_MWh": int(periode["puissance_MW"].sum()),
        "puissance_moyenne_MW": round(marche["puissance_MW"].mean(), 1) if len(marche) else 0,
        "disponibilite_pct": round(100 * periode["en_marche"].mean(), 1),
        "rendement_moyen_pct": round(marche["rendement_pct"].mean(), 2) if len(marche) else 0,
        "conso_charbon_t": int(periode["conso_charbon_t_h"].sum()),
        "nb_anomalies": len(ep),
        "nb_anomalies_hautes": int((ep["severite"] == "haute").sum()),
    }

    anomalies = []
    for _, e in ep.iterrows():
        anomalies.append({
            "debut": str(e["date_debut"]), "fin": str(e["date_fin"]),
            "duree_h": int(e["duree_h"]), "variables": e["variables"],
            "methode": e["methode"], "severite": e["severite"],
            "diagnostic": e["diagnostic"],
        })
    return kpis, anomalies


# ------------------------------------------------------------------
# Prompt structure (commun aux differents LLM)
# ------------------------------------------------------------------
def construire_prompt(kpis, anomalies):
    lignes_anomalies = "\n".join(
        f"- du {a['debut']} au {a['fin']} ({a['duree_h']}h), variables : {a['variables']}, "
        f"severite {a['severite']}, methode {a['methode']}. Diagnostic : {a['diagnostic']}"
        for a in anomalies) or "- aucune anomalie detectee sur la periode"

    return f"""Tu es un ingenieur d'exploitation dans une centrale thermique a charbon.
Redige un rapport de suivi de production hebdomadaire, clair et professionnel,
en francais, a partir des donnees ci-dessous. N'invente aucun chiffre.

Donnees de la periode ({kpis['periode']}) :
- Production totale : {kpis['production_MWh']} MWh
- Puissance moyenne en marche : {kpis['puissance_moyenne_MW']} MW
- Disponibilite : {kpis['disponibilite_pct']} %
- Rendement thermique moyen : {kpis['rendement_moyen_pct']} %
- Consommation de charbon : {kpis['conso_charbon_t']} t
- Anomalies detectees : {kpis['nb_anomalies']} (dont {kpis['nb_anomalies_hautes']} de severite haute)

Detail des anomalies :
{lignes_anomalies}

Structure imposee du rapport :
1. Resume de la periode (3-4 phrases)
2. Indicateurs cles (liste avec les chiffres)
3. Anomalies detectees (dates, gravite, diagnostic probable)
4. Recommandations (actions concretes, par priorite)

Reste factuel et concis (300 mots maximum). Pas de tiret long dans le texte.
Ne signe pas le rapport : pas de "Fait a...", pas de nom, pas de champ a
completer entre crochets. Le rapport se termine apres les recommandations."""


# ------------------------------------------------------------------
# Mode 1 : generation via l'API Gemini
# ------------------------------------------------------------------
def generer_via_gemini(prompt, cle_api, modele=MODELE_GEMINI):
    """Appel REST direct (pas besoin du SDK Google, urllib suffit)."""
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{modele}:generateContent?key={cle_api}")
    corps = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    requete = urllib.request.Request(
        url, data=corps, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(requete, timeout=60) as reponse:
        donnees = json.loads(reponse.read())
    return donnees["candidates"][0]["content"]["parts"][0]["text"]


# ------------------------------------------------------------------
# Mode 2 : template local (secours, hors ligne)
# ------------------------------------------------------------------
RECOMMANDATIONS_PAR_TYPE = {
    "vibrations_mm_s": "Programmer une inspection de la ligne d'arbre turbine "
                       "(equilibrage, paliers) avant aggravation.",
    "temp_vapeur_C": "Verifier la boucle de regulation de desurchauffe et "
                     "l'etalonnage des sondes de temperature.",
    "pression_bar": "Controler les soupapes et la regulation de pression du "
                    "circuit vapeur.",
    "rendement_pct": "Planifier un nettoyage chaudiere (ramonage) et verifier "
                     "la qualite du combustible.",
    "puissance_MW": "Analyser les auxiliaires (broyeurs, pompes) pour "
                    "identifier la cause de la limitation de charge.",
    "conso_charbon_t_h": "Verifier le dosage combustible et la qualite du "
                         "charbon receptionne.",
    "multivariable": "Realiser une inspection generale : la combinaison de "
                     "parametres detectee par le modele ML ne correspond pas "
                     "a un fonctionnement normal.",
}


def generer_via_template(kpis, anomalies):
    """Rapport redige par gabarit : moins souple qu'un LLM mais deterministe,
    et suffisant pour une demonstration hors ligne."""
    sev_txt = (f", dont {kpis['nb_anomalies_hautes']} de severite haute"
               if kpis["nb_anomalies_hautes"] else "")
    if kpis["nb_anomalies"] == 0:
        phrase_anom = ("Aucune anomalie n'a ete detectee : le fonctionnement "
                       "est reste dans les plages normales.")
    else:
        phrase_anom = (f"L'agent de detection a releve {kpis['nb_anomalies']} "
                       f"episode(s) anormal(aux){sev_txt}, detailles ci-dessous.")

    dispo = kpis["disponibilite_pct"]
    appreciation = ("La tranche a ete disponible toute la periode."
                    if dispo >= 99.5 else
                    f"La disponibilite s'etablit a {dispo} %, penalisee par des "
                    "heures d'arret sur la periode.")

    texte = [
        "RAPPORT DE SUIVI DE PRODUCTION",
        f"Periode : {kpis['periode']}",
        "",
        "1. RESUME DE LA PERIODE",
        f"La production s'eleve a {kpis['production_MWh']} MWh pour une "
        f"puissance moyenne de {kpis['puissance_moyenne_MW']} MW en marche. "
        f"{appreciation} Le rendement thermique moyen ressort a "
        f"{kpis['rendement_moyen_pct']} %. {phrase_anom}",
        "",
        "2. INDICATEURS CLES",
        f"- Production totale : {kpis['production_MWh']} MWh",
        f"- Puissance moyenne (en marche) : {kpis['puissance_moyenne_MW']} MW",
        f"- Disponibilite : {kpis['disponibilite_pct']} %",
        f"- Rendement thermique moyen : {kpis['rendement_moyen_pct']} %",
        f"- Consommation charbon : {kpis['conso_charbon_t']} t",
        f"- Episodes d'anomalie : {kpis['nb_anomalies']}",
        "",
        "3. ANOMALIES DETECTEES",
    ]
    if anomalies:
        for a in anomalies:
            texte.append(f"- Du {a['debut']} au {a['fin']} ({a['duree_h']}h), "
                         f"severite {a['severite']}, methode {a['methode']}.")
            texte.append(f"  Diagnostic : {a['diagnostic']}")
    else:
        texte.append("- Aucune anomalie sur la periode.")

    texte += ["", "4. RECOMMANDATIONS"]
    if anomalies:
        deja = set()
        for a in anomalies:
            for var in a["variables"].split("|"):
                reco = RECOMMANDATIONS_PAR_TYPE.get(var)
                if reco and var not in deja:
                    texte.append(f"- {reco}")
                    deja.add(var)
        if not deja:
            texte.append(f"- {RECOMMANDATIONS_PAR_TYPE['multivariable']}")
    else:
        texte.append("- Poursuivre la surveillance selon le plan en vigueur.")
    texte.append("")
    texte.append("Rapport genere automatiquement (mode template, sans LLM).")
    return "\n".join(texte)



# ------------------------------------------------------------------
# Export PDF du rapport (le LLM renvoie du texte, le PDF est fabrique ici)
# ------------------------------------------------------------------
def sauvegarder_pdf(texte, chemin):
    """Convertit le rapport texte en PDF simple avec reportlab.
    Gere le peu de mise en forme presente : titres, sections, puces,
    gras markdown (**...**) renvoye par le LLM."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    except ImportError:
        print("reportlab n'est pas installe (pip install reportlab) : PDF ignore")
        return False

    import html
    import re

    styles = getSampleStyleSheet()
    style_titre = ParagraphStyle("titre", parent=styles["Heading1"],
                                 fontSize=14, spaceAfter=12)
    style_section = ParagraphStyle("section", parent=styles["Heading2"],
                                   fontSize=12, spaceBefore=10, spaceAfter=6)
    style_texte = ParagraphStyle("texte", parent=styles["BodyText"],
                                 fontSize=10.5, leading=15)
    style_puce = ParagraphStyle("puce", parent=style_texte,
                                leftIndent=16, bulletIndent=6)

    doc = SimpleDocTemplate(chemin, pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm)
    elements = []
    titre_pose = False
    for brute in texte.split("\n"):
        ligne = brute.strip()
        if not ligne:
            elements.append(Spacer(1, 6))
            continue
        ligne = html.escape(ligne)
        ligne = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", ligne)  # gras markdown

        est_section = ((ligne.startswith("<b>") and ligne.endswith("</b>"))
                       or re.match(r"^\d+\.\s", ligne) and ligne == ligne.upper())
        if not titre_pose:
            elements.append(Paragraph(ligne, style_titre))
            titre_pose = True
        elif est_section:
            elements.append(Paragraph(ligne, style_section))
        elif ligne.startswith(("* ", "- ")):
            elements.append(Paragraph(ligne[2:], style_puce, bulletText="\u2022"))
        else:
            elements.append(Paragraph(ligne, style_texte))
    doc.build(elements)
    return True


# ------------------------------------------------------------------
# Agent complet
# ------------------------------------------------------------------
def generer_rapport(debut=None, fin=None):
    dossier_data = os.path.join(RACINE, "data")
    df = pd.read_csv(os.path.join(dossier_data, "detection_horaire.csv"),
                     parse_dates=["date"])
    episodes = pd.read_csv(os.path.join(dossier_data, "anomalies_detectees.csv"))

    # par defaut : les 7 derniers jours complets du dataset
    if fin is None:
        fin = str(df["date"].max().date())
    if debut is None:
        debut = str((pd.Timestamp(fin) - pd.Timedelta(days=6)).date())

    kpis, anomalies = calculer_kpis(df, episodes, debut, fin)

    cle = charger_cle_api()
    if cle:
        print(f"Mode : API Gemini ({MODELE_GEMINI})")
        prompt = construire_prompt(kpis, anomalies)
        try:
            rapport = generer_via_gemini(prompt, cle)
        except Exception as e:
            print(f"Echec de l'appel API ({e}), bascule en mode template.")
            rapport = generer_via_template(kpis, anomalies)
    else:
        print("Pas de cle GEMINI_API_KEY trouvee : mode template (hors ligne).")
        rapport = generer_via_template(kpis, anomalies)

    dossier_rapports = os.path.join(RACINE, "rapports")
    os.makedirs(dossier_rapports, exist_ok=True)
    chemin = os.path.join(dossier_rapports, f"rapport_{debut}_{fin}.txt")
    with open(chemin, "w", encoding="utf-8") as f:
        f.write(rapport)
    print(f"Rapport ecrit : {chemin}")

    chemin_pdf = chemin.replace(".txt", ".pdf")
    if sauvegarder_pdf(rapport, chemin_pdf):
        print(f"Version PDF   : {chemin_pdf}")
    print()
    return rapport, kpis


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="Agent de reporting automatique")
    parseur.add_argument("--debut", help="date de debut (AAAA-MM-JJ)")
    parseur.add_argument("--fin", help="date de fin (AAAA-MM-JJ)")
    args = parseur.parse_args()

    rapport, _ = generer_rapport(args.debut, args.fin)
    print(rapport)
    print("\nPour consulter les rapports sur le tableau de bord :")
    print("  python -m streamlit run dashboard\\app_streamlit.py")
