from typing import List, Dict, Any
import numpy as np
import os

from .retriever import retrieve_top_k


# ----------------------
# Listes pour nettoyage
# ----------------------

WEIRD_CHARS = [
    "", "", "", "", "", "", "", "",
    "", "", "", "", ""
]

UNWANTED_PATTERNS = [
    "Lire aussi",
    "Cet article vous a-t-il été utile",
    "Assuré Entreprise Professionnel de santé",
    "Sources",
    "Sites utiles",
    "Oui Non",
    "Copier le lien",
]

BIBLIO_PATTERNS = [
    "santé publique france",
    "consulté le",
    "site internet",
    "saint-maurice",
    "document de référence",
    "pdf ,",
]

MONTHS = [
    "janvier", "février", "fevrier", "mars", "avril", "mai", "juin",
    "juillet", "août", "aout", "septembre", "octobre", "novembre", "décembre", "decembre"
]


# ----------------------
# Helpers de nettoyage
# ----------------------

def _clean_text(text: str) -> str:
    """
    Nettoie le texte brut issu des PDF :
    - supprime les retours à la ligne bizarres
    - supprime les caractères spéciaux
    - réduit les espaces multiples
    """
    if not text:
        return ""

    # Retours à la ligne -> espace
    text = text.replace("\r", " ").replace("\n", " ")

    # Caractères étranges des PDF
    for ch in WEIRD_CHARS:
        text = text.replace(ch, " ")

    # Espaces multiples
    while "  " in text:
        text = text.replace("  ", " ")

    return text.strip()


def _split_sentences(text: str) -> List[str]:
    """
    Découpe grossièrement le texte en phrases.
    (Découpage sur le point, suffisant pour des brochures simples.)
    """
    text = _clean_text(text)
    parts = text.split(".")
    sentences = []
    for p in parts:
        s = p.strip()
        if len(s) > 0:
            sentences.append(s)
    return sentences


def _filter_sentences(sentences: List[str]) -> List[str]:
    """
    Retire les phrases parasites (menus, boutons, bibliographie, entêtes).
    Garde uniquement les phrases informatives.
    """
    kept: List[str] = []
    for s in sentences:
        s_clean = s.strip()
        s_lower = s_clean.lower()

        # Très courte → souvent du bruit
        if len(s_clean) < 30:
            continue

        # Menus / boutons / phrases non médicales
        if any(p.lower() in s_lower for p in UNWANTED_PATTERNS):
            continue

        # Phrases bibliographiques
        if any(p in s_lower for p in BIBLIO_PATTERNS):
            continue

        # Commence par un mois → souvent entête ou référence
        if any(s_lower.startswith(m) for m in MONTHS):
            continue

        kept.append(s_clean)

    return kept


def _summarize_snippet(text: str, max_sentences: int = 3) -> str:
    """
    À partir d'un chunk brut, on :
    - nettoie
    - découpe en phrases
    - filtre les phrases parasites
    - garde les 2-3 premières phrases informatives
    """
    sentences = _split_sentences(text)
    sentences = _filter_sentences(sentences)

    if not sentences:
        return ""

    kept = sentences[:max_sentences]
    summary = ". ".join(kept).strip()
    if not summary.endswith("."):
        summary += "."
    return summary


def _merge_snippets(snippets: List[str], max_chars: int = 900) -> str:
    """
    Fusionne plusieurs petits résumés en un bloc cohérent,
    sans dépasser une certaine longueur.
    """
    merged: List[str] = []
    current_len = 0

    for s in snippets:
        s = s.strip()
        if not s:
            continue
        if current_len + len(s) + 1 > max_chars:
            break
        merged.append(s)
        current_len += len(s) + 1

    merged_text = " ".join(merged).strip()
    return merged_text


# ----------------------
# Détection du sujet / maladie
# ----------------------

def _get_topic_keywords(query: str) -> List[str]:
    """
    Déduit le ou les sujets probables à partir de la question utilisateur
    et retourne des mots-clés qui doivent apparaître dans le nom de fichier.
    Ceci permet d’éviter de mélanger plusieurs maladies dans la même réponse.
    """
    q = query.lower()

    # ORL / pédiatrie / infectieux / chroniques
    if "otite" in q:
        return ["otite"]
    if "rhinopharyng" in q:
        return ["rhinopharyngite"]
    if "angine" in q:
        return ["angine"]
    if "fièvre" in q or "fievre" in q:
        return ["fievre"]
    if "gastro" in q:
        return ["gastro"]
    if "bronchiolite" in q:
        return ["bronchiolite"]
    if "hypertension" in q or "tension" in q:
        return ["hypertension"]
    if "diabète" in q or "diabete" in q:
        return ["diabete"]
    if "migraine" in q:
        return ["migraine"]
    if "grippe" in q:
        return ["grippe"]
    if "covid" in q:
        return ["covid"]
    if "asthme" in q:
        return ["asthme"]
    if "allerg" in q:
        return ["allergie", "allergies"]
    if "cholestérol" in q or "cholesterol" in q:
        return ["cholesterol"]

    # Aucun sujet clairement détecté
    return []


def _filter_results_by_topic(results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Si on détecte un sujet (ex : diabète), on garde en priorité
    les chunks dont le filename contient ce sujet.
    Si le filtrage supprime tout, on revient aux résultats d'origine.
    """
    topic_keywords = _get_topic_keywords(query)
    if not topic_keywords:
        return results

    filtered = []
    for r in results:
        filename = r.get("filename", "")
        filename_lower = filename.lower()
        if any(kw in filename_lower for kw in topic_keywords):
            filtered.append(r)

    # Si aucun résultat ne correspond aux mots-clés, on garde les originaux
    return filtered if filtered else results


# ----------------------
# Pipeline RAG EXTRACTIF
# ----------------------

def answer_question_extractive(
    query: str,
    doc_embeddings: np.ndarray,
    metadata: List[Dict[str, Any]],
    k: int = 3,
    max_chunk_chars: int = 800,
) -> Dict[str, Any]:
    """
    Pipeline RAG EXTRACTIF (sans LLM) :

    1) récupère les k chunks les plus proches de la question
    2) filtre les résultats pour éviter de mélanger plusieurs maladies
    3) nettoie et résume chaque chunk
    4) fusionne les extraits en une seule réponse lisible
    5) renvoie aussi la liste des sources pour l'interface Streamlit

    Retourne :
        {
            "question": str,
            "answer": str,
            "sources": [
                {
                    "filename": str,
                    "score": float,
                    "chunk_index": int,
                    "snippet": str,
                },
                ...
            ]
        }
    """

    # 1) Récupération des meilleurs chunks
    results = retrieve_top_k(
        query=query,
        doc_embeddings=doc_embeddings,
        metadata=metadata,
        k=k,
    )

    if not results:
        return {
            "question": query,
            "answer": (
                "Je n’ai trouvé aucune information pertinente sur ce sujet dans les documents chargés. "
                "Merci de vérifier que les brochures PDF contiennent bien des informations sur cette question."
            ),
            "sources": [],
        }

    # 2) Filtrage par sujet / maladie (pour ne pas mélanger diabète, fièvre, etc.)
    results = _filter_results_by_topic(results, query)

    # 3) Traitement des extraits
    snippets_for_answer: List[str] = []
    sources_list: List[Dict[str, Any]] = []

    for r in results:
        raw_text = r.get("text", "")[:max_chunk_chars]
        summarized = _summarize_snippet(raw_text, max_sentences=3)

        if not summarized:
            # Si le résumé est vide, on tente au moins un nettoyage simple
            summarized = _clean_text(raw_text)

        filename = os.path.basename(r.get("filename", "document inconnu"))

        if summarized:
            snippets_for_answer.append(summarized)

        sources_list.append(
            {
                "filename": filename,
                "score": float(r.get("score", 0.0)),
                "chunk_index": int(r.get("chunk_index", -1)),
                "snippet": summarized,
            }
        )

    # 4) Fusionner tous les petits résumés en un seul texte
    combined_text = _merge_snippets(snippets_for_answer, max_chars=900)

    if not combined_text:
        combined_text = (
            "Les documents contiennent des informations, mais elles n’ont pas pu être "
            "reformulées correctement. Merci de reformuler votre question ou de consulter "
            "un professionnel de santé."
        )

    # 5) Construction de la réponse finale
    answer_text = (
        "Voici une réponse basée sur les brochures disponibles concernant votre question :\n\n"
        f"{combined_text}\n\n"
        "Ce résumé est directement élaboré à partir des brochures. "
        "Il fournit une information générale et ne remplace **en aucun cas** l’avis d’un professionnel de santé."
    )

    return {
        "question": query,
        "answer": answer_text,
        "sources": sources_list,
    }
