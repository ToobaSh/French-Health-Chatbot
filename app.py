import os
import json
import numpy as np
import streamlit as st

from src.rag_pipeline import answer_question_extractive


# ----------------------
# Configuration Streamlit
# ----------------------
st.set_page_config(
    page_title="Chatbot santé (RAG)",
    page_icon="💬",
    layout="wide",
)

st.title("Chatbot d'information santé pour les patients (RAG)")
st.write(
    "Ce chatbot répond à des questions de santé générale en se basant uniquement "
    "sur des brochures patients (PDF/TXT) stockées dans le dossier `data/brochures`. "
    "Il n'utilise pas Internet et ne remplace pas l'avis d'un professionnel de santé."
)


# ----------------------
# Chemins et état de session
# ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BROCHURES_DIR = os.path.join(BASE_DIR, "data", "brochures")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
EMB_PATH = os.path.join(VECTOR_STORE_DIR, "embeddings.npy")
META_PATH = os.path.join(VECTOR_STORE_DIR, "metadata.json")

if "index_ready" not in st.session_state:
    st.session_state["index_ready"] = False
if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = None
if "metadata" not in st.session_state:
    st.session_state["metadata"] = None
if "chat_history" not in st.session_state:
    # chaque élément : {question, answer, sources}
    st.session_state["chat_history"] = []


# ----------------------
# Chargement du vector store pré-calculé
# ----------------------
def load_vector_store():
    if not os.path.exists(EMB_PATH) or not os.path.exists(META_PATH):
        st.error(
            "Vector store introuvable.\n\n"
            "Veuillez exécuter `build_vector_store.py` en local pour pré-calculer les embeddings "
            "et committer le dossier `vector_store/` dans le dépôt."
        )
        return False

    try:
        embeddings = np.load(EMB_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        st.error(f"Erreur lors du chargement du vector store : {e}")
        return False

    st.session_state["embeddings"] = embeddings
    st.session_state["metadata"] = metadata
    st.session_state["index_ready"] = True
    return True


if not st.session_state["index_ready"]:
    load_vector_store()


# ----------------------
# Sidebar : liste des brochures
# ----------------------
with st.sidebar:
    st.header("📂 Brochures chargées")

    if os.path.isdir(BROCHURES_DIR):
        files_in_dir = os.listdir(BROCHURES_DIR)
        brochure_files = [
            f for f in files_in_dir if f.lower().endswith((".pdf", ".txt"))
        ]
        if brochure_files:
            for f in brochure_files:
                st.write("• " + f)
        else:
            st.warning("Aucune brochure PDF/TXT trouvée dans `data/brochures`.")
    else:
        st.error("Le dossier `data/brochures` n'existe pas.")


# ----------------------
# Interface principale : chat
# ----------------------
st.markdown("## 💬 Posez une question de santé (en français)")

if not st.session_state["index_ready"]:
    st.warning(
        "L'index des documents n'est pas prêt.\n\n"
        "Le vector store est manquant ou n'a pas pu être chargé."
    )
else:
    # Afficher l'historique des échanges
    for turn in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            with st.expander("Sources", expanded=False):
                if not turn["sources"]:
                    st.write("Aucune source trouvée.")
                else:
                    for i, src in enumerate(turn["sources"], start=1):
                        display_name = os.path.basename(src["filename"])
                        st.markdown(f"**Source {i} : {display_name}**")
                        st.write(f"Score : `{src['score']:.3f}`")
                        st.write(f"Index du chunk : `{src['chunk_index']}`")
                        st.text(src["snippet"])
                        st.markdown("---")

    # Nouvelle question
    user_question = st.chat_input("Votre question...")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Recherche dans les brochures..."):
                result = answer_question_extractive(
                    query=user_question,
                    doc_embeddings=st.session_state["embeddings"],
                    metadata=st.session_state["metadata"],
                    k=3,
                    max_chunk_chars=600,
                )

            st.write(result["answer"])

            with st.expander("Sources", expanded=False):
                if not result["sources"]:
                    st.write("Aucune source trouvée.")
                else:
                    for i, src in enumerate(result["sources"], start=1):
                        display_name = os.path.basename(src["filename"])
                        st.markdown(f"**Source {i} : {display_name}**")
                        st.write(f"Score : `{src['score']:.3f}`")
                        st.write(f"Index du chunk : `{src['chunk_index']}`")
                        st.text(src["snippet"])
                        st.markdown("---")

        # Sauvegarder dans l'historique
        st.session_state["chat_history"].append(
            {
                "question": user_question,
                "answer": result["answer"],
                "sources": result["sources"],
            }
        )

st.markdown("---")
st.caption(
    "Ce chatbot fournit uniquement une information générale et ne remplace pas l'avis d'un professionnel de santé. "
    "En cas de doute ou de symptômes, consultez un médecin."
)
