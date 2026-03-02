import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# UI
# -----------------------------
st.title("🎓 AI Kursuse Nõustaja - Samm 5")
st.caption("RAG süsteem koos metaandmete eel-filtreerimisega (Variant B).")

# -----------------------------
# Cache: model + data + embeddings
# -----------------------------
@st.cache_resource
def get_models():
    embedder = SentenceTransformer("BAAI/bge-m3")
    df = pd.read_csv("Data/puhtad_andmed.csv")
    embeddings_df = pd.read_pickle("Data/puhtad_andmed_embeddings.pkl")

    # merge once, keep in cache
    merged_df = pd.merge(df, embeddings_df, on="unique_ID")

    # normalize types for filters
    if "eap" in merged_df.columns:
        merged_df["eap"] = pd.to_numeric(merged_df["eap"], errors="coerce")
    for c in ["semester", "keel", "hindamisviis", "linn", "oppeaste", "veebiope"]:
        if c in merged_df.columns:
            merged_df[c] = merged_df[c].fillna("—").astype(str)

    return embedder, merged_df

embedder, merged_df = get_models()

# -----------------------------
# Sidebar: API + Filters (Variant B)
# -----------------------------
with st.sidebar:
    api_key = st.text_input("OpenRouter API Key", type="password")

    st.subheader("Metaandmete filtrid (valikulised)")
    st.caption("Jäta tühjaks, kui ei taha filtreerida.")

    def uniq(col):
        if col not in merged_df.columns:
            return []
        vals = merged_df[col].dropna().astype(str).unique().tolist()
        vals = [v for v in vals if v.strip() != ""]
        return sorted(vals)

    # Multiselect dropdowns
    selected_semester = st.multiselect("Semester", options=uniq("semester"), default=[])
    selected_keel = st.multiselect("Õppekeel", options=uniq("keel"), default=[])
    selected_oppeaste = st.multiselect("Õppeaste", options=uniq("oppeaste"), default=[])
    selected_linn = st.multiselect("Linn", options=uniq("linn"), default=[])
    selected_veebiope = st.multiselect("Veebiõpe / vorm", options=uniq("veebiope"), default=[])
    selected_hindamisviis = st.multiselect("Hindamisviis", options=uniq("hindamisviis"), default=[])

    # EAP range slider
    if "eap" in merged_df.columns and merged_df["eap"].notna().any():
        eap_min = float(np.nanmin(merged_df["eap"].values))
        eap_max = float(np.nanmax(merged_df["eap"].values))
        eap_range = st.slider("EAP vahemik", min_value=float(np.floor(eap_min)),
                              max_value=float(np.ceil(eap_max)),
                              value=(float(np.floor(eap_min)), float(np.ceil(eap_max))),
                              step=0.5)
    else:
        eap_range = None

    st.divider()
    results_N = st.slider("Top-N tulemused", 3, 12, 5, 1)

    st.divider()
    st.subheader("Kulu / tokenid")
    st.caption("Kui OpenRouter ei tagasta usage infot streamis, teeme hinnangu (umbes).")

# -----------------------------
# Token + cost tracking (simple estimate)
# -----------------------------
if "usage" not in st.session_state:
    st.session_state.usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

# Put your *real* OpenRouter pricing here if you want accurate cost.
# Otherwise it stays as a consistent estimate.
MODEL_NAME = "google/gemma-3-27b-it"
PRICE_USD_PER_1M_INPUT = 0.089   # <-- set if you know it
PRICE_USD_PER_1M_OUTPUT = 0.242  # <-- set if you know it

def estimate_tokens(text: str) -> int:
    # crude but works when tokenizer isn't available
    return max(1, len(text) // 4)

def add_cost(in_tok: int, out_tok: int):
    st.session_state.usage["input_tokens"] += in_tok
    st.session_state.usage["output_tokens"] += out_tok
    st.session_state.usage["cost_usd"] += (in_tok / 1_000_000) * PRICE_USD_PER_1M_INPUT + (out_tok / 1_000_000) * PRICE_USD_PER_1M_OUTPUT

# -----------------------------
# Chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# Filtering function
# -----------------------------
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df

    if selected_semester:
        out = out[out["semester"].isin(selected_semester)]
    if selected_keel:
        out = out[out["keel"].isin(selected_keel)]
    if selected_oppeaste:
        out = out[out["oppeaste"].isin(selected_oppeaste)]
    if selected_linn:
        out = out[out["linn"].isin(selected_linn)]
    if selected_veebiope:
        out = out[out["veebiope"].isin(selected_veebiope)]
    if selected_hindamisviis:
        out = out[out["hindamisviis"].isin(selected_hindamisviis)]

    if eap_range is not None and "eap" in out.columns:
        lo, hi = eap_range
        out = out[(out["eap"] >= lo) & (out["eap"] <= hi)]

    return out

# -----------------------------
# User input
# -----------------------------
if prompt := st.chat_input("Kirjelda, mida soovid õppida..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            error_msg = "Palun sisesta API võti!"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            with st.spinner("Otsin sobivaid kursusi..."):
                # 1) Apply metadata filters FIRST
                filtered_df = apply_filters(merged_df).copy()

                if filtered_df.empty:
                    st.warning("Ühtegi kursust ei vasta valitud filtritele. Tee filtrid lõdvemaks.")
                    context_text = "Sobivaid kursusi ei leitud."
                else:
                    # 2) Semantic search on filtered subset
                    query_vec = embedder.encode([prompt])[0]

                    # Ensure embeddings are stacked correctly
                    # Assumes embeddings column name is 'embedding' and each entry is a vector
                    emb_mat = np.stack(filtered_df["embedding"].values)
                    scores = cosine_similarity([query_vec], emb_mat)[0]

                    tmp = filtered_df.copy()
                    tmp["score"] = scores
                    tmp = tmp.sort_values("score", ascending=False).head(results_N)

                    # Keep context compact-ish (LLMs don’t need your whole dataframe dump)
                    show_cols = [c for c in [
                        "aine_kood", "nimi_et", "semester", "eap", "keel",
                        "oppeaste", "linn", "veebiope", "hindamisviis",
                        "kirjeldus"
                    ] if c in tmp.columns]

                    # Shorten long text fields
                    if "kirjeldus" in tmp.columns:
                        tmp["kirjeldus"] = tmp["kirjeldus"].astype(str).str.slice(0, 800)

                    context_text = tmp[show_cols].to_string(index=False)

                # 3) LLM call (RAG)
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

                system_prompt = {
                    "role": "system",
                    "content": (
                        "Oled kursusenõustaja. Kasuta AINULT allolevat konteksti (metaandmete filtrid + vektorotsing).\n\n"
                        f"KONTEKST:\n{context_text}\n\n"
                        "Kui kontekstist ei piisa, ütle seda ja küsi täpsustust."
                    )
                }

                messages_to_send = [system_prompt] + st.session_state.messages

                # token estimate BEFORE call
                approx_in = sum(estimate_tokens(m["content"]) for m in messages_to_send if "content" in m)

                try:
                    stream = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages_to_send,
                        stream=True
                    )
                    response = st.write_stream(stream)

                    # token estimate AFTER
                    approx_out = estimate_tokens(response)
                    add_cost(approx_in, approx_out)

                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    st.error(f"Viga: {e}")

# -----------------------------
# Usage display
# -----------------------------
st.divider()
u = st.session_state.usage
st.subheader("Jooksev kasutus (hinnanguline)")
st.write(
    f"**Sisendtokenid:** {u['input_tokens']}\n\n"
    f"**Väljundtokenid:** {u['output_tokens']}\n\n"
    f"**Kulu (USD):** ${u['cost_usd']:.6f}"
)