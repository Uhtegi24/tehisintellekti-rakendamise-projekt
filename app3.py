import streamlit as st
import pandas as pd
from openai import OpenAI

st.title("🎓 AI Kursuse Nõustaja")
st.caption("AI kasutab kursuste andmeid (esimesed 10 rida).")

DATA_PATH = "Data/andmed_aasta.csv"  # muuda kui su fail asub mujal
MODEL_NAME = "google/gemma-3-27b-it"

# Külgriba API võtme jaoks
with st.sidebar:
    api_key = st.text_input("OpenRouter API Key", type="password")
    st.caption("Kasuta OpenRouteri võtit. Mudel on praegu Gemma (free).")


# ---------- UUS: Andmete laadimine + cache ----------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df

try:
    df = load_data(DATA_PATH)
    st.success(f"Andmed laetud: {df.shape[0]} rida, {df.shape[1]} veergu.")
except Exception as e:
    st.error(f"Ei suutnud andmeid laadida failist '{DATA_PATH}'. Viga: {e}")
    st.stop()

# Näitame esimesed 10 rida (valikuline, aga kasulik)
st.subheader("Esimesed 10 rida andmestikust")
st.table(df.head(10))  # avoids pyarrow issues


# ---------- UUS: muutmine tekstiks AI jaoks ----------
def df_to_compact_text(df10: pd.DataFrame) -> str:
    """
    Teeme 10 rea põhjal lühikese ja loetava tekstibloki.
    Eelistame olulisi veerge, kui need olemas.
    """
    preferred_cols = [
        "aine_kood", "nimi_et", "eap", "semester", "oppeaste_et",
        "oppekeel_et", "hindamine_tuup", "kirjeldus_et"
    ]
    cols = [c for c in preferred_cols if c in df10.columns]
    if not cols:
        cols = list(df10.columns)[:8]  # fallback: esimesed 8 veergu

    lines = []
    for i, row in df10[cols].iterrows():
        parts = []
        for c in cols:
            val = row.get(c)
            if pd.isna(val):
                continue
            s = str(val).strip()
            if not s:
                continue
            # hoia kirjeldus lühike, et prompt ei plahvataks
            if c in ("kirjeldus_et", "kirjeldus") and len(s) > 280:
                s = s[:280] + "..."
            parts.append(f"{c}={s}")
        lines.append(f"- {', '.join(parts)}")

    return "\n".join(lines)


# ---------- Streaming adapter ----------
def stream_to_text(stream):
    """Converts OpenAI/OpenRouter streaming chunks into text for st.write_stream."""
    for chunk in stream:
        try:
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                yield delta.content
        except Exception:
            continue


# ---------- JUBA OLEMAS: chat history ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------- Chat input ----------
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
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

            # UUS: 10 rea tekst
            df10 = df.head(10)
            data_context = df_to_compact_text(df10)

            # IMPORTANT: Gemma route may reject system role -> put instructions + data into user message
            instruction = (
                "Sa oled ülikooli kursuse nõustaja. Kasuta ainult allolevat kursuste loetelu (10 rida), "
                "et vastata kasutaja küsimusele. Kui info puudub, ütle ausalt, et ei saa nende 10 rea põhjal kindlalt väita.\n\n"
                "KURSUSEANDMED (10 rida):\n"
                f"{data_context}\n\n"
                "KASUTAJA KÜSIMUS:\n"
                f"{prompt}"
            )

            messages_to_send = [
                {"role": "user", "content": instruction}
            ]

            try:
                stream = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages_to_send,
                    stream=True
                )
                response = st.write_stream(stream_to_text(stream))
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Viga: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Viga: {e}"})
