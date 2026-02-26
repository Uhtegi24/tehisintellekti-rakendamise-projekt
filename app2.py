import streamlit as st
from openai import OpenAI

st.title("🎓 AI Kursuse Nõustaja - Samm 2")
st.caption("Vestlus päris tehisintellektiga (Gemma 3 / OpenRouter vms).")

# ---------- Helpers ----------
def safe_header_value(s: str | None) -> str | None:
    """
    HTTP headers should be ASCII-safe in many clients.
    If user enters non-ascii, strip it to avoid encoding errors.
    """
    if not s:
        return None
    try:
        s.encode("ascii")
        return s
    except UnicodeEncodeError:
        cleaned = s.encode("ascii", "ignore").decode("ascii")
        return cleaned if cleaned else None


def supports_system_role(model_name: str) -> bool:
    """
    Some OpenRouter routes (notably some Gemma/Google AI Studio ones) reject system/developer roles.
    Conservative heuristic: disable system role for Gemma models.
    """
    m = (model_name or "").lower()
    if "gemma" in m:
        return False
    return True


def stream_to_text(stream):
    """Convert streaming chunks into plain text tokens for st.write_stream."""
    for chunk in stream:
        try:
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                yield delta.content
        except Exception:
            continue


def build_messages(history, system_prompt, model_name):
    """
    If model supports system role: [system] + history
    Else: fold system prompt into last user message.
    """
    hist = list(history)  # copy

    if supports_system_role(model_name):
        return [{"role": "system", "content": system_prompt}] + hist

    # No system role allowed -> inject instructions into user message
    # Expect newest message to be the user's prompt
    if len(hist) > 0 and hist[-1]["role"] == "user":
        hist[-1]["content"] = (
            "JUHIS (järgi seda):\n"
            f"{system_prompt}\n\n"
            "KASUTAJA:\n"
            f"{hist[-1]['content']}"
        )
    else:
        hist.append(
            {"role": "user", "content": f"JUHIS:\n{system_prompt}\n\nKASUTAJA:\n"}
        )
    return hist


# ---------- Sidebar settings ----------
with st.sidebar:
    st.header("Seaded")

    provider = st.selectbox(
        "Provider",
        ["OpenRouter", "OpenAI (default)"],
        help="Kui kasutad OpenRouter API võtit, vali OpenRouter."
    )

    api_key = st.text_input(
        "API võti",
        type="password",
        help="OpenRouter või OpenAI võtme jaoks."
    )

    model_name = st.text_input(
        "Mudel",
        value="google/gemma-3-27b-it:free",
        help="OpenRouter mudelinimed stiilis 'provider/model' või 'provider/model:free'."
    )

    site_url = st.text_input(
        "Site URL (optional)",
        value="http://localhost:8501",
        help="OpenRouter: HTTP-Referer (valikuline, soovituslik)."
    )

    app_title = st.text_input(
        "App title (optional)",
        value="AI Kursuse Noustaja",
        help="OpenRouter: X-Title (valikuline). Hoia ASCII, muidu võib header encoding error tulla."
    )

    st.divider()
    st.caption("Kui API võtit pole, kuvatakse veateade ja vastust ei genereerita.")


# ---------- Session history ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# render history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------- New user input ----------
if prompt := st.chat_input("Kirjelda, mida soovid õppida..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    system_prompt = (
        "Sa oled ülikooli kursuse nõustaja. Küsi 1–2 täpsustavat küsimust, "
        "siis soovita sobivaid kursusi ning põhjenda valikut lühidalt. "
        "Vastus olgu eesti keeles, konkreetselt ja sõbralikult."
    )

    with st.chat_message("assistant"):
        if not api_key:
            err = "❌ API võti puudub. Sisesta see vasakul külgribal, et saaksin vastata."
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
        else:
            try:
                # build client
                if provider == "OpenRouter":
                    headers = {}
                    ref = safe_header_value(site_url)
                    title = safe_header_value(app_title)
                    if ref:
                        headers["HTTP-Referer"] = ref
                    if title:
                        headers["X-Title"] = title

                    client = OpenAI(
                        api_key=api_key,
                        base_url="https://openrouter.ai/api/v1",
                        default_headers=headers
                    )
                else:
                    client = OpenAI(api_key=api_key)

                # build messages with system-compat handling
                messages = build_messages(
                    history=st.session_state.messages,
                    system_prompt=system_prompt,
                    model_name=model_name
                )

                # stream response
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=True,
                )

                response = st.write_stream(stream_to_text(stream))
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"Viga: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Viga: {e}"})
