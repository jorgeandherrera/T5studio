import html
import time

import streamlit as st

# ============================================================
# Configuración general de la aplicación
# ============================================================
st.set_page_config(
    page_title="T5 Studio | Inferencia",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_NAME = "google-t5/t5-small"

EXAMPLE_SUMMARY = (
    "Artificial intelligence is transforming many industries by automating repetitive tasks, "
    "supporting data-driven decisions, and enabling new products. However, its responsible "
    "adoption requires attention to data quality, privacy, fairness, and human oversight. "
    "Organizations that combine technical capabilities with clear governance are better "
    "prepared to obtain value from these systems while reducing their risks."
)

EXAMPLE_TRANSLATION = "Artificial intelligence can support better decisions."

TASKS = {
    "Resumen": {
        "prefix": "summarize",
        "example": EXAMPLE_SUMMARY,
        "default_output": 80,
    },
    "Traducción EN → DE": {
        "prefix": "translate English to German",
        "example": EXAMPLE_TRANSLATION,
        "default_output": 40,
    },
}

# ============================================================
# Estilos: apariencia equivalente a las capturas del proyecto
# ============================================================
st.markdown(
    """
    <style>
    .stApp {background: #f7f9fc; color: #0f172a;}
    .block-container {
        max-width: 1380px;
        padding-top: 0.3rem;
        padding-bottom: 3rem;
    }
    header[data-testid="stHeader"] {background: transparent;}
    #MainMenu, footer {visibility: hidden;}

    .topline {
        border-top: 3px solid #20242c;
        padding-top: 0.55rem;
        color: #b64b00;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: .24rem;
        text-transform: uppercase;
    }
    .hero {
        background: rgba(255,255,255,.92);
        border: 1px solid #edf0f5;
        border-radius: 0 0 28px 28px;
        padding: 2rem 2.7rem 2.2rem 2.7rem;
        box-shadow: 0 18px 34px rgba(15,23,42,.08);
        margin-bottom: 2.6rem;
    }
    .hero-title {
        font-size: clamp(3rem, 6vw, 4.8rem);
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.06em;
        margin: .45rem 0 1.7rem 0;
        color: #0f172a;
    }
    .hero-copy {
        font-size: 1.05rem;
        line-height: 1.8;
        max-width: 960px;
        margin-bottom: 1.4rem;
    }
    .chip {
        display: inline-block;
        border: 1px solid #d5dbe6;
        border-radius: 999px;
        padding: .52rem .9rem;
        margin: .25rem .35rem .15rem 0;
        background: white;
        font-weight: 700;
        font-size: .82rem;
        box-shadow: 0 2px 4px rgba(15,23,42,.03);
    }
    .section-title {
        font-size: 1.45rem;
        font-weight: 800;
        margin: 0 0 1rem 0;
    }
    .result-box {
        background: #f4f8ff;
        border: 1px solid #dbe7fb;
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        min-height: 125px;
        box-shadow: 0 3px 10px rgba(37,99,235,.04);
    }
    .result-label {
        color: #2456a6;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .08rem;
        text-transform: uppercase;
        margin-bottom: .65rem;
    }
    .metric-card {
        background: #fff;
        border: 1px solid #e5e9f0;
        border-radius: 13px;
        padding: .95rem 1rem;
        min-height: 90px;
        box-shadow: 0 3px 9px rgba(15,23,42,.03);
    }
    .metric-value {font-size: 1.25rem; font-weight: 800; color: #0f172a;}
    .metric-label {
        margin-top: .35rem;
        color: #7b8494;
        font-size: .66rem;
        font-weight: 700;
        letter-spacing: .05rem;
        text-transform: uppercase;
    }
    .note {
        color: #7b8494;
        font-size: .83rem;
        margin-top: .5rem;
    }
    div.stButton > button {
        border-radius: 14px;
        min-height: 3.25rem;
        font-weight: 800;
        border: 1px solid #111827;
    }
    div.stButton > button[kind="primary"] {
        background: #111827;
        color: white;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    textarea {
        border-radius: 14px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Carga de T5-small
# Se almacena en caché para evitar recargar el modelo en cada clic.
# ============================================================
@st.cache_resource(show_spinner="Cargando T5-small por primera vez...")
def load_model():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def run_inference(text: str, prefix: str, max_input_tokens: int, max_output_tokens: int):
    """Ejecuta el mismo flujo de inferencia usado en el notebook del proyecto."""
    import torch

    if not isinstance(text, str) or not text.strip():
        raise ValueError("La entrada debe contener texto.")

    tokenizer, model, device = load_model()
    model_input = f"{prefix}: {text.strip()}"

    inputs = tokenizer(
        model_input,
        return_tensors="pt",
        max_length=max_input_tokens,
        truncation=True,
    ).to(device)

    input_tokens = int(inputs["input_ids"].shape[-1])

    # Sincronización necesaria para medir correctamente cuando se usa GPU.
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_output_tokens,
            num_beams=4,
            do_sample=False,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Se resta el <pad> inicial del decoder, que no lo genera el modelo
    output_tokens = int(generated.shape[-1]) - 1
    prediction = tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    return {
        "prediction": prediction,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed": elapsed,
        "device": device.type,
        "model_input": model_input,
        "max_input_tokens": max_input_tokens,
        "generated_ids": generated[:1].cpu(),
    }


def cross_attention_chart(result, layer):
    """Mapa de atención cruzada: qué tokens de la entrada mira el decoder en cada paso."""
    import torch
    import pandas as pd
    import altair as alt

    tokenizer, model, device = load_model()
    enc = tokenizer(
        result["model_input"],
        return_tensors="pt",
        max_length=result["max_input_tokens"],
        truncation=True,
    ).to(device)
    dec_ids = result["generated_ids"].to(device)

    with torch.inference_mode():
        out = model(**enc, decoder_input_ids=dec_ids, output_attentions=True)

    # (cabezas, pasos del decoder, tokens de entrada) -> promedio de las 8 cabezas.
    # La posición i del decoder es la que produce el token i+1.
    att = out.cross_attentions[layer][0].mean(0)[:-1].float().cpu()
    src = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    tgt = tokenizer.convert_ids_to_tokens(dec_ids[0])[1:]

    rows = [
        {"generado": f"{i:02d} {t}", "entrada": f"{j:03d} {s}", "peso": float(att[i, j])}
        for i, t in enumerate(tgt)
        for j, s in enumerate(src)
    ]
    return alt.Chart(pd.DataFrame(rows)).mark_rect().encode(
        x=alt.X("entrada:O", sort=None, title="Tokens de entrada", axis=alt.Axis(labelAngle=-60)),
        y=alt.Y("generado:O", sort=None, title="Token generado"),
        color=alt.Color("peso:Q", scale=alt.Scale(scheme="blues"), title="Peso"),
        tooltip=["entrada", "generado", alt.Tooltip("peso:Q", format=".3f")],
    ).properties(height=max(220, 18 * len(tgt)))


# ============================================================
# Encabezado
# ============================================================
st.markdown('<div class="topline">NLP · TRANSFORMER · INFERENCIA</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">T5 Studio</div>
        <div class="hero-copy">
            Interfaz de despliegue académico para <b>google-t5/t5-small</b>. Convierte una
            instrucción y un texto en una salida generada con la misma arquitectura
            encoder-decoder utilizada en el notebook del proyecto.
        </div>
        <span class="chip">Text-to-Text</span>
        <span class="chip">6 Encoder + 6 Decoder</span>
        <span class="chip">d_model 512</span>
        <span class="chip">8 cabezas</span>
        <span class="chip">~60.5 M parámetros</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Laboratorio de inferencia</div>', unsafe_allow_html=True)

left, right = st.columns([1.45, 1], gap="large")

with left:
    task_name = st.selectbox("Tarea", list(TASKS.keys()), key="task_name")
    task_info = TASKS[task_name]

    # Cambia automáticamente el ejemplo al cambiar de tarea, salvo que el usuario ya esté escribiendo.
    if "last_task" not in st.session_state:
        st.session_state.last_task = task_name
        st.session_state.input_text = task_info["example"]
    elif st.session_state.last_task != task_name:
        st.session_state.last_task = task_name
        st.session_state.input_text = task_info["example"]
        st.session_state.pop("result", None)

    # Carga de datos: un archivo .txt reemplaza el texto del área de entrada
    uploaded = st.file_uploader("Cargar un archivo de texto (.txt)", type=["txt"])
    if uploaded is not None:
        file_key = f"{uploaded.name}-{uploaded.size}"
        if st.session_state.get("last_file") != file_key:
            st.session_state.last_file = file_key
            st.session_state.input_text = uploaded.read().decode("utf-8", errors="ignore")
            st.session_state.pop("result", None)

    text = st.text_area(
        "Texto de entrada",
        key="input_text",
        height=265,
        placeholder="Escriba o pegue aquí el texto que desea procesar...",
    )

with right:
    st.markdown("**Configuración**")
    st.text_input("Prefijo enviado a T5", value=task_info["prefix"], disabled=True)

    max_input_tokens = st.slider(
        "Máximo de tokens de entrada",
        min_value=64,
        max_value=512,
        value=512,
        step=32,
    )

    default_output = 80 if task_name == "Resumen" else 40
    max_output_tokens = st.slider(
        "Máximo de tokens de salida",
        min_value=20,
        max_value=160,
        value=default_output,
        step=10,
    )

    st.markdown(
        '<div class="note">Beam search = 4 · deterministic · no_repeat_ngram_size = 2</div>',
        unsafe_allow_html=True,
    )
    execute = st.button("✦ Ejecutar inferencia", type="primary", use_container_width=True)

if execute:
    if not text.strip():
        st.warning("Ingrese un texto antes de ejecutar la inferencia.")
    else:
        try:
            with st.spinner("Procesando con T5-small..."):
                st.session_state.result = run_inference(
                    text=text,
                    prefix=task_info["prefix"],
                    max_input_tokens=max_input_tokens,
                    max_output_tokens=max_output_tokens,
                )
        except Exception as exc:
            st.error("No fue posible ejecutar T5-small.")
            st.exception(exc)

# ============================================================
# Salida y métricas
# ============================================================
if "result" in st.session_state:
    result = st.session_state.result
    st.write("")
    st.markdown('<div class="section-title">Salida del modelo</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label">Predicción T5-small</div>
            <div>{html.escape(result['prediction'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        (m1, result["input_tokens"], "Tokens entrada"),
        (m2, result["output_tokens"], "Tokens salida"),
        (m3, f"{result['elapsed']:.2f}s", "Inferencia"),
        (m4, result["device"], "Dispositivo"),
    ]
    for col, value, label in metrics:
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Mapa de atención cruzada (decoder → encoder)"):
        st.caption(
            "Cada fila es un token generado y cada columna un token de la entrada. "
            "Entre más oscura la celda, más atención le puso el decoder a ese token "
            "para escribir el suyo (promedio de las 8 cabezas)."
        )
        layer = st.slider("Bloque del decoder", 1, 6, 6)
        st.altair_chart(cross_attention_chart(result, layer - 1), use_container_width=True)

    with st.expander("Ver entrada exacta enviada al modelo"):
        st.code(result["model_input"], language=None)

    st.download_button(
        "Descargar salida (.txt)",
        data=result["prediction"],
        file_name="salida_t5.txt",
        mime="text/plain",
    )
else:
    st.info("Configure la tarea y pulse **Ejecutar inferencia** para generar una salida.")

st.caption(
    "Proyecto académico · T5-small · Hugging Face Transformers · PyTorch · Streamlit"
)
