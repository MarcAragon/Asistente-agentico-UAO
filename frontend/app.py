"""Frontend Streamlit del Asistente RAG UAO (Fase 7).

Consume unicamente POST /ask de la API REST (ver plan-trabajo-tecnico.md).
El historial se guarda solo para mostrarlo en pantalla: es un chat de una
sola vuelta, no se reenvia como contexto al LLM.
"""

import os

import httpx
import streamlit as st

API_BASE_URL = os.getenv("UAO_RAG__API_BASE_URL", "http://localhost:8000")

PREGUNTAS_EJEMPLO = [
    "¿Qué pasa si repruebo tres veces una misma asignatura?",
    "¿Cuáles son los requisitos para obtener el título de magíster?",
    "¿Se puede hacer transferencia interna entre programas?",
]


def preguntar_api(pregunta: str) -> dict:
    """Llama a POST /ask y devuelve un mensaje listo para el historial."""
    try:
        respuesta = httpx.post(
            f"{API_BASE_URL}/ask",
            json={"question": pregunta},
            timeout=30.0,
        )
        respuesta.raise_for_status()
        datos = respuesta.json()
        return {
            "rol": "assistant",
            "texto": datos["answer"],
            "fuentes": datos["sources"],
            "fallback": datos["used_fallback"],
        }
    except httpx.HTTPStatusError as exc:
        codigo = exc.response.status_code
        if codigo == 422:
            texto = "La pregunta no es válida (vacía o muy larga)."
        elif codigo == 503:
            texto = "El asistente no tiene configurada la clave del modelo (CEREBRAS_API_KEY)."
        else:
            texto = f"Error interno de la API ({codigo})."
        return {"rol": "assistant", "texto": texto, "fuentes": [], "fallback": True}
    except httpx.RequestError:
        texto = f"No se pudo conectar con la API. ¿Está corriendo en {API_BASE_URL}?"
        return {"rol": "assistant", "texto": texto, "fuentes": [], "fallback": True}


st.set_page_config(page_title="Asistente RAG UAO", page_icon="🎓")
st.title("Asistente RAG UAO")
st.caption("Preguntas sobre normativa institucional de la UAO")
st.info(
    "Las respuestas son orientativas y citan la fuente oficial de cada dato; "
    "no reemplazan la asesoría de Secretaría Académica o Bienestar Universitario. "
    "Este chat no solicita ni guarda datos personales."
)

if "historial" not in st.session_state:
    st.session_state.historial = []

# El chat_input siempre se ve anclado abajo, sin importar en que parte del
# codigo se declare, asi que lo capturamos aqui arriba antes de dibujar el
# historial.
pregunta_escrita = st.chat_input(
    "Escribe tu pregunta sobre la normativa UAO...", max_chars=500
)

st.caption("O prueba una de estas preguntas:")
columnas = st.columns(len(PREGUNTAS_EJEMPLO))
pregunta_ejemplo = None
for columna, ejemplo in zip(columnas, PREGUNTAS_EJEMPLO):
    if columna.button(ejemplo, use_container_width=True):
        pregunta_ejemplo = ejemplo

pregunta = pregunta_escrita or pregunta_ejemplo
if pregunta:
    st.session_state.historial.append({"rol": "user", "texto": pregunta})
    with st.spinner("Buscando en la normativa..."):
        st.session_state.historial.append(preguntar_api(pregunta))

for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["texto"])
        if mensaje.get("fallback"):
            st.warning("No hay información suficiente en la normativa para esta pregunta.")
        elif mensaje.get("fuentes"):
            with st.expander(f"Fuentes ({len(mensaje['fuentes'])})"):
                for fuente in mensaje["fuentes"]:
                    st.markdown(
                        f"**{fuente['doc_name']}** — {fuente['section']} "
                        f"(similitud {fuente['score']:.2f})"
                    )
                    st.caption(fuente["excerpt"])