"""Frontend Streamlit del Asistente RAG UAO (Fase 7).

Consume unicamente POST /ask de la API REST (ver plan-trabajo-tecnico.md).
El historial se guarda solo para mostrarlo en pantalla: es un chat de una
sola vuelta, no se reenvia como contexto al LLM.
"""

import streamlit as st
from client import preguntar_api

PREGUNTAS_EJEMPLO = [
    "¿Qué pasa si repruebo tres veces una misma asignatura?",
    "¿Cuáles son los requisitos para obtener el título de magíster?",
    "¿Se puede hacer transferencia interna entre programas?",
]

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

with st.sidebar:
    st.header("Asistente RAG UAO")
    st.caption("Universidad Autónoma de Occidente")
    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.historial = []
        
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