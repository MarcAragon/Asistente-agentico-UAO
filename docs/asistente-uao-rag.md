## Capítulo 1. Contexto del proyecto

### 1.1 Descripción del problema

Los estudiantes de la Universidad Autónoma de Occidente (UAO) requieren consultar, con relativa frecuencia, información institucional que se encuentra dispersa en distintos documentos oficiales, tales como: el horario académico, el código de ética, los reglamentos de pregrado y posgrado, la resolución de trabajo de grado y las disposiciones sobre cancelación de asignaturas. Localizar la respuesta a una pregunta puntual exige revisar manualmente documentos extensos en formato PDF o incurrir en tiempos de espera al contactar a Bienestar Universitario o Secretaría Académica.

Este proyecto propone el desarrollo de un asistente conversacional (chatbot) capaz de responder estas preguntas en lenguaje natural, indicando siempre el documento oficial del cual proviene cada respuesta para garantizar trazabilidad y precisión.

### 1.2 Justificación del proyecto

- **Pertinencia y eficiencia:** Resuelve un problema real que afecta a una población concreta. Al no requerir entrenamiento desde cero del modelo generador, aumenta la eficiencia del desarrollo mediante una arquitectura moderna –Retrieval-Augmented Generation (RAG)– estándar en la industria.
- **Impacto operativo:** Reduce la fricción en la búsqueda de información institucional, disminuyendo la carga operativa sobre las dependencias administrativas encargadas de resolver estas consultas recurrentes.
- **Despliegue:** Se integra en un entorno de producción como una aplicación web accesible a través divulgada por los canales independientes disponibles en la institución.

### 1.3 Descripción y Model Card del modelo base

El sistema emplea un enfoque RAG, conectando un modelo de lenguaje preentrenado con una base de conocimiento propia. Esto reduce drásticamente el riesgo de alucinaciones y permite mantener los reglamentos actualizados sin necesidad de reentrenar (fine-tuning) el modelo generador.

#### 1.3.1 Model Card — Modelo generador (LLM)

| Campo | Descripción |
|---|---|
| Modelo | Qwen3.8-27B |
| Desarrollador | Alibaba |
| Tipo | Modelo de lenguaje (LLM); Transformer decoder-only (causal LM) |
| Parámetros | 28 billones |
| Arquitectura | 64 capas; 16 capas GQA (Grouped-Query Attention) y 48 capas de atención lineal (Gated DeltaNet) |
| Contexto | 262,144 tokens (extensible hasta 1,000,000) |
| Licencia | Apache 2.0 |
| Uso | Generación de la respuesta final a partir de los fragmentos recuperados |
| Enlace | [huggingface.co/Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Limitaciones | Su exactitud depende estrictamente de la calidad y pertinencia del contexto recuperado |

#### 1.3.2 Métricas de rendimiento (benchmarks generales)

| Benchmark | Resultado |
|---|---|
| GPQA Diamond | 89.2 |
| Humanity's Last Exam (HLE) | 30.8 |
| IFBench (Instrucciones) | 79.5 |

*Tabla 1.1: Resultados publicados por el desarrollador del modelo generador.*

Estos resultados corresponden a evaluaciones generales del modelo. En la fase de evaluación del proyecto se realizará una medición de métricas específicas sobre el dominio de conocimiento institucional abordado.

#### 1.3.3 Model Card — Modelo de embeddings

| Campo | Descripción |
|---|---|
| Modelo | paraphrase-multilingual-mpnet-base-v2 |
| Desarrollador | UKPLab, Universidad Técnica de Darmstadt |
| Tipo | Modelo de embedding de oraciones (Arquitectura MPNet) |
| Longitud/Vocab | 128 tokens máximos / 250,002 tokens de vocabulario |
| Licencia | Apache 2.0 |
| Uso | Transformación de texto institucional a vectores para búsqueda semántica |
| Enlace | [huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) |

### 1.4 Restricciones y consideraciones técnicas

- **Stack de Desarrollo:** Python 3.13 con gestor de dependencias `uv`, `sentence-transformers`, orquestación mediante LangChain/LlamaIndex, base de datos vectorial (FAISS/ChromaDB) y Streamlit.
- **Infraestructura y Despliegue:** El modelo de embeddings se ejecutará localmente (CPU), mientras que el LLM será consumido vía API. El sistema completo se contenerizará con Docker. Adicionalmente, se integrará un contenedor de Redis para la gestión del caché semántico de consultas frecuentes y el estado de la aplicación. Para exponer el servicio de forma segura, se implementará un servidor proxy inverso que automatice la validación de certificados TLS y garantice un enrutamiento seguro hacia la interfaz.
- **Cumplimiento normativo:** El pipeline de ingesta aplicará filtros para evitar la exposición de datos sensibles, cumpliendo con la Ley 1581 de 2012 de Protección de Datos Personales.

### 1.5 Limitaciones y riesgos

- **Calidad y actualización de los datos:** el sistema es tan confiable como los documentos indexados; si un reglamento se actualiza y el índice no se refresca, el asistente puede dar información desactualizada.
- **Alucinaciones del modelo generador:** como todo LLM, el modelo generador puede producir texto plausible pero incorrecto. Se mitiga citando siempre la fuente de cada respuesta y definiendo un umbral mínimo de similitud semántica: si ningún fragmento recuperado lo supera, el sistema responde que no tiene información suficiente en lugar de inventar una respuesta.
- **Tiempo de procesamiento:** la indexación inicial de los documentos y la latencia de generación de cada respuesta –especialmente al consumir el LLM vía API– pueden afectar la experiencia de uso si no se optimizan.
- **Diferencia entre preguntas de prueba y preguntas reales:** las preguntas diseñadas por el propio equipo para probar el sistema tienden a estar bien formuladas; las preguntas reales de estudiantes pueden incluir jerga, errores de escritura o ambigüedad, lo que puede reducir el desempeño observado en producción respecto a las pruebas internas.

---

## Capítulo 2. Objetivos y alcance

### 2.1 Objetivo general

Desarrollar un asistente conversacional basado en un sistema de Generación Aumentada por Recuperación (RAG) capaz de responder preguntas de los estudiantes de la UAO sobre normativas institucionales, citando de manera precisa la fuente oficial de cada respuesta.

### 2.2 Objetivos específicos

1. Construir un pipeline de extracción y limpieza de documentos oficiales en PDF, dividiéndolos en fragmentos indexables (chunks).
2. Implementar un mecanismo de recuperación semántica utilizando un modelo de embeddings y una base de datos vectorial.
3. Integrar un modelo LLM para la síntesis de respuestas fundamentadas exclusivamente en los documentos recuperados.
4. Desarrollar una interfaz de usuario interactiva y robusta.
5. Contenerizar la solución completa (Backend, Frontend, Caché de Redis y Proxy) usando Docker.
6. Aplicar un protocolo de evaluación cuantitativa y cualitativa del sistema.

### 2.3 Matriz de alcance

| Incluye | Nice to have | No incluye |
|---|---|---|
| Embeddings preentrenados | Historial multiturno | Automatización de actualización de PDFs |
| Pipeline de ingesta PDF | Panel de retroalimentación de respuestas | Integración oficial con sistemas UAO: SIA, Aula Virtual o Portal estudiantil |
| Búsqueda en DB vectorial | | Autenticación de usuarios contra el directorio institucional |
| LLM para síntesis final | | Fine-tuning del LLM |
| Interfaz web (Streamlit) | | |
| Contenerización Docker, caché y proxy | | |

---

## Capítulo 3. Mapa mental del proyecto

### 3.1 Estructura conceptual

Para sintetizar de forma visual los componentes centrales y la arquitectura técnica, el equipo elaboró un mapa mental (ver Figura 3.1). Este diagrama unifica la planeación interna con la lógica de integración de los servicios (pipeline de ingesta, base de datos vectorial, backend de recuperación, caché de respuestas y el frontend servido de forma segura).

*Figura 3.1: Mapa mental del proyecto de asistente virtual RAG para la UAO.*

### 3.2 Cronograma (Diagrama de Gantt) y ruta crítica

Además del mapa mental, el equipo generó un diagrama de Gantt (ver Figura 3.2) que detalla la secuencia temporal de las 14 tareas principales del proyecto, sus subtareas y las dependencias entre ellas, desde la configuración del entorno (07 de septiembre) hasta la entrega final (24 de septiembre).

*Figura 3.2: Diagrama de Gantt del proyecto, con dependencias entre tareas.*

La ruta crítica –la secuencia de tareas que determina la duración mínima del proyecto y que no puede retrasarse sin atrasar la entrega– pasa por el pipeline de datos y recuperación: recolección y limpieza de documentos, fragmentación, generación de embeddings, construcción del índice vectorial e implementación de la recuperación semántica. Cualquier retraso en esta cadena se traslada directamente a la integración del modelo generador y a las fases posteriores de pruebas, documentación y entrega, por lo que el equipo priorizará mantener esta parte del cronograma sin desviaciones.

---

## Capítulo 4. Marco de investigación y metodología

### 4.1 Área de conocimiento y enfoque metodológico

El proyecto se ubica en el área de Procesamiento de Lenguaje Natural (Natural Language Processing, NLP), específicamente en las subáreas de recuperación semántica de información (dense retrieval) y generación aumentada por recuperación (RAG). No se realiza entrenamiento propio de modelos de Machine Learning o Deep Learning: se reutilizan modelos preentrenados mediante transferencia directa (sin fine-tuning), en línea con el enfoque del curso, centrado en el desarrollo del sistema más que en la construcción del modelo.

### 4.2 Datasets y fuentes de datos

La fuente de datos del proyecto son los documentos oficiales de la propia Universidad Autónoma de Occidente, obtenidos directamente de las páginas públicas oficiales de la institución:

- Políticas de permanencia y graduación.
- Política de internacionalización.
- Reglamento general de estudiantes de pregrado profesional.
- Reglamento general de estudiantes de posgrado profesional.
- Reglamento general de estudiantes de pregrado profesional para programas virtuales.
- Reglamento de trabajo de grado para programas de pregrado.
- Reglamento de propiedad intelectual.
- Reglamento general de estudiantes de programas técnicos profesionales, tecnológicos y de especialización de nivel tecnológico.
- Reglamento académico para el doctorado en ingeniería.
- Código de ética y buen gobierno.
- Código de ética para estudiantes.
- Política de tratamiento y protección de datos personales.
- Calendario académico para programas semestrales, periodo comprendido entre el 13 de enero de 2026 y el 25 de enero de 2027.
- Calendario académico para programas cuatrimestrales, periodo comprendido entre el 15 de diciembre de 2025 y el 18 de diciembre de 2026.
- Calendario académico para programas bimestrales, periodo comprendido entre el 15 de diciembre de 2025 y el 18 de diciembre de 2026.

### 4.3 Técnicas

El proyecto emplea recuperación semántica densa (dense retrieval) apoyada en representaciones vectoriales de los documentos y de las consultas del usuario, junto con generación aumentada por recuperación (RAG) para la síntesis de la respuesta final. Como técnica complementaria, se contempla la incorporación de un mecanismo de re-ranking sobre los fragmentos recuperados en primera instancia, con el fin de mejorar la precisión del contexto entregado al modelo generador antes de la síntesis de la respuesta.

### 4.4 Frameworks y librerías

- **LangChain** o **LlamaIndex** — orquestación del pipeline RAG.
- **FAISS** o **ChromaDB** — base de datos vectorial para la búsqueda semántica.
- **Streamlit** — interfaz de usuario tipo chat.
- **Docker** — contenerización y despliegue.
- **uv** y `pyproject.toml` — gestión de dependencias sobre Python 3.13.
- **RAGAS** (opcional) — framework de evaluación automática de sistemas RAG, útil para calcular métricas de fidelidad y relevancia de las respuestas generadas.

### 4.5 Fases CRISP-DM

| Fase | Aplicación en el Proyecto |
|---|---|
| 1. Business Understanding | Definición del problema de accesibilidad a la normativa UAO. |
| 2. Data Understanding | Análisis de estructura, formatos y artefactos en los PDFs oficiales. |
| 3. Data Preparation | Extracción, limpieza y chunking (300-500 tokens con solapamiento). Retención de metadatos (página, sección) para citas exactas. |
| 4. Modeling | Construcción del índice vectorial y despliegue del pipeline RAG. |
| 5. Evaluation | Medición de Recall@k, MRR, fidelidad de generación (Faithfulness) y exactitud en las citas mediante frameworks como RAGAS. |
| 6. Deployment | Empaquetado de la solución vía docker-compose, configurando el enrutamiento HTTP, la persistencia de datos y el motor de caché (Redis). |

---

## Capítulo 5. Gestión de tareas (Kanban)

El seguimiento se gestiona en un tablero Kanban estructurado con estimación en puntos de historia (SP, secuencia de Fibonacci). La carga total proyectada es de 38 SP.

| # | Descripción de la Tarea | Categoría | SP |
|---|---|---|---|
| 001 | Configurar entorno base (uv, pyproject.toml). | DevOps | 2 |
| 002 | Recolección y catalogación de PDFs oficiales UAO. | Datos | 2 |
| 003 | Pipeline de extracción de texto y limpieza estructurada. | Backend | 3 |
| 004 | Lógica de fragmentación (chunking) y preservación de metadatos. | Backend | 2 |
| 005 | Generación de embeddings e indexación en base vectorial. | ML | 3 |
| 006 | Implementación del motor de recuperación (top-k) y re-ranking. | ML | 3 |
| 007 | Integración de Qwen vía API y diseño del prompt de síntesis. | ML | 5 |
| 008 | Desarrollo de la interfaz gráfica interactiva en Streamlit. | Frontend | 5 |
| 009 | Pruebas unitarias (chunking, precisión de búsqueda). | QA | 3 |
| 010 | Contenerización completa (Docker, configuración de Redis y Proxy). | DevOps | 5 |
| 011 | Redacción de documentación técnica (README, arquitectura, uso). | Docs | 2 |
| 012 | Preparación de demo y sustentación final del proyecto. | Equipo | 3 |