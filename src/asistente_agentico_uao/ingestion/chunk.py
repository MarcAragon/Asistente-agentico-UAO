"""Chunking del markdown limpio de LlamaParse (Fase 2).

Estrategia (plan-trabajo-tecnico.md, Fase 2):

1. Fronteras duras: encabezados markdown ``#``..``####`` y pseudo-encabezados
   detectados por regex: artículos como línea propia (``**ARTÍCULO 1º:**`` o
   ``ARTÍCULO 37°.``), ``PARÁGRAFO`` y ``Artículo Único``. Cada frontera
   actualiza la sección vigente (metadata ``section``).
2. Empaquetado de párrafos hasta el objetivo de tokens, medidos con el
   tokenizador real del modelo de embeddings (inyectable para tests).
3. Overlap intra-sección: las últimas oraciones (~70 tokens) se repiten en el
   chunk siguiente. Nunca cruza una frontera dura: la cola del Artículo 21
   no contamina el chunk del Artículo 22.
4. Tablas ``<table>`` atómicas si caben en un chunk; si no, troceadas por
   filas ``<tr>`` repitiendo ``<thead>`` en cada sub-chunk.
5. El texto final lleva un prefijo contextual (``Documento:``/``Sección:``)
   para que chunks cortos y tablas recuperen con contexto. El presupuesto de
   tokens del cuerpo se reduce según el prefijo, de modo que el chunk final
   nunca exceda ``max_tokens``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

# --- Patrones de frontera ---------------------------------------------------

HEADER_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")

# Artículos/parágrafos como línea propia (negrita o no). Mayúsculas para no
# chocar con menciones en minúscula dentro de párrafos ("... el artículo 41°").
PSEUDO_HEADER_RE = re.compile(
    r"^(?:\*\*)?\s*(?:ART[IÍ]CULO\s+(?:[ÚU]NICO|\d+)|[Aa]rt[íi]culo\s+[ÚU]nico|PAR[ÁA]GRAFO\b)"
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
ROW_RE = re.compile(r"<tr\b.*?</tr>", re.IGNORECASE | re.DOTALL)
THEAD_RE = re.compile(r"<thead\b.*?</thead>", re.IGNORECASE | re.DOTALL)
PLACEHOLDER_SPLIT_RE = re.compile(r"(\x00T\d+\x00)")


@dataclass
class Chunk:
    """Chunk final listo para indexar en ChromaDB."""

    doc_name: str
    section: str
    text: str  # texto final (con prefijo contextual Documento/Sección)
    chunk_index: int  # posición dentro del documento
    n_tokens: int  # tokens del texto final (tokenizador real)
    is_table: bool = False


# --- 1. Fronteras duras ------------------------------------------------------


def _pseudo_title(raw: str) -> str:
    """Título de sección a partir de un pseudo-encabezado (artículo/parágrafo).

    La línea completa puede contener el texto del artículo, así que se recorta
    a la primera oración y a 160 caracteres para metadata manejable.
    """
    clean = re.sub(r"\*+", "", raw).strip()
    first = SENTENCE_SPLIT_RE.split(clean, maxsplit=1)[0]
    return first[:160].rstrip()


def split_sections(md: str) -> list[tuple[str, str]]:
    """Divide el markdown en ``(sección, cuerpo)`` usando fronteras duras.

    La línea del pseudo-encabezado se conserva en el cuerpo (suele contener el
    texto del artículo); solo actualiza el título de sección vigente.
    """
    sections: list[tuple[str, str]] = []
    title = ""
    lines: list[str] = []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            sections.append((title, body))
        lines.clear()

    for raw in md.splitlines():
        header = HEADER_RE.match(raw)
        if header:
            flush()
            title = header.group(2).strip()
            continue
        if PSEUDO_HEADER_RE.match(raw):
            flush()
            title = _pseudo_title(raw)
        lines.append(raw)
    flush()
    return sections


# --- 2. Unidades de empaquetado ----------------------------------------------


def _blocks(body: str) -> list[tuple[str, bool]]:
    """Párrafos y tablas ``<table>`` de una sección, en orden. bool = es tabla."""
    stashed: dict[str, str] = {}

    def _stash(m: re.Match[str]) -> str:
        key = f"\x00T{len(stashed)}\x00"
        stashed[key] = m.group(0)
        return f"\n\n{key}\n\n"

    text = TABLE_RE.sub(_stash, body)
    out: list[tuple[str, bool]] = []
    for para in re.split(r"\n\s*\n", text):
        for part in PLACEHOLDER_SPLIT_RE.split(para):
            part = part.strip()
            if not part:
                continue
            if part in stashed:
                out.append((stashed[part], True))
            else:
                out.append((part, False))
    return out


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _hard_split(text: str, count_tokens: Callable[[str], int], limit: int) -> list[str]:
    """Trocea por palabras una oración sin puntuación que excede el límite."""
    words = text.split()
    pieces: list[str] = []
    buf: list[str] = []
    for word in words:
        buf.append(word)
        if count_tokens(" ".join(buf)) >= limit:
            pieces.append(" ".join(buf))
            buf = []
    if buf:
        pieces.append(" ".join(buf))
    return pieces or [text]


def _split_table(table: str, count_tokens: Callable[[str], int], max_tokens: int) -> list[str]:
    """Trocea una tabla grande por filas ``<tr>``, repitiendo ``<thead>``."""
    thead_m = THEAD_RE.search(table)
    thead = thead_m.group(0) if thead_m else ""
    rows = [r for r in ROW_RE.findall(table) if not (thead and r in thead)]
    if not rows:
        return [table]
    base = "<table>\n" + (thead + "\n" if thead else "")
    closing = "\n</table>"
    base_tokens = count_tokens(base) + count_tokens(closing)
    out: list[str] = []
    buf: list[str] = []
    size = base_tokens
    for row in rows:
        row_tokens = count_tokens(row) + 1
        if buf and size + row_tokens > max_tokens:
            out.append(base + "\n".join(buf) + closing)
            buf, size = [], base_tokens
        buf.append(row)
        size += row_tokens
    if buf:
        out.append(base + "\n".join(buf) + closing)
    return out


def _units(
    body: str, count_tokens: Callable[[str], int], max_tokens: int
) -> list[tuple[str, bool]]:
    """Aplana la sección en unidades (texto, es_tabla) de a lo sumo max_tokens."""
    units: list[tuple[str, bool]] = []
    for text, is_table in _blocks(body):
        if is_table:
            if count_tokens(text) <= max_tokens:
                units.append((text, True))
            else:
                for sub in _split_table(text, count_tokens, max_tokens):
                    units.append((sub, True))
            continue
        if count_tokens(text) <= max_tokens:
            units.append((text, False))
            continue
        for sentence in _sentences(text):
            if count_tokens(sentence) <= max_tokens:
                units.append((sentence, False))
            else:
                units.extend(
                    (piece, False)
                    for piece in _hard_split(sentence, count_tokens, max_tokens)
                )
    return units


# --- 3. Overlap intra-sección -------------------------------------------------


def _overlap_tail(
    units: list[str], count_tokens: Callable[[str], int], overlap_tokens: int
) -> list[str]:
    """Últimas oraciones de los units que caben en overlap_tokens."""
    picked: list[str] = []
    total = 0
    for unit in reversed(units):
        for sentence in reversed(_sentences(unit) or [unit]):
            cost = count_tokens(sentence)
            if total + cost > overlap_tokens:
                return list(reversed(picked))
            picked.append(sentence)
            total += cost
    return list(reversed(picked))


def _section_chunks(
    body: str,
    count_tokens: Callable[[str], int],
    target_tokens: int,
    max_tokens: int,
    min_tokens: int,
    overlap_tokens: int,
) -> list[tuple[str, bool]]:
    """Empaqueta el cuerpo de UNA sección en chunks (texto, es_tabla)."""
    units = _units(body, count_tokens, max_tokens)
    results: list[tuple[str, bool]] = []

    current: list[str] = []
    cur_tokens = 0
    carried = False  # current proviene solo del tail repetido (sin contenido nuevo)

    def emit() -> None:
        nonlocal current, cur_tokens, carried
        if current:
            results.append(("\n".join(current), False))
        current, cur_tokens, carried = [], 0, False

    def carry_tail() -> None:
        nonlocal current, cur_tokens, carried
        tail = _overlap_tail(current, count_tokens, overlap_tokens)
        emit()
        current, carried = tail, bool(tail)
        cur_tokens = sum(count_tokens(s) for s in tail)

    for text, is_table in units:
        if is_table:
            if current and not carried:
                emit()
            else:
                current, cur_tokens, carried = [], 0, False
            results.append((text, True))
            continue
        cost = count_tokens(text)
        if current and not carried and cur_tokens + cost > max_tokens:
            carry_tail()
        elif carried and cur_tokens + cost > max_tokens:
            # el tail repetido ya no cabe junto al nuevo unit: se descarta
            current, cur_tokens, carried = [], 0, False
        current.append(text)
        carried = False
        cur_tokens += cost
        if cur_tokens >= target_tokens:
            carry_tail()
    if not carried:
        emit()

    # Fusión del último chunk pequeño con su anterior (misma sección).
    merged: list[tuple[str, bool]] = []
    for text, is_table in results:
        if (
            not is_table
            and merged
            and not merged[-1][1]
            and count_tokens(text) < min_tokens
            and count_tokens(merged[-1][0]) + count_tokens(text) <= max_tokens
        ):
            prev = merged.pop()
            merged.append((prev[0] + "\n" + text, False))
        else:
            merged.append((text, is_table))
    return merged


# --- 5. Prefijo contextual y API pública --------------------------------------


def _prefix(doc_name: str, section: str) -> str:
    if section:
        return f"Documento: {doc_name}\nSección: {section}"
    return f"Documento: {doc_name}"


def chunk_markdown(
    md: str,
    doc_name: str,
    count_tokens: Callable[[str], int],
    target_tokens: int = 400,
    max_tokens: int = 450,
    min_tokens: int = 80,
    overlap_tokens: int = 70,
) -> list[Chunk]:
    """Convierte el markdown limpio de un documento en chunks indexables.

    ``count_tokens`` debe ser el tokenizador real del modelo de embeddings
    (ver ``embeddings.token_counter``); para tests puede ser un aproximador.
    """
    chunks: list[Chunk] = []
    index = 0
    for title, body in split_sections(md):
        if not body.strip():
            continue
        prefix = _prefix(doc_name, title)
        prefix_tokens = count_tokens(prefix)
        body_target = max(32, target_tokens - prefix_tokens)
        body_max = max(body_target + 16, max_tokens - prefix_tokens)
        body_min = min(min_tokens, body_target)
        body_overlap = min(overlap_tokens, body_target // 2)
        for text, is_table in _section_chunks(
            body, count_tokens, body_target, body_max, body_min, body_overlap
        ):
            final = prefix + "\n\n" + text
            n_tokens = count_tokens(final)
            if n_tokens < 3:  # ruido puro, sin contenido semántico
                continue
            chunks.append(
                Chunk(
                    doc_name=doc_name,
                    section=title,
                    text=final,
                    chunk_index=index,
                    n_tokens=n_tokens,
                    is_table=is_table,
                )
            )
            index += 1
    return chunks
