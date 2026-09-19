"""CLI: parsea todos los PDFs oficiales UAO con LlamaCloud Parse -> Data/Documentos_MD/*.md.

Reemplaza el pipeline anterior (PyMuPDF + EasyOCR): el servicio agentic de
LlamaCloud Parse resuelve internamente OCR y tablas, y devuelve markdown
limpio que es el formato de entrada del pipeline RAG.

Uso:
    uv run python scripts/llama_cloud_parsing.py             # todos los PDFs
    uv run python scripts/llama_cloud_parsing.py --file X    # coincidencia parcial de nombre
    uv run python scripts/llama_cloud_parsing.py --redo      # reprocesa aunque exista el .md

La API key se lee de la variable de entorno LLAMA_CLOUD_API_KEY (o del
archivo .env en la raíz del proyecto; ver .env.example).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from llama_cloud import LlamaCloud

from asistente_agentico_uao.core.config import settings


def load_api_key() -> str:
    """Obtiene LLAMA_CLOUD_API_KEY vía Settings (entorno o archivo .env)."""
    key = settings.llama_cloud_api_key.strip()
    if key:
        return key
    print(
        "ERROR: LLAMA_CLOUD_API_KEY no está configurada. "
        "Defínela en .env (copia .env.example) o en el entorno.",
        file=sys.stderr,
    )
    raise SystemExit(1)


# Ruido típico del markdown de LlamaParse en documentos UAO:
# etiquetas de elementos visuales, numeración de página huérfana e imágenes.
_NOISE_PATTERNS = [
    # Etiquetas de elementos no textuales: "logo: ...", "signature: ..."
    re.compile(
        r"^\s*(logo|signature|stamp|handwriting|handwritten)\s*:.*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    # Numeración de página huérfana: "Página 1 de 2"
    re.compile(
        r"^\s*p[áa]gina\s+\d+(\s+de\s+\d+)?\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    # Imágenes embebidas ![alt](url): sin valor para el RAG textual
    re.compile(r"!\[[^\]]*\]\([^)]*\)"),
]


def clean_markdown(md: str) -> str:
    """Limpieza ligera del markdown de LlamaParse.

    El servicio agentic ya entrega texto con ortografía, acentos y tablas
    correctos; aquí solo se retira el ruido de layout y se normaliza el
    espaciado. Conserva encabezados markdown (##, ###) porque serán las
    fronteras duras del chunking (Fase 2).
    """
    for pattern in _NOISE_PATTERNS:
        md = pattern.sub("", md)
    md = md.replace("\u00a0", " ")  # espacios no separables
    md = re.sub(r"[ \t]+", " ", md)  # colapsa espacios horizontales
    md = re.sub(r" ?\n ?", "\n", md)  # sin espacios en bordes de línea
    md = re.sub(r"\n{3,}", "\n\n", md)  # máximo una línea en blanco
    return md.strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parsea los PDFs de Documentos/ a markdown con LlamaCloud Parse."
    )
    parser.add_argument("--docs-dir", type=Path, default=settings.docs_dir)
    parser.add_argument("--out-dir", type=Path, default=settings.markdown_dir)
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Procesa un solo PDF (coincidencia parcial de nombre, sin distinguir mayúsculas)",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Reprocesa PDFs aunque ya exista su .md de salida",
    )
    args = parser.parse_args()

    pdfs = sorted(args.docs_dir.glob("*.pdf"))
    if args.file:
        pdfs = [p for p in pdfs if args.file.lower() in p.name.lower()]
    if not pdfs:
        print(f"No se encontraron PDFs en {args.docs_dir}")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        p for p in pdfs if args.redo or not (args.out_dir / f"{p.stem}.md").exists()
    ]
    skipped = len(pdfs) - len(pending)
    if skipped:
        print(f"Omitidos (ya parseados): {skipped} — usa --redo para reprocesar")
    if not pending:
        print("Nada por procesar.")
        return 0

    client = LlamaCloud(api_key=load_api_key())
    print(f"Procesando {len(pending)} PDF(s) -> {args.out_dir}\n")

    failures = 0
    for pdf in pending:
        out_path = args.out_dir / f"{pdf.stem}.md"
        started = time.perf_counter()
        try:
            file_obj = client.files.create(file=str(pdf), purpose="parse")
            result = client.parsing.parse(
                file_id=file_obj.id,
                tier="agentic",
                version="latest",
                expand=["markdown_full"],
            )
            md = clean_markdown(result.markdown_full or "")
            if not md:
                raise ValueError("LlamaCloud devolvió markdown vacío")
            out_path.write_text(md, encoding="utf-8")
            elapsed = time.perf_counter() - started
            print(
                f"OK  {pdf.name} -> {out_path.name} ({len(md)} chars, {elapsed:.1f}s)"
            )
            # Limpieza: borrar el archivo subido para no acumular en LlamaCloud
            try:
                client.files.delete(file_id=file_obj.id)
            except Exception as del_exc:  # noqa: BLE001 - no interrumpe el pipeline
                print(
                    f"AVISO  {pdf.name}: no se pudo borrar el archivo remoto: {del_exc}"
                )
        except Exception as exc:  # noqa: BLE001 - reportar y continuar
            failures += 1
            print(f"ERROR {pdf.name}: {exc}", file=sys.stderr)

    print(
        f"\nListo. {len(pending) - failures}/{len(pending)} markdown en {args.out_dir}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
