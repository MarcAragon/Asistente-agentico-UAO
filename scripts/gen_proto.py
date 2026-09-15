"""Regenera los stubs gRPC desde ``grpc_impl/protos/index_admin.proto``.

Uso:
    uv run python scripts/gen_proto.py

Los stubs generados (``index_admin_pb2.py``/``index_admin_pb2_grpc.py`` y
sus ``.pyi``) se commitean para que no haga falta regenerarlos en cada
clon; correr este script solo es necesario si cambia el .proto.

Dos ajustes que hace este script sobre la salida cruda de protoc:

1. ``--pyi_out``: los mensajes de ``_pb2.py`` se construyen dinámicamente
   en runtime (``_builder.BuildMessageAndEnumDescriptors``), por lo que el
   analizador estático (Pylance/mypy) no vería ``IngestRequest``, ``Empty``,
   etc. Los ``.pyi`` declaran esas clases y eliminan esos avisos.
2. grpc_tools emite un import absoluto (``import index_admin_pb2``) que
   rompe dentro del paquete: se reescribe al import del paquete instalado.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from grpc_tools import protoc

ROOT = Path(__file__).resolve().parents[1]
PROTOS = ROOT / "src" / "asistente_agentico_uao" / "grpc_impl" / "protos"
STUBS = ROOT / "src" / "asistente_agentico_uao" / "grpc_impl" / "stubs"

STALE_IMPORT = "import index_admin_pb2 as index__admin__pb2"
FIXED_IMPORT = (
    "from asistente_agentico_uao.grpc_impl.stubs import ("
    "index_admin_pb2 as index__admin__pb2)"
)


def main() -> int:
    STUBS.mkdir(parents=True, exist_ok=True)
    args = [
        f"-I{PROTOS}",
        f"--python_out={STUBS}",
        f"--grpc_python_out={STUBS}",
        f"--pyi_out={STUBS}",  # tipos para el analizador estático (Pylance)
        "index_admin.proto",  # ruta relativa al -I (exigencia de protoc)
    ]
    cwd = os.getcwd()
    os.chdir(PROTOS)
    try:
        code = protoc.main(args)
    finally:
        os.chdir(cwd)
    if code != 0:
        print("ERROR: protoc falló; revisa el .proto", file=sys.stderr)
        return 1

    grpc_file = STUBS / "index_admin_pb2_grpc.py"
    text = grpc_file.read_text(encoding="utf-8")
    if STALE_IMPORT in text:
        text = text.replace(STALE_IMPORT, FIXED_IMPORT)
        grpc_file.write_text(text, encoding="utf-8")
    print(f"Stubs gRPC regenerados en {STUBS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
