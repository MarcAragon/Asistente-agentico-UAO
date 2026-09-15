from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class IngestRequest(_message.Message):
    __slots__ = ("file_match", "rebuild")
    FILE_MATCH_FIELD_NUMBER: _ClassVar[int]
    REBUILD_FIELD_NUMBER: _ClassVar[int]
    file_match: str
    rebuild: bool
    def __init__(self, file_match: _Optional[str] = ..., rebuild: _Optional[bool] = ...) -> None: ...

class IngestProgress(_message.Message):
    __slots__ = ("doc_name", "chunks", "seconds", "token_p50", "token_p90", "token_max", "done", "total_docs")
    DOC_NAME_FIELD_NUMBER: _ClassVar[int]
    CHUNKS_FIELD_NUMBER: _ClassVar[int]
    SECONDS_FIELD_NUMBER: _ClassVar[int]
    TOKEN_P50_FIELD_NUMBER: _ClassVar[int]
    TOKEN_P90_FIELD_NUMBER: _ClassVar[int]
    TOKEN_MAX_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_DOCS_FIELD_NUMBER: _ClassVar[int]
    doc_name: str
    chunks: int
    seconds: float
    token_p50: int
    token_p90: int
    token_max: int
    done: int
    total_docs: int
    def __init__(self, doc_name: _Optional[str] = ..., chunks: _Optional[int] = ..., seconds: _Optional[float] = ..., token_p50: _Optional[int] = ..., token_p90: _Optional[int] = ..., token_max: _Optional[int] = ..., done: _Optional[int] = ..., total_docs: _Optional[int] = ...) -> None: ...

class PruneResult(_message.Message):
    __slots__ = ("removed_chunks",)
    REMOVED_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    removed_chunks: int
    def __init__(self, removed_chunks: _Optional[int] = ...) -> None: ...

class IndexStatusResponse(_message.Message):
    __slots__ = ("chunks", "documents", "embedding_model", "device")
    CHUNKS_FIELD_NUMBER: _ClassVar[int]
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_MODEL_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    chunks: int
    documents: int
    embedding_model: str
    device: str
    def __init__(self, chunks: _Optional[int] = ..., documents: _Optional[int] = ..., embedding_model: _Optional[str] = ..., device: _Optional[str] = ...) -> None: ...
