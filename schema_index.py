import sqlite3
import faiss
from sentence_transformers import SentenceTransformer
from schema_enrich import enrich_column

# embedding model
_model = SentenceTransformer("BAAI/bge-m3")

# global state
_faiss_index = None
_schema_columns = []


def build_schema_index(
    db_path: str = "data/retail.db",
    table_name: str = "transactions"
):
    """
    Build FAISS index over table schema.
    This must run before any query.
    """
    global _faiss_index, _schema_columns

    if _faiss_index is not None:
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    conn.close()

    if not columns:
        raise RuntimeError("No schema found in database")

    texts = []
    _schema_columns = []

    for col in columns:
        col_name = col[1]
        texts.append(enrich_column(col_name))
        _schema_columns.append(col_name)

    embeddings = _model.encode(
        texts,
        normalize_embeddings=True
    )

    dim = embeddings.shape[1]
    _faiss_index = faiss.IndexFlatIP(dim)
    _faiss_index.add(embeddings)


def select_schema(query: str, top_k: int = 6):
    """
    Return top-k relevant schema columns.
    """
    if _faiss_index is None:
        raise RuntimeError(
            "Schema index not initialized. "
            "Call build_schema_index() first."
        )

    query_vec = _model.encode(
        [query],
        normalize_embeddings=True
    )

    _, indices = _faiss_index.search(query_vec, top_k)

    return [_schema_columns[i] for i in indices[0]]
