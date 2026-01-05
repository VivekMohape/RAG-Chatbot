import sqlite3
import faiss
from sentence_transformers import SentenceTransformer
from schema_enrich import enrich_column

# embedding model
_model = SentenceTransformer("BAAI/bge-m3")

# global state
_index = None
_schema_cols = []


def build_schema_index(
    db_path: str = "data/retail.db",
    table_name: str = "transactions"
):
    """
    Build FAISS index over schema columns.
    Runs once per app lifecycle.
    """
    global _index, _schema_cols

    # do not rebuild if already exists
    if _index is not None:
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # fetch column names
    cursor.execute(f"PRAGMA table_info({table_name})")
    cols = cursor.fetchall()
    conn.close()

    if not cols:
        raise RuntimeError("No columns found in database table")

    texts = []
    _schema_cols = []

    for col in cols:
        col_name = col[1]
        texts.append(enrich_column(col_name))
        _schema_cols.append(col_name)

    # create embeddings
    vectors = _model.encode(
        texts,
        normalize_embeddings=True
    )

    # create FAISS index
    dim = vectors.shape[1]
    _index = faiss.IndexFlatIP(dim)
    _index.add(vectors)


def select_schema(query: str, top_k: int = 6):
    """
    Select relevant columns based on query semantics.
    """
    if _index is None:
        raise RuntimeError("Schema index not initialized")

    query_vec = _model.encode(
        [query],
        normalize_embeddings=True
    )

    _, indices = _index.search(query_vec, top_k)

    return [_schema_cols[i] for i in indices[0]]
