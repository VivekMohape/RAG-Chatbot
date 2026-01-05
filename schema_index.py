import sqlite3
import faiss
from sentence_transformers import SentenceTransformer
from schema_enrich import enrich_column

_model = SentenceTransformer("BAAI/bge-m3")

_faiss_index = None
_schema_cols = []

def build_schema_index(db_path, table_name):
    global _faiss_index, _schema_cols

    if _faiss_index is not None:
        return

    conn = sqlite3.connect(db_path)
    cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    conn.close()

    texts = []
    _schema_cols = []

    for col in cols:
        texts.append(enrich_column(col[1]))
        _schema_cols.append(col[1])

    embeddings = _model.encode(texts, normalize_embeddings=True)

    _faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    _faiss_index.add(embeddings)

def select_schema(query, top_k=6):
    if _faiss_index is None:
        raise RuntimeError("Schema index not built")

    q_vec = _model.encode([query], normalize_embeddings=True)
    _, ids = _faiss_index.search(q_vec, top_k)

    return [_schema_cols[i] for i in ids[0]]
