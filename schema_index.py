import faiss
from sentence_transformers import SentenceTransformer
from schema_enrich import enrich_column

model = SentenceTransformer("BAAI/bge-m3")
schema_vectors = []
schema_meta = []
index = None

def build_schema_index(columns):
    global index
    texts = [enrich_column(c) for c in columns]
    vectors = model.encode(texts, normalize_embeddings=True)
    index = faiss.IndexFlatIP(len(vectors[0]))
    index.add(vectors)
    schema_meta.extend(columns)

def select_schema(query, k=6):
    q_vec = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(q_vec, k)
    return [schema_meta[i] for i in ids[0]]
