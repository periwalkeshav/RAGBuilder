"""A/B: does the cross-lingual retrieval failure come from the embedding model?

Run inside the API container:

    docker compose exec api python scripts/ab_embedding_models.py

Isolates one variable. Both models are asked the same question in German and in
English against three German passages, one of which is the right answer. If the
model handles cross-lingual retrieval, the English query ranks the same passage
first as the German one does.

Measured result on this corpus (see README):

    e5-small   DE query -> correct    EN query -> WRONG
    e5-large   DE query -> correct    EN query -> correct

That is the concrete cost of the -small default, and the reason to switch to
-large for a bilingual deployment.
"""

import logging

import numpy as np

logging.disable(logging.INFO)
from sentence_transformers import SentenceTransformer

QUERY_EN = "query: What is the maximum daily working time under German law?"
QUERY_DE = "query: Wie lange darf die werktägliche Arbeitszeit höchstens sein?"
PASSAGES = {
    "ArbZG §3 (correct)": "passage: § 3 Arbeitszeit der Arbeitnehmer Die werktägliche Arbeitszeit der Arbeitnehmer darf acht Stunden nicht überschreiten. Sie kann auf bis zu zehn Stunden nur verlängert werden, wenn innerhalb von sechs Kalendermonaten oder innerhalb von 24 Wochen im Durchschnitt acht Stunden werktäglich nicht überschritten werden.",
    "MuSchG §4 (retrieved)": "passage: § 4 Verbot der Mehrarbeit; Ruhezeit Der Arbeitgeber darf eine schwangere oder stillende Frau nicht mit Mehrarbeit beschäftigen. Mehrarbeit ist jede Arbeit, die über acht- einhalb Stunden täglich oder über 90 Stunden in der Doppelwoche hinaus geleistet wird.",
    "JArbSchG §8": "passage: § 8 Dauer der Arbeitszeit Jugendliche dürfen nicht mehr als acht Stunden täglich und nicht mehr als 40 Stunden wöchentlich beschäftigt werden.",
}

for name in ("intfloat/multilingual-e5-small", "intfloat/multilingual-e5-large"):
    model = SentenceTransformer(name)
    model.max_seq_length = 512
    texts = [QUERY_EN, QUERY_DE] + list(PASSAGES.values())
    v = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=8)
    print(f"\n{name}")
    for qi, qlabel in ((0, "EN query"), (1, "DE query")):
        scores = {label: float(np.dot(v[qi], v[2 + i])) for i, label in enumerate(PASSAGES)}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        winner = ranked[0][0]
        mark = "CORRECT" if "correct" in winner else "WRONG"
        print(f"  {qlabel}: top = {winner:<24} [{mark}]")
        for label, score in ranked:
            print(f"      {score:.4f}  {label}")
