"""Step 3: Combine keyword and semantic rankings without a database."""
import re
import numpy as np
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
import config
from api_calls import call_with_timeout_retry


def tokens(text):
    return re.findall(r'\w+', text.lower()) or ['']


class HybridSearch:
    def __init__(self, chunks, embeddings=None):
        self.chunks = chunks
        self.embeddings = embeddings or OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL, max_retries=config.MAX_RETRIES,
            request_timeout=config.REQUEST_TIMEOUT)
        self.vectors = np.asarray(call_with_timeout_retry(self.embeddings.embed_documents,
            [c.page_content for c in chunks]), dtype=float)
        self.bm25 = BM25Okapi([tokens(c.page_content) for c in chunks])

    def search(self, query):
        query_vector = np.asarray(call_with_timeout_retry(self.embeddings.embed_query, query), dtype=float)
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vector)
        cosine = (self.vectors @ query_vector) / np.maximum(norms, 1e-12)
        keyword = self.bm25.get_scores(tokens(query))
        scores = np.zeros(len(self.chunks))
        # Reciprocal rank fusion avoids mixing incomparable raw scores.
        for values in (cosine, keyword):
            ranking = sorted(range(len(values)), key=lambda i: (-values[i], i))
            for rank, index in enumerate(ranking, start=1):
                scores[index] += 1 / (config.RRF_CONSTANT + rank)
        best = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:config.TOP_K]
        return [self.chunks[i] for i in best]
