# Document Check Agent

A small Python course project using Streamlit, LangChain, LangGraph, OpenAI,
and in-memory embeddings + BM25. No external vector database is needed.

## Run

Python 3.11–3.13 is recommended. From this folder:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your key in the existing `.env` file:

```dotenv
OPENAI_API_KEY=your-key-here
```

For a fresh clone, copy `.env.example` to `.env` first. `.env` and `.venv/`
are ignored by Git. Never put a real key in `.env.example`.

```sh
streamlit run app.py
```

Upload a PDF, modern Word `.docx`, or UTF-8 text document. Enter one rule per
line or upload a UTF-8 CSV with `rule` and optional unique `id` columns.
Review and approve the rulebook. Download results as CSV when complete.
Restart Streamlit after changing `.env` or `config.py`.

## Evaluate

The default evaluation runs the smoke test in `examples/` and both PulseCare
sets in `test_data/`, printing each score and an aggregate score.

```sh
python evaluate.py
python evaluate.py --document examples/document.txt --rules examples/rules.csv --answers examples/answers.csv --approve
```

The first command waits for approval. `--approve` explicitly approves the
rulebook for automated runs; it still resumes the same LangGraph interrupt.
The script prints reasons for verdicts and every mismatched rule. Failed sets
show the exception and produce exit status 2; totals then cover completed sets only.
OpenAI chat and embedding calls use a 60-second request timeout and one retry on timeout only. The judge uses temperature 0 (results may still vary).
The answer key needs `id,verdict` or `id,expected` columns and exactly the same IDs as the rules CSV. The answer key
is used only for scoring and is never supplied to the agent.

## How the steps work

1. `read_document` extracts text with PDF page, Word block, or text offsets,
   then splits it into overlapping chunks.
2. `load_rules` preserves user wording and validates rule IDs.
3. `ask_user` pauses with LangGraph `interrupt`; `InMemorySaver` keeps the
   thread state. Rejection ends the run without model calls.
4. `search_evidence` combines cosine similarity and BM25 with reciprocal rank
   fusion. The index is built only after approval.
5. `judge_rule` requests structured output and validates quotes against the cited chunk, allowing whitespace and
   ≤/<=, °, µ/μ/u, and dash typography differences. Returned quotes and
   character locations always use the original source text. Locations are computed locally.
6. Weak, invalid, or ambiguous evidence triggers one reworded search. Both
   attempts' evidence is considered. Related but partial or ambiguous text
   remains `UNCLEAR`. Placeholder-only or blank required sections are missing.
   Prohibitions use the full document text; no violation after the retry means
   `MET`. For requirements to provide content, no evidence after both searches becomes
   `NOT MET` with "No evidence found in document" and no quote or location.

All models and settings are in `config.py`. At most 50 rules, two searches per
rule, bounded API retries, request timeouts, and a graph step limit prevent
unbounded execution. The five operations are explicit graph nodes/functions;
the model does not choose arbitrary tools or control the loop.

## Limits

`MET` requires clear evidence of fulfillment. `NOT MET` means explicit
contradictory text or no relevant evidence after both searches. `UNCLEAR` means
related text is partial or ambiguous (or its quote could not be validated).
Retrieval can miss relevant text, so an absence verdict reports a search-based
compliance gap rather than exhaustive proof of absence. Exact quote validation
prevents invented quotations but does not guarantee the model's interpretation
is correct. Quotes match extracted text; PDF extraction can differ from visual
layout. Scanned PDFs need OCR, and legacy `.doc` files need conversion to
`.docx`. Word locations use body block numbers, not page numbers; headers,
footers, and text boxes are not extracted. In-memory state disappears on restart.
Uploaded text and rules go to OpenAI; runs require a funded API key and network.

## Offline checks

```sh
python -m unittest discover -s tests -v
```

These use fake model and embedding responses to check approval, rejection,
retry limits, quote validation, retrieval, loaders, and scoring without API costs.

Implementation references: [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
and [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
