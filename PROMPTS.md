# Project prompt log

## 2026-10-25 — Correct compliance verdict logic

```text
One correction to the verdict logic: in compliance checking, a required item
that is absent is a gap. So:
- MET: clear evidence the requirement is fulfilled
- NOT MET: evidence contradicts the rule, OR nothing relevant is found after
  both searches (original + reworded) - explain "No evidence found in document"
- UNCLEAR: related text exists but is partial or ambiguous
Please update judge_rule and the agent flow accordingly.
```

Dates use Europe/Berlin time. This file records the user's project prompts
verbatim; keep adding future project prompts here for the course write-up.

## 2026-10-25 — Initial project request

```text
I'm building my Week 3 course project: an AI agent called "Document Check Agent".
I have about 6 hours, so keep it simple and working over perfect.

What it does:
A user uploads a document (PDF, Word or text) and a list of rules (typed or a CSV).
An agent checks the document rule by rule and returns MET / NOT MET / UNCLEAR
for each rule, with the exact quote and where it was found.

Stack (please use exactly this):
- Python, LangChain + LangGraph, OpenAI models, Streamlit for the screen
- Hybrid search: embeddings + BM25 (in memory is fine, no external database)
- Put all model names and settings (chat model, embedding model, chunk size,
  overlap, top-k, max retries) in one file called config.py

The agent (LangGraph) needs these tools:
1. read_document – load and chunk the uploaded file
2. load_rules – read the rules the user gave
3. search_evidence – hybrid search for one rule
4. judge_rule – decide MET / NOT MET / UNCLEAR with the quote
5. ask_user – pause and ask the user a question (use LangGraph interrupt)

Rules for the agent:
- Before checking, show the rulebook and wait for the user to approve it
  (human in the loop, using interrupt + an in-memory checkpointer)
- If the evidence for a rule is weak, search once more with different wording;
  if still weak, mark it UNCLEAR. Never guess.
- Set a maximum number of loops so it can't run forever.
- Never invent a quote: every quote must exist in the document.

Also:
- Create a file PROMPTS.md and add every prompt I give you to it, with the date,
  so I can use them in my project write-up.
- Organise the code by step with plain-English comments.

First, show me a short plan (folder structure + steps) in under 15 lines.
Don't write code until I say "go".
```

## 2026-10-25 — Additions and approval to build

```text
Looks good. Two additions:
1. Store the OpenAI key in a .env file, add a .env.example, and a .gitignore
   so .env and .venv are never pushed to GitHub.
2. Add evaluate.py: it runs the agent on a test document + rules CSV,
   compares the verdicts with an answer key CSV, and prints the score
   (e.g. 13/16 correct) plus the rules it got wrong.
Use a virtual environment in .venv. Then go.
```

## 2026-10-25 — Topic relevance, retry judgment, and verdict reasons

```text
The judge returned UNCLEAR for R4 (governing law) although the document has
no governing-law clause. Fix:
1. In the judge prompt: text only counts as "related" if it addresses the
   same topic as the rule. General contract text is NOT related.
   If nothing addresses the topic -> NOT MET, "No evidence found in document".
2. Make the final verdict use the second (retry) judgment, not the strictest
   of both.
3. Add a short "reason" field to every verdict and print it in evaluate.py,
   so I can see why each rule got its verdict.
Then run the offline tests.
```

## 2026-10-25 — PulseCare evaluation and API failures

```text
I added a test_data folder with two PulseCare test sets (technical file and
test report): a .docx, a rules CSV and an answer key CSV each.
1. Update evaluate.py so it runs both sets and the smoke test, and prints the
   score per set and in total, with the reason for every wrong verdict.
2. R4 flips between NOT MET and UNCLEAR between runs. When the judge finds no
   relevant evidence and returns no quote, treat it as absence (NOT MET), not as
   a failed quote check. Set the judge's temperature to 0 for stable results.
3. If an OpenAI call fails or hangs, show the error instead of waiting forever.
Then run the offline tests.
```

## 2026-10-25 — Table quotes, placeholders, prohibitions, timeout retry

```text
Results: TF4 still fails the quote check, TF5 and TF7 are judged UNCLEAR,
and the test report timed out. Please:
1. Make sure the earlier fix (DOCX tables as text + tolerant quote matching
   for ≤ ° µ – ) is applied, and tell me if it was the cause.
2. Judge prompt: a section that only says "To be completed", "TBD" or is
   blank counts as missing -> NOT MET.
3. Judge prompt: for prohibition rules ("must not ..."), if nothing in the
   document violates the rule -> MET.
4. Raise the request timeout to 60 seconds and retry once on timeout.
Then run the offline tests.
```
