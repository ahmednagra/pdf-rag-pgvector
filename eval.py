"""Evaluate the running service on a small labelled question set.

    python eval.py                      # against http://localhost:8080
    python eval.py --base-url http://api:8000 --top-k 5 --json eval_results.json

Steps: (1) ingest the sample PDFs (idempotent, so re-running is cheap); (2) for each question,
call /search and /answer; (3) print metrics.

Metrics
  hit_rate@1   share of answerable questions whose labelled page is the top result
  hit_rate@k   share of answerable questions whose labelled page is in the top-k results
  mrr          mean reciprocal rank of that page (1.0 = always ranked first)
  answer_match share of answerable questions whose answer contains the expected key fact
  groundedness share of answer sentences whose words are mostly found in the cited passages
  citation_rate share of answer sentences that carry a valid [n] citation
  abstention   share of unanswerable questions answered with "I don't know"
"""

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

import httpx

from app.evaluation import groundedness, is_abstention, retrieval_rank

ROOT = Path(__file__).resolve().parent


def wait_for_health(client: httpx.Client, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            if client.get("/health").status_code == 200:
                return
        except httpx.TransportError:
            pass
        if time.monotonic() > deadline:
            sys.exit(f"service at {client.base_url} is not healthy after {timeout:.0f}s")
        time.sleep(2)


def ingest(client: httpx.Client, pdf_dir: Path) -> None:
    for path in sorted(pdf_dir.glob("*.pdf")):
        with path.open("rb") as handle:
            response = client.post(
                "/documents", files={"file": (path.name, handle, "application/pdf")}
            )
        response.raise_for_status()
        body = response.json()
        print(f"  {body['status']:<9} {path.name} ({body['chunk_count']} chunks)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/eval/questions.jsonl")
    parser.add_argument("--pdf-dir", type=Path, default=ROOT / "data/sample_pdfs")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", type=Path, help="also write per-question results here")
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.questions.read_text().splitlines() if line]
    client = httpx.Client(base_url=args.base_url, timeout=120.0)
    wait_for_health(client)
    print("Ingesting sample PDFs:")
    ingest(client, args.pdf_dir)

    rows = []
    for item in questions:
        hits = client.post("/search", json={"query": item["question"], "top_k": args.top_k})
        hits.raise_for_status()
        reply = client.post("/answer", json={"question": item["question"], "top_k": args.top_k})
        reply.raise_for_status()
        body = reply.json()
        passages = {n: s["content"] for n, s in enumerate(body["sources"], start=1)}
        cited = {c["source"]: passages[c["source"]] for c in body["citations"]}
        g = groundedness(body["answer"], cited)
        answerable = item["file"] is not None
        rank = retrieval_rank(hits.json(), item["file"], item["page"]) if answerable else None
        rows.append(
            {
                "question": item["question"],
                "answerable": answerable,
                "rank": rank,
                "answer": body["answer"],
                "answer_match": answerable
                and item["answer_contains"].lower() in body["answer"].lower(),
                "abstained": is_abstention(body["answer"]),
                "groundedness": g.score,
                "citation_rate": g.citation_rate,
            }
        )

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    answered = [r for r in answerable if not r["abstained"]]
    metrics = {
        "hit_rate@1": mean(r["rank"] == 1 for r in answerable),
        f"hit_rate@{args.top_k}": mean(r["rank"] is not None for r in answerable),
        "mrr": mean(1 / r["rank"] if r["rank"] else 0.0 for r in answerable),
        "answer_match": mean(r["answer_match"] for r in answerable),
        "groundedness": mean(r["groundedness"] for r in answered) if answered else 0.0,
        "citation_rate": mean(r["citation_rate"] for r in answered) if answered else 0.0,
        "abstention": mean(r["abstained"] for r in unanswerable) if unanswerable else 0.0,
    }

    print(f"\nPer question (rank = position of the labelled page in the top {args.top_k}):")
    for r in rows:
        if r["answerable"]:
            flag = "ok " if r["answer_match"] else "MISS"
            print(f"  [{flag}] rank={r['rank'] or '-'}  {r['question']}")
        else:
            flag = "ok " if r["abstained"] else "MISS"
            print(f"  [{flag}] (unanswerable)  {r['question']}")

    print(f"\nResults ({len(answerable)} answerable, {len(unanswerable)} unanswerable questions):")
    for name, value in metrics.items():
        print(f"  {name:<14} {value:.2f}")

    if args.json:
        args.json.write_text(json.dumps({"metrics": metrics, "questions": rows}, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
