import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "day-5"))
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("JWT_SECRET", "eval-secret")

from app.config import get_settings
from app.db.database import SessionLocal, engine
from app.db.models import Base, Chunk, Document, User
from app.security import hash_password
from app.services.eval import EvalCase, aggregate, score_case
from app.services.rag import chunk_text, embed_text, generate_answer, retrieve


async def seed(session, dataset) -> int:
    user = User(username="eval-user", hashed_password=hash_password("password123"))
    session.add(user)
    await session.flush()
    for item in dataset:
        doc = Document(title=item["document_title"], owner_id=user.id)
        session.add(doc)
        await session.flush()
        for idx, chunk in enumerate(chunk_text(item["document_content"])):
            session.add(Chunk(document_id=doc.id, chunk_index=idx, text=chunk, embedding=embed_text(chunk)))
    await session.commit()
    return user.id


async def run() -> dict:
    settings = get_settings()
    dataset = json.loads((Path(__file__).parent / "dataset.json").read_text())

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with SessionLocal() as session:
        await seed(session, dataset)

        from sqlalchemy import select

        all_chunks = list((await session.execute(select(Chunk))).scalars().all())
        all_docs = {d.id: d.title for d in (await session.execute(select(Document))).scalars().all()}

        results = []
        for item in dataset:
            case = EvalCase(
                query=item["query"],
                expected_keywords=item["expected_keywords"],
                relevant_document_title=item["document_title"],
            )
            query_embedding = embed_text(item["query"])
            retrieved = retrieve(query_embedding, all_chunks, top_k=settings.top_k)
            answer = generate_answer(item["query"], retrieved)
            retrieved_titles = [all_docs[c.document_id] for c, _ in retrieved]
            result = score_case(case, retrieved_titles, answer)
            results.append(result)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    overall = aggregate(results)
    report = {
        "overall_score": round(overall, 4),
        "baseline": settings.eval_baseline_score,
        "passed": overall >= settings.eval_baseline_score,
        "cases": [
            {
                "query": r.query,
                "retrieval_hit": r.retrieval_hit,
                "keyword_overlap": round(r.keyword_overlap, 4),
                "score": round(r.score, 4),
            }
            for r in results
        ],
    }
    return report


def main() -> None:
    report = asyncio.run(run())
    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
