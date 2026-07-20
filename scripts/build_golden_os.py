"""풀링 골든셋 빌더(025 G2) — 판정 provenance + 재검색(rank 조인)으로 `golden_os.json` 생성.

2026-06-11 전수 평가 판정(`golden_os.judgments.json`, 1392행)을 정답으로 변환한다:
질의마다 3채널(KoSimCSE/BGE-M3/KURE-v1) top-8 을 **결정적으로 재검색**해 (channel, rank)→asset_id 를
복원하고, 'valid' 판정 자산의 합집합을 그 질의의 정답(IR 풀링 관례)으로 둔다. partial 은 별도 보존.
absent 질의는 `expect_empty: true`. 토픽 태그(coverage guard 용)는 코퍼스 파일명 슬러그 1토큰 기준.

⚠️ 운영 규칙: fixture 의 ``topics`` 는 손수정 금지 — 본 빌더의 ``_QUERY_TOPICS`` 가 정본이며 재생성으로만
갱신한다(이중 보유 divergence 방지). coverage guard(RUN_OS_E2E) 실패 시 고칠 곳도 여기다: 신규 토픽
질의를 judgments 에 추가·판정 후 ``_QUERY_TOPICS`` 에 토픽을 등재하고 재생성한다.

읽기 전용(헌법 6조)·temperature 무관(임베딩 결정적) — 같은 코퍼스에서 2회 실행 동일 출력.

실행: conda run -n AuroraFS python scripts/build_golden_os.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 질의 id → 커버 토픽(코퍼스 파일명 슬러그 1토큰). coverage guard 가 코퍼스 토픽과 대조한다.
_QUERY_TOPICS: dict[str, list[str]] = {
    "P01": ["등산"], "P02": ["재활용"], "P03": ["전기차"], "P04": ["태양광"], "P05": ["수영"],
    "P06": ["반려식물"], "P07": ["커피"], "P08": ["라면"], "P09": ["천체"], "P10": ["보드게임"],
    "P11": ["목공"], "P12": ["무선"], "P13": ["자전거"], "P14": ["텃밭"], "P15": ["낚시"],
    "P16": ["드론"], "P17": ["기타"], "P18": ["주식"], "P19": ["김치"], "P20": ["베이킹"],
    "P21": ["에어컨"], "P22": ["캠핑"], "P23": ["인테리어"], "P24": ["골프"], "P25": ["요가"],
    "P26": ["선거"], "P27": ["술과"], "P28": ["다이어트"],
    "E01": ["스마트폰"], "E02": ["스마트폰"], "E03": ["무선", "전기차"], "E04": ["천체"],
    "E05": ["낚시"], "E06": ["김치"],
}

CHANNELS = [
    ("KoSimCSE", "st", "BM-K/KoSimCSE-roberta-multitask"),
    ("BGE-M3", "st_bge", "BAAI/bge-m3"),
    ("KURE-v1", "st_kure", "nlpai-lab/KURE-v1"),
]

# 전수 평가 하니스와 동일한 자산 단위 top-8 SQL(MAX 청크 코사인·NaN 제외) — rank 재현의 정본.
SQL = """SELECT t.asset_id, MAX(t.sim) AS sim FROM (
  SELECT a.asset_id, 1-(ae.embedding <=> %(qv)s::vector) AS sim
  FROM asset a
  JOIN asset_embedding ae ON ae.asset_id=a.asset_id AND ae.channel=%(ch)s AND ae.embedding IS NOT NULL
  WHERE a.status='registered'
) t WHERE t.sim <> double precision 'NaN'
GROUP BY t.asset_id ORDER BY sim DESC LIMIT 8"""


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env.dev", override=False)
    from src.config.settings import get_current_settings, init_settings

    init_settings("dev")
    from src.database.postgres_util import PostgresUtil
    from src.embedders.text_embedder import embed_texts, pad_embedding_to_storage_dim

    cfg = get_current_settings()
    fx = _REPO_ROOT / "tests" / "fixtures" / "search"
    prov = json.loads((fx / "golden_os.judgments.json").read_text(encoding="utf-8"))
    queries = prov["queries"]
    verdicts: dict[str, str] = prov["verdicts"]

    texts = [q["query"] for q in queries]
    qvecs: dict[str, list[str]] = {}
    for _name, ch, model in CHANNELS:
        raws = embed_texts(texts, model_name=model, normalize_embeddings=cfg.embed.normalize)
        qvecs[ch] = [
            "[" + ",".join(repr(float(x)) for x in pad_embedding_to_storage_dim(list(r))) + "]"
            for r in raws
        ]
        print(f"[embed] {_name} 완료", file=sys.stderr)

    golden = {"version": "2026-06-11-pooled-v1", "provenance": "golden_os.judgments.json", "queries": []}
    db = PostgresUtil()
    with db, db.connection() as conn:
        for qi, q in enumerate(queries):
            entry: dict = {
                "id": q["id"], "query": q["query"], "category": q["category"],
                "topics": _QUERY_TOPICS.get(q["id"], []),
            }
            if q["category"] == "absent":
                entry["expect_empty"] = True
            else:
                relevant: set[str] = set()
                partial: set[str] = set()
                for name, ch, _m in CHANNELS:
                    with conn.cursor() as cur:
                        cur.execute(SQL, {"qv": qvecs[ch][qi], "ch": ch})
                        rows = cur.fetchall()
                    for rank, (aid, _sim) in enumerate(rows, start=1):
                        v = verdicts.get(f"{q['id']}|{name}|{rank}")
                        if v == "valid":
                            relevant.add(str(aid))
                        elif v == "partial":
                            partial.add(str(aid))
                entry["relevant"] = sorted(relevant)
                entry["partial"] = sorted(partial - relevant)
            golden["queries"].append(entry)
            print(f"[golden] {q['id']} 완료", file=sys.stderr)

    out = fx / "golden_os.json"
    out.write_text(json.dumps(golden, ensure_ascii=False, indent=1), encoding="utf-8")
    n_rel = sum(len(e.get("relevant", [])) for e in golden["queries"])
    print(f"DONE {out} — 질의 {len(golden['queries'])} · 정답 자산 연 {n_rel}")


if __name__ == "__main__":
    main()
