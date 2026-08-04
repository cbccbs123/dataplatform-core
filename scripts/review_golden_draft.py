"""골든 초안 사람 검수 — 초안 쌍을 하나씩 보여주고 채택·기각·종류 수정을 받는다.

**왜 사람이 해야 하나**: 031 ADR 결정1 이 *"silver 자동채택 금지"* 를 못 박았다. `curate` 가 만든
초안은 **시스템이 이미 만든 관계**(고신뢰 `graph_edge`)와 경로 신호 쌍을 긁어온 것이므로, 그대로
쓰면 "시스템이 자기 출력을 정답으로 삼는" 순환이 된다. 사람이 걸러야 골든이 된다.

**흐름에서의 위치**
    `measure_relation_quality curate` → **이 스크립트(사람 검수)** → `relation_golden.json`
    → `measure_relation_quality snapshot`/`measure` 가 그 골든으로 품질을 잰다.

**깨지면 안 되는 것**
- **매 응답마다 저장한다.** 69쌍을 30분 검수하다 끊기면 처음부터 다시 하게 된다 — 실제로
  중단 위험이 큰 작업이라 재개 가능해야 한다(`--resume`).
- 산출 골든에서 `_review`·`_NOTE` 같은 **초안 표식을 제거**한다(031 ADR). 남아 있으면 검수를
  안 한 초안과 구분되지 않는다.
- 기각한 쌍은 **버리지 말고 기록**한다. "사람이 아니라고 판단한 쌍"은 오탐 분석의 근거다.

**주의**: 표시에 자산 요약을 쓴다 — 화면 출력이므로 파일에는 저장하지 않는다(개인정보 노출면).

실행
    conda activate AuroraFS
    python scripts/review_golden_draft.py --env dev --draft <초안.json> --out <골든.json>
    python scripts/review_golden_draft.py --env dev --draft <초안.json> --out <골든.json> --resume
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from psycopg.rows import dict_row

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 검수자가 고를 수 있는 관계 종류 — DB `relation_kind` 의 활성 5종과 같아야 한다.
_KINDS = ("duplicate_near", "same_domain", "references", "derived_from", "same_series")

_HELP = """
  y = 관계 맞음(제안 종류 그대로 채택)      n = 관계 아님(기각)
  k = 관계는 맞지만 종류를 바꾼다            s = 판단 보류(골든에 넣지 않음)
  ? = 이 도움말                              q = 중단(여기까지 저장 — 나중에 --resume)
"""

_ASSET_SQL = """
    SELECT a.fs_path, a.modality, regexp_replace(a.fs_path,'^.*/','') AS name,
           coalesce(t.topic_ko,'-') AS topic, coalesce(t.subtopic_ko,'-') AS subtopic,
           left(coalesce(m.ext_meta->>'summary',''), 320) AS summary
    FROM asset a
    LEFT JOIN asset_metadata m ON m.asset_id = a.asset_id
    LEFT JOIN asset_topic t ON t.asset_id = a.asset_id
    WHERE a.fs_path = ANY(%s)
"""


def fetch_asset_info(db: Any, paths: list[str]) -> dict[str, dict]:
    """검수 화면에 쓸 자산 내용을 읽는다(읽기 전용).

    Args:
        db: DB 핸들.
        paths: 초안에 등장하는 자산 경로 전체.

    Returns:
        ``{fs_path: 행}``. 못 찾은 경로는 키가 없다(호출부가 건너뛴다).
    """
    with db.transaction() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_ASSET_SQL, (paths,))
        # ⚠️ dict_row 커서다 — dict(cur.fetchall()) 로 감싸면 키/값이 뒤집힌다.
        return {r["fs_path"]: r for r in cur.fetchall()}


def display_name(path: str, info: dict[str, dict]) -> str:
    """자산 한 건을 한 줄로 — 재수집 명명(`<uuid>__<제목>`)에서 제목만 남긴다.

    Args:
        path: 자산 경로.
        info: ``fetch_asset_info`` 결과.

    Returns:
        ``제목(모달리티)`` 형태. 정보가 없으면 경로 끝부분.
    """
    r = info.get(path)
    if not r:
        return f"{path[-50:]}(?)"
    nm = str(r["name"])
    nm = nm.split("__", 1)[-1] if "__" in nm else nm
    return f"{nm[:52]}({r['modality']})"


def render_pair(idx: int, total: int, pr: dict, info: dict[str, dict]) -> str:
    """한 쌍을 검수 화면 문자열로 만든다(순수 함수).

    ⚠️ **요약을 함께 보여준다.** 이름만으로는 "제주도 vs 섬 일반" 같은 개체 수준 구분이 안 된다 —
    앞선 측정에서 판정 사유 36자만 주는 화면이 건당 30초의 원인이었다.

    Args:
        idx: 1-based 순번.
        total: 전체 건수.
        pr: 초안 쌍(``a``·``b``·``kind``).
        info: 자산 내용 맵.

    Returns:
        출력할 여러 줄 문자열.
    """
    a, b = info.get(pr["a"], {}), info.get(pr["b"], {})
    return "\n".join([
        "",
        f"[{idx}/{total}]  제안 종류: {pr.get('kind') or '?'}",
        f"  A  {display_name(pr['a'], info)}",
        f"     주제 {a.get('topic','-')}>{a.get('subtopic','-')} · {str(a.get('summary',''))[:150]}",
        f"  B  {display_name(pr['b'], info)}",
        f"     주제 {b.get('topic','-')}>{b.get('subtopic','-')} · {str(b.get('summary',''))[:150]}",
    ])


def build_golden(decided: list[dict], isolated: list[str]) -> dict:
    """검수 결과를 골든 dict 로 조립한다(순수 함수).

    초안 표식(``_review``·``note``)을 **제거**한다 — 031 ADR: 검수를 마친 골든과 초안이
    구분되지 않으면 자동채택 금지 규율이 무의미해진다.

    Args:
        decided: 채택된 쌍 목록(``a``·``b``·``kind`` 만 남긴다).
        isolated: 고립 자산 경로 목록.

    Returns:
        `parse_golden` 이 받는 형태의 dict.
    """
    return {
        "version": 1,
        "key_type": "fs_path",
        "pairs": [{"a": p["a"], "b": p["b"], "kind": p["kind"]} for p in decided],
        "isolated": list(isolated),
    }


def _load_progress(path: Path) -> dict:
    """중간 저장을 읽는다. 없으면 빈 진행 상태.

    Args:
        path: 진행 파일 경로.

    Returns:
        ``{answers: {인덱스: 결정}}``.
    """
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"answers": {}}


def main(argv: list[str] | None = None) -> int:
    """초안을 사람에게 보여주고 검수 결과로 골든을 만든다.

    Args:
        argv: 명령행 인자. ``None`` 이면 실제 인자를 읽는다(테스트 주입용).

    Returns:
        0=완료(골든 저장), 2=중단(진행 저장 — ``--resume`` 로 재개).
    """
    ap = argparse.ArgumentParser(description="골든 초안 사람 검수(031 ADR — silver 자동채택 금지)")
    ap.add_argument("--env", choices=["dev", "prod"], default="dev",
                    help="설정 프로파일(기본: dev). .env.<env> 를 읽어 초기화한다")
    ap.add_argument("--draft", required=True, help="curate 가 만든 초안 JSON 경로")
    ap.add_argument("--out", required=True, help="검수 완료 골든 저장 경로")
    ap.add_argument("--limit", type=int, default=None,
                    help="검수할 쌍 수 상한(미지정=초안 전체). 층화 선별본을 넣었다면 불필요")
    ap.add_argument("--isolated-limit", dest="iso_limit", type=int, default=10,
                    help="골든에 담을 고립 자산 수 상한(기본 10). 초안 고립은 수백 건이라 전수는 비현실")
    ap.add_argument("--resume", action="store_true", help="중단한 지점부터 재개")
    args = ap.parse_args(argv)

    from dotenv import load_dotenv

    from src.config.settings import init_settings
    from src.database.postgres_util import PostgresUtil

    dotenv_path = _REPO_ROOT / f".env.{args.env}"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    init_settings(args.env)

    draft = json.loads(Path(args.draft).read_text(encoding="utf-8"))
    pairs = draft.get("pairs", [])[: args.limit] if args.limit else draft.get("pairs", [])
    if not pairs:
        print("⛔ 초안에 검수할 쌍이 없다.")
        return 2

    prog_path = Path(args.out).with_suffix(".progress.json")
    prog = _load_progress(prog_path) if args.resume else {"answers": {}}
    answers: dict[str, Any] = dict(prog.get("answers", {}))

    db = PostgresUtil()
    with db:
        info = fetch_asset_info(db, sorted({p for pr in pairs for p in (pr["a"], pr["b"])}))

    print(f"\n골든 초안 검수 — {len(pairs)}쌍 (이미 결정 {len(answers)}건)")
    print(_HELP)
    stopped = False
    for i, pr in enumerate(pairs, 1):
        if str(i) in answers:
            continue
        print(render_pair(i, len(pairs), pr, info))
        while True:
            key = input("  > ").strip().lower()
            if key == "?":
                print(_HELP)
                continue
            if key == "q":
                stopped = True
                break
            if key in ("y", "n", "s"):
                answers[str(i)] = {"decision": key, "kind": pr.get("kind")}
                break
            if key == "k":
                print("     종류: " + " / ".join(f"{n+1}={k}" for n, k in enumerate(_KINDS)))
                sel = input("     번호 > ").strip()
                if sel.isdigit() and 1 <= int(sel) <= len(_KINDS):
                    answers[str(i)] = {"decision": "y", "kind": _KINDS[int(sel) - 1]}
                    break
                print("     번호가 잘못됐다.")
                continue
            print("  y / n / k / s / ? / q 중 하나를 입력하라.")
        # 매 응답마다 저장 — 30분 검수가 중단으로 날아가지 않게.
        prog_path.write_text(json.dumps({"answers": answers}, ensure_ascii=False), encoding="utf-8")
        if stopped:
            break

    decided = [{"a": pairs[int(i) - 1]["a"], "b": pairs[int(i) - 1]["b"], "kind": v["kind"]}
               for i, v in sorted(answers.items(), key=lambda kv: int(kv[0]))
               if v["decision"] == "y"]
    rejected = [i for i, v in answers.items() if v["decision"] == "n"]
    skipped = [i for i, v in answers.items() if v["decision"] == "s"]

    if stopped:
        print(f"\n중단 — 결정 {len(answers)}/{len(pairs)}건 저장됨. 재개: --resume")
        print(f"  진행 파일: {prog_path}")
        return 2

    golden = build_golden(decided, draft.get("isolated", [])[: args.iso_limit])
    Path(args.out).write_text(json.dumps(golden, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n완료 — 채택 {len(decided)} · 기각 {len(rejected)} · 보류 {len(skipped)}")
    print(f"  고립 {len(golden['isolated'])}건 포함(초안 {len(draft.get('isolated', []))}건 중 상한 적용)")
    print(f"  골든 저장: {args.out}")
    if rejected:
        # 기각분은 오탐 분석 근거다 — 버리지 않고 남긴다.
        rej_path = Path(args.out).with_suffix(".rejected.json")
        rej_path.write_text(json.dumps(
            {"rejected_indexes": rejected,
             "pairs": [{"a": pairs[int(i) - 1]["a"], "b": pairs[int(i) - 1]["b"],
                        "proposed_kind": pairs[int(i) - 1].get("kind")} for i in rejected]},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  기각분 기록: {rej_path} (오탐 분석 근거)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
