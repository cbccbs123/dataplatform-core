"""OpenSearch 동기화 단위 테스트 — DB·OS·opensearch-py 실서버 불필요.

G1 순수 함수(`build_index_body`·`asset_to_doc`·`parse_vector`)는 결정적이고 입력만으로
출력이 정해지므로 CI 단위 게이트에서 항상 돈다.

G2 색인 seam(`ensure_index`·`index_asset`·`sync_all`)은 **가짜 클라이언트/가짜 conn 주입**으로
OS·DB 없이 호출 인자(인덱스 매핑·`_id=asset_id` upsert·읽기전용 SELECT 파라미터)를 검증한다 —
IO 함수의 옳은 액션 조립만 단위로 보증하고, 실제 색인 결과는 G5(실OS·실DB e2e) 책임.
"""
from __future__ import annotations

import unittest

from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION
from src.search.opensearch_sync import (
    asset_to_doc,
    build_index_body,
    clean_file_name,
    parse_vector,
)


class TestParseVector(unittest.TestCase):
    def test_list_passthrough(self) -> None:
        # 이미 리스트면 float 로 정규화해 통과.
        self.assertEqual(parse_vector([0.1, 0.2, 0.3]), [0.1, 0.2, 0.3])

    def test_pgvector_string(self) -> None:
        # pgvector 텍스트 표현 '[..]' 를 float 리스트로 파싱(음수 포함).
        self.assertEqual(parse_vector("[0.5,-0.25,1.0]"), [0.5, -0.25, 1.0])

    def test_empty_and_whitespace(self) -> None:
        # 빈 벡터·공백 토큰을 안전하게 처리.
        self.assertEqual(parse_vector("[]"), [])
        self.assertEqual(parse_vector("[ 0.1 , 0.2 ]"), [0.1, 0.2])


class TestAssetToDoc(unittest.TestCase):
    def _row(self, **over):
        row = {
            "asset_id": "a1",
            "modality": "video",
            "domain_label": "general",
            "status": "registered",
            "fs_path": "/data/sub/무선_충전기_xyz.mp4",
            "ext_meta": {
                "summary": "무선 충전기 리뷰",
                "keywords": ["충전기", "Qi2"],
                "labels": ["전자제품"],
            },
            "emb": "[0.1,0.2,0.3]",
            "chunk_count": 7,
        }
        row.update(over)
        return row

    def test_doc_shape_and_fields(self) -> None:
        doc = asset_to_doc(self._row(), channel="st")
        self.assertEqual(doc["asset_id"], "a1")
        self.assertEqual(doc["modality"], "video")
        self.assertEqual(doc["domain_label"], "general")
        # 047: status·channel·chunk_count 는 OS 문서에 넣지 않음.
        self.assertNotIn("status", doc)
        self.assertNotIn("channel", doc)
        self.assertNotIn("chunk_count", doc)
        # T004: file_name 은 clean_file_name 적용 값(확장자 제거·'_'→공백·ID 토큰 제거). 'xyz'(<8)
        # 는 보존된다.
        self.assertEqual(doc["file_name"], "무선 충전기 xyz")
        self.assertEqual(doc["fs_uri"], "/data/sub/무선_충전기_xyz.mp4")  # 원본 경로는 보존
        self.assertEqual(doc["summary"], "무선 충전기 리뷰")
        self.assertEqual(doc["keywords"], ["충전기", "Qi2"])
        self.assertEqual(doc["labels"], ["전자제품"])
        self.assertEqual(doc["embedding"], [0.1, 0.2, 0.3])

    def test_all_expected_keys_present(self) -> None:
        # T001 이 명시한 문서 필드 집합을 전부 갖는지(누락 방지).
        doc = asset_to_doc(self._row(), channel="st")
        expected = {
            "asset_id", "modality", "domain_label",
            "file_name", "fs_uri", "summary", "keywords", "labels",
            "embedding",
        }
        self.assertTrue(expected.issubset(doc.keys()))
        self.assertNotIn("search_text", doc)
        for omitted in ("status", "channel", "chunk_count"):
            self.assertNotIn(omitted, doc)

    def test_summary_keywords_labels_separate_fields(self) -> None:
        # 047: BM25 교차 필드는 summary·keywords 색인 필드로 쿼리 — 합본 search_text 제거.
        doc = asset_to_doc(self._row(), channel="st")
        self.assertEqual(doc["summary"], "무선 충전기 리뷰")
        self.assertEqual(doc["keywords"], ["충전기", "Qi2"])
        self.assertEqual(doc["labels"], ["전자제품"])

    def test_file_name_not_merged_into_summary(self) -> None:
        # file_name 은 별도 필드 — summary/keywords 와 합치지 않는다(026 FR-003①).
        row = self._row(
            fs_path="/data/NOISETOKEN12.mp4",
            ext_meta={"summary": "주제 요약", "keywords": ["키워드"], "labels": ["라벨"]},
        )
        doc = asset_to_doc(row, channel="st")
        self.assertEqual(doc["summary"], "주제 요약")
        self.assertNotIn("NOISETOKEN12", doc["summary"])
        self.assertNotIn("NOISETOKEN12", " ".join(doc["keywords"]))

    def test_labels_dict_flattened_to_label_string(self) -> None:
        # T003(P0·FR-002): labels 가 {label,score} dict 면 label 문자열만 추출(str(dict) 직렬화 금지).
        # vlm_text_for_embedding 과 동형 — 'score'·숫자가 BM25 를 오염시키고 정확매칭을 무력화하던 버그.
        row = self._row(
            ext_meta={
                "summary": "s",
                "keywords": [],
                "labels": [{"label": "텍스트", "score": 0.51}, {"label": "인물", "score": 0.3}],
            }
        )
        doc = asset_to_doc(row, channel="st")
        self.assertEqual(doc["labels"], ["텍스트", "인물"])
        # labels 는 keyword 배열로만 저장(dict-repr 오염 없음).
        self.assertNotIn("score", doc["labels"])
        self.assertNotIn("{", str(doc["labels"]))

    def test_labels_mixed_dict_and_str(self) -> None:
        # dict·str 혼합 labels 도 모두 문자열로 평탄화. 빈 label/공백은 제외.
        row = self._row(
            ext_meta={"summary": "s", "labels": [{"label": "가방"}, "신발", {"label": ""}, "  "]}
        )
        doc = asset_to_doc(row, channel="st")
        self.assertEqual(doc["labels"], ["가방", "신발"])

    def test_missing_ext_meta_safe(self) -> None:
        # ext_meta 없음/None 도 빈 값으로 안전 처리.
        doc = asset_to_doc(self._row(ext_meta=None), channel="st")
        self.assertEqual(doc["summary"], "")
        self.assertEqual(doc["keywords"], [])
        self.assertEqual(doc["labels"], [])

    def test_non_list_keywords_ignored(self) -> None:
        # keywords/labels 가 리스트 아니면(스키마 위반) 빈 리스트로 방어.
        doc = asset_to_doc(
            self._row(ext_meta={"summary": "s", "keywords": "wrong"}), channel="st"
        )
        self.assertEqual(doc["keywords"], [])

    def test_zero_vector_omits_embedding(self) -> None:
        # 영벡터(퇴화 임베딩)는 embedding 필드를 아예 생략 → 텍스트만 색인(cosinesimil 거부 회피).
        doc = asset_to_doc(self._row(emb="[0.0,0.0,0.0]"), channel="st")
        self.assertNotIn("embedding", doc)
        self.assertEqual(doc["summary"], "무선 충전기 리뷰")  # 텍스트는 정상 색인

    def test_nonzero_vector_includes_embedding(self) -> None:
        # 한 성분이라도 0 이 아니면 embedding 포함(벡터 검색 대상).
        doc = asset_to_doc(self._row(emb="[0.0,0.1,0.0]"), channel="st")
        self.assertEqual(doc["embedding"], [0.0, 0.1, 0.0])

    def test_filter_index_fields_from_row(self) -> None:
        row = self._row(
            fs_path="/sample_data/data3/foo.stt.txt",
            created_at="2026-01-20T08:00:00+00:00",
        )
        doc = asset_to_doc(row, channel="st")
        self.assertEqual(
            doc["filter_kw"],
            {"file_ext": "txt", "source_dataset": "data3"},
        )
        self.assertEqual(doc["filter_date"], {"created_at": "2026-01-20"})


class TestCleanFileName(unittest.TestCase):
    """T004(FR-003②): 파일명 정제 — ID스러움(유튜브 ID 등) 토큰만 보수적으로 제거.

    토큰 단위(위치 무관). 판정: 순수 영숫자([A-Za-z0-9-])·길이≥8·(모음<25% 또는 (대소/숫자 2종 이상
    혼합 and 사전식 단어 아님)). 한글 포함 토큰은 항상 보존. 일반 영단어('Maintenance')는 보존하되
    무작위 ID('HAi1OZD1OMM')는 제거 — 정제는 순수·결정적(헌법 3조)."""

    def test_removes_youtube_id_token(self) -> None:
        # 'HAi1OZD1OMM': 길이 11·대소/숫자 3종 혼합·사전식 아님 → 제거. 한글 토큰만 남는다.
        self.assertEqual(clean_file_name("무선_충전기_HAi1OZD1OMM.mp4"), "무선 충전기")

    def test_preserves_common_english_word(self) -> None:
        # 'Maintenance'(모음 5/11≥0.25·Capitalized 사전식) → 보존(오탐 방지).
        self.assertEqual(clean_file_name("Server_Maintenance.txt"), "Server Maintenance")

    def test_preserves_korean_tokens(self) -> None:
        # 한글 포함 토큰은 ID스러움 판정 대상이 아니다(항상 보존).
        self.assertEqual(clean_file_name("스마트폰_리뷰.mp4"), "스마트폰 리뷰")

    def test_empty_and_no_extension_safe(self) -> None:
        # 빈 문자열·확장자 없음·점만 있는 경우 안전.
        self.assertEqual(clean_file_name(""), "")
        self.assertEqual(clean_file_name("스마트폰"), "스마트폰")

    def test_low_vowel_ratio_token_removed(self) -> None:
        # 모음 희소(<25%) 영숫자 토큰(길이≥8)은 혼합 여부와 무관하게 제거.
        self.assertEqual(clean_file_name("리뷰_QWXZBKLMN.mp4"), "리뷰")

    def test_consonant_heavy_real_word_kept(self) -> None:
        # 모음 희소(<25%)지만 모음을 보유한 규칙 표기 영단어는 보존(리뷰 후속 — 모음 분기의
        # 자연어 가드): 'strength'(e 1개·0.125). 모음·y 전무('QWXZBKLMN')만 ID 로 제거.
        self.assertEqual(clean_file_name("강도_strength_가이드.pdf"), "강도 strength 가이드")

    def test_short_alnum_token_preserved(self) -> None:
        # 길이<8 영숫자 토큰(예: 'Qi2'·'xyz')은 ID 판정에서 제외(보수적) — 보존.
        self.assertEqual(clean_file_name("충전기_Qi2.mp4"), "충전기 Qi2")

    def test_all_id_tokens_yields_empty(self) -> None:
        # 모든 토큰이 ID스러우면 빈 문자열(파일명 신호 0) — 안전.
        self.assertEqual(clean_file_name("HAi1OZD1OMM.mp4"), "")

    def test_noise_pattern_token_removed(self) -> None:
        # settings 잡음 패턴 목록(수집원 규약)도 토큰 제거 — 새 명명 규약을 코드 수정 없이 대응.
        out = clean_file_name("리뷰_shorts.mp4", noise_patterns=[r"^shorts$"])
        self.assertEqual(out, "리뷰")


class TestIndexBody(unittest.TestCase):
    def test_knn_and_nori_mapping(self) -> None:
        body = build_index_body()
        props = body["mappings"]["properties"]
        # kNN 검색을 켠다.
        self.assertTrue(body["settings"]["index"]["knn"])
        # 임베딩은 knn_vector, 차원은 단일 출처 상수와 일치(헌법 6조·FR-005).
        self.assertEqual(props["embedding"]["type"], "knn_vector")
        self.assertEqual(props["embedding"]["dimension"], FIX_EMBEDDING_DIMENSION)
        self.assertEqual(props["embedding"]["method"]["space_type"], "cosinesimil")
        # T006(FR-004): 한국어 텍스트 필드는 user_dictionary 를 받는 **커스텀** nori analyzer('nori_user').
        # 내장 'nori' 는 user_dictionary 를 못 받으므로 반드시 커스텀 정의를 쓴다.
        for f in ("summary", "keywords", "file_name"):
            self.assertEqual(props[f]["analyzer"], "nori_user")
        # 메타 필터는 keyword.
        for k in ("asset_id", "modality", "domain_label"):
            self.assertEqual(props[k]["type"], "keyword")
        for omitted in ("status", "channel", "chunk_count"):
            self.assertNotIn(omitted, props)
        self.assertEqual(props["filter_kw"]["properties"]["file_ext"]["type"], "keyword")
        self.assertEqual(props["filter_kw"]["properties"]["source_dataset"]["type"], "keyword")
        self.assertEqual(props["filter_date"]["properties"]["created_at"]["type"], "date")

    def test_custom_nori_analyzer_with_user_dictionary(self) -> None:
        # T006(FR-004): settings.analysis 에 nori_tokenizer + user_dictionary_rules 기반 커스텀
        # analyzer 'nori_user' 가 정의되고, 기본 외래어 고유명사 목록이 사전 규칙으로 들어간다
        # (아이패드·아이폰 등이 분해되지 않게 — '아이패드' BM25 가짜매칭 0).
        analysis = build_index_body()["settings"]["analysis"]
        tok = analysis["tokenizer"]["nori_user_tokenizer"]
        self.assertEqual(tok["type"], "nori_tokenizer")
        self.assertEqual(
            tok["user_dictionary_rules"],
            ["아이패드", "아이폰", "스마트워치", "맥세이프", "에어팟", "갤럭시", "애플워치"],
        )
        analyzer = analysis["analyzer"]["nori_user"]
        self.assertEqual(analyzer["tokenizer"], "nori_user_tokenizer")

    def test_nori_user_words_override(self) -> None:
        # 사전 목록은 인자로 주입 가능(settings 단일 출처가 IO 층에서 전달) — 결정적 반영.
        body = build_index_body(nori_user_words=["갤럭시탭", "버즈"])
        tok = body["settings"]["analysis"]["tokenizer"]["nori_user_tokenizer"]
        self.assertEqual(tok["user_dictionary_rules"], ["갤럭시탭", "버즈"])

    def test_dim_override(self) -> None:
        # 차원은 인자로 덮어쓸 수 있다(테스트·향후 모델 교체 대비).
        body = build_index_body(dim=8)
        self.assertEqual(body["mappings"]["properties"]["embedding"]["dimension"], 8)


# ── G2 색인 seam 테스트용 주입 대역(가짜 OpenSearch 클라이언트 / 가짜 psycopg conn) ──
# 실제 opensearch-py·DB 없이 IO 함수가 부르는 메서드·인자만 기록·검증한다.


class _FakeIndices:
    """`client.indices` 대역 — exists/create/delete/refresh 호출과 인자를 기록."""

    def __init__(self, existing: bool = False) -> None:
        self._existing = existing
        self.created: list[tuple[str, dict]] = []
        self.deleted: list[str] = []
        self.refreshed: list[str] = []
        self.exists_calls: list[str] = []

    def exists(self, index: str) -> bool:
        self.exists_calls.append(index)
        return self._existing

    def create(self, index: str, body: dict) -> None:
        self.created.append((index, body))
        self._existing = True

    def delete(self, index: str) -> None:
        self.deleted.append(index)
        self._existing = False

    def refresh(self, index: str) -> None:
        self.refreshed.append(index)


class _FakeClient:
    """OpenSearch 클라이언트 대역 — indices·index(단건)·bulk 호출 기록."""

    def __init__(self, existing: bool = False) -> None:
        self.indices = _FakeIndices(existing)
        self.indexed: list[dict] = []
        self.bulk_calls: list[dict] = []

    def index(self, *, index: str, id: str, body: dict) -> dict:
        self.indexed.append({"index": index, "id": id, "body": body})
        return {"result": "updated"}

    def bulk(self, actions, **kwargs) -> tuple[int, list]:
        acts = list(actions)
        self.bulk_calls.append({"actions": acts, "kwargs": kwargs})
        return (len(acts), [])


class _FakeCursor:
    """psycopg 커서 대역 — execute(sql, params) 기록, fetchone/iter 로 주입 행 반환."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = list(rows)
        self.executed: list[tuple[str, object]] = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def execute(self, sql: str, params=None) -> None:
        self.executed.append((sql, params))

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


class _FakeConn:
    """psycopg conn 대역 — cursor(row_factory=)→커서. 실제 DB 연결 불필요."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = list(rows)
        self.cursors: list[_FakeCursor] = []

    def cursor(self, row_factory=None) -> _FakeCursor:
        cur = _FakeCursor(self._rows)
        self.cursors.append(cur)
        return cur


def _bulk_via_client(client, actions, **kwargs):
    """sync_all 의 bulk seam 주입 대역 — helpers.bulk 흉내(가짜 client.bulk 위임)."""
    return client.bulk(actions, **kwargs)


def _no_topics(_conn, _asset_id):
    """065 topics seam 주입 대역 — 주제 미부여 자산(주제 0). 색인 경로 기본 배선(fetch_asset_topic)은
    실 DB 를 타므로, 이 순수 색인 seam 테스트는 topics_fn 을 [] 로 주입해 기존 문서 형상을 유지한다.
    주제 수록/정본 읽기 자체는 tests/test_opensearch_topics.py 가 별도로 덮는다(065 T402)."""
    return []


def _asset_row(**over) -> dict:
    """index_asset/sync_all SQL 이 돌려줄 1행(asset+meta+평균임베딩) 대역."""
    row = {
        "asset_id": "a1",
        "modality": "video",
        "domain_label": "general",
        "status": "registered",
        "fs_path": "/data/무선_충전기.mp4",
        "ext_meta": {"summary": "무선 충전기 리뷰", "keywords": ["충전기"], "labels": ["전자제품"]},
        "emb": "[0.1,0.2,0.3]",
        "chunk_count": 3,
    }
    row.update(over)
    return row


def _is_read_only(sql: str) -> bool:
    """SQL 이 읽기전용 SELECT 인지(쓰기 키워드 부재) 확인 — FR-004(헌법 6조) 가드."""
    up = sql.upper()
    if "SELECT" not in up:
        return False
    return not any(w in up for w in ("INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "MERGE"))


class TestEnsureIndex(unittest.TestCase):
    def test_creates_when_absent(self) -> None:
        # 인덱스 없으면 build_index_body 매핑으로 create, 반환 'created'.
        from src.search.opensearch_sync import ensure_index

        client = _FakeClient(existing=False)
        self.assertEqual(ensure_index(client, "assets"), "created")
        self.assertEqual(len(client.indices.created), 1)
        idx, body = client.indices.created[0]
        self.assertEqual(idx, "assets")
        self.assertEqual(body, build_index_body())
        self.assertEqual(client.indices.deleted, [])

    def test_noop_when_exists(self) -> None:
        # 이미 있으면 생성하지 않고 'exists'.
        from src.search.opensearch_sync import ensure_index

        client = _FakeClient(existing=True)
        self.assertEqual(ensure_index(client, "assets"), "exists")
        self.assertEqual(client.indices.created, [])
        self.assertEqual(client.indices.deleted, [])

    def test_recreate_deletes_then_creates(self) -> None:
        # recreate=True 면 delete 후 재생성(파괴적·옵트인), 반환 'recreated'.
        from src.search.opensearch_sync import ensure_index

        client = _FakeClient(existing=True)
        self.assertEqual(ensure_index(client, "assets", recreate=True), "recreated")
        self.assertEqual(client.indices.deleted, ["assets"])
        self.assertEqual(len(client.indices.created), 1)


class TestIndexAsset(unittest.TestCase):
    def test_indexes_single_with_id_upsert(self) -> None:
        # PG 1행 조회 → asset_to_doc → client.index(_id=asset_id) upsert.
        from src.search.opensearch_sync import index_asset

        client = _FakeClient(existing=True)
        conn = _FakeConn([_asset_row()])
        doc = index_asset(client, conn, "a1", index="assets", channel="st", topics_fn=_no_topics)

        self.assertEqual(len(client.indexed), 1)
        call = client.indexed[0]
        self.assertEqual(call["index"], "assets")
        self.assertEqual(call["id"], "a1")  # _id=asset_id → 재실행 멱등(upsert)
        self.assertEqual(call["body"]["asset_id"], "a1")
        self.assertEqual(call["body"]["embedding"], [0.1, 0.2, 0.3])
        self.assertEqual(doc, call["body"])

    def test_select_is_read_only_with_params(self) -> None:
        # 단건 SQL 은 읽기전용 SELECT 이고 (channel, asset_id) 로 파라미터화(FR-004).
        from src.search.opensearch_sync import index_asset

        conn = _FakeConn([_asset_row()])
        index_asset(
            _FakeClient(existing=True), conn, "a1", index="assets", channel="st",
            topics_fn=_no_topics,
        )
        sql, params = conn.cursors[0].executed[0]
        self.assertTrue(_is_read_only(sql))
        self.assertEqual(params, ("st", "a1"))

    def test_select_filters_registered_status(self) -> None:
        # 단건 색인도 전체 재동기화(_SYNC_SQL)와 **대칭**으로 status='registered' 만 색인한다 —
        # 비-registered(deferred/failed/medical)는 행이 없어 no-op. 증분 경로로만 비-registered 가
        # 새던 비대칭(번들 게이트 우회와 같은 결의 누출)을 SQL 단에서 차단.
        from src.search.opensearch_sync import index_asset

        conn = _FakeConn([_asset_row()])
        index_asset(
            _FakeClient(existing=True), conn, "a1", index="assets", channel="st",
            topics_fn=_no_topics,
        )
        sql, _params = conn.cursors[0].executed[0]
        self.assertIn("REGISTERED", sql.upper())

    def test_noop_when_no_row(self) -> None:
        # 자산/임베딩 없음(INNER JOIN 제외) → 색인 안 함, 명시적 None 반환.
        from src.search.opensearch_sync import index_asset

        client = _FakeClient(existing=True)
        result = index_asset(client, _FakeConn([]), "missing", index="assets", channel="st")
        self.assertIsNone(result)
        self.assertEqual(client.indexed, [])


class TestCheckPgvectorVersion(unittest.TestCase):
    """pgvector>=0.5 선검사 — 동기화 SELECT 의 avg(embedding) 집계가 pgvector 0.5.0 도입 의존이라,
    복구 도구 시작 시 한 번 읽기전용으로 버전을 확인해 구버전/미설치를 원인 분명한 오류로 막는다."""

    def test_passes_when_version_meets_minimum(self) -> None:
        from src.search.opensearch_sync import check_pgvector_version

        self.assertEqual(check_pgvector_version(_FakeConn([{"extversion": "0.5.0"}])), "0.5.0")

    def test_passes_for_newer_version(self) -> None:
        # 0.7.x 등 상위 버전은 통과(메이저·마이너 튜플 비교).
        from src.search.opensearch_sync import check_pgvector_version

        self.assertEqual(check_pgvector_version(_FakeConn([{"extversion": "0.7.4"}])), "0.7.4")

    def test_raises_when_below_minimum(self) -> None:
        # 0.4.x 는 vector 타입 집계(avg/sum)가 없어 동기화가 깨진다 → 선검사에서 차단.
        from src.search.opensearch_sync import check_pgvector_version

        with self.assertRaises(RuntimeError):
            check_pgvector_version(_FakeConn([{"extversion": "0.4.4"}]))

    def test_rejects_below_minimum_with_nonnumeric_prefix(self) -> None:
        # 비숫자 접두('v0.4' 등)도 **위치 보존** 파싱 — minor 가 major 자리로 밀려 0.4 가 4.x 로
        # 오인·통과되면 안 된다. 0.5 미만이므로 정확히 차단돼야 한다(파싱 견고성).
        from src.search.opensearch_sync import check_pgvector_version

        with self.assertRaises(RuntimeError):
            check_pgvector_version(_FakeConn([{"extversion": "v0.4"}]))

    def test_rejects_unparseable_version_conservatively(self) -> None:
        # semver 로 파싱 불가한 값은 보수적으로 차단(미달 취급) — 모호한 통과보다 명확한 거부.
        from src.search.opensearch_sync import check_pgvector_version

        with self.assertRaises(RuntimeError):
            check_pgvector_version(_FakeConn([{"extversion": "garbage"}]))

    def test_raises_when_not_installed(self) -> None:
        # pg_extension 에 vector 행 없음(미설치) → 명확한 RuntimeError.
        from src.search.opensearch_sync import check_pgvector_version

        with self.assertRaises(RuntimeError):
            check_pgvector_version(_FakeConn([]))

    def test_query_is_read_only(self) -> None:
        # 선검사도 PG 무수정(FR-004·헌법 6조) — 읽기전용 SELECT.
        from src.search.opensearch_sync import check_pgvector_version

        conn = _FakeConn([{"extversion": "0.5.0"}])
        check_pgvector_version(conn)
        sql, _ = conn.cursors[0].executed[0]
        self.assertTrue(_is_read_only(sql))


class TestSyncAll(unittest.TestCase):
    def test_bulk_actions_use_id_upsert(self) -> None:
        # 전체 registered 를 bulk 색인 — 각 액션이 _index/_id=asset_id/_source.
        from src.search.opensearch_sync import sync_all

        client = _FakeClient(existing=False)
        conn = _FakeConn([_asset_row(asset_id="a1"), _asset_row(asset_id="a2")])
        status, ok, errors = sync_all(
            client, conn, index="assets", channel="st",
            bulk_fn=_bulk_via_client, topics_fn=_no_topics,
        )

        self.assertEqual(status, "created")  # 없던 인덱스를 ensure_index 가 생성
        self.assertEqual(ok, 2)
        self.assertEqual(errors, [])
        self.assertEqual(len(client.bulk_calls), 1)
        actions = client.bulk_calls[0]["actions"]
        self.assertEqual([a["_id"] for a in actions], ["a1", "a2"])
        for a in actions:
            self.assertEqual(a["_index"], "assets")
            self.assertEqual(a["_id"], a["_source"]["asset_id"])  # _id=asset_id 멱등
        client.indices.refreshed and self.assertIn("assets", client.indices.refreshed)

    def test_select_is_read_only_with_channel_param(self) -> None:
        # 전체 SQL 은 읽기전용 SELECT(registered 필터)이고 (channel,) 파라미터(FR-004).
        from src.search.opensearch_sync import sync_all

        conn = _FakeConn([_asset_row()])
        sync_all(
            _FakeClient(), conn, index="assets", channel="st",
            bulk_fn=_bulk_via_client, topics_fn=_no_topics,
        )
        sql, params = conn.cursors[0].executed[0]
        self.assertTrue(_is_read_only(sql))
        self.assertEqual(params, ("st",))
        self.assertIn("REGISTERED", sql.upper())

    def test_recreate_forwarded_to_ensure(self) -> None:
        # --recreate 는 ensure_index 로 전달되어 인덱스 재생성(스키마 변경 복구).
        from src.search.opensearch_sync import sync_all

        client = _FakeClient(existing=True)
        conn = _FakeConn([_asset_row()])
        status, _ok, _errors = sync_all(
            client, conn, index="assets", channel="st", recreate=True,
            bulk_fn=_bulk_via_client, topics_fn=_no_topics,
        )
        self.assertEqual(status, "recreated")
        self.assertEqual(client.indices.deleted, ["assets"])


if __name__ == "__main__":
    unittest.main()
