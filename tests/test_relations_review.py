import unittest
from unittest.mock import MagicMock, patch


class TestReview(unittest.TestCase):
    def _conn(self, rowcount=1):
        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__.return_value = cur
        cur.rowcount = rowcount
        conn.cursor.return_value = cur
        return conn, cur

    def test_approve_sets_active_and_reviewer_with_guard(self):
        from src.relations.review import approve_edge
        conn, cur = self._conn()
        self.assertTrue(approve_edge(conn, edge_id="e1", reviewer="bc"))
        sql = cur.execute.call_args[0][0]
        params = cur.execute.call_args[0][1]
        self.assertIn("reviewed_by", sql)
        self.assertIn("status = 'proposed'", sql)  # 이미 결정된 엣지 재결정 방지 가드
        self.assertEqual(params[0], "active")
        self.assertEqual(params[1], "bc")

    def test_reject_sets_rejected(self):
        from src.relations.review import reject_edge
        conn, cur = self._conn()
        self.assertTrue(reject_edge(conn, edge_id="e1", reviewer="bc"))
        self.assertEqual(cur.execute.call_args[0][1][0], "rejected")

    def test_promote_kind_only_inactive(self):
        from src.relations.review import promote_relation_kind
        conn, cur = self._conn()
        self.assertTrue(promote_relation_kind(conn, kind_code="gaming_hardware", reviewer="bc"))
        self.assertIn("status='inactive'", cur.execute.call_args[0][0].replace(" ", ""))


class TestListProposedEdges(unittest.TestCase):
    """069 B3(P2-3) — CLI(run_relations_review --list) 큐 조회에 양끝 의료 제외 +
    결정적 정렬(confidence 뒤 edge_id 2차키). JSON shape(컬럼 키)는 불변.
    """

    def _conn(self, rows=None):
        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__.return_value = cur
        cur.fetchall.return_value = rows if rows is not None else []
        conn.cursor.return_value = cur
        return conn, cur

    def test_excludes_medical_both_ends_and_edge_id_tiebreaker(self):
        from src.relations.review import list_proposed_edges
        conn, cur = self._conn(rows=[])
        list_proposed_edges(conn, limit=100)
        sql = str(cur.execute.call_args.args[0])
        # 양끝(src·dst) 자산 의료 제외 — NULL 도메인 노출 위해 IS DISTINCT FROM(= 은 NULL 누락).
        self.assertEqual(sql.count("IS DISTINCT FROM 'medical'"), 2)
        self.assertIn("JOIN node", sql)   # 양끝 자산 도메인 확인용 조인(_REVIEW_FROM 재사용)
        self.assertIn("JOIN asset", sql)
        self.assertIn("e.status = 'proposed'", sql)
        # 결정적 정렬: confidence 1차 + edge_id 2차키(동점 순서 고정·헌법 3조).
        order = sql.split("ORDER BY", 1)[1]
        self.assertIn("confidence DESC NULLS LAST", order)
        self.assertIn("edge_id", order)
        self.assertIn("LIMIT %s", sql)
        self.assertEqual(cur.execute.call_args.args[1], (100,))

    def test_json_shape_columns_preserved(self):
        # CLI 소비 계약 불변 — SELECT 컬럼(=행 dict 키)이 그대로.
        from src.relations.review import list_proposed_edges
        row = {"edge_id": "e1", "src_node": "n1", "dst_node": "n2",
               "kind_code": "same_domain", "confidence": 0.9, "reason": "r", "topic": {}}
        conn, cur = self._conn(rows=[row])
        out = list_proposed_edges(conn, limit=10)
        self.assertEqual(out, [row])
        sql = str(cur.execute.call_args.args[0])
        for col in ("e.edge_id", "e.src_node", "e.dst_node", "rk.kind_code",
                    "e.confidence", "e.reason", "e.topic"):
            self.assertIn(col, sql)


class TestListEdgesForReview(unittest.TestCase):
    """FR-101/102/103 — status별 식별보강 페이징 조회(엣지 행 그대로·C6·의료 제외)."""

    def _conn(self, *, total=1, rows=None):
        """dict_row 커서 대역 — COUNT 1회 + rows 1회 순서로 fetchone/fetchall 를 돌려준다."""
        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__.return_value = cur
        cur.fetchone.return_value = {"count": total}
        cur.fetchall.return_value = rows if rows is not None else []
        conn.cursor.return_value = cur
        return conn, cur

    def _sample_row(self):
        import datetime as _dt
        return {
            "edge_id": "e1",
            "kind_code": "same_domain",
            "confidence": 0.9,
            "reason": "유사",
            "topic": {"topic_ko": "게임"},
            "status": "proposed",
            "reviewed_by": None,
            "reviewed_at": None,
            "created_at": _dt.datetime(2026, 6, 30, 12, 0, 0),
            "src_asset_id": "as1",
            "src_fs_path": "/data/문서A.txt",
            "src_modality": "text",
            "dst_asset_id": "as2",
            "dst_fs_path": "/data/영상B.mp4",
            "dst_modality": "video",
        }

    def test_review_statuses_constant(self):
        from src.relations.review import _REVIEW_STATUSES
        self.assertEqual(_REVIEW_STATUSES, ("proposed", "active", "rejected"))

    def test_sql_binds_status_join_order_medical(self):
        from src.relations.review import list_edges_for_review
        conn, cur = self._conn(total=7, rows=[self._sample_row()])
        list_edges_for_review(conn, status="proposed", limit=50, offset=10)
        # 마지막 execute = 행 조회 SQL(그 앞은 COUNT). 두 SQL 을 합쳐 검사한다.
        sqls = " ".join(str(c.args[0]) for c in cur.execute.call_args_list)
        self.assertIn("JOIN node", sqls)
        self.assertIn("JOIN asset", sqls)
        self.assertIn("IS DISTINCT FROM 'medical'", sqls)  # 양끝 의료 제외
        # tiebreaker 포함 결정적 정렬
        rows_sql = str(cur.execute.call_args_list[-1].args[0])
        self.assertIn("ORDER BY", rows_sql)
        self.assertIn("confidence DESC NULLS LAST", rows_sql)
        self.assertIn("edge_id", rows_sql.split("ORDER BY", 1)[1])
        self.assertIn("LIMIT", rows_sql)
        self.assertIn("OFFSET", rows_sql)
        # status 는 %s 바인딩(f-string 인젝션 아님)
        self.assertIn("status = %s", rows_sql)
        rows_params = cur.execute.call_args_list[-1].args[1]
        self.assertEqual(rows_params[0], "proposed")
        self.assertEqual(rows_params[-2], 50)  # limit
        self.assertEqual(rows_params[-1], 10)  # offset

    def test_row_shape_with_src_dst_and_basename(self):
        from src.relations.review import list_edges_for_review
        conn, _cur = self._conn(total=1, rows=[self._sample_row()])
        result = list_edges_for_review(conn, status="proposed", limit=50, offset=0)
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["status"], "proposed")
        self.assertEqual(result["limit"], 50)
        self.assertEqual(result["offset"], 0)
        self.assertEqual(len(result["rows"]), 1)
        row = result["rows"][0]
        self.assertEqual(row["edge_id"], "e1")
        self.assertEqual(row["kind_code"], "same_domain")
        self.assertEqual(row["confidence"], 0.9)
        self.assertEqual(row["reason"], "유사")
        self.assertEqual(row["topic"], {"topic_ko": "게임"})
        self.assertEqual(row["status"], "proposed")
        self.assertIsNone(row["reviewed_by"])
        self.assertIsNone(row["reviewed_at"])
        # src/dst 각 {asset_id, file_name(basename), modality}
        self.assertEqual(row["src"], {"asset_id": "as1", "file_name": "문서A.txt", "modality": "text"})
        self.assertEqual(row["dst"], {"asset_id": "as2", "file_name": "영상B.mp4", "modality": "video"})

    def test_created_at_included_in_row(self):
        # FR-761 — 응답 행에 created_at 추가(additive·datetime 그대로). SELECT 에 e.created_at.
        import datetime as _dt

        from src.relations.review import list_edges_for_review
        conn, cur = self._conn(total=1, rows=[self._sample_row()])
        result = list_edges_for_review(conn, status="proposed", limit=50, offset=0)
        rows_sql = str(cur.execute.call_args_list[-1].args[0])
        self.assertIn("e.created_at", rows_sql)
        self.assertEqual(result["rows"][0]["created_at"], _dt.datetime(2026, 6, 30, 12, 0, 0))

    def test_ids_normalized_to_str(self):
        # psycopg 는 UUID 컬럼을 uuid.UUID 로 반환 — edge_id/asset_id 는 str 로 정규화해야
        # 파이썬 소비자 문자열 비교·JSON 직렬화가 일관된다(graph_query seam 관례). 미변환 시
        # UUID(...) == "..." 가 False 가 되는 함정을 이 가드로 고정(실DB e2e 회귀 유래).
        import uuid as _uuid

        from src.relations.review import list_edges_for_review
        row = self._sample_row()
        row["edge_id"] = _uuid.UUID("018f0000-0000-7000-8000-000000000263")
        row["src_asset_id"] = _uuid.UUID("018f0000-0000-7000-8000-000000000261")
        row["dst_asset_id"] = _uuid.UUID("018f0000-0000-7000-8000-000000000262")
        conn, _cur = self._conn(total=1, rows=[row])
        out = list_edges_for_review(conn, status="proposed", limit=50, offset=0)["rows"][0]
        self.assertIsInstance(out["edge_id"], str)
        self.assertEqual(out["edge_id"], "018f0000-0000-7000-8000-000000000263")
        self.assertIsInstance(out["src"]["asset_id"], str)
        self.assertIsInstance(out["dst"]["asset_id"], str)

    def test_count_uses_same_where(self):
        from src.relations.review import list_edges_for_review
        conn, cur = self._conn(total=42, rows=[])
        result = list_edges_for_review(conn, status="active", limit=10, offset=0)
        self.assertEqual(result["total"], 42)
        count_sql = str(cur.execute.call_args_list[0].args[0])
        self.assertIn("COUNT", count_sql.upper())
        self.assertIn("status = %s", count_sql)
        self.assertIn("IS DISTINCT FROM 'medical'", count_sql)
        self.assertEqual(cur.execute.call_args_list[0].args[1][0], "active")


class TestListEdgesForReviewFilters(unittest.TestCase):
    """G7 확장(FR-701~705) — 검색·필터 인자가 실행 SQL·params 에 %s 로 반영되고,
    미지정 시 조건이 붙지 않는다(하위 호환·SC-011). COUNT·rows 는 동일 WHERE·params 공유.
    """

    def _conn(self, *, total=1, rows=None):
        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__.return_value = cur
        cur.fetchone.return_value = {"count": total}
        cur.fetchall.return_value = rows if rows is not None else []
        conn.cursor.return_value = cur
        return conn, cur

    def _both_sql(self, cur):
        """(count_sql, count_params, rows_sql, rows_params) — 두 execute 콜 캡처."""
        count_call = cur.execute.call_args_list[0]
        rows_call = cur.execute.call_args_list[-1]
        return (
            str(count_call.args[0]),
            count_call.args[1],
            str(rows_call.args[0]),
            rows_call.args[1],
        )

    def test_no_filter_args_backward_compatible(self):
        # SC-011 — 확장 인자 전부 생략 시 WHERE 는 status + 의료 제외 2개만(현행).
        from src.relations.review import list_edges_for_review
        conn, cur = self._conn(total=1, rows=[])
        list_edges_for_review(conn, status="proposed", limit=50, offset=0)
        count_sql, count_params, rows_sql, rows_params = self._both_sql(cur)
        for sql in (count_sql, rows_sql):
            self.assertIn("e.status = %s", sql)
            self.assertIn("IS DISTINCT FROM 'medical'", sql)
            # 확장 필터 조건은 없어야 한다
            self.assertNotIn("ILIKE", sql)
            self.assertNotIn("rk.kind_code = %s", sql)
            self.assertNotIn("e.confidence >=", sql)
            self.assertNotIn("e.confidence <=", sql)
        # COUNT params 는 status 하나(page 파라미터 없음)
        self.assertEqual(count_params, ("proposed",))
        # rows params 는 status + limit + offset
        self.assertEqual(rows_params, ("proposed", 50, 0))

    def test_q_generates_eight_or_ilike_bindings(self):
        # FR-702 — q 통합 텍스트는 8개 OR ILIKE(edge_id/asset_id ::text·fs_path·reason·topic ko/en).
        from src.relations.review import list_edges_for_review
        conn, cur = self._conn(total=1, rows=[])
        list_edges_for_review(conn, status="proposed", limit=10, offset=0, q="게임")
        count_sql, count_params, rows_sql, rows_params = self._both_sql(cur)
        for sql in (count_sql, rows_sql):
            self.assertIn("e.edge_id::text ILIKE %s", sql)
            self.assertIn("sn.asset_id::text ILIKE %s", sql)
            self.assertIn("dn.asset_id::text ILIKE %s", sql)
            self.assertIn("sa.fs_path ILIKE %s", sql)
            self.assertIn("da.fs_path ILIKE %s", sql)
            self.assertIn("e.reason ILIKE %s", sql)
            self.assertIn("e.topic->>'topic_ko' ILIKE %s", sql)
            self.assertIn("e.topic->>'topic_en' ILIKE %s", sql)
        # q_pat = %게임% 8회 바인딩(COUNT 는 status 뒤 8개)
        self.assertEqual(count_params, ("proposed",) + ("%게임%",) * 8)
        # rows 는 status + 8×q_pat + limit + offset
        self.assertEqual(rows_params, ("proposed",) + ("%게임%",) * 8 + (10, 0))

    def test_asset_id_reviewed_by_use_text_equals(self):
        # FR-703 — asset_id/reviewed_by 는 ::text = (비-UUID 입력에 500 아닌 0건).
        from src.relations.review import list_edges_for_review
        conn, cur = self._conn(total=1, rows=[])
        list_edges_for_review(conn, status="active", limit=5, offset=0,
                              asset_id="not-a-uuid", reviewed_by="bc")
        _c_sql, _c_p, rows_sql, rows_params = self._both_sql(cur)
        self.assertIn("sn.asset_id::text = %s", rows_sql)
        self.assertIn("dn.asset_id::text = %s", rows_sql)
        self.assertIn("e.reviewed_by::text = %s", rows_sql)
        self.assertIn("not-a-uuid", rows_params)
        self.assertIn("bc", rows_params)

    def test_kind_code_and_modality_filters(self):
        # FR-703 — kind_code 정확 일치·modality 양끝 중 하나.
        from src.relations.review import list_edges_for_review
        conn, cur = self._conn(total=1, rows=[])
        list_edges_for_review(conn, status="proposed", limit=5, offset=0,
                              kind_code="same_domain", modality="text")
        _c_sql, _c_p, rows_sql, rows_params = self._both_sql(cur)
        self.assertIn("rk.kind_code = %s", rows_sql)
        self.assertIn("sa.modality = %s", rows_sql)
        self.assertIn("da.modality = %s", rows_sql)
        self.assertIn("same_domain", rows_params)
        self.assertEqual(rows_params.count("text"), 2)  # 양끝 각각 바인딩

    def test_confidence_range_filters(self):
        # FR-704 — min/max_confidence 는 >= / <= (범위 검증은 호출자 책임).
        from src.relations.review import list_edges_for_review
        conn, cur = self._conn(total=1, rows=[])
        list_edges_for_review(conn, status="proposed", limit=5, offset=0,
                              min_confidence=0.3, max_confidence=0.8)
        _c_sql, _c_p, rows_sql, rows_params = self._both_sql(cur)
        self.assertIn("e.confidence >= %s", rows_sql)
        self.assertIn("e.confidence <= %s", rows_sql)
        self.assertIn(0.3, rows_params)
        self.assertIn(0.8, rows_params)

    def test_count_and_rows_share_where_and_params(self):
        # FR-705 — COUNT 와 rows 가 동일 WHERE·같은 필터 params(페이징 total 일치).
        from src.relations.review import list_edges_for_review
        conn, cur = self._conn(total=1, rows=[])
        list_edges_for_review(conn, status="proposed", limit=5, offset=0,
                              q="a", kind_code="same_domain", min_confidence=0.5)
        count_sql, count_params, rows_sql, rows_params = self._both_sql(cur)
        # WHERE 절(FROM 이후, ORDER BY 전)이 동일해야 한다
        count_where = count_sql.split("WHERE", 1)[1]
        rows_where = rows_sql.split("WHERE", 1)[1].split("ORDER BY", 1)[0]
        self.assertEqual(count_where.strip(), rows_where.strip())
        # rows_params = count_params + (limit, offset)
        self.assertEqual(rows_params, count_params + (5, 0))

    def test_period_filter_uses_whitelisted_date_col(self):
        # FR-751/752 — since/until 은 date_col(화이트리스트) 기준 >= / < 바인딩.
        import datetime as _dt

        from src.relations.review import list_edges_for_review
        conn, cur = self._conn(total=1, rows=[])
        since = _dt.datetime(2026, 6, 1)
        until = _dt.datetime(2026, 7, 1)
        list_edges_for_review(conn, status="active", limit=5, offset=0,
                              since=since, until=until, date_col="reviewed_at")
        _c_sql, _c_p, rows_sql, rows_params = self._both_sql(cur)
        self.assertIn("e.reviewed_at >= %s", rows_sql)
        self.assertIn("e.reviewed_at < %s", rows_sql)
        self.assertIn(since, rows_params)
        self.assertIn(until, rows_params)

    def test_period_filter_created_at_col(self):
        import datetime as _dt

        from src.relations.review import list_edges_for_review
        conn, cur = self._conn(total=1, rows=[])
        list_edges_for_review(conn, status="proposed", limit=5, offset=0,
                              since=_dt.datetime(2026, 6, 1), date_col="created_at")
        _c_sql, _c_p, rows_sql, _rows_params = self._both_sql(cur)
        self.assertIn("e.created_at >= %s", rows_sql)
        self.assertNotIn("e.created_at < %s", rows_sql)  # until 미지정

    def test_invalid_date_col_raises_value_error(self):
        # date_col 화이트리스트 위반 → ValueError(f-string 인젝션 방지·호출자 검증 전제).
        from src.relations.review import list_edges_for_review
        conn, _cur = self._conn(total=1, rows=[])
        with self.assertRaises(ValueError):
            list_edges_for_review(conn, status="proposed", limit=5, offset=0,
                                  date_col="created_at; DROP TABLE graph_edge")


class TestListRelationKinds(unittest.TestCase):
    """G7 확장(FR-801) — relation_kind 목록(kind_code 정렬·status 필터·shape)."""

    def _conn(self, *, rows=None):
        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__.return_value = cur
        cur.fetchall.return_value = rows if rows is not None else []
        conn.cursor.return_value = cur
        return conn, cur

    def test_all_kinds_ordered_by_kind_code(self):
        from src.relations.review import list_relation_kinds
        rows = [
            {"kind_code": "a_kind", "kind_name_ko": "가", "description": "가 설명",
             "status": "active"},
            {"kind_code": "b_kind", "kind_name_ko": "나", "description": None,
             "status": "inactive"},  # description 은 nullable → None 가능
        ]
        conn, cur = self._conn(rows=rows)
        result = list_relation_kinds(conn)
        sql = str(cur.execute.call_args.args[0])
        self.assertIn("ORDER BY kind_code", sql)
        self.assertNotIn("WHERE", sql)  # status 미지정 → 전체
        self.assertIn("description", sql)  # 관계 설명도 DB 에서 읽어 전달(FR-801)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["rows"], rows)
        self.assertEqual(result["rows"][0]["description"], "가 설명")

    def test_status_filter_binds_where(self):
        from src.relations.review import list_relation_kinds
        conn, cur = self._conn(rows=[
            {"kind_code": "a_kind", "kind_name_ko": "가", "status": "active"}])
        result = list_relation_kinds(conn, status="active")
        sql = str(cur.execute.call_args.args[0])
        params = cur.execute.call_args.args[1]
        self.assertIn("WHERE status = %s", sql)
        self.assertEqual(params, ("active",))
        self.assertEqual(result["total"], 1)


class TestBulkReview(unittest.TestCase):
    """FR-201/202/203 — 일괄 승인/반려는 건별 단건 함수 디스패치·per-id 결과."""

    def test_approve_dispatches_approve_edge_per_id(self):
        from src.relations import review
        conn = MagicMock()
        with patch.object(review, "approve_edge", return_value=True) as m_ap, \
             patch.object(review, "reject_edge") as m_rj:
            out = review.bulk_review(conn, edge_ids=["e1", "e2"], reviewer="bc", action="approve")
        self.assertEqual(out, [{"edge_id": "e1", "ok": True}, {"edge_id": "e2", "ok": True}])
        m_rj.assert_not_called()
        self.assertEqual(m_ap.call_count, 2)
        m_ap.assert_any_call(conn, edge_id="e1", reviewer="bc")
        m_ap.assert_any_call(conn, edge_id="e2", reviewer="bc")

    def test_reject_dispatches_reject_edge(self):
        from src.relations import review
        conn = MagicMock()
        with patch.object(review, "approve_edge") as m_ap, \
             patch.object(review, "reject_edge", return_value=True) as m_rj:
            out = review.bulk_review(conn, edge_ids=["e1"], reviewer="bc", action="reject")
        self.assertEqual(out, [{"edge_id": "e1", "ok": True}])
        m_ap.assert_not_called()
        m_rj.assert_called_once_with(conn, edge_id="e1", reviewer="bc")

    def test_partial_ok_false_does_not_stop(self):
        # ok=False(엣지 없음/proposed 아님)는 예외가 아니라 결과값 — 나머지 계속 진행(FR-203).
        from src.relations import review
        conn = MagicMock()
        with patch.object(review, "approve_edge", side_effect=[False, True]):
            out = review.bulk_review(conn, edge_ids=["e1", "e2"], reviewer="bc", action="approve")
        self.assertEqual(out, [{"edge_id": "e1", "ok": False}, {"edge_id": "e2", "ok": True}])

    def test_unknown_action_raises_not_silent_reject(self):
        # action 화이트리스트 가드 — 오타·미지 값이 조용히 reject 로 처리되면 안 된다(놀람 최소화).
        from src.relations import review
        conn = MagicMock()
        with patch.object(review, "approve_edge") as m_ap, \
             patch.object(review, "reject_edge") as m_rj:
            with self.assertRaises(ValueError):
                review.bulk_review(conn, edge_ids=["e1"], reviewer="bc", action="approv")
        m_ap.assert_not_called()
        m_rj.assert_not_called()


class TestReviseEdge(unittest.TestCase):
    """FR-301 — 사람 전용 결정 정정(proposed 가드 우회·전 방향 전이)."""

    def _conn(self, rowcount=1):
        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__.return_value = cur
        cur.rowcount = rowcount
        conn.cursor.return_value = cur
        return conn, cur

    def test_revise_no_proposed_guard_updates_reviewer(self):
        from src.relations.review import revise_edge
        conn, cur = self._conn(rowcount=1)
        self.assertTrue(revise_edge(conn, edge_id="e1", reviewer="bc", to_status="rejected"))
        sql = cur.execute.call_args[0][0]
        params = cur.execute.call_args[0][1]
        # 사람 전용 — proposed 가드 없음(그것이 _decide_edge 와의 차이·C4)
        self.assertNotIn("status = 'proposed'", sql)
        self.assertNotIn("status='proposed'", sql.replace(" ", ""))
        self.assertIn("reviewed_by", sql)
        self.assertIn("reviewed_at", sql)
        self.assertIn("updated_at", sql)
        self.assertIn("WHERE edge_id = %s", sql)
        # status·reviewer 바인딩
        self.assertIn("rejected", params)
        self.assertIn("bc", params)
        self.assertIn("e1", params)

    def test_revise_rowcount_zero_returns_false(self):
        from src.relations.review import revise_edge
        conn, _cur = self._conn(rowcount=0)
        self.assertFalse(revise_edge(conn, edge_id="missing", reviewer="bc", to_status="active"))


class TestHumanLlmBoundary(unittest.TestCase):
    """FR-302 — revise(사람 정정) 도입 후에도 LLM sync ON CONFLICT 는 status 미갱신."""

    def test_sync_on_conflict_does_not_update_status(self):
        # sync_graph_edges 의 upsert SQL 을 mock 커서로 캡처 — ON CONFLICT DO UPDATE SET 절에
        # status 가 없어야 한다. 사람의 검토 결정(특히 rejected·revise)을 LLM 재제안이 덮지 않음.
        from unittest import mock

        from src.relations import graph_persist
        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__.return_value = cur
        conn.cursor.return_value = cur
        edge = {"target_media_item_id": "018f0000-0000-7000-8000-000000000278",
                "relation_type_code": "same_domain", "reason": "유사", "confidence": 0.5}
        with mock.patch.object(graph_persist, "ensure_asset_node", side_effect=lambda c, aid: "n_" + aid), \
             mock.patch.object(graph_persist, "fetch_relation_kind",
                               return_value={"relation_kind_id": "k1", "is_symmetric": True}):
            graph_persist.sync_graph_edges(
                conn, source_asset_id="018f0000-0000-7000-8000-000000000004",
                edges=[edge],
                allowed_target_ids=frozenset({"018f0000-0000-7000-8000-000000000278"}))
        upsert_sql = next(
            str(c.args[0]) for c in cur.execute.call_args_list
            if "INSERT INTO graph_edge" in str(c.args[0]))
        # 069 B4: SET 절 뒤에 RETURNING status 가 붙으므로(=DB 실제 status 회수), "SET 대상에
        # status 가 없다"는 원 의도를 보존하려면 RETURNING 이전(=DO UPDATE SET 할당부)만 검사한다.
        set_clause = upsert_sql.split("DO UPDATE SET", 1)[1].split("RETURNING", 1)[0]
        self.assertNotIn("status", set_clause)  # ON CONFLICT 시 status 는 사람 결정 보존(미갱신)
