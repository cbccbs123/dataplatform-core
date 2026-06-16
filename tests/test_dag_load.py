"""030 G3(T008) — Airflow DAG 무결성·얇은 래퍼 단위 테스트 [US5·FR-011·SC-008].

목적
    `airflow/dags/` 의 세 DAG(dag_collect·dag_process·dag_relations)이
      1. **Airflow 미기동(메타DB·스케줄러 없이)에서 DagBag 파싱만으로** 무오류로 로드되고
         (import_errors 0·cycle 0·3개 DAG 존재·max_active_runs==1·catchup==False),
      2. **각 태스크 callable 이 G1/G2 순수 함수의 얇은 래퍼**임(collect_file·process_received_batch·
         scan_unresolved_assets+run_relations 를 호출하고 DAG 에 비즈니스 로직 0 — FR-011)
    을 단언한다.

DagBag 파싱은 임시 ``AIRFLOW_HOME`` + ``LOAD_EXAMPLES=False`` 로 메타DB·스케줄러 없이 수행한다
    (SC-008: Airflow 미기동 단위 테스트). 실 DB·LLM·파일·모델 불필요. 래퍼 검증은 callable 을
    DagBag 에서 꺼내 ``init_settings``/``PostgresUtil``/G1·G2 함수를 모킹한 뒤 호출해, 위임 호출이
    실제로 일어남을 확인한다(소스 문자열 검사보다 강함).

Airflow 가 설치되지 않은 환경(거버넌스상 apache-airflow 미설치)에서는 전 클래스를 skip 한다 —
    헌법 8조 회귀 0(SC-010): 단위 집합은 의존 부재에도 green 을 유지한다.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

# Airflow 가용성 게이트 — DagBag 파싱 전에 임시 AIRFLOW_HOME 을 세팅해야 config 가 그 경로로 초기화된다.
_AIRFLOW_HOME = tempfile.mkdtemp(prefix="airflow_dagbag_test_")
os.environ.setdefault("AIRFLOW_HOME", _AIRFLOW_HOME)
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")

try:
    from airflow.dag_processing.dagbag import (
        DagBag,  # Airflow 3.x 신경로(models.dagbag 는 deprecated)
    )

    _AIRFLOW_AVAILABLE = True
except Exception:  # noqa: BLE001 — airflow 미설치/임포트 실패 시 전 테스트 skip(회귀 0)
    _AIRFLOW_AVAILABLE = False

# 프로젝트 루트 기준 DAG 폴더(테스트가 어디서 실행돼도 절대경로).
_DAG_FOLDER = str(Path(__file__).resolve().parents[1] / "airflow" / "dags")

_EXPECTED_DAGS = {"dag_collect", "dag_process", "dag_relations"}
_COLLECT_TASK = "collect_inbox"
_PROCESS_TASK = "process_batch"
_RELATIONS_TASK = "propose_relations"

_AID = uuid.UUID("018f0000-0000-7000-8000-000000000010")

_dagbag_cache: dict[str, object] = {}


def _dagbag():
    """임시 AIRFLOW_HOME·LOAD_EXAMPLES=False 로 DagBag 1회 파싱(메타DB 불요·캐시 재사용)."""
    if "bag" not in _dagbag_cache:
        _dagbag_cache["bag"] = DagBag(dag_folder=_DAG_FOLDER, include_examples=False, safe_mode=False)
    return _dagbag_cache["bag"]


def _callable(dag_id: str, task_id: str):
    return _dagbag().dags[dag_id].get_task(task_id).python_callable


def _fake_db(*, fetchone=None):
    """가짜 PostgresUtil — ``with db:`` · ``with db.transaction() as conn:`` · ``with conn.cursor() as cur``
    컨텍스트를 모두 지원하고 cur.fetchone/fetchall 을 제어한다(실 DB 불요)."""
    db = mock.MagicMock(name="db")
    db.__enter__.return_value = db
    db.__exit__.return_value = False
    txn = db.transaction.return_value
    conn = txn.__enter__.return_value
    txn.__exit__.return_value = False
    curcm = conn.cursor.return_value
    cur = curcm.__enter__.return_value
    curcm.__exit__.return_value = False
    cur.fetchone.return_value = fetchone
    cur.fetchall.return_value = []
    return db, conn, cur


@unittest.skipUnless(_AIRFLOW_AVAILABLE, "apache-airflow 미설치 — DagBag 파싱 불가")
class TestDagBagIntegrity(unittest.TestCase):
    """SC-008: Airflow 미기동에서 DagBag 파싱만으로 세 DAG 가 무오류·안전 가드 유지."""

    def test_no_import_errors(self) -> None:
        # import_errors 에는 파싱 실패·DAG cycle(AirflowDagCycleException)이 모두 기록된다 →
        # 빈 dict 이면 import 오류 0 + cycle 0 을 함께 보증한다.
        self.assertEqual(_dagbag().import_errors, {})

    def test_three_expected_dags_present(self) -> None:
        self.assertTrue(_EXPECTED_DAGS.issubset(set(_dagbag().dags)),
                        msg=f"누락: {_EXPECTED_DAGS - set(_dagbag().dags)}")

    def test_singleton_run_and_no_catchup(self) -> None:
        bag = _dagbag()
        for dag_id in _EXPECTED_DAGS:
            dag = bag.dags[dag_id]
            # 단일 GPU·중복 처리 차단 — 동시 1 run·과거 미실행 보충 금지(FR-010·US4·US3).
            self.assertEqual(dag.max_active_runs, 1, msg=f"{dag_id} max_active_runs")
            self.assertFalse(dag.catchup, msg=f"{dag_id} catchup")
            self.assertIsNotNone(dag.schedule, msg=f"{dag_id} schedule")

    def test_each_dag_has_single_python_task(self) -> None:
        bag = _dagbag()
        expected_task = {
            "dag_collect": _COLLECT_TASK,
            "dag_process": _PROCESS_TASK,
            "dag_relations": _RELATIONS_TASK,
        }
        for dag_id, task_id in expected_task.items():
            dag = bag.dags[dag_id]
            self.assertEqual([t.task_id for t in dag.tasks], [task_id],
                             msg=f"{dag_id} 태스크 구성")
            task = dag.get_task(task_id)
            # PythonOperator(또는 동급) — python_callable 이 있는 얇은 래퍼.
            self.assertTrue(callable(getattr(task, "python_callable", None)),
                            msg=f"{dag_id}.{task_id} python_callable")

    def test_process_dag_pins_gpu_pool(self) -> None:
        # 단일 GPU OOM 차단 — dag_process 태스크는 크기 1 Pool('gpu')에 묶인다(FR-010).
        task = _dagbag().dags["dag_process"].get_task(_PROCESS_TASK)
        self.assertEqual(task.pool, "gpu")

    def test_dag_modules_keep_top_level_light(self) -> None:
        # 모듈 최상위는 정의만(스케줄러가 자주 파싱) — 무거운 src 심볼(설정·DB·배치)은 callable 안에서
        # 런타임 import 해야 한다. 모듈 전역에 노출돼 있으면 top-level 에서 import 된 것이라 위반.
        import inspect
        import sys

        heavy = ("PostgresUtil", "init_settings", "process_received_batch",
                 "run_relations", "collect_file", "scan_unresolved_assets")
        for dag_id, task_id in (
            ("dag_collect", _COLLECT_TASK),
            ("dag_process", _PROCESS_TASK),
            ("dag_relations", _RELATIONS_TASK),
        ):
            cb = _callable(dag_id, task_id)
            module = inspect.getmodule(cb) or sys.modules.get(cb.__module__)
            self.assertIsNotNone(module, msg=f"{dag_id} 모듈 해소")
            for name in heavy:
                self.assertNotIn(name, vars(module),
                                 msg=f"{dag_id}: '{name}' 이 모듈 최상위에 노출됨(런타임 import 여야)")


@unittest.skipUnless(_AIRFLOW_AVAILABLE, "apache-airflow 미설치 — DagBag 파싱 불가")
class TestThinWrappers(unittest.TestCase):
    """FR-011: 각 DAG 태스크 callable 이 G1/G2 순수 함수를 호출하는 얇은 래퍼임을 호출로 증명."""

    def test_collect_callable_calls_collect_file_per_inbox_file(self) -> None:
        cb = _callable("dag_collect", _COLLECT_TASK)
        db, _conn, _cur = _fake_db(fetchone=None)  # fs_path 미존재 → 두 파일 모두 수집 진행
        with mock.patch.dict(os.environ, {"WATCHER_INBOX_DIR": "/inbox", "META_ENV": "dev"}), \
                mock.patch("src.config.settings.init_settings") as m_init, \
                mock.patch("src.database.postgres_util.PostgresUtil", return_value=db), \
                mock.patch("src.ingest.collector.collect_files",
                           return_value=["/inbox/a.txt", "/inbox/b.txt"]) as m_files, \
                mock.patch("src.app.run_ingest.collect_file") as m_collect:
            cb()
        m_init.assert_called_once()
        m_files.assert_called_once()
        # 인입 각 파일마다 G1 collect_file(conn, path) 호출 — 래퍼는 루프·트랜잭션만, 수집 로직 0.
        self.assertEqual(m_collect.call_count, 2)
        self.assertEqual([c.args[1] for c in m_collect.call_args_list],
                         ["/inbox/a.txt", "/inbox/b.txt"])

    def test_collect_callable_skips_already_collected(self) -> None:
        cb = _callable("dag_collect", _COLLECT_TASK)
        db, _conn, _cur = _fake_db(fetchone=(1,))  # fs_path 이미 존재 → 멱등 스킵(FR-008)
        with mock.patch.dict(os.environ, {"WATCHER_INBOX_DIR": "/inbox", "META_ENV": "dev"}), \
                mock.patch("src.config.settings.init_settings"), \
                mock.patch("src.database.postgres_util.PostgresUtil", return_value=db), \
                mock.patch("src.ingest.collector.collect_files",
                           return_value=["/inbox/a.txt"]), \
                mock.patch("src.app.run_ingest.collect_file") as m_collect:
            cb()
        m_collect.assert_not_called()

    def test_process_callable_calls_process_received_batch(self) -> None:
        from src.ingest.batch_runner import BatchReport

        cb = _callable("dag_process", _PROCESS_TASK)
        db, _c, _cur = _fake_db()
        with mock.patch.dict(os.environ, {"META_ENV": "dev"}), \
                mock.patch("src.config.settings.init_settings") as m_init, \
                mock.patch("src.database.postgres_util.PostgresUtil", return_value=db), \
                mock.patch("src.ingest.batch_runner.process_received_batch",
                           return_value=BatchReport()) as m_batch:
            cb()
        m_init.assert_called_once()
        m_batch.assert_called_once()
        # db 를 첫 인자로, 배치 한도·cap·고착 임계를 키워드로 전달(얇은 래퍼·배치 로직 0).
        self.assertIs(m_batch.call_args.args[0], db)
        for kw in ("limit", "max_failures", "older_than_s"):
            self.assertIn(kw, m_batch.call_args.kwargs)

    def test_relations_callable_scans_then_delegates_run_relations(self) -> None:
        cb = _callable("dag_relations", _RELATIONS_TASK)
        db, _c, _cur = _fake_db()
        with mock.patch.dict(os.environ, {"META_ENV": "dev"}), \
                mock.patch("src.config.settings.init_settings") as m_init, \
                mock.patch("src.database.postgres_util.PostgresUtil", return_value=db), \
                mock.patch("src.ingest.batch_runner.scan_unresolved_assets",
                           return_value=[_AID]) as m_scan, \
                mock.patch("src.app.run_relations.run_relations",
                           return_value={"done": [(str(_AID), 1, 0)], "failed": []}) as m_rr:
            cb()
        m_init.assert_called_once()
        m_scan.assert_called_once()
        # G2 스캔 → run_relations(내부에서 propose_relations_for_asset 호출 + relation_resolution 전이).
        m_rr.assert_called_once()
        self.assertEqual(m_rr.call_args.args[0], [str(_AID)])

    def test_relations_callable_noop_when_nothing_unresolved(self) -> None:
        cb = _callable("dag_relations", _RELATIONS_TASK)
        db, _c, _cur = _fake_db()
        with mock.patch.dict(os.environ, {"META_ENV": "dev"}), \
                mock.patch("src.config.settings.init_settings"), \
                mock.patch("src.database.postgres_util.PostgresUtil", return_value=db), \
                mock.patch("src.ingest.batch_runner.scan_unresolved_assets",
                           return_value=[]), \
                mock.patch("src.app.run_relations.run_relations") as m_rr:
            cb()
        # 미해소 0건이면 run_relations 를 부르지 않는다(빈 배치 호출 회피).
        m_rr.assert_not_called()


if __name__ == "__main__":
    unittest.main()
