"""069 T310(D2·P2-26) — keyframe dedup 기본값 단일 출처(SSOT) 계약. 실모델·DB 0.

문제(원 리뷰 P2-26): dedup 기본값 7종이 두 곳에 이중 하드코딩돼 있었다 —
``KeyframeDedupConfig`` dataclass 필드 기본값 ↔ ``settings`` 의 ``VIDEO_KEYFRAME_DEDUP_*``
env fallback 리터럴. 한쪽만 바뀌면 조용히 드리프트한다.

처방: 경량 상수 모듈 ``src/config/keyframe_dedup_defaults.py`` 를 유일 출처로 두고, dataclass 필드
기본값과 settings env fallback 이 모두 이 상수를 참조한다(settings 는 cv2 heavy 인 keyframe_dedup 를
import 할 수 없으므로 상수를 별도 경량 모듈에 둔다). 값은 통합 전과 완전히 동일(동작 불변).
"""

from __future__ import annotations

import pathlib
import re
import unittest


class TestKeyframeDedupDefaultsSingleSource(unittest.TestCase):
    def test_constants_pinned_values(self) -> None:
        # 통합 전 실측값을 못박는다 — SSOT 이관이 값을 바꾸지 않았음을 보증.
        from src.config import keyframe_dedup_defaults as d

        self.assertEqual(d.DEFAULT_ENABLED, True)
        self.assertEqual(d.DEFAULT_HASH_MAX, 7)
        self.assertEqual(d.DEFAULT_SSIM_MIN, 0.94)
        self.assertEqual(d.DEFAULT_SSIM_GRAY_LO, 0.90)
        self.assertEqual(d.DEFAULT_HIST_MIN, 0.97)
        self.assertEqual(d.DEFAULT_COMPARE_MODE, "recent")
        self.assertEqual(d.DEFAULT_RECENT_WINDOW, 4)

    def test_dataclass_field_defaults_reference_constants(self) -> None:
        # KeyframeDedupConfig 필드 기본값 == 상수(단일 출처). enabled 는 필수 필드(기본값 없음)로 유지.
        from src.config import keyframe_dedup_defaults as d
        from src.preprocess.keyframe_dedup import KeyframeDedupConfig

        fields = KeyframeDedupConfig.__dataclass_fields__
        self.assertEqual(fields["hash_max"].default, d.DEFAULT_HASH_MAX)
        self.assertEqual(fields["ssim_min"].default, d.DEFAULT_SSIM_MIN)
        self.assertEqual(fields["ssim_gray_lo"].default, d.DEFAULT_SSIM_GRAY_LO)
        self.assertEqual(fields["hist_min"].default, d.DEFAULT_HIST_MIN)
        self.assertEqual(fields["compare_mode"].default, d.DEFAULT_COMPARE_MODE)
        self.assertEqual(fields["recent_window"].default, d.DEFAULT_RECENT_WINDOW)

    def test_settings_defaults_use_constants_not_literals(self) -> None:
        # settings.py 의 VIDEO_KEYFRAME_DEDUP_* fallback 이 리터럴이 아니라 상수를 참조하는지
        # 소스 수준으로 확인(단일 출처 봉인). 숫자 리터럴(0.94 등)이 dedup fallback 라인에 남으면 실패.
        settings_src = (
            pathlib.Path(__file__).resolve().parents[1] / "src" / "config" / "settings.py"
        ).read_text(encoding="utf-8")
        # FR-E4(PR4a): dedup fallback 은 _build_settings 손나열 → _FIELD_SPECS 테이블로 이관됐다.
        # 대문자 env 키 ``VIDEO_KEYFRAME_DEDUP_*`` 는 이제 7개 스펙 행에만 등장한다(소속 attr 은 소문자).
        # 봉인 의도(리터럴 금지·상수 참조)는 그대로 — 탐지 패턴만 새 형태에 맞춘다.
        dedup_lines = [
            ln for ln in settings_src.splitlines()
            if "VIDEO_KEYFRAME_DEDUP_" in ln and "_Spec(" in ln
        ]
        self.assertEqual(len(dedup_lines), 7, "dedup fallback 라인 7개(enabled+6)")
        for ln in dedup_lines:
            # 상수(DEFAULT_*) 참조 필수, 원시 숫자/문자 리터럴 fallback 금지.
            self.assertRegex(ln, r"DEFAULT_[A-Z_]+", f"상수 미참조: {ln.strip()}")
            self.assertNotRegex(
                ln, r",\s*(True|False|\d+\.?\d*|\"recent\")\s*\)",
                f"리터럴 fallback 잔존: {ln.strip()}",
            )

    def test_default_literals_defined_once(self) -> None:
        # ssim_min 대표 리터럴(0.94)이 코드(주석 제외)에서 상수 모듈 1곳에만(단일 정의처).
        src_root = pathlib.Path(__file__).resolve().parents[1] / "src"
        hits = []
        for p in src_root.rglob("*.py"):
            for ln in p.read_text(encoding="utf-8").splitlines():
                code = ln.split("#", 1)[0]  # 주석부 제거 — 산문 속 0.94(예: recall 0.94→0.90)는 무시
                if re.search(r"\b0\.94\b", code):
                    hits.append(str(p.relative_to(src_root)))
        # 코드 리터럴 0.94 는 상수 모듈 파일 1곳에만(라인 번호는 취약하므로 파일 경로만 봉인).
        self.assertEqual(hits, ["config/keyframe_dedup_defaults.py"],
                         f"0.94 코드 리터럴은 상수 모듈 1곳만이어야: {hits}")


if __name__ == "__main__":
    unittest.main()
