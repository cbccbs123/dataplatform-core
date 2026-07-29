"""패키징 봉인 — 코드가 런타임에 읽는 데이터 파일이 wheel 에 담기는지 검사한다.

**왜 필요한가**: `src/relations/prompt.py` 는 `Path(__file__).resolve().parent / "taxonomy_seed.json"`
으로 시드를 읽는다. 이 JSON 이 wheel 에 안 담기면 **설치본에서 import 가 터진다.**
그런데 개발은 editable 설치(`pip install -e .`)로 하고 editable 은 소스 트리를 가리키므로
**로컬에서는 영원히 드러나지 않는다.**

실제로 이 버그가 오래 숨어 있었다(2026-07-29 발견). 파이프라인 CI 가 인증 문제로 테스트를 0개
돌리던 동안에는 발견될 수 없었고, 인증을 고치자마자 `FileNotFoundError` 로 드러났다.

**이 테스트는 파일시스템만 본다** — wheel 을 실제로 빌드하지 않는다(빌드는 느리고 네트워크·
setuptools 버전에 좌우된다). 대신 두 가지를 대조한다:
  ① 코드가 `Path(__file__).parent / "<이름>"` 으로 읽는 데이터 파일을 **소스에서 추출**하고
  ② 그 이름이 `pyproject.toml` 의 `[tool.setuptools.package-data]` 에 **선언돼 있는지** 확인.

즉 "코드가 읽는데 선언되지 않은 파일"을 잡는다. 새 시드를 추가하고 선언을 빠뜨리면 여기서 막힌다.
"""
from __future__ import annotations

import re
import sys
import tomllib
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# `Path(__file__).resolve().parent / "무엇.json"` 형태를 찾는다. 코드가 자기 옆의 데이터 파일을
# 읽는 관용 표현이며, 이 형태가 곧 "설치본에 함께 있어야 한다"는 신호다.
_SIBLING_DATA = re.compile(
    r'Path\(__file__\)\.resolve\(\)\.parent\s*/\s*["\']([^"\']+\.(?:json|md|txt|yaml|yml))["\']')


def _declared_package_data() -> dict[str, set[str]]:
    """`pyproject.toml` 의 package-data 선언을 읽는다.

    Returns:
        ``{패키지 경로: {파일명…}}``. 선언이 없으면 빈 dict.
    """
    with open(_ROOT / "pyproject.toml", "rb") as f:
        cfg = tomllib.load(f)
    raw = cfg.get("tool", {}).get("setuptools", {}).get("package-data", {})
    return {k: set(v) for k, v in raw.items()}


def _code_read_data() -> dict[str, set[str]]:
    """소스에서 "옆 파일을 읽는" 코드를 찾아 ``{패키지 경로: {파일명…}}`` 으로 모은다.

    Returns:
        패키지 경로는 ``src.relations`` 형태(package-data 키와 같은 표기).
    """
    found: dict[str, set[str]] = {}
    for py in sorted((_ROOT / "src").rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        for name in _SIBLING_DATA.findall(py.read_text(encoding="utf-8")):
            pkg = ".".join(py.relative_to(_ROOT).parent.parts)
            found.setdefault(pkg, set()).add(name)
    return found


class TestPackageDataDeclared(unittest.TestCase):
    """코드가 읽는 데이터 파일이 전부 package-data 에 선언돼 있는지."""

    def test_옆_파일을_읽는_코드가_있으면_package_data_에_선언돼_있다(self):
        need = _code_read_data()
        self.assertTrue(need, "옆 데이터 파일을 읽는 코드를 하나도 못 찾았다 — 정규식이 낡았을 수 있다")
        declared = _declared_package_data()
        missing: list[str] = []
        for pkg, names in sorted(need.items()):
            have = declared.get(pkg, set())
            for n in sorted(names - have):
                missing.append(f"{pkg}/{n}")
        self.assertEqual(
            missing, [],
            "코드가 읽는데 pyproject [tool.setuptools.package-data] 에 없는 파일이다 — "
            "비-editable 설치에서 FileNotFoundError 가 난다:\n  " + "\n  ".join(missing))

    def test_선언된_파일이_실제로_존재한다(self):
        # 선언은 있는데 파일이 없으면 오탈자다 — 그래도 설치는 되고 런타임에 터진다.
        for pkg, names in sorted(_declared_package_data().items()):
            base = _ROOT / Path(*pkg.split("."))
            for n in sorted(names):
                with self.subTest(f"{pkg}/{n}"):
                    self.assertTrue((base / n).is_file(), f"{pkg}/{n} 선언됐으나 파일이 없다")


class TestSeedFilesReadable(unittest.TestCase):
    """시드 JSON 이 실제로 읽히는지 — 선언과 무관하게 내용이 온전한지 본다."""

    def test_taxonomy_seed_가_import_시점에_읽힌다(self):
        # prompt.py 는 모듈 로드 시 시드를 읽는다 — import 가 되면 파일이 온전한 것이다.
        from src.relations import prompt
        self.assertTrue(hasattr(prompt, "build_relation_proposal_prompt"))

    def test_설치_경로에서도_시드가_옆에_있다(self):
        # editable 이 아닌 설치본에서 돌면 site-packages 안을 보게 된다 — 그 경우에도 존재해야 한다.
        from src.relations import prompt
        pkg_dir = Path(prompt.__file__).resolve().parent
        for n in ("taxonomy_seed.json", "subtopic_seed.json"):
            with self.subTest(n):
                self.assertTrue((pkg_dir / n).is_file(),
                                f"{pkg_dir}/{n} 없음 — 설치본이면 package-data 누락이다")


if __name__ == "__main__":
    sys.exit(unittest.main())
