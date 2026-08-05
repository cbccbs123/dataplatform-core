# dataplatform-core

멀티모달(텍스트·이미지·영상·오디오) 데이터 통합 플랫폼의 **공유 코어 라이브러리**입니다.
파일에서 추출한 메타데이터와 청크 임베딩을 PostgreSQL + pgvector 에 적재하고, 자산 간 관계(graph)와
하이브리드 검색(BM25 + kNN)을 위한 규약·계약·순수 로직과 **DB 스키마 정본**을 소유합니다.

> 국책과제 **RS-2025-02215256** 산출물.

## 세 레포의 관계

이 플랫폼은 세 레포로 나뉩니다. 이 레포는 그중 **코어(라이브러리)** 입니다.

| 레포 | 파이썬 패키지 | 역할 |
|---|---|---|
| **dataplatform-core**(이 레포) | `src.*` | 규약·계약·순수 로직 + DB 스키마 정본(`migrations/`) |
| dataplatform-pipeline | `processing.*` | 실행 오케스트레이션(수집·분류·추출·임베딩·적재·색인·관계 생성) |
| dataplatform-service | `service.*` | HTTP API(검색·자산 상세·다운로드·관계 검토 serving) |

**이 레포에는 실행 진입점이 없습니다.** 라이브러리이므로 CLI·Airflow·HTTP 계층은 위 두 레포에 있습니다.
여기서 유효한 명령은 테스트·마이그레이션·시드입니다.

## 요구사항

| 항목 | 버전 |
|---|---|
| Python | **3.13 이상** |
| PostgreSQL | **17** + `pgvector` 확장 |
| OpenSearch | `analysis-nori`(한국어 형태소) 플러그인 · kNN 사용 시 k-NN 플러그인 |

임베딩 차원은 **1536D 로 고정**돼 있습니다(`src/config/embedding_constants.py`).

## 설치

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[migrate]" -c constraints.txt    # [migrate] = alembic (마이그레이션용)
```

`constraints.txt` 는 `pyproject.toml` 의 추상 선언(`>=`)으로부터 **해소된 정확 버전**을 고정합니다.
재현성이 필요하면 위처럼 `-c constraints.txt` 를 함께 쓰십시오.

## 환경변수

템플릿이 있습니다 — 복사해서 값만 채우면 됩니다:

```bash
cp .env.example .env.dev      # .env.dev 는 커밋되지 않습니다(.gitignore)
```

### 설정을 주는 두 가지 방법

| 방법 | 어디에 | 우선순위 |
|---|---|---|
| **A. `.env.<환경>` 파일** | **실행하는 디렉터리** → 없으면 레포 루트 순으로 찾습니다 | 낮음 |
| **B. 환경변수 직접 주입** | 배포·컨테이너·CI(`export` · `env_file:` · `env:`) | **높음**(A 를 덮어씁니다) |

방법 B 로 파일 값을 그대로 올리려면:

```bash
set -a; . ./.env.dev; set +a
```

> `--env dev` 는 `.env.dev` 를, `--env prod` 는 `.env.prod` 를 찾습니다.

### 🔴 필수 — 없으면 기동 시점에 실패합니다

설정 로더가 다음 11개를 **필수로 요구**합니다(미설정 시 `ValueError: 필수 환경변수 누락: <이름>` 으로
즉시 중단 — 잘못된 설정으로 조용히 도는 것을 막기 위한 fail-fast 입니다).

```dotenv
META_MODEL=              # 온프레미스 LLM 모델 이름
ENCODING=utf-8
CHUNK_SIZE=1000          # 텍스트 청킹
OVERLAP_SIZE=100
SUMMARY_MAX_CHARS=500
TOP_K_KEYWORDS=10
TEXT_EMBED_MODEL=        # 텍스트 임베딩 모델
TEXT_EMBED_CHUNK_SIZE=512
TEXT_EMBED_NORMALIZE=true
OPENAI_BASE_URL=         # OpenAI 호환 엔드포인트(온프레미스 LLM 서버)
OPENAI_API_KEY=          # 위 엔드포인트용 키(온프레미스면 임의값도 가능)
```

> `OPENAI_*` 라는 이름은 **OpenAI 호환 프로토콜**을 뜻합니다 — 외부 OpenAI 서비스가 아니라
> 온프레미스 LLM 서버를 가리킵니다(설계 제약: 의료 데이터는 외부 LLM 호출 금지).

### 그 외

| 변수 | 용도 |
|---|---|
| `POSTGRES_HOST` · `POSTGRES_PORT` · `POSTGRES_DB` · `POSTGRES_USER` · `POSTGRES_PASSWORD` | PostgreSQL 접속 |
| `OPENSEARCH_HOST` · `OPENSEARCH_PORT` | OpenSearch 접속(색인·검색) |
| `LLM_BASE_URL` · `LLM_MODEL` | 온프레미스 LLM 엔드포인트(zero-shot 보조) |
| `EMBEDDING_API_URL` | 원격 임베딩 서버(선택 — 미설정 시 로컬 모델 로드) |

## 스키마 생성과 시드

```bash
alembic -c alembic.ini upgrade head          # ① DB 스키마 생성
python -m scripts.seed_topic_registry --env dev --apply   # ② 닫힌 주제 분류체계 시드
```

> ⚠️ **②를 생략하면 관계 생성 결과가 0건이 됩니다.** 관계 제안은 닫힌 주제 어휘(taxonomy)를
> 전제로 동작하므로, 어휘가 비어 있으면 후보가 만들어지지 않습니다.
> `--apply` 없이 실행하면 dry-run(DB 미접촉)으로 무엇이 적재될지만 보여줍니다.

## 테스트

```bash
python -m unittest discover -s tests     # 순수 단위 테스트(실 DB·모델 불필요 — 해당 테스트는 자동 skip)
```

실 DB 통합 테스트는 환경변수 게이트로만 실행됩니다(`RUN_DB_E2E=1`, `RUN_OS_E2E=1`).

## 구조

```
src/
  config/       설정·상수(임베딩 차원 등)
  database/     PostgreSQL 풀·트랜잭션·ID(UUIDv7)·lineage
  domain/       닫힌 어휘(DB CHECK 와 동기)
  embedders/    임베딩 어댑터(SentenceTransformer·CLIP·원격)
  file/         파일 식별·해시
  llm/          LLM 단일 seam + 요약기(text/image/video)
  registry/     레지스트리
  relations/    자산 간 관계 — 후보 생성·제안·저장·품질 + 주제 시드
  search/       하이브리드 검색(BM25 + kNN 융합)·색인 동기
  topic/        자산 자기주제 조회
migrations/     DB 스키마 정본(alembic + 손작성 SQL)
scripts/        시드·게이트·측정 도구
tests/          단위 테스트
```

## 설계 제약

- **학습 기반 방식을 쓰지 않습니다** — 학습·파인튜닝·지도학습·능동학습 없음. 사전학습 모델은 **추론 전용**으로만 사용합니다.
- **LLM 호출은 단일 seam(`src/llm/client.py`)을 경유**하며 `temperature=0` 입니다. 결정 재현성이 요구사항입니다.
- 임베딩 차원 **1536D 고정** · DB 는 **PostgreSQL + pgvector** 고정.
- 코드·주석·로그는 한국어로 작성합니다.

## 그래프 조회 주의

관계 엣지는 **대칭 저장**됩니다. 조회는 반드시 `src/relations/graph_query.py` 를 경유하십시오 —
`WHERE src_node = X` 같은 단방향 쿼리는 dst 쪽으로 접힌 대칭 엣지를 누락합니다.

## 트러블슈팅

### `ValueError: 필수 환경변수 누락: META_MODEL`

설정이 **하나도** 로드되지 않았다는 뜻입니다. 값이 틀린 게 아니라 대개 `.env` 파일을 못 찾은 것입니다.

1. `.env.dev` 가 **실행하는 디렉터리** 또는 레포 루트에 있는지 확인하십시오(`cp .env.example .env.dev`).
2. `--env dev` 로 실행했는지 확인하십시오 — `--env prod` 는 `.env.prod` 를 찾습니다.
3. 그래도 안 되면 환경변수를 직접 주입하십시오: `set -a; . ./.env.dev; set +a`
   (§환경변수 › 방법 B — 설치 방식과 무관하게 항상 동작합니다).

### 관계 생성 결과가 0건입니다

닫힌 주제 분류체계(taxonomy) 시드를 적재하지 않았을 때 나타납니다. 관계 제안은 그 어휘를
전제로 후보를 만들기 때문에 어휘가 비어 있으면 후보가 0이 됩니다.

```bash
python -m scripts.seed_topic_registry --env dev --apply
```

### 검색 결과가 비어 있습니다

적재는 됐어도 OpenSearch 색인이 없으면 검색은 빈 결과입니다. `OPENSEARCH_SYNC_ENABLED=true` 인지,
`analysis-nori` 플러그인이 설치돼 있는지, 그리고 인덱스가 존재하는지 확인하십시오.
