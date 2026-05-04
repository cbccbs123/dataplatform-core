from __future__ import annotations

import time
import traceback
from typing import Iterable

from src.embedders.text_embedder import embedding_text_meta
from src.file.file_type_defs import ALLOWED_TEXT_META_FILE_KINDS, MediaKind
from src.file.file_type_detector import detect_file_kind


def run_embedding(file_list: Iterable[str]) -> None:
    for file in file_list:
        start_time = time.time()
        file_kind = detect_file_kind(file)
        print("-" * 100)
        print(f"파일명: {file.split('/')[-1]}, 파일 종류: {file_kind}")
        try:
            if file_kind in ALLOWED_TEXT_META_FILE_KINDS:
                result = embedding_text_meta(
                    file_path=file,
                    file_kind=file_kind
                )
                print(
                    "result:",
                    f"embedding_vector_len={len(result['embedding_vector'])}",
                )
            elif file_kind == MediaKind.IMAGE.value:
                print(f"파일 종류: {file_kind}.")
            elif file_kind == MediaKind.VIDEO.value:
                print(f"파일 종류: {file_kind}.")
            elif file_kind == MediaKind.AUDIO.value:
                print(f"파일 종류: {file_kind}.")
            else:
                raise ValueError(f"파일 종류: {file_kind}는 지원하지 않습니다.")
        except ValueError as e:
            print(f"파일 요약 추출 오류: {e}")
            continue
        except Exception as e:
            print(f"예외 발생: {e}")
            print(traceback.format_exc())
            continue
        finally:
            elapsed_time = time.time() - start_time
            print(f"소요 시간: {elapsed_time:.2f}초")
            print("-" * 100)
