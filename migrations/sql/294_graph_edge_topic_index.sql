-- 056 (v294): graph_edge.topic 표현식 인덱스 2 — 주제 기반 검색·탐색(topic_query seam) 가속.
-- topic_query 의 find_topic_neighbors/list_topics/assets_in_topic 는 topic jsonb 를
-- (topic->>'topic_ko' = %s / = ANY(%s)) · (topic->>'subtopic_ko' = %s) 술어로 조회한다.
-- 인덱스 없이는 graph_edge(~수천 행) seq scan → 주제 패싯·페이징이 매 조회 전체 스캔이 된다.
-- 표현식 인덱스로 topic_ko/subtopic_ko 등가 조회를 인덱스 프로브화(SC-06 성능).
-- 멱등(IF NOT EXISTS) — 재적용·부트스트랩 안전.
CREATE INDEX IF NOT EXISTS ix_graph_edge_topic_ko
    ON graph_edge ((topic->>'topic_ko'));

CREATE INDEX IF NOT EXISTS ix_graph_edge_subtopic_ko
    ON graph_edge ((topic->>'subtopic_ko'));
