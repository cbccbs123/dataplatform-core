-- MVP phase: remove edge tables; catalog (relation_kind, relation_type, topic tables per migration order) only.
-- relation_proposal references media_relation — drop proposal first.

DROP TABLE IF EXISTS relation_proposal;
DROP TABLE IF EXISTS media_relation;
