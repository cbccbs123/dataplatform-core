-- Upgrade from pre-trim 004: drop display columns on relation_type (if present).
ALTER TABLE relation_type
    DROP COLUMN IF EXISTS type_name,
    DROP COLUMN IF EXISTS description;
