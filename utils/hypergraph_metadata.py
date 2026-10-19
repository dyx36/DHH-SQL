#coding=utf8

HYPERGRAPH_METADATA_KEY = '__dhh_hypergraph_metadata__'
HYPERGRAPH_METADATA_VERSION = 1
REQUIRED_DB_KEYS = ('composite_primary_keys', 'foreign_key_hyperedges', 'join_path_hyperedges')


def build_hypergraph_metadata(max_fk_path_len, paths_per_table_pair):
    return {
        'version': HYPERGRAPH_METADATA_VERSION,
        'max_fk_path_len': int(max_fk_path_len),
        'paths_per_table_pair': int(paths_per_table_pair),
    }


def attach_hypergraph_metadata(tables, metadata):
    tables[HYPERGRAPH_METADATA_KEY] = dict(metadata)
    return tables


def iter_database_items(tables):
    for db_id, db in tables.items():
        if db_id == HYPERGRAPH_METADATA_KEY:
            continue
        yield db_id, db


def validate_hypergraph_metadata(tables, table_path='<memory>', expected=None):
    metadata = tables.get(HYPERGRAPH_METADATA_KEY)
    if metadata is None:
        raise ValueError(
            'Processed table file %s is missing DHH-SQL hypergraph metadata. '
            'Regenerate it from raw tables with --raw_table_path.' % table_path
        )
    if metadata.get('version') != HYPERGRAPH_METADATA_VERSION:
        raise ValueError(
            'Processed table file %s has hypergraph metadata version %s, expected %s. '
            'Regenerate tables.bin with the current code.' %
            (table_path, metadata.get('version'), HYPERGRAPH_METADATA_VERSION)
        )
    if expected is not None:
        for key in ('max_fk_path_len', 'paths_per_table_pair'):
            if int(metadata.get(key, -1)) != int(expected[key]):
                raise ValueError(
                    'Processed table file %s was generated with %s=%s, expected %s. '
                    'Regenerate tables.bin/train.bin/dev.bin with matching preprocessing settings.' %
                    (table_path, key, metadata.get(key), expected[key])
                )
    missing = {
        db_id: [key for key in REQUIRED_DB_KEYS if key not in db]
        for db_id, db in iter_database_items(tables)
        if any(key not in db for key in REQUIRED_DB_KEYS)
    }
    if missing:
        preview = ', '.join('%s:%s' % (db_id, '/'.join(keys)) for db_id, keys in list(missing.items())[:5])
        raise ValueError(
            'Processed table file %s is missing hypergraph fields (%s). '
            'Regenerate it from raw tables with the current preprocessing code.' % (table_path, preview)
        )
    return metadata
