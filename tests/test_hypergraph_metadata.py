from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.hypergraph_metadata import (
    attach_hypergraph_metadata,
    build_hypergraph_metadata,
    validate_hypergraph_metadata,
)


def test_validate_hypergraph_metadata_rejects_old_tables():
    tables = {
        'db': {
            'composite_primary_keys': [],
            'foreign_key_hyperedges': [],
            'join_path_hyperedges': [],
        }
    }

    try:
        validate_hypergraph_metadata(tables, 'tables.bin')
    except ValueError as exc:
        assert 'missing DHH-SQL hypergraph metadata' in str(exc)
    else:
        raise AssertionError('old tables without metadata should be rejected')


def test_validate_hypergraph_metadata_checks_settings():
    tables = {
        'db': {
            'composite_primary_keys': [],
            'foreign_key_hyperedges': [],
            'join_path_hyperedges': [],
        }
    }
    attach_hypergraph_metadata(tables, build_hypergraph_metadata(3, 2))

    validate_hypergraph_metadata(tables, 'tables.bin', expected=build_hypergraph_metadata(3, 2))
    try:
        validate_hypergraph_metadata(tables, 'tables.bin', expected=build_hypergraph_metadata(4, 2))
    except ValueError as exc:
        assert 'max_fk_path_len' in str(exc)
    else:
        raise AssertionError('mismatched hypergraph settings should be rejected')


if __name__ == "__main__":
    test_validate_hypergraph_metadata_rejects_old_tables()
    test_validate_hypergraph_metadata_checks_settings()
