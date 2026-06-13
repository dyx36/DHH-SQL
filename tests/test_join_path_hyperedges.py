from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocess.common_utils import Preprocessor
from preprocess.graph_utils import GraphProcessor
from utils.constants import RELATIONS


class DummyArgs:
    semantic_graph = False
    max_fk_path_len = 3
    paths_per_table_pair = 2


def build_schema_relations(table_num, column_num):
    relations = []
    for i in range(table_num + column_num):
        row = []
        for j in range(table_num + column_num):
            if i < table_num and j < table_num:
                row.append('table-table-identity' if i == j else 'table-table-generic')
            elif i < table_num:
                row.append('table-column-generic')
            elif j < table_num:
                row.append('column-table-generic')
            else:
                row.append('column-column-identity' if i == j else 'column-column-generic')
        relations.append(row)
    return relations


def test_build_join_path_hyperedges_keeps_short_multihop_paths_per_pair():
    processor = Preprocessor.__new__(Preprocessor)
    processor.max_fk_path_len = 2
    processor.paths_per_table_pair = 1
    fks = [
        {'src_table_id': 0, 'dst_table_id': 1, 'src_column_id': 1, 'dst_column_id': 2},
        {'src_table_id': 1, 'dst_table_id': 2, 'src_column_id': 3, 'dst_column_id': 4},
        {'src_table_id': 0, 'dst_table_id': 2, 'src_column_id': 5, 'dst_column_id': 6},
    ]

    paths = processor._build_join_path_hyperedges(3, fks)

    assert any(path['table_ids'] == [0, 1, 2] and path['fk_ids'] == [0, 1] for path in paths)
    assert all(len(path['fk_ids']) >= 2 for path in paths)
    assert not any(path['fk_ids'] == [2] for path in paths)


def test_graph_processor_adds_path_hyperedge_relations():
    db = {
        'processed_table_toks': [['a'], ['b'], ['c']],
        'processed_column_toks': [['*'], ['a_id'], ['b_id'], ['b_id2'], ['c_id']],
        'relations': build_schema_relations(3, 5),
        'composite_primary_keys': [],
        'foreign_key_hyperedges': [
            {'src_table_id': 0, 'dst_table_id': 1, 'src_column_id': 1, 'dst_column_id': 2},
            {'src_table_id': 1, 'dst_table_id': 2, 'src_column_id': 3, 'dst_column_id': 4},
        ],
        'join_path_hyperedges': [{'table_ids': [0, 1, 2], 'fk_ids': [0, 1], 'column_ids': [1, 2, 3, 4]}],
    }
    ex = {
        'processed_question_toks': ['show'],
        'relations': [['question-question-identity']],
        'schema_linking': (
            [['question-table-nomatch'] * 3 + ['question-column-nomatch'] * 5],
            [['table-question-nomatch'], ['table-question-nomatch'], ['table-question-nomatch']] + [['column-question-nomatch']] * 5,
        ),
        'schema_linking2': (
            [['question-table-semanticmatch'] * 3 + ['question-column-semanticmatch'] * 5],
            [['table-question-semanticmatch'], ['table-question-semanticmatch'], ['table-question-semanticmatch']] + [['column-question-semanticmatch']] * 5,
        ),
        'schema_weight': [[0.1, 0.2, 0.3, 0.0, 0.4, 0.5, 0.6, 0.7]],
        'used_tables': [0, 1, 2],
        'used_columns': [1, 2, 3, 4],
    }

    processed = GraphProcessor().process_graph_utils(ex, db)
    graph = processed['graph']
    rel_names = [edge[2] for edge in graph.global_edges]
    rel_by_pair = {(edge[0], edge[1]): edge[2] for edge in graph.global_edges}
    q_num = len(ex['processed_question_toks'])
    t_num = len(db['processed_table_toks'])
    c_num = len(db['processed_column_toks'])
    fk_start = q_num + t_num + c_num
    path_idx = fk_start + len(db['foreign_key_hyperedges'])

    assert 'path-table-member' in rel_names
    assert 'table-path-member' in rel_names
    assert 'path-column-member' in rel_names
    assert 'column-path-member' in rel_names
    assert 'path-fk-member' in rel_names
    assert 'fk-path-member' in rel_names
    assert 'question-path-nomatch' not in rel_names
    assert 'path-question-nomatch' not in rel_names
    assert 'path-table-generic' not in rel_names
    assert rel_by_pair[(path_idx, fk_start)] == 'path-fk-member'
    assert rel_by_pair[(fk_start, path_idx)] == 'fk-path-member'
    assert rel_by_pair[(path_idx, q_num + 0)] == 'path-table-member'
    assert rel_by_pair[(path_idx, q_num + t_num + 1)] == 'path-column-member'
    assert len(processed['schema_weight'][0]) == 11
    assert processed['schema_weight'][0][-1] == 0.7
    assert graph.node_label[-1] == 0.0
    for relation in rel_names:
        assert relation in RELATIONS


if __name__ == "__main__":
    test_build_join_path_hyperedges_keeps_short_multihop_paths_per_pair()
    test_graph_processor_adds_path_hyperedge_relations()
