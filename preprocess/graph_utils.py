#coding=utf8
import math, dgl, torch
import numpy as np
from utils.constants import MAX_RELATIVE_DIST
from utils.graph_example import GraphExample
import time

# mapping special column * as an ordinary column
special_column_mapping_dict = {
    'question-*-generic': 'question-column-nomatch',
    '*-question-generic': 'column-question-nomatch',
    'table-*-generic': 'table-column-has',
    '*-table-generic': 'column-table-has',
    '*-column-generic': 'column-column-generic',
    'column-*-generic': 'column-column-generic',
    '*-*-identity': 'column-column-identity'
}
# nonlocal_relations = [
#     'question-question-generic', 'table-table-generic', 'column-column-generic', 'table-column-generic', 'column-table-generic',
#     'table-table-fk', 'table-table-fkr', 'table-table-fkb', 'column-column-sametable',
#     '*-column-generic', 'column-*-generic', '*-*-identity', '*-table-generic',
#     'question-question-identity', 'table-table-identity', 'column-column-identity'] + [
#     'question-question-dist' + str(i) for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1, 1) if i not in [-1, 0, 1]
# ]
class GraphProcessor():

    @staticmethod
    def _aggregate_schema_weight(schema_weight, member_indices):
        member_indices = [idx for idx in member_indices if 0 <= idx < schema_weight.shape[1]]
        if len(member_indices) == 0:
            return np.zeros((schema_weight.shape[0], 1), dtype=schema_weight.dtype)
        return np.max(schema_weight[:, member_indices], axis=1, keepdims=True)

    def process_rgatsql(self, ex: dict, db: dict, relation: list, relation_semantic: list):
        graph = GraphExample()

        num_nodes = int(math.sqrt(len(relation)))
        assert num_nodes * num_nodes == len(relation), 'Relation matrix must be square.'
        assert len(relation) == len(relation_semantic), 'Relation and semantic relation sizes must match.'

        # local_edges = [(idx // num_nodes, idx % num_nodes, (special_column_mapping_dict[r] if r in special_column_mapping_dict else r))
        #     for idx, r in enumerate(relation) if r not in nonlocal_relations]
        # local_edges2 = [(idx // num_nodes, idx % num_nodes,
        #                 (special_column_mapping_dict[r] if r in special_column_mapping_dict else r))
        #                for idx, r in enumerate(relation) if r not in nonlocal_relations]
        # local_edges_mask = [True if r not in nonlocal_relations else False
        #                for idx, r in enumerate(relation)]
        # graph.local_edges_mask = local_edges_mask

        global_edges, global_edges2 = [], []
        for idx, (r, r_sem) in enumerate(zip(relation, relation_semantic)):
            if r == 'no-edge':
                continue
            src, dst = idx // num_nodes, idx % num_nodes
            global_edges.append((src, dst, special_column_mapping_dict[r] if r in special_column_mapping_dict else r))
            global_edges2.append((src, dst, special_column_mapping_dict[r_sem] if r_sem in special_column_mapping_dict else r_sem))

        src_ids, dst_ids = list(map(lambda r: r[0], global_edges)), list(map(lambda r: r[1], global_edges))
        graph.global_g = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes, idtype=torch.int32)
        graph.global_edges = global_edges
        graph.global_edges2 = global_edges2
        # src_ids, dst_ids = list(map(lambda r: r[0], local_edges)), list(map(lambda r: r[1], local_edges))
        # graph.local_g = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes, idtype=torch.int32)
        # graph.local_edges = local_edges
        # graph.local_edges2 = local_edges2

        # graph pruning for nodes
        q_num = len(ex['processed_question_toks'])
        s_num = num_nodes - q_num
        graph.question_mask = [1] * q_num + [0] * s_num
        graph.schema_mask = [0] * q_num + [1] * s_num
        t_num = len(db['processed_table_toks'])
        c_num = len(db['processed_column_toks'])
        pk_groups = db.get('composite_primary_keys', [])
        pk_num = len(pk_groups)
        fk_groups = db.get('foreign_key_hyperedges', [])
        fk_num = len(fk_groups)
        path_groups = db.get('join_path_hyperedges', [])
        path_num = len(path_groups)
        graph.keep_mask = [1] * (q_num + t_num + c_num) + [0] * (pk_num + fk_num + path_num)
        graph.gp = dgl.heterograph({
            ('question', 'to', 'schema'): (list(range(q_num)) * s_num,
            [i for i in range(s_num) for _ in range(q_num)])
            }, num_nodes_dict={'question': q_num, 'schema': s_num}, idtype=torch.int32
        )
        def check_node(i):
            if i < t_num and i in ex['used_tables']:
                return 1.0
            if i < t_num + c_num and i - t_num in ex['used_columns']:
                return 1.0
            return 0.0
        graph.node_label = list(map(check_node, range(s_num)))
        graph.schema_weight = ex['schema_weight']
        ex['graph'] = graph

        return ex

    def process_graph_utils(self, ex: dict, db: dict):
        """ Example should be preprocessed by self.pipeline
        """
        q = np.array(ex['relations'], dtype='<U100')

        s = np.array(db['relations'], dtype='<U100')

        q_s = np.array(ex['schema_linking'][0], dtype='<U100')
        s_q = np.array(ex['schema_linking'][1], dtype='<U100')

        q_s2 = np.array(ex['schema_linking2'][0], dtype='<U100')
        s_q2 = np.array(ex['schema_linking2'][1], dtype='<U100')
        dtype = '<U100'
        t_num = len(db['processed_table_toks'])
        c_num = len(db['processed_column_toks'])
        q_s = np.full(q_s.shape, 'no-edge', dtype=dtype)
        s_q = np.full(s_q.shape, 'no-edge', dtype=dtype)
        q_s2 = np.full(q_s2.shape, 'no-edge', dtype=dtype)
        s_q2 = np.full(s_q2.shape, 'no-edge', dtype=dtype)
        pk_groups = db.get('composite_primary_keys', [])
        pk_num = len(pk_groups)
        if pk_num > 0:
            schema_dim = s.shape[0]

            s_ext = np.full((schema_dim + pk_num, schema_dim + pk_num), 'no-edge', dtype=dtype)
            s_ext[:schema_dim, :schema_dim] = s

            for pk_id, group in enumerate(pk_groups):
                pk_idx = schema_dim + pk_id
                s_ext[pk_idx, pk_idx] = 'pk-pk-identity'

                tab_id = group.get('table_id', -1)
                if 0 <= tab_id < t_num:
                    s_ext[pk_idx, tab_id] = 'pk-table-of'
                    s_ext[tab_id, pk_idx] = 'table-pk-of'

                for col_id in group.get('column_ids', []):
                    if 0 <= col_id < c_num:
                        col_idx = t_num + col_id
                        s_ext[pk_idx, col_idx] = 'pk-column-member'
                        s_ext[col_idx, pk_idx] = 'column-pk-member'

            s = s_ext

            q_s = np.concatenate([q_s, np.full((q_s.shape[0], pk_num), 'no-edge', dtype=dtype)], axis=1)
            s_q = np.concatenate([s_q, np.full((pk_num, s_q.shape[1]), 'no-edge', dtype=dtype)], axis=0)
            q_s2 = np.concatenate([q_s2, np.full((q_s2.shape[0], pk_num), 'no-edge', dtype=dtype)], axis=1)
            s_q2 = np.concatenate([s_q2, np.full((pk_num, s_q2.shape[1]), 'no-edge', dtype=dtype)], axis=0)

            if 'schema_weight' in ex:
                sw = np.array(ex['schema_weight'])
                new_weights = []
                for group in pk_groups:
                    member_idx = []
                    tab_id = group.get('table_id', -1)
                    if 0 <= tab_id < t_num:
                        member_idx.append(tab_id)
                    member_idx.extend([t_num + col_id for col_id in group.get('column_ids', []) if 0 <= col_id < c_num])
                    new_weights.append(self._aggregate_schema_weight(sw, member_idx))
                ex['schema_weight'] = np.concatenate([sw] + new_weights, axis=1)
        fk_groups = db.get('foreign_key_hyperedges', [])
        fk_num = len(fk_groups)
        if fk_num > 0:
            schema_dim = s.shape[0]
            pk_start = t_num + c_num
            pk_end = pk_start + pk_num
            s_ext = np.full((schema_dim + fk_num, schema_dim + fk_num), 'no-edge', dtype=dtype)
            s_ext[:schema_dim, :schema_dim] = s
            for fk_id, group in enumerate(fk_groups):
                fk_idx = schema_dim + fk_id
                s_ext[fk_idx, fk_idx] = 'fk-fk-identity'
                src_tab_id = group.get('src_table_id', -1)
                dst_tab_id = group.get('dst_table_id', -1)
                if 0 <= src_tab_id < t_num:
                    s_ext[fk_idx, src_tab_id] = 'fk-table-src'
                    s_ext[src_tab_id, fk_idx] = 'table-fk-src'
                if 0 <= dst_tab_id < t_num:
                    s_ext[fk_idx, dst_tab_id] = 'fk-table-tgt'
                    s_ext[dst_tab_id, fk_idx] = 'table-fk-tgt'
                src_col_id = group.get('src_column_id', -1)
                dst_col_id = group.get('dst_column_id', -1)
                if 0 <= src_col_id < c_num:
                    src_col_idx = t_num + src_col_id
                    s_ext[fk_idx, src_col_idx] = 'fk-column-src'
                    s_ext[src_col_idx, fk_idx] = 'column-fk-src'
                if 0 <= dst_col_id < c_num:
                    dst_col_idx = t_num + dst_col_id
                    s_ext[fk_idx, dst_col_idx] = 'fk-column-tgt'
                    s_ext[dst_col_idx, fk_idx] = 'column-fk-tgt'
            s = s_ext
            q_s = np.concatenate([q_s, np.full((q_s.shape[0], fk_num), 'no-edge', dtype=dtype)], axis=1)
            s_q = np.concatenate([s_q, np.full((fk_num, s_q.shape[1]), 'no-edge', dtype=dtype)], axis=0)
            q_s2 = np.concatenate([q_s2, np.full((q_s2.shape[0], fk_num), 'no-edge', dtype=dtype)], axis=1)
            s_q2 = np.concatenate([s_q2, np.full((fk_num, s_q2.shape[1]), 'no-edge', dtype=dtype)], axis=0)
            if 'schema_weight' in ex:
                sw = np.array(ex['schema_weight'])
                new_weights = []
                for group in fk_groups:
                    member_idx = []
                    for tab_id in (group.get('src_table_id', -1), group.get('dst_table_id', -1)):
                        if 0 <= tab_id < t_num:
                            member_idx.append(tab_id)
                    for col_id in (group.get('src_column_id', -1), group.get('dst_column_id', -1)):
                        if 0 <= col_id < c_num:
                            member_idx.append(t_num + col_id)
                    new_weights.append(self._aggregate_schema_weight(sw, member_idx))
                ex['schema_weight'] = np.concatenate([sw] + new_weights, axis=1)

        path_groups = db.get('join_path_hyperedges', [])
        path_num = len(path_groups)
        if path_num > 0:
            schema_dim = s.shape[0]
            pk_start = t_num + c_num
            pk_end = pk_start + pk_num
            fk_start = pk_end
            fk_end = fk_start + fk_num
            s_ext = np.full((schema_dim + path_num, schema_dim + path_num), 'no-edge', dtype=dtype)
            s_ext[:schema_dim, :schema_dim] = s
            for path_id, group in enumerate(path_groups):
                path_idx = schema_dim + path_id
                s_ext[path_idx, path_idx] = 'path-path-identity'
                if fk_num > 0:
                    for fk_id in group.get('fk_ids', []):
                        if 0 <= fk_id < fk_num:
                            fk_idx = fk_start + fk_id
                            s_ext[path_idx, fk_idx] = 'path-fk-member'
                            s_ext[fk_idx, path_idx] = 'fk-path-member'

                for tab_id in group.get('table_ids', []):
                    if 0 <= tab_id < t_num:
                        s_ext[path_idx, tab_id] = 'path-table-member'
                        s_ext[tab_id, path_idx] = 'table-path-member'
                for col_id in group.get('column_ids', []):
                    if 0 <= col_id < c_num:
                        col_idx = t_num + col_id
                        s_ext[path_idx, col_idx] = 'path-column-member'
                        s_ext[col_idx, path_idx] = 'column-path-member'
            s = s_ext
            q_s = np.concatenate([q_s, np.full((q_s.shape[0], path_num), 'no-edge', dtype=dtype)], axis=1)
            s_q = np.concatenate([s_q, np.full((path_num, s_q.shape[1]), 'no-edge', dtype=dtype)], axis=0)
            q_s2 = np.concatenate([q_s2, np.full((q_s2.shape[0], path_num), 'no-edge', dtype=dtype)], axis=1)
            s_q2 = np.concatenate([s_q2, np.full((path_num, s_q2.shape[1]), 'no-edge', dtype=dtype)], axis=0)
            if 'schema_weight' in ex:
                sw = np.array(ex['schema_weight'])
                new_weights = []
                for group in path_groups:
                    member_idx = []
                    member_idx.extend([tab_id for tab_id in group.get('table_ids', []) if 0 <= tab_id < t_num])
                    member_idx.extend([t_num + col_id for col_id in group.get('column_ids', []) if 0 <= col_id < c_num])
                    member_idx.extend([fk_start + fk_id for fk_id in group.get('fk_ids', []) if 0 <= fk_id < fk_num])
                    new_weights.append(self._aggregate_schema_weight(sw, member_idx))
                ex['schema_weight'] = np.concatenate([sw] + new_weights, axis=1)

        relation = np.concatenate([
            np.concatenate([q, q_s], axis=1),
            np.concatenate([s_q, s], axis=1)
        ], axis=0)

        relation_semantic = np.concatenate([
            np.concatenate([q, q_s2], axis=1),
            np.concatenate([s_q2, s], axis=1)
        ], axis=0)

        assert relation.shape == relation_semantic.shape
        assert relation.shape[0] == relation.shape[1]
        if 'schema_weight' in ex:
            schema_weight = np.array(ex['schema_weight'])
            assert schema_weight.shape == (q.shape[0], s.shape[0]), \
                'schema_weight shape %s does not match question/schema sizes %s.' % (
                    schema_weight.shape, (q.shape[0], s.shape[0]))

        relation = relation.flatten().tolist()

        relation_semantic = relation_semantic.flatten().tolist()
        
        ex = self.process_rgatsql(ex, db, relation, relation_semantic)

        return ex
