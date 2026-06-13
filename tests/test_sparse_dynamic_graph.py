from pathlib import Path
import sys
from types import SimpleNamespace

import dgl
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.model_constructor import Text2SQL
from utils.constants import RELATIONS_INDEX


def test_runtime_graph_contains_only_positive_dynamic_links():
    model = Text2SQL.__new__(Text2SQL)
    model.question_schema_dynamic = RELATIONS_INDEX['question-schema-dynamic']
    model.schema_question_dynamic = RELATIONS_INDEX['schema-question-dynamic']

    graph = SimpleNamespace(
        global_g=dgl.graph(([0, 1], [0, 1]), num_nodes=3),
        global_edges=torch.tensor([
            RELATIONS_INDEX['question-question-identity'],
            RELATIONS_INDEX['table-table-identity'],
        ]),
        global_edges2=torch.tensor([
            RELATIONS_INDEX['question-question-identity'],
            RELATIONS_INDEX['table-table-identity'],
        ]),
        question_mask=torch.tensor([True, False, False]),
    )
    weights = torch.tensor([
        [-1.0, 0.5, 0.0],
        [0.0, -1.0, -1.0],
        [0.7, -1.0, -1.0],
    ])

    graph = model._inject_dynamic_edges(graph, weights, torch.device('cpu'))

    src, dst = graph.global_g.edges()
    pairs = list(zip(src.tolist(), dst.tolist()))
    assert pairs == [(0, 0), (1, 1), (0, 1), (2, 0)]
    assert graph.global_edges.tolist() == [
        RELATIONS_INDEX['question-question-identity'],
        RELATIONS_INDEX['table-table-identity'],
        RELATIONS_INDEX['question-schema-dynamic'],
        RELATIONS_INDEX['schema-question-dynamic'],
    ]
    assert torch.equal(graph.global_g.edata['weight'], torch.tensor([-1.0, -1.0, 0.5, 0.7]))


if __name__ == "__main__":
    test_runtime_graph_contains_only_positive_dynamic_links()
