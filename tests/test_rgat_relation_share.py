from pathlib import Path
import sys
from types import SimpleNamespace

import dgl
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.encoder.rgatsql import RGATSQL


def test_relation_share_layers_supports_optimized_graph():
    args = SimpleNamespace(
        gnn_num_layers=1,
        relation_num=6,
        relation_share_layers=True,
        relation_share_heads=True,
        gnn_hidden_size=8,
        num_heads=2,
        dropout=0.0,
        optimize_graph=True,
    )
    model = RGATSQL(args)
    graph = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
    graph.edata['weight'] = torch.ones(2)
    batch = SimpleNamespace(graph=SimpleNamespace(
        global_g=graph,
        global_edges=torch.tensor([0, 1]),
        global_edges2=torch.tensor([2, 3]),
    ))

    output = model(torch.randn(2, 8), batch)

    assert output.shape == (2, 8)


if __name__ == "__main__":
    test_relation_share_layers_supports_optimized_graph()
