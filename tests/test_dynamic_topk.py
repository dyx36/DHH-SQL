from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.encoder.graph_input import GraphInputLayerPLM


def test_build_dynamic_link_weights_keeps_top_two_question_tokens():
    sims = torch.tensor([
        [0.2, 0.8, 0.5],
        [0.7, 0.1, 0.6],
    ])

    weights, logits, topk_idx, topk_scores = GraphInputLayerPLM._build_dynamic_link_weights(sims, dynamic_topk=2)

    expected = torch.tensor([
        [0.0, 0.8, 0.5],
        [0.7, 0.0, 0.6],
    ])
    assert torch.equal(weights, expected)
    assert torch.equal(logits, torch.tensor([0.8, 0.7]))
    assert topk_idx.shape == (2, 2)
    assert topk_scores.shape == (2, 2)


def test_build_dynamic_link_weights_clamps_invalid_k():
    sims = torch.tensor([[0.1, 0.4, 0.3]])

    weights, logits, topk_idx, topk_scores = GraphInputLayerPLM._build_dynamic_link_weights(sims, dynamic_topk=10)
    assert torch.equal(weights, sims)
    assert torch.equal(logits, torch.tensor([0.4]))
    assert topk_idx.shape == (1, 3)
    assert topk_scores.shape == (1, 3)

    weights, logits, topk_idx, topk_scores = GraphInputLayerPLM._build_dynamic_link_weights(sims, dynamic_topk=0)
    assert torch.equal(weights, torch.tensor([[0.0, 0.4, 0.0]]))
    assert torch.equal(logits, torch.tensor([0.4]))
    assert topk_idx.shape == (1, 1)
    assert topk_scores.shape == (1, 1)


def test_schema_loss_length_excludes_hypernodes():
    class DummyBatch:
        table_lens = torch.tensor([2])
        column_lens = torch.tensor([3])
        pk_lens = torch.tensor([1])
        fk_lens = torch.tensor([1])
        path_lens = torch.tensor([1])

    batch = DummyBatch()

    assert GraphInputLayerPLM._schema_base_len(batch, 0) == 5
    assert GraphInputLayerPLM._schema_full_len(batch, 0) == 8


if __name__ == "__main__":
    test_build_dynamic_link_weights_keeps_top_two_question_tokens()
    test_build_dynamic_link_weights_clamps_invalid_k()
    test_schema_loss_length_excludes_hypernodes()
