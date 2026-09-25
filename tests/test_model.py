"""Forward-pass, ablation and training-loop smoke tests on synthetic tensors."""

import math

import pytest
import torch
from torch.utils.data import DataLoader

from grace.config import CUMULATIVE_ABLATION_ORDER
from grace.data import SequenceDataset, my_collate
from grace.model import CognitiveModel
from grace.train_eval import evaluate, train_one_epoch

NUM_CONCEPTS = 5
NUM_ITEMS = 5
SEQ_LEN = 6


def _make_model(ablations=None, rank=2):
    return CognitiveModel(
        num_concepts=NUM_CONCEPTS,
        num_items=NUM_ITEMS,
        concept_adj=torch.eye(NUM_CONCEPTS),
        q_matrix=torch.eye(NUM_ITEMS),
        concept_emb_dim=8,
        item_emb_dim=8,
        rank=rank,
        ablations=ablations,
    )


def _synthetic_sequences(count=2, length=SEQ_LEN):
    return [
        [
            {"problem_idx": (start + i) % NUM_ITEMS, "label": (start + i) % 2, "dt": float(i)}
            for i in range(length)
        ]
        for start in range(count)
    ]


def _loader(sequences):
    return DataLoader(
        SequenceDataset(sequences), batch_size=1, shuffle=False, collate_fn=my_collate
    )


def test_forward_concept_and_items_shapes():
    model = _make_model()
    concept_emb, item_content_emb = model.forward_concept_and_items()
    assert tuple(concept_emb.shape) == (NUM_CONCEPTS, NUM_CONCEPTS)
    assert tuple(item_content_emb.shape) == (NUM_ITEMS, 8)


def test_meta_optimizer_respects_the_paper_bounds():
    model = _make_model()
    theta = torch.zeros(1, NUM_CONCEPTS, requires_grad=True)
    grad = torch.full((1, NUM_CONCEPTS), 0.5)
    item_vec = torch.zeros(1, 8)
    y = torch.tensor([1.0])
    dt = torch.tensor([0.0])
    r_t = torch.tensor([[0.5]])

    theta_next, aux = model.meta_opt(theta, grad, item_vec, y, dt, r_t, model.graph_prop)

    assert torch.all(aux["eta"] >= 1e-6) and torch.all(aux["eta"] <= 0.1)
    assert torch.all(aux["diag_precond"] >= 1e-6) and torch.all(aux["diag_precond"] <= 10.0)
    assert torch.all(aux["U"] >= -10.0) and torch.all(aux["U"] <= 10.0)
    assert tuple(aux["U"].shape) == (1, NUM_CONCEPTS, 2)
    assert not torch.allclose(theta_next, theta)


def test_get_global_concept_shape():
    model = _make_model()
    concept_emb, _ = model.forward_concept_and_items()
    global_concept = model.get_global_concept(torch.zeros(1, NUM_CONCEPTS), concept_emb)
    assert tuple(global_concept.shape) == (1, NUM_CONCEPTS)


@pytest.mark.parametrize("ablations", CUMULATIVE_ABLATION_ORDER)
def test_every_ablation_variant_trains_and_evaluates(ablations):
    torch.manual_seed(0)
    model = _make_model(ablations=ablations.split(","))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = _loader(_synthetic_sequences())

    loss, auc, acc = train_one_epoch(model, loader, torch.device("cpu"), optimizer)
    assert math.isfinite(loss)
    assert 0.0 <= auc <= 1.0
    assert 0.0 <= acc <= 1.0

    eval_loss, eval_auc, eval_acc = evaluate(model, loader, torch.device("cpu"))
    assert math.isfinite(eval_loss)
    assert 0.0 <= eval_auc <= 1.0
    assert 0.0 <= eval_acc <= 1.0


def test_ablated_components_are_switched_off():
    full = _make_model()
    assert (full.use_item_encoder, full.use_concept_attn, full.use_gnn) == (True, True, True)
    assert (full.use_graph_prop, full.use_precond) == (True, True)

    ablated = _make_model(ablations=["e", "d", "c", "b", "a"])
    assert (ablated.use_item_encoder, ablated.use_concept_attn) == (False, False)
    assert (ablated.use_gnn, ablated.use_graph_prop, ablated.use_precond) == (False, False, False)

    _, item_content_emb = ablated.forward_concept_and_items()
    assert torch.count_nonzero(item_content_emb) == 0


def test_training_step_updates_the_parameters():
    torch.manual_seed(0)
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    before = model.head.fc1.weight.detach().clone()

    train_one_epoch(model, _loader(_synthetic_sequences()), torch.device("cpu"), optimizer)
    assert not torch.allclose(before, model.head.fc1.weight.detach())
