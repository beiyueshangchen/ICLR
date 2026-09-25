"""GRACE: graph-conditioned meta-optimization of the cognitive state.

The module is organised as

* representation modules -- :class:`ConceptGNN`, :class:`ItemEncoder`,
  :class:`ConceptAttention`,
* the prediction head -- :class:`StudentItemHead`,
* the learned state transition -- :class:`GraphConditionedMetaOptimizer`,
* and :class:`CognitiveModel`, which wires them together and owns the constant
  concept graph and Q-matrix buffers.

Each representation/update component can be switched off through the ablation
letters defined in :mod:`grace.config`:

===== ========================== ==========================================
Letter Component                  Effect when ablated
===== ========================== ==========================================
a      Item encoder               Item content vector is all zeros
b      Concept attention          Global context is the mean concept embedding
c      Concept graph encoder      Concept embeddings come from a linear map
d      Graph-diffused gradient    The meta-optimizer sees the raw gradient
e      Preconditioner             Update uses the raw (un-preconditioned) gradient
===== ========================== ==========================================
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptGNN(nn.Module):
    """Stack of graph-propagation layers producing the concept embeddings ``C``."""

    def __init__(self, num_concepts: int, in_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.linears = nn.ModuleList()
        last_dim = in_dim
        for _ in range(num_layers):
            self.linears.append(nn.Linear(last_dim, hidden_dim))
            last_dim = hidden_dim

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        for linear in self.linears:
            h = F.relu(linear(torch.matmul(adj, h)))
        return h


class ItemEncoder(nn.Module):
    """Two-layer MLP mapping the Q-masked concept embedding of an item to its content vector."""

    def __init__(self, concept_dim: int, out_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(concept_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, concept_emb: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return self.mlp(q @ concept_emb)


class ConceptAttention(nn.Module):
    """Attention over concept embeddings, conditioned on the current state ``theta``.

    Produces the global concept context ``g_t`` used by the prediction head.
    """

    def __init__(self, student_dim: int, concept_dim: int):
        super().__init__()
        self.W_s = nn.Linear(student_dim, concept_dim, bias=False)
        self.W_c = nn.Linear(concept_dim, concept_dim, bias=False)
        self.w_g = nn.Linear(concept_dim, 1, bias=False)

    def forward(self, theta: torch.Tensor, concept_emb: torch.Tensor) -> torch.Tensor:
        state_term = self.W_s(theta).unsqueeze(1)
        concept_term = self.W_c(concept_emb).unsqueeze(0)
        scores = self.w_g(torch.tanh(state_term + concept_term)).squeeze(-1)
        gamma = torch.softmax(scores, dim=1)
        return gamma @ concept_emb


class StudentItemHead(nn.Module):
    """Predicts the next response from the state, the item and the global context."""

    def __init__(self, student_dim: int, item_dim: int, concept_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.item_proj = nn.Linear(item_dim, student_dim)
        self.concept_proj = nn.Linear(concept_dim, student_dim)
        self.fc1 = nn.Linear(student_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(student_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(
        self, theta: torch.Tensor, item_emb: torch.Tensor, global_concept_repr: torch.Tensor
    ) -> torch.Tensor:
        item_p = self.item_proj(item_emb)
        concept_p = self.concept_proj(global_concept_repr)
        h1 = torch.relu(self.fc1(torch.cat([theta, item_p, concept_p], dim=-1)))
        h2 = torch.relu(self.fc2(torch.cat([theta * item_p, theta * concept_p], dim=-1)))
        return self.out(h1 + h2).squeeze(-1)


class GraphConditionedMetaOptimizer(nn.Module):
    """Learned state transition: a preconditioned step on the prediction-error gradient.

    Given the state ``theta``, the gradient ``g_t`` of the prediction loss with
    respect to it, the item vector, the response ``y``, the time gap ``dt`` and the
    running accuracy ``r_t``, the module produces

    .. math::

        \\theta_{t+1} = \\rho_t \\odot \\theta_t - \\eta_t P_t \\tilde g_t,
        \\quad \\tilde g_t = (1 - \\beta_t) g_t + \\beta_t G g_t,

    where ``rho`` is the per-concept forgetting gate, ``eta`` the step size,
    ``beta`` the graph-diffusion coefficient and ``P`` the diagonal-plus-low-rank
    preconditioner.
    """

    def __init__(
        self,
        state_dim: int,
        item_dim: int,
        hidden_dim: int = 64,
        rank: int = 4,
        time_dim: int = 16,
        use_precond: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.item_dim = item_dim
        self.rank = rank
        self.use_precond = use_precond

        ctx_dim = state_dim + state_dim + item_dim + 1 + time_dim + 1
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim), nn.ReLU(),
            nn.Linear(time_dim, time_dim), nn.ReLU(),
        )
        self.ctx_mlp = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

        self.rho_head = nn.Linear(hidden_dim, state_dim)
        self.eta_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.diag_head = nn.Sequential(nn.Linear(hidden_dim, state_dim), nn.Softplus())
        self.U_head = nn.Linear(hidden_dim, state_dim * rank)

    def forward(
        self,
        theta: torch.Tensor,
        grad_theta: torch.Tensor,
        item_emb: torch.Tensor,
        y: torch.Tensor,
        dt: torch.Tensor,
        r_t: torch.Tensor,
        graph_prop: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        if dt.dim() == 1:
            dt = dt.unsqueeze(-1)
        if r_t.dim() == 1:
            r_t = r_t.unsqueeze(-1)

        dt_feat = self.time_mlp(dt)
        ctx = torch.cat([theta, grad_theta, item_emb, y, dt_feat, r_t], dim=-1)
        h = self.ctx_mlp(ctx)

        rho = torch.sigmoid(self.rho_head(h))
        eta = torch.clamp(self.eta_head(h), min=1e-6, max=0.1)
        beta = self.beta_head(h)

        if graph_prop is not None:
            g_graph = torch.matmul(grad_theta, graph_prop.T)
            g_tilde = (1.0 - beta) * grad_theta + beta * g_graph
        else:
            g_tilde = grad_theta

        if self.use_precond:
            d = torch.clamp(self.diag_head(h), min=1e-6, max=10.0)
            U = torch.clamp(self.U_head(h), min=-10.0, max=10.0).view(
                theta.size(0), self.state_dim, self.rank
            )
            Ug = torch.bmm(U.transpose(1, 2), g_tilde.unsqueeze(-1))
            low_rank_term = torch.bmm(U, Ug).squeeze(-1) / max(self.rank, 1)
            Pg = d * g_tilde + low_rank_term
        else:
            d = torch.ones_like(g_tilde)
            U = torch.zeros(
                theta.size(0), self.state_dim, self.rank, device=theta.device, dtype=theta.dtype
            )
            Pg = g_tilde

        theta_new = rho * theta - eta * Pg
        aux = {
            "rho": rho,
            "eta": eta,
            "beta": beta,
            "diag_precond": d,
            "g_tilde": g_tilde,
            "U": U,
        }
        return theta_new, aux


class CognitiveModel(nn.Module):
    """Full GRACE model: representation modules, prediction head and learned transition.

    Args:
        num_concepts: size ``K`` of the cognitive state / number of concepts.
        num_items: number of distinct questions.
        concept_adj: row-normalised concept graph ``A`` (``K x K``).
        q_matrix: row-normalised Q-matrix ``Q`` (``num_items x K``).
        concept_emb_dim: width ``d_c`` of the concept embeddings.
        item_emb_dim: width ``d`` of the item embeddings.
        rank: rank ``r`` of the low-rank preconditioner factor.
        ablations: iterable of ablation letters (see the module docstring).
    """

    def __init__(
        self,
        num_concepts: int,
        num_items: int,
        concept_adj: torch.Tensor,
        q_matrix: torch.Tensor,
        concept_emb_dim: int = 32,
        item_emb_dim: int = 32,
        rank: int = 4,
        ablations: Optional[Set[str]] = None,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_items = num_items
        self.student_dim = num_concepts
        self.item_dim = item_emb_dim

        ablations = set() if ablations is None else set(ablations)
        self.use_item_encoder = "a" not in ablations
        self.use_concept_attn = "b" not in ablations
        self.use_gnn = "c" not in ablations
        self.use_graph_prop = "d" not in ablations
        self.use_precond = "e" not in ablations

        self.register_buffer("adj", concept_adj)
        self.register_buffer("Q", q_matrix)
        self.register_buffer("graph_prop", concept_adj)

        self.concept_init = nn.Parameter(torch.randn(num_concepts, concept_emb_dim) * 0.1)
        self.gnn = ConceptGNN(num_concepts, concept_emb_dim, hidden_dim=num_concepts)
        self.concept_proj = nn.Linear(concept_emb_dim, num_concepts)
        self.item_encoder = ItemEncoder(concept_dim=num_concepts, out_dim=item_emb_dim)
        self.item_id_emb = nn.Embedding(num_items, item_emb_dim)
        self.concept_attn = ConceptAttention(student_dim=num_concepts, concept_dim=num_concepts)
        self.head = StudentItemHead(
            student_dim=num_concepts, item_dim=item_emb_dim, concept_dim=num_concepts
        )
        self.meta_opt = GraphConditionedMetaOptimizer(
            state_dim=num_concepts,
            item_dim=item_emb_dim,
            rank=rank,
            use_precond=self.use_precond,
        )

    def forward_concept_and_items(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the concept embeddings ``C`` and the item content vectors."""
        if self.use_gnn:
            concept_emb = self.gnn(self.concept_init, self.adj)
        else:
            concept_emb = self.concept_proj(self.concept_init)

        if self.use_item_encoder:
            item_content_emb = self.item_encoder(concept_emb, self.Q)
        else:
            item_content_emb = torch.zeros(
                self.num_items,
                self.item_dim,
                device=concept_emb.device,
                dtype=concept_emb.dtype,
            )
        return concept_emb, item_content_emb

    def get_global_concept(self, theta: torch.Tensor, concept_emb: torch.Tensor) -> torch.Tensor:
        """Global concept context ``g_t`` for the current (Q-masked) state."""
        if self.use_concept_attn:
            return self.concept_attn(theta, concept_emb)
        return concept_emb.mean(dim=0, keepdim=True)

    def item_vector(self, item_content_emb: torch.Tensor, idx: int, device: torch.device) -> torch.Tensor:
        """Item vector for question ``idx``: content embedding plus ID embedding."""
        idx_tensor = torch.tensor([idx], device=device)
        return item_content_emb[idx : idx + 1] + self.item_id_emb(idx_tensor)
