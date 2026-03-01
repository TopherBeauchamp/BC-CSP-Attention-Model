"""
attention_model.py — Attention Model (Encoder + Decoder) for BC-CSP

This file implements the full neural network that reads a BC-CSP instance and
outputs a tour one node at a time. It contains:

    1. The ENCODER call  (Section I of the LaTeX)  — delegates to GraphAttentionEncoder
    2. The DECODER logic  (Section II of the LaTeX) — GRU + dynamic modulation + glimpse attention
    3. Node selection     (Eq. 14-16)               — greedy or sampling
    4. Log-likelihood computation for REINFORCE      (Section IV)

The key BC-CSP adaptations vs. the original Attention Model (Kool et al., ICLR 2019):
    - Separate depot/sensor embeddings with packets as a 3rd feature  → _init_embed()
    - GRU-based decoder hidden state instead of concatenation context  → _get_log_p_rnn()
    - Dynamic marginal coverage gain modulating logit keys            → Eq. (10)-(11)

HIGH-LEVEL FORWARD PASS:
    input (dict) → _init_embed() → encoder → _inner_rnn() (decoder loop) → cost, log_likelihood

See the LaTeX document sections:
    Section I:   Encoder (graph_encoder.py)
    Section II:  Decoder (this file, _get_log_p_rnn and _one_to_many_logits)
    Section III: Problem Environment (state_bccsp.py, problem_bccsp.py)
    Section IV:  REINFORCE Training (run.py, reinforce_baselines.py)
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches
from options import get_options
from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_decode_type(model, decode_type):
    """Helper to set decode_type on the inner model (handles DataParallel wrapping)."""
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Precomputed decoder context that stays FIXED across all decoding steps.
    Computed once in _precompute() to avoid redundant work.

    Fields (all derived from encoder output h_i^(L)):
        node_embeddings:          (B, N+1, d)     — raw encoder embeddings, used as e_t input to GRU
        context_node_projected:   (B, 1, d)       — W_c · h̄ , the projected graph embedding
                                                     used as initial GRU hidden state s_0
                                                     [LaTeX Section II: s_0 = W_c · h̄]
        glimpse_key:              (M, B, 1, N+1, d_k) — node embeddings projected for glimpse attention K
        glimpse_val:              (M, B, 1, N+1, d_k) — node embeddings projected for glimpse attention V
        logit_key:                (B, 1, N+1, d)      — node embeddings projected for final logit computation
                                                        This is K_logit before dynamic modulation [Eq. (11)]
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        """Index all tensors at once (used during batch shrinking)."""
        assert torch.is_tensor(key) or isinstance(key, slice)
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],   # dim 0 = heads
                glimpse_val=self.glimpse_val[:, key],   # dim 0 = heads
                logit_key=self.logit_key[key]
            )


class AttentionModel(nn.Module):
    """
    The complete Attention Model for BC-CSP, containing both encoder and decoder.

    Architecture overview:
        ENCODER (Section I):
            _init_embed() → embedder (GraphAttentionEncoder with L=3 layers)
            Produces h_i^(L) for each node and h̄ (graph mean)

        DECODER (Section II):
            _inner_rnn() loop → at each step t:
                _get_log_p_rnn():
                    1. GRU step          → Eq. (9):   q_t, s_t = GRU(e_t, s_{t-1})
                    2. Dynamic modulation → Eq. (10)-(11): K'_logit = K_logit ⊙ (1 + W_δ · δ_t)
                    3. Glimpse attention  → Eq. (12):  g_t via multi-head attention
                    4. Final logits       → Eq. (13)-(15): u_j, tanh clip, softmax
                _select_node():
                    5. Node selection     → Eq. (16):  sample or greedy
    """

    def __init__(self,
                 embedding_dim,       # d = 128
                 hidden_dim,          # (unused directly, kept for API compatibility)
                 problem,             # Problem class (BCCSP)
                 n_encode_layers=2,   # L = number of encoder layers (overridden to 3 in practice)
                 tanh_clipping=10.,   # C = 10, clipping constant for logits [Eq. (14)]
                 mask_inner=True,     # Whether to mask infeasible nodes in glimpse attention
                 mask_logits=True,    # Whether to mask infeasible nodes in final logits
                 normalization='batch',
                 n_heads=8,           # M = 8 attention heads
                 checkpoint_encoder=False,
                 shrink_size=None):   # Efficiency: remove finished instances from batch
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim   # d = 128
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None              # Set to "greedy" or "sampling" before forward
        self.temp = 1.0                      # τ in Eq. (15): temperature for softmax
        self.allow_partial = problem.NAME == 'sdvrp'
        self.is_bccsp = problem.NAME == 'bccsp'
        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'
        self.is_orienteering = problem.NAME == 'op'
        self.is_pctsp = problem.NAME == 'pctsp'
        self.is_csp = problem.NAME == 'csp'
        self.tanh_clipping = tanh_clipping   # C = 10 [Eq. (14)]

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads               # M = 8
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.tmp = None

        # ====================================================================
        # PROBLEM-SPECIFIC EMBEDDING LAYERS — Eq. (1)-(2)
        # ====================================================================
        if self.is_vrp or self.is_orienteering or self.is_pctsp or self.is_bccsp:
            step_context_dim = embedding_dim + 1

            if self.is_pctsp:
                node_dim = 4  # x, y, expected_prize, penalty
            else:
                node_dim = 3  # x, y, packets (for BC-CSP) or demand/prize (for others)

            # Eq. (1): W_depot · [x_0, y_0] + b_depot  →  d-dim depot embedding
            self.init_embed_depot = nn.Linear(2, embedding_dim)

            if self.is_vrp and self.allow_partial:
                self.project_node_step = nn.Linear(1, 3 * embedding_dim, bias=False)
        elif self.is_csp:
            node_dim = 2
            self.first_placeholder = nn.Parameter(torch.Tensor(embedding_dim))
            self.first_placeholder.data.uniform_(-1, 1)
        else:  # TSP
            assert problem.NAME == "tsp" or problem.NAME == "csp", \
                "Unsupported problem: {}".format(problem.NAME)
            step_context_dim = 2 * embedding_dim
            node_dim = 2
            self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
            self.W_placeholder.data.uniform_(-1, 1)

        # Eq. (2): W_node · [x_i, y_i, p_i] + b_node  →  d-dim sensor embedding
        self.init_embed = nn.Linear(node_dim, embedding_dim)

        # ====================================================================
        # DYNAMIC FEATURE PROJECTION — used in Eq. (11)
        # W_δ: projects scalar marginal coverage δ_t to d-dim for multiplicative gating
        # ====================================================================
        self.init_dynamic = nn.Linear(1, embedding_dim, bias=False)  # W_δ

        # ====================================================================
        # GRU DECODER — Eq. (9)
        # Single-layer GRU with input_size=d, hidden_size=d
        # q_t, s_t = GRU(e_t, s_{t-1})
        # ====================================================================
        self.gru = nn.GRU(embedding_dim, embedding_dim, 1, batch_first=True)
        self.x0 = torch.zeros((1, 1, embedding_dim), requires_grad=True, device=device)

        # ====================================================================
        # ENCODER — Section I (delegates to GraphAttentionEncoder)
        # Note: node_dim is NOT passed here (None), because _init_embed() handles
        # the initial embedding separately for depot vs. sensors. The encoder
        # receives pre-embedded (B, N+1, d) tensors.
        # ====================================================================
        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,  # L = 3
            normalization=normalization
        )

        # ====================================================================
        # DECODER PROJECTION LAYERS (precomputed once, used every decoding step)
        # ====================================================================
        # Projects each node embedding h_i to three vectors:
        #   glimpse_key (d), glimpse_val (d), logit_key (d)  — total 3d
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)

        # Projects graph embedding h̄ to initial GRU hidden state s_0 = W_c · h̄
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)  # W_c

        # (Unused in current code but kept) Would project concatenation of rnn_out + context
        self.project_query = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)

        assert embedding_dim % n_heads == 0
        # Projects concatenated glimpse heads (M * d_k = d) back to d-dim
        # This is W_O in Eq. (12)
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)  # W_O for glimpse

    def set_decode_type(self, decode_type, temp=None):
        """Set decoding strategy: 'greedy' (eval) or 'sampling' (training)."""
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp  # τ in Eq. (15)

    def forward(self, input, return_pi=False):
        """
        Full forward pass: encode → decode → compute cost and log-likelihood.

        This is called during TRAINING (Section IV of LaTeX):
            1. Encode the input graph                          [Section I]
            2. Decode a tour via _inner_rnn                    [Section II]
            3. Compute cost = -covered_packets via get_costs   [Section III, Eq. (16)]
            4. Compute normalized log-likelihood for REINFORCE [Eq. (18)]

        Args:
            input: dict with keys 'loc', 'depot', 'packets', 'max_length', 'radius'
            return_pi: if True, also return the tour sequence

        Returns:
            cost: (B,) — negative covered packets [used as L(π) in Eq. (17)]
            ll:   (B,) — normalized log-likelihood [used as log p_θ(π|s) in Eq. (17)]
            pi:   (B, T) — tour sequence (only if return_pi=True)
        """
        # ====================================================================
        # STEP 1: ENCODER — Section I
        # _init_embed: applies Eq. (1)-(2) to get (B, N+1, d) initial embeddings
        # embedder: applies L=3 attention layers [Eq. (3)-(8)]
        # ====================================================================
        if self.checkpoint_encoder:
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        else:
            embeddings, _ = self.embedder(self._init_embed(input))

        # ====================================================================
        # STEP 2: DECODER — Section II
        # Autoregressive tour construction, returns log-probs and actions
        # ====================================================================
        _log_p, pi = self._inner_rnn(input, embeddings)

        # ====================================================================
        # STEP 3: COST — Section III, Eq. (16)
        # cost = -covered_packets (negated because framework minimizes)
        # ====================================================================
        cost, mask = self.problem.get_costs(input, pi)

        # ====================================================================
        # STEP 4: LOG-LIKELIHOOD — Section IV, Eq. (18)
        # LL = (1/T) × Σ log p(π_t | s, π_{1:t-1}) × 50
        # The ×50 and per-step normalization stabilizes REINFORCE across
        # tours of varying length.
        # ====================================================================
        ll = _log_p.sum(-1)
        for i in range(ll.size(0)):
            ll[i] = ll[i] / torch.nonzero(_log_p[i]).size(0) * 50

        if return_pi:
            return cost, ll, pi
        return cost, ll

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """Sample multiple solutions for the same instance (used in eval)."""
        return sample_many(
            inner_func=lambda x: self._inner_rnn(x, self.embedder(self._init_embed(x))[0]),
            get_cost_func=self.problem.get_costs,
            input=input,
            batch_rep=batch_rep,
            iter_rep=iter_rep
        )

    def precompute_fixed(self, input):
        """Precompute encoder + fixed decoder context (used in beam search)."""
        embeddings, _ = self.embedder(self._init_embed(input))
        return CachedLookup(self._precompute(embeddings))

    def _init_embed(self, input):
        """
        Initial Embedding — Eq. (1) and (2) from Section I.

        For BC-CSP, creates separate embeddings for depot and sensors:
            Eq. (1): h_0^(0) = W_depot · [x_0, y_0] + b_depot     (depot, 2→d)
            Eq. (2): h_i^(0) = W_node · [x_i, y_i, p_i] + b_node  (sensors, 3→d)

        Then concatenates them: (B, N+1, d) with depot at index 0.

        Returns:
            (B, N+1, d) tensor of initial node embeddings, ready for the encoder.
        """
        if self.is_csp:
            return self.init_embed(input['loc'])
        elif self.is_bccsp:
            return torch.cat((
                # Eq. (1): depot embedding — Linear(2 → d) applied to [x_0, y_0]
                self.init_embed_depot(input['depot'])[:, None, :],    # (B, 1, d)
                # Eq. (2): sensor embeddings — Linear(3 → d) applied to [x_i, y_i, p_i]
                self.init_embed(torch.cat((
                    input['loc'],                        # (B, N, 2) — sensor coordinates
                    input['packets'][:, :, None]          # (B, N, 1) — packet counts
                ), -1))                                   # → (B, N, 3) → Linear → (B, N, d)
            ), 1)  # Concatenate along node dim → (B, N+1, d)
        else:
            return self.init_embed(input)

    def _select_node(self, probs, mask):
        """
        Node Selection — Eq. (15)-(16) from Section II.

        Two modes:
            "greedy":   π_t = argmax p(·)     [used during evaluation and baseline rollout]
            "sampling": π_t ~ softmax(u/τ)    [used during training for REINFORCE exploration]

        Args:
            probs: (B, N+1) — probability distribution over nodes (after softmax)
            mask:  (B, N+1) — True for infeasible nodes

        Returns:
            logp:     (B,) — log-probability of the selected node
            selected: (B,) — index of the selected node
        """
        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            # Eq. (16) greedy: pick the node with highest probability
            logp, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            # Eq. (16) sampling: draw from the categorical distribution
            selected = probs.multinomial(1).squeeze(1)
            logp = probs.gather(-1, selected.unsqueeze(1)).squeeze(1)
            # Safety: resample if an infeasible node was selected (rare GPU bug)
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)
                logp = probs.gather(-1, selected.unsqueeze(1)).squeeze(1)
        else:
            assert False, "Unknown decode type"
        return logp.log(), selected

    def _precompute(self, embeddings, num_steps=1):
        """
        Precompute fixed decoder context from encoder output (done ONCE per instance).

        From the encoder's node embeddings h_i^(L):
            1. graph_embed = h̄ = mean(h_i^(L))                       [end of Section I]
            2. context_node_projected = W_c · h̄                       [Section II: s_0 init]
            3. glimpse_key, glimpse_val, logit_key from projection     [used in Eq. (12)-(13)]

        The projection: project_node_embeddings maps each h_i to 3 vectors of size d:
            [glimpse_key_i ; glimpse_val_i ; logit_key_i] = W_proj · h_i

        Returns:
            AttentionModelFixed — named tuple with all precomputed tensors
        """
        # h̄ = (1/(N+1)) Σ h_i^(L)  — graph-level embedding
        graph_embed = embeddings.mean(1)  # (B, d)

        # s_0 = W_c · h̄  — initial GRU hidden state [Section II]
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]  # (B, 1, d)

        # Project node embeddings for decoder attention
        # Each node h_i → [glimpse_key_i, glimpse_val_i, logit_key_i]
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # Reshape glimpse keys/values into multi-head format: (M, B, 1, N+1, d_k)
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),   # (M, B, 1, N+1, d_k)
            self._make_heads(glimpse_val_fixed, num_steps),   # (M, B, 1, N+1, d_k)
            logit_key_fixed.contiguous()                       # (B, 1, N+1, d) — single head
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _inner_rnn(self, input, embeddings):
        """
        Main DECODER LOOP — Section II of the LaTeX.

        Autoregressively constructs a tour by repeatedly calling _get_log_p_rnn()
        and _select_node() until all instances in the batch have returned to the depot.

        Flow at each step t:
            1. _get_log_p_rnn() computes log-probabilities over all nodes  [Eq. (9)-(15)]
            2. _select_node() picks the next node (greedy or sampling)     [Eq. (16)]
            3. state.update() transitions the environment                  [Section III]

        The shrink_size optimization removes finished instances from the batch
        to avoid wasting compute on instances that have already returned to depot.

        Args:
            input:      dict — the raw problem instance
            embeddings: (B, N+1, d) — encoder output h_i^(L)

        Returns:
            outputs:   (B, T) — log-probabilities of selected nodes at each step
            sequences: (B, T) — selected node indices at each step (the tour π)
        """
        outputs = []
        sequences = []

        # Initialize the problem state (Section III: StateBCCSP)
        state = self.problem.make_state(input)

        # ====================================================================
        # PRECOMPUTE fixed decoder context (done once)
        # ====================================================================
        # fixed contains:
        #   node_embeddings:         (B, N+1, d)       — encoder output, used as GRU input e_t
        #   context_node_projected:  (B, 1, d)         — W_c · h̄, used as GRU initial hidden s_0
        #   glimpse_key/val:         (M, B, 1, N+1, d_k) — for glimpse attention [Eq. (12)]
        #   logit_key:               (B, 1, N+1, d)    — for final logit computation [Eq. (13)]
        fixed = self._precompute(embeddings)

        batch_size = state.ids.size(0)

        last_hh = None  # GRU hidden state s_t; initialized to W_c · h̄ on first step

        # ====================================================================
        # AUTOREGRESSIVE DECODING LOOP
        # ====================================================================
        i = 0
        while not (self.shrink_size is None and state.all_finished()):

            # --- Batch shrinking optimization ---
            # Remove finished instances to save compute (instances that returned to depot)
            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                if 1 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    state = state[unfinished]
                    fixed = fixed[unfinished]
                    last_hh = last_hh.transpose(0, 1)[unfinished].transpose(0, 1)

            # --- One decoding step: compute log-probs ---
            # This calls the GRU, dynamic modulation, glimpse attention, and logit computation
            # Implements Eq. (9) through (15) of the LaTeX
            log_p, mask, last_hh = self._get_log_p_rnn(fixed, state, normalize=True, last_hh=last_hh)

            # --- Select next node ---
            # Eq. (16): greedy argmax or categorical sampling
            logp_selected, selected = self._select_node(
                log_p.exp()[:, 0, :],  # Convert log-probs to probs, squeeze step dim
                mask[:, 0, :]
            )

            # --- Update environment state ---
            # Section III: move to selected node, update coverage, tour length, visited mask
            state = state.update(selected)

            # --- Unshrink: pad back to original batch size if shrunk ---
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = logp_selected, selected
                logp_selected = log_p_.new_zeros(batch_size)
                selected = selected_.new_zeros(batch_size) - 1  # -1 = padding

                logp_selected[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            outputs.append(logp_selected)
            sequences.append(selected)
            i += 1

        # Stack across time steps: (B, T)
        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def _get_log_p_rnn(self, fixed, state, normalize=True, last_hh=None):
        """
        One decoding step — implements Eq. (9) through (15) of Section II.

        Steps:
            1. Get decoder input e_t (embedding of current node)
            2. GRU forward:    q_t, s_t = GRU(e_t, s_{t-1})            [Eq. (9)]
            3. Dynamic modulation: K'_logit = K_logit ⊙ (1 + W_δ·δ_t)  [Eq. (10)-(11)]
            4. Compute mask (feasibility from Section III)
            5. Glimpse attention + logits                                [Eq. (12)-(15)]

        Args:
            fixed:    AttentionModelFixed — precomputed decoder context
            state:    StateBCCSP — current environment state
            normalize: if True, apply log_softmax with temperature τ
            last_hh:  (1, B, d) — previous GRU hidden state s_{t-1}

        Returns:
            log_p:   (B, 1, N+1) — log-probabilities over all nodes
            mask:    (B, 1, N+1) — infeasibility mask
            last_hh: (1, B, d)   — updated GRU hidden state s_t
        """
        current_node = state.get_current_node()  # (B, 1) — index of current position
        batch_size, num_steps = current_node.size()

        # ====================================================================
        # STEP 1: Get decoder input e_t — embedding of the current node
        # ====================================================================
        if state.i.item() == 0:
            # First step: use depot embedding (index 0) as input
            if self.is_csp:
                decoder_input = self.first_placeholder[None, None, :].expand(batch_size, 1, -1)
            else:
                # e_0 = h_0^(L)  (depot's encoder embedding)
                decoder_input = fixed.node_embeddings[:, 0:1, :]  # (B, 1, d)
        else:
            # Subsequent steps: look up the embedding of the previously selected node
            # e_t = h_{π_{t-1}}^(L)
            decoder_input = fixed.node_embeddings.gather(
                1,
                current_node[:, :, None].expand(batch_size, 1, fixed.node_embeddings.size(-1))
            )  # (B, 1, d)

        # ====================================================================
        # STEP 2: GRU forward — Eq. (9): q_t, s_t = GRU(e_t, s_{t-1})
        # ====================================================================
        if last_hh is None:
            # Initialize hidden state: s_0 = W_c · h̄  (projected graph embedding)
            # context_node_projected is (B, 1, d), GRU expects (1, B, d) for hidden
            last_hh = fixed.context_node_projected.transpose(1, 0)  # (1, B, d)

        self.gru.flatten_parameters()  # Optimize for CUDA
        rnn_out, last_hh = self.gru(decoder_input, last_hh)
        # rnn_out: (B, 1, d) — this is the query q_t
        # last_hh: (1, B, d) — this is the updated hidden state s_t

        query = rnn_out  # q_t ∈ ℝ^d — used for glimpse attention [Eq. (12)]

        # ====================================================================
        # STEP 3: Get attention keys/values and apply DYNAMIC MODULATION
        # ====================================================================
        # Get the (static) glimpse K, V and logit K from precomputed fixed context
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # --- Eq. (10)-(11): Dynamic Feature Modulation ---
        # δ_t = state.get_dynamic() → (B, 1, N+1) marginal coverage gain per node
        # W_δ · δ_t → project scalar to d-dim: (B, N+1, 1) → (B, N+1, d)
        # K'_logit = K_logit ⊙ (1 + W_δ · δ_t)  [multiplicative gating]
        dynamic_hidden = self.init_dynamic(state.get_dynamic().transpose(1, 2))  # (B, N+1, d)
        logit_K = logit_K.mul(1 + dynamic_hidden.unsqueeze(1))  # (B, 1, N+1, d)
        # Nodes with high marginal coverage gain get amplified logit keys;
        # nodes that add little new coverage get dampened.

        # ====================================================================
        # STEP 4: Compute feasibility mask (from Section III)
        # ====================================================================
        mask = state.get_mask()  # (B, 1, N+1) — True = infeasible

        # ====================================================================
        # STEP 5: Glimpse attention + final logits — Eq. (12)-(15)
        # ====================================================================
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        # Apply temperature and log-softmax: log softmax(u / τ) [Eq. (15)]
        if normalize:
            log_p = F.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask, last_hh

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        """
        Glimpse Attention + Final Logit Computation — Eq. (12)-(14) from Section II.

        This function:
            1. Computes multi-head attention (glimpse) between query q_t and all nodes
               → produces context vector g_t                           [Eq. (12)]
            2. Computes final logits: u_j = g_t · K'_logit,j / √d     [Eq. (13)]
            3. Applies tanh clipping: u_j ← C · tanh(u_j), C=10       [Eq. (14)]
            4. Masks infeasible nodes to -∞

        Args:
            query:     (B, 1, d)           — q_t from the GRU
            glimpse_K: (M, B, 1, N+1, d_k) — glimpse attention keys
            glimpse_V: (M, B, 1, N+1, d_k) — glimpse attention values
            logit_K:   (B, 1, N+1, d)      — logit keys (after dynamic modulation)
            mask:      (B, 1, N+1)         — True where infeasible

        Returns:
            logits:  (B, 1, N+1) — raw logits (before log_softmax)
            glimpse: (B, 1, d)   — the glimpse context vector g_t
        """
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads  # d_k = 128/8 = 16

        # ====================================================================
        # Eq. (12): GLIMPSE ATTENTION — multi-head attention from q_t to all nodes
        # ====================================================================
        # Reshape query into M heads: (B, 1, d) → (M, B, 1, 1, d_k)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Attention compatibility: Q · K^T / √d_k
        # (M, B, 1, 1, d_k) × (M, B, 1, d_k, N+1) → (M, B, 1, 1, N+1)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))

        # Mask infeasible nodes in the glimpse attention
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Weighted sum of values: softmax(compatibility) · V
        # (M, B, 1, 1, N+1) × (M, B, 1, N+1, d_k) → (M, B, 1, 1, d_k)
        heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)

        # Concatenate M heads and project with W_O: g_t ∈ ℝ^d
        # (M, B, 1, 1, d_k) → (B, 1, 1, M*d_k) → (B, 1, 1, d)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        final_Q = glimpse  # g_t — the glimpse context vector

        # ====================================================================
        # Eq. (13): FINAL LOGITS  u_j = g_t^T · K'_logit,j / √d
        # ====================================================================
        # (B, 1, 1, d) × (B, 1, d, N+1) → (B, 1, 1, N+1) → squeeze → (B, 1, N+1)
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # ====================================================================
        # Eq. (14): TANH CLIPPING  u_j ← C · tanh(u_j), C = 10
        # ====================================================================
        if self.tanh_clipping > 0:
            logits = F.tanh(logits) * self.tanh_clipping

        # Mask infeasible nodes → -∞ (so softmax gives them 0 probability)
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):
        """
        Retrieve the precomputed glimpse K, V and logit K.

        For most problems (including BC-CSP), these are simply the static
        projections from _precompute(). The SDVRP case adds step-dependent
        demand information.

        Returns:
            glimpse_K: (M, B, 1, N+1, d_k)
            glimpse_V: (M, B, 1, N+1, d_k)
            logit_K:   (B, 1, N+1, d)
        """
        if self.is_vrp and self.allow_partial:
            glimpse_key_step, glimpse_val_step, logit_key_step = \
                self.project_node_step(state.demands_with_depot[:, :, :, None].clone()).chunk(3, dim=-1)
            return (
                fixed.glimpse_key + self._make_heads(glimpse_key_step),
                fixed.glimpse_val + self._make_heads(glimpse_val_step),
                fixed.logit_key + logit_key_step,
            )
        # For BC-CSP / TSP / OP / etc.: return static precomputed projections
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        """Reshape a tensor into multi-head format: (B, steps, N+1, d) → (M, B, steps, N+1, d_k)."""
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (M, B, steps, N+1, d_k)
        )