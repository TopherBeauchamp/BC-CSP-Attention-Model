import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateCSP(NamedTuple):
    # Fixed input
    loc: torch.Tensor
    dist: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance,
    # loc and dist are not duplicated; ids index the original rows.
    ids: torch.Tensor  # (batch, 1)

    # State
    first_a: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor     # bool mask, (batch, 1, n_loc)
    mask_cover: torch.Tensor   # bool mask, (batch, 1, n_loc)

    cover_range: torch.Tensor  # scalar tensor (kept for compatibility, not used by radius-CSP)
    radius: torch.Tensor       # scalar tensor, coverage distance

    dynamic: torch.Tensor
    dynamic_updation: torch.Tensor

    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor

    @property
    def visited(self):
        # Always return boolean mask
        if self.visited_.dtype == torch.bool:
            return self.visited_
        return mask_long2bool(self.visited_, n=self.loc.size(-2))

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return self._replace(
                ids=self.ids[key],
                first_a=self.first_a[key],
                prev_a=self.prev_a[key],
                visited_=self.visited_[key],
                mask_cover=self.mask_cover[key],
                dynamic=self.dynamic[key],
                dynamic_updation=self.dynamic_updation[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
            )
        assert False, "Fallback __getitem__ removed for Python 3.8+ NamedTuple compatibility"

    @staticmethod
    def _scalar_int(x, default: int):
        if x is None:
            return default
        if torch.is_tensor(x):
            return int(x.reshape(-1)[0].item())
        return int(x)

    @staticmethod
    def _scalar_float(x, default: float):
        if x is None:
            return default
        if torch.is_tensor(x):
            return float(x.reshape(-1)[0].item())
        return float(x)

    @staticmethod
    def initialize(input, visited_dtype=torch.bool):
        """
        Radius-based CSP coverage:
        - selecting node j covers all nodes within `radius` of node j (including j).
        - episode ends when all nodes are covered.
        """
        loc = input["loc"]                       # (batch, n, 2)
        cover_range = input.get("cover_range")   # might be int/tensor/shape weird
        radius = input.get("radius")             # might be float/tensor/shape weird

        batch_size, n_loc, _ = loc.size()
        device = loc.device

        # make them scalars safely
        cover_range_val = StateCSP._scalar_int(cover_range, default=7)
        radius_val = StateCSP._scalar_float(radius, default=0.15)

        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        # bool masks (modern PyTorch expects bool)
        visited_mask = torch.zeros(batch_size, 1, n_loc, dtype=torch.bool, device=device)
        cover_mask = torch.zeros(batch_size, 1, n_loc, dtype=torch.bool, device=device)

        # keep these as tensors so existing code doesn't break
        cover_range_tensor = torch.tensor(cover_range_val, device=device, dtype=torch.long)
        radius_tensor = torch.tensor(radius_val, device=device, dtype=torch.float)

        # dynamic stuff is repo-specific; keep it stable and shape-safe
        if cover_range_val <= 0:
            dyn_upd = torch.zeros(batch_size, 1, 0, dtype=torch.float, device=device)
        else:
            dyn_upd = torch.arange(cover_range_val, device=device, dtype=torch.float).view(1, 1, -1)
            dyn_upd = dyn_upd.expand(batch_size, 1, -1) / float(cover_range_val)

        return StateCSP(
            loc=loc,
            dist=(loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1),
            ids=torch.arange(batch_size, dtype=torch.int64, device=device)[:, None],
            first_a=prev_a,
            prev_a=prev_a,
            visited_=visited_mask,
            mask_cover=cover_mask,
            cover_range=cover_range_tensor,
            radius=radius_tensor,
            dynamic=torch.ones(batch_size, 1, n_loc, dtype=torch.float, device=device),
            dynamic_updation=dyn_upd,
            lengths=torch.zeros(batch_size, 1, device=device),
            cur_coord=None,
            i=torch.zeros(1, dtype=torch.int64, device=device),
        )

    def get_final_cost(self):
        assert self.all_finished()
        return self.lengths + (self.loc[self.ids, self.first_a, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):
        """
        selected: (batch,) long indices
        After visiting node j, cover all nodes within radius of node j.
        """
        batch_size = selected.size(0)
        prev_a = selected[:, None]  # (batch, 1)

        cur_coord = self.loc[self.ids, prev_a]  # (batch, 1, 2)

        lengths = self.lengths
        if self.cur_coord is not None:
            lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)

        first_a = prev_a if self.i.item() == 0 else self.first_a

        # visited
        if self.visited_.dtype == torch.bool:
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], True)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a)

        # mask covered cities (radius-based CSP coverage)
        # Properly index into loc using ids
        # self.loc shape: (original_batch, n_loc, 2)
        # self.ids shape: (current_batch, 1)
        # cur_coord shape: (current_batch, 1, 2)
        
        # Get locations for current batch instances
        batch_locs = self.loc[self.ids.squeeze(-1)]  # (current_batch, n_loc, 2)
        
        # Calculate distances from current coordinate to all nodes
        # batch_locs: (current_batch, n_loc, 2)
        # cur_coord: (current_batch, 1, 2)
        dists = (batch_locs - cur_coord).norm(p=2, dim=-1).squeeze(1)  # (current_batch, n_loc)

        mask_cover = self.mask_cover.clone()
        dynamic = self.dynamic.clone()

        # radius can be scalar tensor or shape (1,1) etc.
        r = self.radius
        if torch.is_tensor(r) and r.numel() == 1:
            r_val = float(r.item())
        else:
            # If you later want per-node radii, you can support it here.
            # For now, treat as scalar-ish.
            r_val = float(r.reshape(-1)[0].item())

        # Vectorized coverage update
        covered = dists < r_val  # (current_batch, n_loc) boolean mask
        
        # DEBUG: prove radius is used (print once)
        if not hasattr(StateCSP, "_radius_debug_printed"):
            StateCSP._radius_debug_printed = True
            total_covered = covered.sum().item()
            total_nodes = covered.numel()
            print(f"[DEBUG state_csp_cover] radius={r_val:.4f} covered_after_step={total_covered}/{total_nodes}")
        
        # Update mask_cover for all covered nodes
        mask_cover[:, 0, :] = mask_cover[:, 0, :] | covered
        
        # Update dynamic values for covered nodes
        # For each batch, update dynamics
        for b in range(batch_size):
            n_idx = covered[b].nonzero(as_tuple=False).squeeze(-1)
            
            if n_idx.numel() == 0:
                continue
            
            # Optional: dynamic update (keep your intention, but safe)
            dynamic[b, 0, n_idx[0]] = 0.0
            if n_idx.numel() > 1:
                rest = n_idx[1:]
                upd = torch.arange(rest.numel(), device=rest.device, dtype=torch.float32)
                upd = upd / max(rest.numel(), 1)
                dynamic[b, 0, rest] = dynamic[b, 0, rest] * upd

        return self._replace(
            first_a=first_a,
            prev_a=prev_a,
            visited_=visited_,
            mask_cover=mask_cover,
            dynamic=dynamic,
            lengths=lengths,
            cur_coord=cur_coord,
            i=self.i + 1,
        )

    def all_finished(self):
        return self.mask_cover.all()

    def get_finished(self):
        return self.mask_cover.sum(-1) == self.mask_cover.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_dynamic(self):
        return self.dynamic

    def get_mask(self):
        return self.visited

    def get_nn(self, k=None):
        if k is None:
            k = self.loc.size(-2) - self.i.item()
        return (self.dist[self.ids, :, :] + self.visited.float()[:, :, None, :] * 1e6).topk(
            k, dim=-1, largest=False
        )[1]

    def get_nn_current(self, k=None):
        assert False, "Not implemented"

    def construct_solutions(self, actions):
        return actions