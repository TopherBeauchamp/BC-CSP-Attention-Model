import torch 
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter

class StateBCCSP(NamedTuple):
    # Fixed Input 
    coords: torch.Tensor      # (B, N+1, 2) depot + loc
    loc: torch.Tensor         # (B, N, 2) loc only
    packets: torch.Tensor     # (B, N) packets per sensor node
    radius: torch.Tensor      # (B,) scalar radius per instance
    max_length: torch.Tensor  # (B, N+1) per-node feasibility threshold (like OP's max_length trick)

    # For beam/multi-copy
    ids: torch.Tensor

    # State
    prev_a: torch.Tensor
    visited_: torch.Tensor
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    covered_: torch.Tensor       # (B, 1, N) bool mask of covered sensors
    cur_total_covered: torch.Tensor
    i: torch.Tensor

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        return mask_long2bool(self.visited_, n=self.coords.size(-2))

    @property
    def covered(self):
        # covered_ always stored as bool
        return self.covered_

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
                covered_=self.covered_[key],
                cur_total_covered=self.cur_total_covered[key],
            )
        assert False, "Fallback __getitem__ removed for Python 3.8+ NamedTuple compatibility"

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        """
        input keys:
          - depot: (B,2) expected (0,0)
          - loc: (B,N,2)
          - packets: (B,N)
          - max_length: (B,) or (B,1)
          - radius: scalar or (B,) or (B,1)
        """

        depot = input["depot"]
        loc = input["loc"]
        packets = input["packets"].float()

        B, N, _ = loc.size()
        coords = torch.cat((depot[:, None, :], loc), dim=-2)  # (B, N+1, 2)

        # Normalize radius to shape (B,)
        r = input["radius"]
        if torch.is_tensor(r):
            r = r.to(loc.device).float()
            if r.numel() == 1:
                r = r.view(1).repeat(B)
            elif r.dim() > 1:
                r = r.view(B, -1)[:, 0]
        else:
            r = torch.tensor(float(r), device=loc.device).repeat(B)

        # Like OP: store feasibility threshold per node = max_length - dist(node, depot) - eps :contentReference[oaicite:5]{index=5}
        ml = input["max_length"]
        if torch.is_tensor(ml):
            ml = ml.to(loc.device).float()
            if ml.dim() > 1:
                ml = ml.squeeze(-1)
        else:
            ml = torch.tensor(float(ml), device=loc.device).repeat(B)

        max_length = ml[:, None] - (depot[:, None, :] - coords).norm(p=2, dim=-1) - 1e-6  # (B, N+1)

        return StateBCCSP(
            coords=coords,
            loc=loc,
            packets=packets,
            radius=r,
            max_length=max_length,
            ids=torch.arange(B, dtype=torch.int64, device=loc.device)[:, None],
            prev_a=torch.zeros(B, 1, dtype=torch.long, device=loc.device),
            visited_=(
                torch.zeros(B, 1, N + 1, dtype=torch.uint8, device=loc.device)
                if visited_dtype == torch.uint8
                else torch.zeros(B, 1, (N + 1 + 63) // 64, dtype=torch.int64, device=loc.device)
            ),
            lengths=torch.zeros(B, 1, device=loc.device),
            cur_coord=depot[:, None, :],
            covered_=torch.zeros(B, 1, N, dtype=torch.bool, device=loc.device),
            cur_total_covered=torch.zeros(B, 1, device=loc.device),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),
        )

    def all_finished(self):
        # Same end rule as OP: once we returned to depot (0) after at least one step, we stop. :contentReference[oaicite:6]{index=6}
        return self.i.item() > 0 and (self.prev_a == 0).all()
    
    def get_finished(self):
        """
        Returns a (B,) tensor indicating which instances have finished.
        An instance is finished when it has returned to the depot after at least one step.
        """
        return (self.i.item() > 0) & (self.prev_a == 0).squeeze(-1)
    
    def get_current_node(self):
        return self.prev_a

    def update(self, selected):
        assert self.i.size(0) == 1, "Can only update if state represents single step"

        selected = selected[:, None]  # (B,1)
        prev_a = selected

        # Move + update length
        cur_coord = self.coords[self.ids, selected]  # (B,1,2)
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (B,1)

        # Update coverage ONLY if selecting a sensor node (a>0)
        covered_ = self.covered_
        cur_total_covered = self.cur_total_covered

        a = selected.squeeze(1)  # (B,)
        is_node = a > 0
        if is_node.any():
            # Get the actual batch indices (accounting for shrinking)
            batch_idx = torch.arange(a.size(0), device=a.device)[is_node]  # Which instances in current batch
            original_idx = self.ids[batch_idx, 0]  # Map to original batch indices
            
            idx = (a[is_node] - 1).long()  # Sensor indices (0..N-1)
            centers = self.loc[original_idx, idx, :]  # (b', 2)
            loc_sub = self.loc[original_idx, :, :]    # (b', N, 2)

            dist = (loc_sub - centers[:, None, :]).norm(p=2, dim=-1)  # (b',N)
            newly = dist <= self.radius[original_idx][:, None]  # ✅ FIXED (b',N)

            old_cov = covered_[batch_idx, 0, :]  # (b',N)
            add = newly & (~old_cov)

            # add newly-covered packets once
            add_packets = (self.packets[original_idx, :] * add.float()).sum(-1, keepdim=True)  # ✅ FIXED (b',1)
            cur_total_covered = cur_total_covered.clone()
            cur_total_covered[batch_idx, :] = cur_total_covered[batch_idx, :] + add_packets

            covered_ = covered_.clone()
            covered_[batch_idx, 0, :] = old_cov | newly

        # Update visited mask (same approach as OP) :contentReference[oaicite:7]{index=7}
        if self.visited_.dtype == torch.uint8:
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a, check_unset=False)

        return self._replace(
            prev_a=prev_a,
            visited_=visited_,
            lengths=lengths,
            cur_coord=cur_coord,
            covered_=covered_,
            cur_total_covered=cur_total_covered,
            i=self.i + 1,
        )

    def get_mask(self):
        """
        Feasible actions:
          - not already visited (except depot can be revisited to terminate)
          - must be able to go to that node AND still be within budget to return to depot,
            implemented the same way as OP (length upon arrival <= max_length[node]) :contentReference[oaicite:8]{index=8}
          - after depot is visited (i.e. returned), nothing else allowed (terminate)
        """
        visited_ = self.visited

        # predicted arrival length if go from cur_coord -> each node
        travel = (self.coords[self.ids, :, :] - self.cur_coord[:, :, None, :]).norm(p=2, dim=-1)  # (B,1,N+1)

        mask = (
            visited_ |
            visited_[:, :, 0:1] |                     # if depot already visited (returned), stop
            (self.lengths[:, :, None] + travel > self.max_length[self.ids, :])  # budget feasibility
        )

        # depot can always be visited (to terminate)
        mask[:, :, 0] = 0
        return mask.bool()

    def construct_solutions(self, actions):
        return actions
    
    def get_dynamic(self):
        """
        Dynamic context: (B, 1, N+1) 
        - Depot gets 0
        - Sensors get value based on uncovered packets in their radius
        """
        B = self.ids.size(0)
        
        # For each sensor node, estimate marginal coverage value
        # This is approximate: how many uncovered packets are in radius?
        dynamic = torch.zeros(B, 1, self.coords.size(1), device=self.ids.device)
        
        # Simple heuristic: nodes with more uncovered packets nearby = higher value
        for b in range(B):
            for j in range(self.loc.size(1)):  # for each sensor
                if self.visited[b, 0, j + 1]:  # +1 because depot is index 0
                    continue
                
                # Count uncovered packets within radius of node j
                center = self.loc[b, j, :]
                dist = (self.loc[b, :, :] - center[None, :]).norm(p=2, dim=-1)
                in_range = dist <= self.radius[b]
                uncovered_in_range = in_range & (~self.covered_[b, 0, :])
                
                marginal_packets = (self.packets[b, :] * uncovered_in_range.float()).sum()
                dynamic[b, 0, j + 1] = marginal_packets  # j+1 because depot is 0
        
        # Normalize to [0, 1] range for stability
        dynamic = dynamic / (dynamic.max() + 1e-8)
        return dynamic