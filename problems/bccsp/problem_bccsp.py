from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.bccsp.state_bccsp import StateBCCSP
from utils.beam_search import beam_search

class BCCSP(object):
    """
    Budget-Constrained Covering Salesman Problem (BC-CSP)

    - Depot fixed at (0,0)
    - Choose a route that starts at depot and ends at depot 
    - Must respect max_length (budget)
    - Objective: maximize covered packets (union of radius coverage of visited nodes)
    """

    NAME = "bccsp"

    @staticmethod
    def get_costs(dataset, pi):
        """
        dataset: dict with keys

        - loc: (B, N, 2)
        - packets: (B, N) float (originally integer 1...100)
        - depot: (B, 2) (expected to be zeros)
        - max_length: (B,) or (B, 1)
        - radius: scalar tensor or (B,) or (B,1)
        
        pi: (B, T) sequence of actions in {0..N} where 0 is depot, 1..N correspond to 
        loc[*, idx-1] should end at depot (0). (state enforces this during decoding)
        """

        if pi.size(-1) == 1:
            assert (pi == 0).all(), "If all length-1 tours, they should be depot=0"
            return torch.zeros(pi.size(0), dtype=torch.float, device=pi.device), None
        
        # Validate: no duplicates except depot zeros (same as OP style)
        sorted_pi = pi.data.sort(1)[0]

        # Filter out -1 padding for validation
        for i in range(sorted_pi.size(0)):
            tour = sorted_pi[i][sorted_pi[i] >= 0]  # Only valid nodes (including depot=0)
            if tour.numel() > 1:
                # Check no duplicates except depot (0)
                if not ((tour[1:] == 0) | (tour[1:] > tour[:-1])).all():
                    assert False, f"Duplicates in tour {i}: {pi[i].tolist()}"

        B, N, _ = dataset["loc"].size()

        # Build coords with depot at index 0
        depot = dataset["depot"]
        loc_with_depot = torch.cat((depot[:, None, :], dataset["loc"]), dim=1)  # (B, N+1, 2)

        # Path length: depot -> ... -> depot (same formula as OP problem)
        # BUT we need to handle -1 padding!
        lengths = torch.zeros(B, device=pi.device)
        for i in range(B):
            tour = pi[i][pi[i] >= 0]  # Filter out -1 padding
            if tour.numel() == 1:
                # Just depot, no movement
                lengths[i] = 0.0
            else:
                # Get coordinates for this tour
                coords = loc_with_depot[i, tour, :]  # (T, 2)
                # Calculate tour length
                lengths[i] = (
                    (coords[1:] - coords[:-1]).norm(p=2, dim=-1).sum()  # Segment lengths
                    + (coords[0] - depot[i]).norm(p=2, dim=-1)           # Start from depot
                    + (coords[-1] - depot[i]).norm(p=2, dim=-1)          # Return to depot
                )

        max_length = dataset["max_length"]
        if max_length.dim() > 1:
            max_length = max_length.squeeze(-1)
        assert (lengths <= max_length + 1e-5).all(), \
            "Max length exceeded by {}".format((lengths - max_length).max())

        # Coverage reward: union of radius disks centered at visited non-depot nodes.
        # Important: coverage is over ALL sensors (the N nodes), not just visited.
        packets = dataset["packets"]  # (B, N)

        radius = dataset["radius"]
        # normalize radius to shape (B,) float
        if isinstance(radius, (float, int)):
            radius = torch.tensor(radius, device=pi.device, dtype=torch.float).repeat(B)
        elif torch.is_tensor(radius):
            radius = radius.to(pi.device).float()
            if radius.numel() == 1:
                radius = radius.view(1).repeat(B)
            elif radius.dim() > 1:
                radius = radius.view(B, -1)[:, 0]  # allow (B,1) or (1,B) etc
        else:
            raise ValueError("Unsupported radius type")

        covered = torch.zeros(B, N, dtype=torch.bool, device=pi.device)

        # Iterate through steps; N is small (20/50/100), so this is fine.
        # For each selected node a in pi: if a > 0, center = loc[a-1]
        for t in range(pi.size(1)):
            a = pi[:, t]  # (B,)
            is_node = a > 0
            if not is_node.any():
                continue
            idx = (a[is_node] - 1).long()  # indices into loc
            centers = dataset["loc"][is_node, idx, :]  # (b', 2)

            # distances from each center to all nodes in that instance
            # loc_sub: (b', N, 2)
            loc_sub = dataset["loc"][is_node, :, :]
            dist = (loc_sub - centers[:, None, :]).norm(p=2, dim=-1)  # (b', N)

            newly = dist <= radius[is_node][:, None]
            covered[is_node] |= newly

        covered_prize = (packets * covered.float()).sum(-1)  # (B,)

        # We want to maximize covered_prize but framework minimizes cost
        return -covered_prize, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return BCCSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateBCCSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = BCCSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def generate_instance(size, radius=0.15, max_length=2.0, packets_low=1, packets_high=100):
    """
    size: number of sensor nodes
    radius: scalar
    max_length: budget (total travel length constraint)
    packets: integer in [packets_low, packets_high]
    """
    loc = torch.rand(size, 2)
    depot = torch.zeros(2)  # (0,0)

    packets = torch.randint(low=packets_low, high=packets_high + 1, size=(size,), dtype=torch.int64).float()

    return {
        "loc": loc,
        "depot": depot,
        "packets": packets,
        "max_length": torch.tensor(max_length, dtype=torch.float),
        "radius": torch.tensor(radius, dtype=torch.float),
    }


class BCCSPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0,
                 radius=0.15, max_length=None, distribution=None):
        """
        If filename is provided, expects pkl containing tuples:
          (loc, packets, max_length, radius)
        where:
          loc: (N,2)
          packets: (N,)
          max_length: float
          radius: float

        depot is always reconstructed as (0,0) on load.

        distribution: ignored for BC-CSP (kept for compatibility)
        """
        super(BCCSPDataset, self).__init__()

        if max_length is None:
            # You can tune these; OP uses {20:2,50:3,100:4}. :contentReference[oaicite:4]{index=4}
            MAX_LENGTHS = {20: 2.0, 50: 3.0, 100: 4.0}
            max_length = MAX_LENGTHS.get(size, 3.0)

        if filename is not None:
            assert os.path.splitext(filename)[1] == ".pkl"
            with open(filename, "rb") as f:
                data = pickle.load(f)

            self.data = [
                {
                    "loc": torch.FloatTensor(loc),
                    "depot": torch.zeros(2, dtype=torch.float),
                    "packets": torch.FloatTensor(packets),
                    "max_length": torch.tensor(ml, dtype=torch.float),
                    "radius": torch.tensor(r, dtype=torch.float),
                }
                for (loc, packets, ml, r) in data[offset:offset + num_samples]
            ]
        else:
            self.data = [
                generate_instance(size=size, radius=radius, max_length=max_length)
                for _ in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

