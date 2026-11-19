import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data as GeometricData

import torch


# -------------------------------------------------------------------
# 1) Two-hop exclusion using a sparse edge list
# -------------------------------------------------------------------
def mask_hop_neighbors(edge_index: torch.Tensor,
                       y: torch.Tensor,
                       num_nodes: int,
                       max_hop: int = 2):
    """
    Return a boolean mask of positive nodes and their max_hop-hop neighbors.

    True entries mark nodes excluded from negative sampling. This implementation
    uses an undirected graph; the local follow setting controls traversal direction.
    """
    undirected: bool = True,
    follow: str = "out"
    device = edge_index.device
    assert y.numel() == num_nodes, f"y length ({y.numel()}) must equal num_nodes ({num_nodes})"

    # 1) Create a boolean indicator for positive nodes.
    v = (y == 1).to(torch.bool).view(-1, 1).to(device)

    # 2) Store A[dst, src] = 1 so A @ frontier follows outgoing edges.
    ei = edge_index
    if undirected:
        ei = torch.cat([ei, ei[[1, 0]]], dim=1)  # Symmetrize the edge list.

    if follow == "out":  # src -> dst
        rows, cols = ei[1], ei[0]  # A[dst,src]
    elif follow == "in":  # dst -> src
        rows, cols = ei[0], ei[1]  # A[src,dst]
    else:
        raise ValueError('follow must be "out" or "in"')

    values = torch.ones(rows.size(0), dtype=torch.float32, device=device)
    A = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), values, (num_nodes, num_nodes), device=device
    ).coalesce()

    # 3) Expand one hop at a time, using boolean masks for unions.
    reached = v.clone()  # Include the positive nodes themselves.
    frontier = v.clone()
    for _ in range(max_hop):
        frontier = torch.sparse.mm(A, frontier.float())  # float for mm
        frontier = (frontier > 0)  # bool for union
        reached |= frontier  # bool OR
        frontier = frontier.to(torch.float32)  # float for next mm

    mask = reached.view(-1)
    node_ids = mask.nonzero(as_tuple=False).view(-1)

    return reached.squeeze(1).bool()


# -------------------------------------------------------------------
# 2) Build edge_index and edge_weight from an edge list
# -------------------------------------------------------------------
def read_edge_list(edge_list_filename):
    """
    Read a CSV edge list and return edge indices, weights, and relation IDs.

    The first two columns contain one-based node IDs. An optional third column
    provides edge weights; otherwise all weights are set to one.
    """
    df_edges = pd.read_csv(edge_list_filename)
    dst = torch.tensor(df_edges.iloc[:, 0].to_numpy() - 1, dtype=torch.long)
    src = torch.tensor(df_edges.iloc[:, 1].to_numpy() - 1, dtype=torch.long)

    if df_edges.shape[1] >= 3:
        weight = torch.tensor(df_edges.iloc[:, 2].to_numpy(), dtype=torch.float32)
    else:
        weight = torch.ones(src.size(0), dtype=torch.float32)

    edge_index = torch.stack([src, dst], dim=0)
    edge_weight = weight

    edge_type = np.zeros(len(src), dtype=np.int64).tolist()
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    return edge_index, edge_weight, edge_type


# -------------------------------------------------------------------
# 3) Balance and split the dataset using stratification
# -------------------------------------------------------------------
def get_split_indices(content_filename,
                      edge_list_filename,
                      seed=42,
                      ratios=(0.6, 0.2, 0.2)):
    """
    Read labels and edges, then exclude negative nodes within two hops of positives.

    Randomly sample as many eligible negatives as positives and split the balanced
    sample into training, validation, and test sets using stratification.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"

    # --- 1) Labels ---
    df = pd.read_csv(content_filename)
    y = torch.tensor(df['Presence'].to_numpy(), dtype=torch.long)
    N = y.size(0)

    # --- 2) edge list ---
    edge_index, _, _ = read_edge_list(edge_list_filename)

    # Identify nodes within two hops of positive nodes for exclusion.
    remove_mask = mask_hop_neighbors(edge_index, y, num_nodes=N, max_hop=2)

    pos_idx_all = torch.nonzero(y == 1, as_tuple=False).squeeze(1).numpy()
    # neg_idx_all = torch.nonzero(y == 0, as_tuple=False).squeeze(1).numpy()
    neg_idx_all = torch.nonzero((y == 0) & (~remove_mask), as_tuple=False).squeeze(1).numpy()

    rng = np.random.RandomState(seed)
    rng.shuffle(pos_idx_all)
    rng.shuffle(neg_idx_all)

    if len(neg_idx_all) < len(pos_idx_all):
        raise ValueError(
            f"Only {len(neg_idx_all)} negative samples remain outside the two-hop exclusion zone; "
            f"this is insufficient to balance {len(pos_idx_all)} positive samples. "
            "Reduce the negative-to-positive sampling ratio or the exclusion hop count.")

    neg_idx_balanced = neg_idx_all[:len(pos_idx_all)]
    balanced_idx = np.concatenate([pos_idx_all, neg_idx_balanced])
    balanced_y = y[balanced_idx].numpy()

    train_ratio, val_ratio, test_ratio = ratios
    train_idx, temp_idx = train_test_split(
        balanced_idx,
        test_size=val_ratio + test_ratio,
        stratify=balanced_y,
        random_state=seed,
    )
    temp_y = balanced_y[np.isin(balanced_idx, temp_idx)]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_y,
        random_state=seed,
    )

    # ---------- Compute and print split statistics ----------
    def count_stats(indices):
        subset_y = y[indices]
        pos = int((subset_y == 1).sum().item())
        neg = int((subset_y == 0).sum().item())
        total = len(indices)
        return total, pos, neg, pos / total, neg / total

    sets = {'Train': train_idx, 'Validation': val_idx, 'Test': test_idx}
    print("\n=== Dataset statistics ===")
    for name, indices in sets.items():
        total, pos, neg, pos_r, neg_r = count_stats(indices)
        print(f"{name:10s}: Total={total:4d} | "
              f"Positive={pos:4d} ({pos_r:5.2%}) | "
              f"Negative={neg:4d} ({neg_r:5.2%})")

    return train_idx, val_idx, test_idx


# -------------------------------------------------------------------
# 4) Load the full dataset and return GeometricData
# -------------------------------------------------------------------
def read_data(content_filename, edge_list_filename, split_indices=None):
    # ------- Node features X and labels y -------
    df = pd.read_csv(content_filename)
    y = df['Presence']
    X = df.drop(columns=['OID_', 'Shape_Length', 'Shape_Area', 'Presence'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # ------- Split masks -------
    train_indices, val_indices, test_indices = split_indices
    num_samples = len(y_tensor)
    train_mask = torch.zeros(num_samples, dtype=torch.bool)
    val_mask = torch.zeros(num_samples, dtype=torch.bool)
    test_mask = torch.zeros(num_samples, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    # ------- edge list -------
    edge_index, edge_weight, edge_type = read_edge_list(edge_list_filename)

    # ------- GeometricData -------
    data = GeometricData(
        x=x_tensor,
        y=y_tensor,
        edge_type=edge_type,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        edge_index=edge_index,
        edge_weight=edge_weight
    )

    data.num_features = x_tensor.shape[1]
    data.num_classes = len(torch.unique(y_tensor))
    data.num_relations = 1

    return data
