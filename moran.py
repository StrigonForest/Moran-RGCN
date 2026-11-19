from __future__ import annotations
import numbers
from collections.abc import Sequence
import torch
from torch_geometric.data import Data
from libpysal.weights import W
import esda
from sklearn.preprocessing import StandardScaler
import numpy as np


# -------- Main function --------
def apply_moran_mutli_graph(
        data: Data,
        *,
        feature_idxs: Sequence[int] | int | None = None,  # None = all data.x columns; otherwise data.x[:, idx]
        scale: bool = False,  # Z-score the variables used for LISA
        alpha: float = 0.01,  # Significance threshold
        cluster_keep: tuple[int, ...] = (1, 3),  # Retain the specified quadrants: 1=HH, 2=LH, 3=LL, 4=HL
) -> Data:
    """
    Construct a multirelational graph from local Moran's I.

    Use all data.x columns when feature_idxs is None, or the selected columns
    otherwise. For each feature, connect significant source nodes in retained
    quadrants to their existing neighbors. Relation IDs follow the selected
    feature order, and all edge weights are one. If no edges are selected, reuse
    the original edges with a single relation.

    Return a Data object with x, y, edge_index, edge_type, edge_weight, the input
    split masks, and num_features, num_classes, and num_relations metadata.
    """
    # -------- 1. Select the variables for LISA --------
    if feature_idxs is None:
        feature_idxs = range(data.x.size(1))
    elif isinstance(feature_idxs, numbers.Integral):
        feature_idxs = [feature_idxs]
    else:  # tuple / list
        feature_idxs = list(feature_idxs)

    num_relations = len(feature_idxs)
    rel2feat = {rel: feat for rel, feat in enumerate(feature_idxs)}

    # -------- 2. Convert edge_index to PySAL spatial weights --------
    N = data.x.size(0)
    neighbors = {i: [] for i in range(N)}
    weights = {i: [] for i in range(N)}
    src, dst = data.edge_index.cpu().numpy()
    ew = data.edge_weight.cpu().numpy()
    for s, d, w_val in zip(src, dst, ew):
        neighbors[s].append(d)
        weights[s].append(float(w_val))
    w = W(neighbors, weights)

    # -------- 3. Compute LISA --------
    edge_src, edge_dst, edge_type, edge_weight = [], [], [], []

    for type_idx, feat_idx in rel2feat.items():
        vals = data.x[:, feat_idx].float().cpu().numpy()
        if scale:
            vals = StandardScaler().fit_transform(vals.reshape(-1, 1)).ravel()

        lisa = esda.Moran_Local(vals, w)
        sig_mask = lisa.p_sim < alpha
        q = lisa.q
        Ii = lisa.Is

        # -------- 4. Construct new edges --------
        for i, neigh in w.neighbors.items():
            if sig_mask[i] and q[i] in cluster_keep:
                for j in neigh:
                    edge_src.append(i)
                    edge_dst.append(j)
                    edge_type.append(type_idx)
                    weight = 1
                    edge_weight.append(weight)

            # # Check that the source node is significant and in a retained quadrant.
            # if not (sig_mask[i] and q[i] in cluster_keep):
            #     continue
            #
            # for j in neigh:
            #     # Connect only neighbors that are also significant and in the same quadrant.
            #     if sig_mask[j] and q[j] == q[i]:
            #         edge_src.append(j)
            #         edge_dst.append(i)
            #         edge_type.append(type_idx)
            #         weight = (Ii[i] + Ii[j]) / 2
            #         edge_weight.append(weight)

    if not edge_src:
        edge_src, edge_dst = data.edge_index.cpu().tolist()
        edge_type = np.zeros(len(edge_src), dtype=np.int64).tolist()
        edge_weight = np.ones(len(edge_src)).tolist()
        num_relations = 1
        rel2feat = {0: "fallback"}

    # -------- 5. Convert edges and weights to tensors --------
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    # -------- 6. Create the new Data object --------
    new_data = Data(
        x=data.x,
        y=data.y,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_weight=edge_weight,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
    )

    # Add metadata
    new_data.num_features = data.num_features
    new_data.num_classes = data.num_classes
    new_data.num_relations = num_relations

    return new_data
