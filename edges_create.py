import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


def build_sparse_adj_from_edgelist(edge_df, N, src_col='src_id', dst_col='nbr_id', undirected=True):
    """Build a binary sparse adjacency matrix from an edge list, excluding self-loops."""
    # Detect zero-based IDs; subtract 1 from one-based IDs.
    min_id = edge_df[[src_col, dst_col]].min().min()
    offset = 1 if min_id == 1 else 0

    rows = edge_df[src_col].values - offset
    cols = edge_df[dst_col].values - offset

    data = np.ones_like(rows, dtype=np.float32)
    A = csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)
    if undirected:
        A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def adjacency_order_sparse(attributes_scaled, edge_df, order=1,
                           src_col='src_id', dst_col='nbr_id', undirected=True):
    N = attributes_scaled.shape[0]
    A = build_sparse_adj_from_edgelist(edge_df, N, src_col, dst_col, undirected)

    seen = csr_matrix((N, N), dtype=bool)
    weighted_adj = csr_matrix((N, N), dtype=np.float32)
    prev_power = A.copy()

    for k in range(1, order + 1):
        # Reachable at this hop
        reachable = prev_power.astype(bool)

        # New edges = reachable \ seen
        new_links_bool = reachable - reachable.multiply(seen)
        new_links_bool.eliminate_zeros()

        if new_links_bool.nnz > 0:
            weighted_adj += new_links_bool.astype(np.float32) * (1.0 / k)
            # seen = seen | new_links_bool   # Not supported for sparse matrices
            seen = seen.maximum(new_links_bool)  # Elementwise OR

        if k < order:
            prev_power = prev_power @ A
            prev_power.data[:] = 1.0  # Treat as boolean
            prev_power.eliminate_zeros()

    weighted_adj.setdiag(0)
    weighted_adj.eliminate_zeros()
    return weighted_adj


def environment_similarity_sparse(attributes_scaled, edge_df, edge_num: int = 10, edge_thresh: float = 0.3):
    attributes_scaled_df = pd.DataFrame(attributes_scaled, columns=X.columns)

    edge_cols = [col for col in attributes_scaled_df.columns if any(tag in col for tag in ['DTR', 'LIT', 'DTF'])]
    node_cols = [col for col in attributes_scaled_df.columns if col not in edge_cols]
    f_node = attributes_scaled_df[node_cols]
    f_edge = attributes_scaled_df[edge_cols]

    # Environmental feature grouping
    # edge_gp = f_edge.groupby(f_edge.columns.to_list())

    f_node = f_node.to_numpy()
    f_edge = f_edge.to_numpy()

    # i = 0
    # e_class = np.zeros([attributes_scaled_df.shape[0], 1], dtype=int)
    # for key, value in edge_gp.indices.items():
    #     # Select neighbors from the closest groups in order.
    #     eachEnv = value.tolist()
    #     e_class[eachEnv] = i
    #     i = i + 1
    # attributes_scaled_df['ENV'] = e_class

    node_num = attributes_scaled_df.shape[0]
    edges = np.zeros(shape=[1, 4 + f_edge.shape[1]])
    for self in range(node_num):
        pt = self * 100 / node_num
        # if pt % 10 == 0:
        print('\r', '{:.1f}%'.format(pt), end='')
        # Environmental feature similarity
        dist_edge = pd.Series(cdist(f_edge, f_edge[self].reshape((1, f_edge.shape[1])), metric='cosine').flatten())
        # Attribute feature similarity
        dist_node = pd.Series(cdist(f_node, f_node[self].reshape((1, f_node.shape[1])), metric='cosine').flatten())

        dist_edge_m = dist_edge.where(dist_edge < edge_thresh).dropna()

        # edgegroup = dist_edge_m.groupby(dist_edge_m)
        max_edge_num = edge_num

        # Similarity bin width; smaller values produce finer bins.
        bin_width = 0.001  # Adjustable, e.g. 0.01 / 0.02 / 0.05
        # Compute bin labels (0, 1, 2, ...); values in the same bin share a label.
        labels = np.floor(dist_edge_m.values / bin_width).astype(int)
        edgegroup = {}
        for lab, idx in zip(labels, dist_edge_m.index):
            edgegroup.setdefault(lab, []).append(idx)

        edgegroup = dict(sorted(edgegroup.items(), key=lambda x: x[0]))

        closeENid = []
        for key, value in edgegroup.items():
            # Select neighbors from the closest groups in order.
            closeEid = value
            # Rank candidate neighbors by feature distance and select the nearest ones.
            cand = dist_node[closeEid]
            cand = cand[cand <= 1]
            closeENid.extend(cand.sort_values().head(max_edge_num - len(closeENid) + 1).index.tolist())
            if len(closeENid) < 2:
                continue
            else:
                break
        closeENid = closeENid[1:]
        if len(closeENid) < max_edge_num:
            max_edge_num = len(closeENid)
        weight = f_edge[closeENid]
        selflist = np.ones(max_edge_num, dtype=int) * self
        edge = np.c_[closeENid, selflist[:, None], 1 - dist_node[closeENid], 1 - dist_edge[closeENid], weight]
        edges = np.r_[edges, edge]
    edge_index = np.array(edges[1:, :4])
    # # Set all adjacency matrix values to 0.
    # adj_matrix_np = np.zeros_like(adj_matrix.to_numpy(), dtype=float)
    # for row in edge_index:
    #     end_node, start_node, node_sim, env_sim = row
    #     adj_matrix_np[int(end_node), int(start_node)] = node_sim * env_sim

    # === Construct a csr_matrix ===
    row = edge_index[:, 0].astype(int)
    col = edge_index[:, 1].astype(int)
    data = (edge_index[:, 2] * edge_index[:, 3])  # node_sim * env_sim
    weighted_ecc = csr_matrix((data, (row, col)), shape=(node_num, node_num))

    return weighted_ecc


def sparse_to_edge_csv_with_weight(sparse_mat: csr_matrix,
                                   edges_raw: pd.DataFrame,
                                   path: str,
                                   offset: int = 1,
                                   src_col='src_id', dst_col='nbr_id',
                                   weight_col='weight'):
    """
    Export nonzero entries of a sparse weighted adjacency matrix to a CSV edge list.

    Parameters
    ----------
    sparse_mat : csr_matrix
        Sparse matrix whose nonzero entries provide the edge weights.
    edges_raw : DataFrame
        Original edge table. Currently unused; the optional merge is commented out.
    path : str
        Output CSV path.
    offset : int
        Add 1 for one-based node IDs or 0 for zero-based node IDs.
    src_col, dst_col : str
        Output column names for source and destination node IDs.
    weight_col : str
        Output column name for edge weights.

    Returns
    -------
    DataFrame
        Source IDs, destination IDs, and weights written to the CSV file.
    """
    # Extract nonzero entries from the weight matrix.
    coo = sparse_mat.tocoo()
    w_df = pd.DataFrame({
        src_col: coo.row + offset,
        dst_col: coo.col + offset,
        weight_col: coo.data
    })

    # # Optionally merge with the original edges; choose the join type for the desired edge coverage.
    # merged = edges_raw.merge(w_df, on=[src_col, dst_col], how='outer')
    #
    # # For edges beyond k hops with missing weights, fill with 0 or drop them.
    # merged[weight_col] = merged[weight_col].fillna(0.0)
    #
    # # Sort (optional)
    # merged = merged.sort_values([src_col, weight_col], ascending=[True, False])

    w_df.to_csv(path, index=False)
    return w_df


if __name__ == '__main__':
    # Read attributes and the sparse edge list.
    attributes = pd.read_csv('../../datasets/Yaan/Yaan_content.csv')
    edges = pd.read_csv('../../datasets/Yaan/Yaan_edge_list_queen.csv')

    y = attributes['Presence']
    X = attributes.drop(columns=['OID_', 'Shape_Length', 'Shape_Area', 'Presence'])

    scaler = StandardScaler()
    attributes_scaled = scaler.fit_transform(X)

    # # Compute the weighted adjacency matrix up to k hops.
    order = 1  # Set the desired number of hops.
    weighted_adj_sparse = adjacency_order_sparse(attributes_scaled, edges, order=order)

    # max_edge_num = 10
    # edge_thresh = 0.3
    # weighted_ecc_sparse = environment_similarity_sparse(attributes_scaled, edges, edge_num=max_edge_num,
    #                                                     edge_thresh=edge_thresh)

    # Save as CSV; large graphs can produce large files.
    out_path = '../../datasets/Yaan/Yaan_edge_list_AO_A1.csv'
    _ = sparse_to_edge_csv_with_weight(weighted_adj_sparse, edges, out_path,
                                       offset=1,
                                       src_col='src_id', dst_col='nbr_id',
                                       weight_col='weight')
    print(f'Saved to {out_path}')
