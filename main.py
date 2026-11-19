from __future__ import annotations

import argparse
import copy
import json
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import is_undirected

from Landslide_Task.model_compare import run_RandomForest, run_SVM, run_LR, GCN
from Landslide_Task.moran import apply_moran_mutli_graph
from Landslide_Task.utils import (
    compute_acc,
    compute_auc,
    compute_prec,
    compute_rec,
    compute_f1,
    set_seed
)
from model import MoranRGCN
from data_process import read_data, get_split_indices


# ────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────

def run_experiment(
        content_filename: str,
        edge_list_filename: str,
        split_indices: Optional[dict[str, torch.Tensor]] = None,
        **hparams,
):
    """Train/test LS2GNN once and persist metrics.

    Parameters
    ----------
    content_filename, edge_list_filename : csv paths for the dataset
    split_indices : pre‑computed masks (optional)
    hparams : all hyper‑parameters to override defaults
    """
    # Default hyper‑parameters (can be overridden by **hparams)
    cfg = dict(
        n_layers=2,
        dim_h=16,
        lr=5e-3,
        w_decay=1e-4,
        dropout=0.4,
        max_epochs=1000,
        grad_clip=5.0,
        scheduler_factor=0.5,
        scheduler_patience=20,
        early_stop_patience=100,
        warmup_epochs=5,
        csv_path="./comparison/Moran-RGCN_results.csv",
    )
    cfg.update(hparams)

    print("\n▶︎ Hyper‑parameters")
    print(json.dumps(cfg, indent=4, ensure_ascii=False))

    # ─── Data ───────────────────────────────────────────────
    data = read_data(
        content_filename,
        edge_list_filename,
        split_indices=split_indices,
    )

    data = apply_moran_mutli_graph(
        data,
        scale=False,
        alpha=0.01,
        cluster_keep=(1, 3),          # HH + LL
    )

    # Ablation: remove all edges.
    # data.edge_index = torch.empty((2, 0), dtype=torch.long, device=data.x.device)
    # data.edge_weight = torch.empty((0,), dtype=torch.float, device=data.x.device)

    # Compute label homophily.
    src, dst = data.edge_index
    homophily_ratio = (data.y[src] == data.y[dst]).float().mean().item()
    print(f"Homophily ratio: {homophily_ratio:.4f}")

    # Count nodes incident to at least one edge.
    nodes_with_edges = torch.unique(data.edge_index).numel()
    print(f"\nNodes with edges: {nodes_with_edges} / {data.x.size(0)}")

    # Check whether the graph is undirected.
    is_ud = is_undirected(
        data.edge_index,
        data.edge_weight if data.edge_weight.numel() else None,
        num_nodes=data.x.size(0)
    )
    print(f"\nIs the graph undirected (topology{'+weights' if data.edge_weight.numel() else ''})?  {is_ud}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoranRGCN(
        cfg["n_layers"],
        data.num_relations,
        data.num_features,
        cfg["dim_h"],
        data.num_classes,
        cfg["dropout"]
    ).to(device)
    # model = GCN(
    #     cfg["n_layers"],
    #     data.num_features,
    #     cfg["dim_h"],
    #     data.num_classes,
    #     cfg["dropout"]
    # ).to(device)

    data = data.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["w_decay"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg["scheduler_factor"],
        patience=cfg["scheduler_patience"],
        verbose=True,
    )

    best_state, best_val_auc, es_counter = None, 0.0, 0

    # ─── Training loop ─────────────────────────────────────
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad()

        # Warm‑up: linearly scale LR for the first few epochs
        if epoch <= cfg["warmup_epochs"]:
            lr_scale = epoch / float(cfg["warmup_epochs"])
            for pg in optimizer.param_groups:
                pg["lr"] = cfg["lr"] * lr_scale

        out = model(data.x, data.edge_index, data.edge_type, data.edge_weight)
        # out = model(data.x, data.edge_index, data.edge_weight)

        train_loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        train_auc = compute_auc(out[data.train_mask], data.y[data.train_mask])

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        # ─ Validation ──────────────────────────────────────
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_type, data.edge_weight)
            # out = model(data.x, data.edge_index, data.edge_weight)

            val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])
            val_auc = compute_auc(out[data.val_mask], data.y[data.val_mask])

        scheduler.step(val_auc)

        # ─ Early‑stopping logic (by val_auc) ──────────────
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= cfg["early_stop_patience"]:
                print("Early stopping!")
                break

        # ─ Logging every 10 epochs ─────────────────────────
        if epoch % 10 == 0:
            print(
                f"\n"
                f"\033[47;30m[{epoch:4d}]\033[0m\n"
                f" Train_loss: {train_loss:.4f}, Val_loss: {val_loss:.4f}\n"
                f" Train_auc : {train_auc:.4f}, Val_auc : {val_auc:.4f}, "
                f"Best Val Auc: {best_val_auc:.4f}"
            )

    # ─── Test on best model ───────────────────────────────
    assert best_state is not None, "Training loop finished without any validation improvement."
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_test = model(data.x, data.edge_index, data.edge_type, data.edge_weight)[data.test_mask]
        # logits_test = model(data.x, data.edge_index, data.edge_weight)[data.test_mask]

        pred_test = logits_test.argmax(dim=1)
        true_test = data.y[data.test_mask]

    final_metrics = {
        "Acc": compute_acc(pred_test, true_test),
        "AUC": compute_auc(logits_test, true_test),
        "Precision": compute_prec(true_test, pred_test),
        "Recall": compute_rec(true_test, pred_test),
        "F1": compute_f1(true_test, pred_test),
    }

    print("\n──── The GNN Final ───")
    for k, v in final_metrics.items():
        print(f"{k:<6}: {v:.4f}")

    # # ─── Persist results to CSV ───────────────────────────
    # results_row = {
    #     "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    #     **final_metrics,
    #     **cfg,
    # }
    # df = pd.DataFrame([results_row])
    # csv_path = Path(cfg["csv_path"])
    # csv_path.parent.mkdir(parents=True, exist_ok=True)
    # header = not csv_path.exists()
    # df.to_csv(csv_path, mode="a", header=header, index=False)

    # # Export predictions for plotting.
    # model.eval()
    # with torch.no_grad():
    #     logits_all = model(data.x, data.edge_index, data.edge_weight)  # (N, C)
    #
    # # Convert logits to probabilities.
    # probs_all = logits_all.softmax(dim=1)  # Use [:, 1] for the positive class in binary classification.
    #
    # # Collect predictions in a DataFrame.
    # probs_df = pd.DataFrame({
    #     "slope_idx": torch.arange(data.num_nodes).cpu().numpy() + 1,  # Alternatively, use data['OID_'].
    #     "true_label": data.y.cpu().numpy(),
    #     # Save the positive-class probability for binary classification.
    #     "K8-GCN_prob": probs_all[:, 1].cpu().numpy(),
    #     # For multiclass classification, add a column for each class:
    #     # **{f"prob_{c}": probs_all[:, c].cpu().numpy() for c in range(probs_all.size(1))}
    # })
    #
    # # Write predictions to CSV.
    # prob_path = Path("./comparison/K8-GCN_probs.csv")
    # probs_df.to_csv(prob_path, index=False)
    # print(f"Per-sample probabilities written to {prob_path}")

    return best_val_auc, final_metrics


# ────────────────────────────────────────────────────────────
# Main entry
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", type=str, help="Path to node/feature CSV",
                        default="../../datasets/Yaan/Yaan_content.csv")
    parser.add_argument("--edge_list", type=str, help="Path to edge list CSV", default="../../datasets"
                                                                                       "/Yaan/Yaan_edge_list_AO_A1.csv")
    parser.add_argument("--seed", type=int, default=12)
    # Model hyper‑parameters
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dim_h", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    # Optimizer hyper‑parameters
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--w_decay", type=float, default=1e-4)
    # Training control`
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--early_stop_patience", type=int, default=30)
    # Scheduler
    parser.add_argument("--scheduler_patience", type=int, default=20)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    # Warm‑up
    parser.add_argument("--warmup_epochs", type=int, default=5)
    # Misc
    parser.add_argument("--csv_path", type=str, default="./comparison/Moran-RGCN_results.csv")

    args = parser.parse_args()

    set_seed(args.seed)

    split_indices = get_split_indices(content_filename=args.content, edge_list_filename='../../datasets/Yaan/Yaan_edge_list_queen.csv', seed=args.seed)

    run_experiment(
        content_filename=args.content,
        edge_list_filename=args.edge_list,
        split_indices=split_indices,
        # Forward all remaining args as hparams
        **{k: v for k, v in vars(args).items() if k not in {"content", "edge_list", "seed"}}
    )

    run_RandomForest(
        content_filename=args.content,
        seed=args.seed,
        split_indices=split_indices
    )

    # run_SVM(
    #     content_filename=args.content,
    #     seed=args.seed,
    #     split_indices=split_indices
    # )

    # run_LR(
    #     content_filename=args.content,
    #     seed=args.seed,
    #     split_indices=split_indices
    # )
