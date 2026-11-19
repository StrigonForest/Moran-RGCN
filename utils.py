import random
import numpy as np
import torch

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, \
    average_precision_score


# ───────── Basic utilities ─────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_numpy(x):
    """tensor / list -> 1-D numpy array"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ───────── Metric functions ─────────

def compute_acc(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Classification accuracy"""
    return accuracy_score(_to_numpy(true), _to_numpy(pred))


def compute_auc(logits: torch.Tensor, true: torch.Tensor) -> float:
    """
    AUC-ROC.
    - Binary classification: use positive-class probabilities directly
    - Multiclass classification: use one-vs-rest with macro averaging
    """
    y_true = _to_numpy(true)
    y_prob = _to_numpy(torch.softmax(logits, dim=1))
    if y_prob.shape[1] == 2:  # binary
        return roc_auc_score(y_true, y_prob[:, 1])
    return roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")


def compute_prec(true: torch.Tensor, pred: torch.Tensor) -> float:
    return precision_score(_to_numpy(true), _to_numpy(pred), average="macro")


def compute_rec(true: torch.Tensor, pred: torch.Tensor) -> float:
    return recall_score(_to_numpy(true), _to_numpy(pred), average="macro")


def compute_f1(true: torch.Tensor, pred: torch.Tensor) -> float:
    return f1_score(_to_numpy(true), _to_numpy(pred), average="macro")
