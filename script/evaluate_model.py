import sys
    
import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)

from imblearn.under_sampling import RandomUnderSampler

# import os
# print(os.getcwd())
# os.chdir('../')
# python script/evaluate_model.py model.rinalmo_dora_20260507_171051 model/rinalmo_dora_20260507_171051.pt


# from model.rinalmo_dora_20260507_171051 import (
#     hyperparameters_queryer,
#     get_dataset,
#     predict_batch,
#     DEVICE,
#     load_model,
# ) 


def load_model_script(script_name):
    import sys
    from pathlib import Path
    lib_path = Path(__file__).resolve().parent.parent / "model/prev_experiments"

    sys.path.append(str(lib_path))

    import importlib

    vars = [
        'hyperparameters_queryer',
        'get_dataset',
        'predict_batch',
        'DEVICE',
        'load_model',
    ]

    module = importlib.import_module(script_name)
    # module = importlib.import_module('model', script_name)

    for name in vars:
        globals()[name] = getattr(module, name)


def _compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, cutoff=0.5):
    """
    Compute binary classification metrics from true labels and predicted probabilities.
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred_prob = np.asarray(y_pred_prob)

    y_pred = (y_pred_prob > cutoff).astype(int)

    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if len(np.unique(y_true)) < 2:
        auc = float("nan")
        aupr = float("nan")
    else:
        auc = roc_auc_score(y_true, y_pred_prob)
        aupr = average_precision_score(y_true, y_pred_prob)

    return {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "auc": auc,
        "aupr": aupr,
        "f1": f1,
        "mcc": mcc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "total": int(tp + tn + fp + fn),
    }


def _nan_downsample_summary(metric_keys):
    """
    Return NaN mean/std for balanced downsample metrics.
    """

    out = {}

    for key in metric_keys:
        out[f"balanced_downsample_{key}_mean"] = float("nan")
        out[f"balanced_downsample_{key}_std"] = float("nan")

    return out


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    cutoff=0.5,
    n_downsample_runs: int = 100,
    random_state: int = 42,
):
    """
    Compute binary classification metrics.

    This function returns:
    1. Metrics on the original data distribution.
    2. Repeated random downsampling metrics, summarized by mean and standard deviation.

    Downsampling is applied only for auxiliary balanced evaluation.
    It should not replace metrics computed on the original validation/test distribution.
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred_prob = np.asarray(y_pred_prob)

    if y_true.shape[0] != y_pred_prob.shape[0]:
        raise ValueError("y_true and y_pred_prob must have the same length.")

    # Metrics on the original distribution
    metrics = _compute_metrics(
        y_true=y_true,
        y_pred_prob=y_pred_prob,
        cutoff=cutoff,
    )

    metric_keys = list(metrics.keys())

    # If only one class exists, downsampling is impossible
    if len(np.unique(y_true)) < 2:
        metrics.update(_nan_downsample_summary(metric_keys))
        return metrics

    # Repeated random downsampling
    downsampled_results = []

    for i in range(n_downsample_runs):
        try:
            X = y_pred_prob.reshape(-1, 1)
            y = y_true

            rus = RandomUnderSampler(
                random_state=random_state + i,
            )

            X_resampled, y_resampled = rus.fit_resample(X, y)

            y_pred_prob_resampled = X_resampled.ravel()
            y_true_resampled = y_resampled

            run_metrics = _compute_metrics(
                y_true=y_true_resampled,
                y_pred_prob=y_pred_prob_resampled,
                cutoff=cutoff,
            )

            downsampled_results.append(run_metrics)

        except Exception:
            metrics.update(_nan_downsample_summary(metric_keys))
            return metrics

    # Summarize downsample metrics by mean and std
    for key in metric_keys:
        values = np.array(
            [run[key] for run in downsampled_results],
            dtype=float,
        )

        metrics[f"balanced_downsample_{key}_mean"] = np.nanmean(values)
        metrics[f"balanced_downsample_{key}_std"] = np.nanstd(values, ddof=1)

    return metrics


def evaluate_group(df, cutoff=0.5, sec_cufoff=dict()):
    rows = []
    for sec in df.section.unique():
        df_temp = df.loc[df.section == sec]
        cutoff = sec_cufoff.get(sec, cutoff)  # select specified cutoff by section
        perf_result = compute_binary_metrics(
            df_temp.label,
            df_temp.yhat_prob,
            cutoff
        )
        # df_temp.label, df_temp.yhat, df_temp.yhat_prob)
        # pprint(perf_result)
        perf_result = {'section': sec} | perf_result
        rows.append(perf_result)
        # print('----')
    return pd.DataFrame.from_dict(rows, ).sort_values(by='section')

def main():
    import sys
    load_model_script(sys.argv[1])
    hp = hyperparameters_queryer()

    df = pd.read_csv("input_data/df_data.csv")
    print(df.shape, flush=True)

    # (
    #     train_loader, val_loader, test_loader,
    #     # section_ratio, section_neg_pos_ratio,
    #     pos_ratio,
    #     train_idx, val_idx, test_idx,
    # ) = get_dataset(hp)
    obj = get_dataset(hp)
    train_loader, val_loader, test_loader = obj[:3]
    train_idx, val_idx, test_idx = obj[-3:]
    
    df_val = df.iloc[val_idx]
    df_test = df.iloc[test_idx]
    print(df_test.shape, df_val.shape)
    
    model = load_model(sys.argv[2], hp, DEVICE)
    print('======== test set ========')
    yhat_prob, yhat, labels_t = predict_batch(model, test_loader, DEVICE)
    pprint(compute_binary_metrics(labels_t.numpy(), yhat_prob.numpy()))
    df_test['yhat_prob'] = yhat_prob.numpy().tolist()
    df_test['yhat'] = yhat.numpy().tolist()
    print(evaluate_group(df_test).to_string())
    
    print('======== val set ========')
    yhat_prob, yhat, labels_t = predict_batch(model, val_loader, DEVICE)
    pprint(compute_binary_metrics(labels_t.numpy(), yhat_prob.numpy()))
    df_val['yhat_prob'] = yhat_prob.numpy().tolist()
    df_val['yhat'] = yhat.numpy().tolist()
    print(evaluate_group(df_val).to_string())
    

if __name__ == '__main__':
    main()
