import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score
)

from train_nn_model_by_site_rnafm_transformer_RiNALMo_noncanonicaltis import (
    hyperparameters_queryer,
    get_dataset,
    predict_batch,
    DEVICE,
    load_model,
) 

def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_prob: np.ndarray):
    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    denom = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    mcc   = (tp*tn - fp*fn) / np.sqrt(denom) if denom else 0.0

    if len(np.unique(y_true)) < 2:
        auc = aupr = float("nan")
    else:
        auc  = roc_auc_score(y_true, y_pred_prob)
        aupr = average_precision_score(y_true, y_pred_prob)

    return {
        "accuracy": acc, "precision": precision, "recall": recall,
        "specificity": specificity, "auc": auc, "aupr": aupr,
        "f1": f1, "mcc": mcc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, 'total': tp + tn + fp + fn,
    }


def evaluate_group(df, cut=0.5):
    rows = []
    for sec in df.section.unique():
        df_temp = df.loc[df.section == sec]
        # print(sec)
        perf_result = compute_binary_metrics(
            df_temp.label, [(1 if x > cut else 0) for x in df_temp.yhat_prob], df_temp.yhat_prob)
        # df_temp.label, df_temp.yhat, df_temp.yhat_prob)
        # pprint(perf_result)
        perf_result = {'section': sec} | perf_result
        rows.append(perf_result)
        # print('----')
    return pd.DataFrame.from_dict(rows, )


def main():
    hp = hyperparameters_queryer()
    train_loader, val_loader, test_loader, pos_weight, train_idx, val_idx, test_idx = get_dataset(hp)
    yhat_prob, yhat, labels_t = predict_batch(model, test_loader, DEVICE)
    model = load_model("../model/rinalmo_dora_20260505_100930.pt", hp, DEVICE)
    
    
    df = pd.read_csv("../input_data/df_data.csv")
    df_test = df.iloc[test_idx]
    df_test['yhat_prob'] = yhat_prob.numpy().tolist()
    df_test['yhat'] = yhat.numpy().tolist()
    
    df_perf = evaluate_group(df_test)
    df_perf.to_csv('rinalmo_dora_20260505_100930.evaluate')
