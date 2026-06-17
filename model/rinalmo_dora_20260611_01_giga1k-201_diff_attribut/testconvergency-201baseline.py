import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional
from captum.attr import LayerIntegratedGradients
from pathlib import Path
import json
from ushuffle import Shuffler, set_seed

from main import *
from main import _tokenizer, _PAD_ID, centralize_transcript

# # ── plug in your model ────────────────────────────────────────────────────────
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device : {DEVICE}")
print(f"captum : OK")




RINALMO_MODEL_ID = "multimolecule/rinalmo-giga"

# ── tokenizer ─────────────────────────────────────────────────────────────────
# tokenizer = RnaTokenizer.from_pretrained(RINALMO_MODEL_ID)
tokenizer = _tokenizer

# ── model ─────────────────────────────────────────────────────────────────────
save_path = r'model_1k.pt'
hp = hyperparameters_queryer()

model = load_model(save_path, hp, DEVICE)
model.to(DEVICE).eval()


# ── forward function ──────────────────────────────────────────────────────────

def make_forward_func(model, attention_mask):
    """
    Returns a forward_func compatible with LayerIntegratedGradients.

    LayerIntegratedGradients calls forward_func(input_ids) internally,
    so attention_mask must be captured via closure.

    Returns raw logit (scalar) — LIG will backprop through this.
    """
    def forward_func(input_ids):
        output = model(input_ids, attention_mask=attention_mask)
        return output.squeeze(-1).unsqueeze(0)    # plain tensor
    return forward_func


def build_baseline_keep_tis_window(
    input_ids: torch.Tensor,
    baseline_token_id: int,
    keep_left_window: int,
    keep_right_window: int,
    tis_position: int,
) -> torch.Tensor:
    """
    Build baseline by masking tokens outside a TIS-centered window.

    Assumptions:
      - input_ids shape: [1, L]
      - tokenizer adds CLS at token index 0 and EOS at token index -1
      - tis_position is the raw sequence index, not token index
      - keep window includes the TIS position itself:
            [tis_position - keep_left_window, tis_position + keep_right_window]
    """
    baseline = input_ids.clone()

    seq_len = input_ids.size(1) - 2  # exclude CLS and EOS

    # raw sequence coordinate
    keep_start_nt = max(0, tis_position - keep_left_window)
    keep_end_nt = min(seq_len, tis_position + keep_right_window + 1)  # exclusive

    # convert raw sequence coordinate to token coordinate
    keep_start_tok = keep_start_nt + 1
    keep_end_tok = keep_end_nt + 1

    # mask all real sequence tokens first; keep CLS and EOS unchanged
    baseline[:, 1:-1] = baseline_token_id

    # restore the TIS-centered window
    baseline[:, keep_start_tok:keep_end_tok] = input_ids[:, keep_start_tok:keep_end_tok]

    return baseline

# ── IGResult ─────────────────────────────────────────────────────────────────────

@dataclass
class IGResult:
    sequence:          str
    attributions:      np.ndarray   # (seq_len,)
    pred_prob:         float
    convergence_delta: float
    baseline_type:     str
    dimension_reduction: str
    baseline_logit : float
    score_diff: float
    relative_delta: float

def sanity_check(
        sequence: str,
    model,
    tokenizer,
):
    enc = tokenizer(sequence, return_tensors="pt", padding=False, truncation=True)
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    attention_mask = attention_mask.masked_fill(input_ids == _PAD_ID, 0)

    # sanity check
    embedding_layer = model.backbone.base_model.model.embeddings.word_embeddings

    print("embedding_layer:", embedding_layer, flush=True)

    called = {"fwd": False}
    grads = {}

    def fwd_hook(module, inputs, output):
        called["fwd"] = True
        print("[forward hook] embedding output:", output.shape)

    def bwd_hook(module, grad_input, grad_output):
        grads["bwd"] = grad_output[0]
        print("[backward hook] grad output norm:", grad_output[0].norm().item())

    h1 = embedding_layer.register_forward_hook(fwd_hook)
    h2 = embedding_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad(set_to_none=True)

    logit = model(input_ids, attention_mask=attention_mask).squeeze()
    logit.backward()

    h1.remove()
    h2.remove()

    print("forward called:", called["fwd"], flush=True)
    print("backward grad exists:", "bwd" in grads, flush=True)

    
def run_lig_mask(
    sequence: str,
    model,
    tokenizer,
    baseline_token_id: int,
    n_steps: int = 100,
    dim_reduction: str = 'sum',
) -> IGResult:
    """
    LayerIntegratedGradients with [MASK] baseline.
    Attribution layer: model.encoder.embeddings.word_embeddings
    """
    model.eval()

    enc = tokenizer(sequence, return_tensors="pt", padding=False, truncation=True)
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    attention_mask = attention_mask.masked_fill(input_ids == _PAD_ID, 0)

    print()
    print('baseline token: ', baseline_token_id)
    print()
    baseline_ids = build_baseline_keep_tis_window(
        input_ids, baseline_token_id,
        keep_left_window=99, keep_right_window=102,
        tis_position=500,)

    forward_func = make_forward_func(model, attention_mask)

    lig = LayerIntegratedGradients(
        forward_func,
        model.backbone.base_model.model.embeddings.word_embeddings,
    )

    attrs, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        n_steps=n_steps,
        method="gausslegendre",
        # method="riemann_trapezoid",
        internal_batch_size=1,
        return_convergence_delta=True,
    )  # attrs: (1, L, D)

    # dim reduction over embedding dim → per-token scalar, strip CLS/EOS
    if dim_reduction == 'l2-norm':
        attrs_per_token = attrs.norm(dim=-1).squeeze(0)[1:-1].detach().cpu().numpy()
    elif dim_reduction == 'sum':
        attrs_per_token = attrs.sum(dim=-1).squeeze(0)[1:-1].detach().cpu().numpy()
    else:
        raise ValueError(f"unknown: {dim_reduction}")

    with torch.no_grad():
        logit = forward_func(input_ids)
        baseline_logit = forward_func(baseline_ids)

        score_diff = logit - baseline_logit

        abs_delta = delta.abs()
        relative_delta = abs_delta / score_diff.abs().clamp_min(1e-8)

        logit_value = logit.item()
        baseline_logit_value = baseline_logit.item()
        score_diff_value = score_diff.item()
        delta_value = delta.item()
        relative_delta_value = relative_delta.item()

    prob = torch.sigmoid(torch.tensor(logit_value)).item()

    print("logit:", logit_value, flush=True)
    print("baseline_logit:", baseline_logit_value, flush=True)
    print("score_diff:", score_diff_value, flush=True)
    print("delta:", delta_value, flush=True)
    print("relative_delta:", relative_delta_value, flush=True)

    return IGResult(
        sequence=sequence,
        attributions=attrs_per_token,
        pred_prob=prob,
        convergence_delta=delta_value,
        baseline_type="[MASK]",
        dimension_reduction=dim_reduction,
        baseline_logit=baseline_logit_value,
        score_diff=score_diff_value,
        relative_delta=relative_delta_value,
    )

def dinucleotide_shuffle(
    seq: str,
    n_shuffles: int = 1,
    seed: int | None = None,
) -> list[str]:
    """
    Generate dinucleotide-preserving shuffled sequences using ushuffle.

    Parameters
    ----------
    seq : str
        Input sequence.
    n_shuffles : int
        Number of shuffled sequences to generate.
    seed : int | None
        Random seed. If provided, results are reproducible.

    Returns
    -------
    list[str]
        Shuffled sequences preserving dinucleotide composition.
    """
    if not isinstance(seq, str):
        raise TypeError("seq must be a str")

    if not isinstance(n_shuffles, int):
        raise TypeError("n_shuffles must be an int")

    if n_shuffles < 1:
        return []

    if seed is not None:
        set_seed(seed)

    if len(seq) < 2:
        return [seq for _ in range(n_shuffles)]

    seq_bytes = seq.encode("ascii")
    shuffler = Shuffler(seq_bytes, 2)

    return [
        shuffler.shuffle().decode("ascii")
        for _ in range(n_shuffles)
    ]

def run_lig_dinuc(
    sequence: str,
    model,
    tokenizer,
    n_shuffles: int = 10,
    n_steps: int = 100,
    seed: int = 42,
    dim_reduction: str = 'sum'
) -> IGResult:
    """
    LayerIntegratedGradients with averaged dinucleotide-shuffle baselines.

    For each shuffled baseline:
      1. tokenize the shuffled sequence
      2. run LIG against that baseline
      3. collect attribution tensor

    Final attribution = mean over all shuffled baselines.
    Delta / relative_delta are averaged over valid shuffled baselines.
    """
    model.eval()

    enc = tokenizer(sequence, return_tensors="pt", padding=False, truncation=True)
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    attention_mask = attention_mask.masked_fill(input_ids == _PAD_ID, 0)

    forward_func = make_forward_func(model, attention_mask)

    lig = LayerIntegratedGradients(
        forward_func,
        model.backbone.base_model.model.embeddings.word_embeddings,
    )

    lpad = len(sequence) - len(sequence.lstrip('<pad>'))
    rpad = len(sequence) - len(sequence.rstrip('<pad>'))

    shuffled_seqs = [
        "<pad>" * (lpad // 5) + shuf + "<pad>" * (rpad // 5)
        for shuf in dinucleotide_shuffle(
            sequence[lpad: len(sequence) - rpad] if rpad > 0 else sequence[lpad:],
            n_shuffles=n_shuffles,
            seed=seed,
        )
    ]

    all_attrs = []

    deltas = []
    relative_deltas = []
    baseline_logits = []
    score_diffs = []

    with torch.no_grad():
        logit = forward_func(input_ids)
        logit_value = logit.item()

    for shuf_seq in shuffled_seqs:
        shuf_enc = tokenizer(
            shuf_seq,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        baseline_ids = shuf_enc["input_ids"].to(DEVICE)

        # Ensure same length as input
        if baseline_ids.shape != input_ids.shape:
            print(
                f"  [warn] baseline length mismatch — skipping this shuffle | "
                f"input={input_ids.shape}, baseline={baseline_ids.shape}"
            )
            continue

        attrs, delta = lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            n_steps=n_steps,
            method="gausslegendre",
            internal_batch_size=1,
            return_convergence_delta=True,
        )  # attrs: (1, L, D)

        if dim_reduction == 'l2-norm':
            attrs_per_token = (
                attrs.norm(dim=-1)
                .squeeze(0)[1:-1]
                .detach()
                .cpu()
                .numpy()
            )
        elif dim_reduction == 'sum':
            attrs_per_token = (
                attrs.sum(dim=-1)
                .squeeze(0)[1:-1]
                .detach()
                .cpu()
                .numpy()
            )
        else:
            raise ValueError(f"unknown: {dim_reduction}")

        all_attrs.append(attrs_per_token)

        with torch.no_grad():
            baseline_logit = forward_func(baseline_ids)
            score_diff = logit - baseline_logit

            abs_delta = delta.abs()
            relative_delta = abs_delta / score_diff.abs().clamp_min(1e-8)

        deltas.append(delta.item())
        relative_deltas.append(relative_delta.item())
        baseline_logits.append(baseline_logit.item())
        score_diffs.append(score_diff.item())

    if len(all_attrs) == 0:
        raise RuntimeError("No valid dinucleotide-shuffled baselines were generated.")

    mean_attrs = np.mean(all_attrs, axis=0)

    prob = torch.sigmoid(torch.tensor(logit_value)).item()

    delta_mean = float(np.mean(deltas))
    delta_abs_mean = float(np.mean(np.abs(deltas)))
    delta_abs_min = float(np.min(np.abs(deltas)))
    delta_abs_max = float(np.max(np.abs(deltas)))

    relative_delta_mean = float(np.mean(relative_deltas))
    relative_delta_min = float(np.min(relative_deltas))
    relative_delta_max = float(np.max(relative_deltas))

    baseline_logit_mean = float(np.mean(baseline_logits))
    score_diff_mean = float(np.mean(score_diffs))

    print(
        f"logit: {logit_value:.6f}\n"
        f"baseline_logit_mean: {baseline_logit_mean:.6f}\n"
        f"score_diff_mean: {score_diff_mean:.6f}\n"
        f"delta_mean: {delta_mean:.6f}\n"
        f"delta_abs_mean: {delta_abs_mean:.6f}\n"
        f"delta_abs_min: {delta_abs_min:.6f}\n"
        f"delta_abs_max: {delta_abs_max:.6f}\n"
        f"relative_delta_mean: {relative_delta_mean:.6f}\n"
        f"relative_delta_min: {relative_delta_min:.6f}\n"
        f"relative_delta_max: {relative_delta_max:.6f}"
    )

    return IGResult(
        sequence=sequence,
        attributions=mean_attrs,
        pred_prob=prob,
        convergence_delta=delta_mean,
        baseline_type="dinuc-shuffle",
        dimension_reduction=dim_reduction,
        baseline_logit=baseline_logit_mean,
        score_diff=score_diff_mean,
        relative_delta=relative_delta_mean,
    )


def main():
    # import sys
    import pandas as pd
    # ── example usage (requires real model) ──────────────────────────────────────
    NEUTRAL_TOKEN_ID = tokenizer.mask_token_id
    # NEUTRAL_TOKEN_ID = 0  # for pad token
    # NEUTRAL_TOKEN_ID = 10   # for N token
    dim_reduction = 'sum'
    # seq = sys.argv[1]
    # seq = 'TATGAGTATAGATGAAGTATAGTAATAGTACTTAATATGTCGGATCCGTGCTATAAGCAATTTGGATTGTATGAAAAAATCTCTTAAATTTGATTGAAATATTATTCGTCCAATAATTATCAACTCAATACTAAATCGACCTATATCTAATTAATAATTTAACAAATTAATATTTTAAAAAATTGAAATATCAATAAGTCGAATCATTAAGCATAATTATCGGCCTATAAAAAAGATTTAGTCGTATTATCTGGAAAAGATTAAATAAATCCGGCGCCATATTCATAGTGATTTATGGCTGAAGCCCACACGTTATAACAACAACTACTCCATCAATGGAAGCTTCCATTTCTTGAATTTCTCAAACTCTTACTAAATTCAACTCCGGCGGTGGAACCCTTGTAATCTTCAACATATTTCGTTAATTAATTCTTATATACATATATATAAAGAAATTTGTTTTCTACTGCCGAAAGTTTCTTCTTCTCATCGGAGTAGATATGCCGAGCTTAATTGTCAAAGTTTACAGCTTACTCTTCAAGTATAACCTTAATCGCCGATTGCAATCACTAATCCAATCCCCAATTTCATACCCTTTTAACGGTGTCGTCTCACGCGCCGATGAATCGATTATCACTTCTAACCCTAGTTTCTCTACCGACGGTGTTGCAACTAAGGACCTGCATATTGATTCTTTGACTTGTCTATCTCTCAGGATTTACCTCCCTCAATCTGCACTTATTTCGTTGAGAAATTTGGAATCTGGTGAAGGGGTTTATGGGGGTTATGTACCGGGAAAAAATGGGAAAAATTGTAAGAAATTGCCGGTGATTTTGCAGTTTCATGGTGGTGCTTGGGTGACTGGGGGTATTGATACGGTTTCCAATGATGTTTTTTGTAGGAAATTGGCGAAATCTTGTGATGCTATTGTGATTGCTGTTGGGTATAGATTGGCACCGGAGAGTAGGTTTCCGGCTGCGTTTGAAGATGGGGTTGCGGCG'
    df_test = pd.read_csv('df_test_giga_1k-diff-201truncated.csv')
    df_test["seq"] = [
        centralize_transcript(hp["left_window"], hp["right_window"], s, p)
        for s, p in zip(df_test["seq"].values, df_test["position"].values)    
    ]
    
    # SAVE_DIR = "ig_data"
    # import os; os.makedirs(SAVE_DIR, exist_ok=True)
        

    N_SAMPLES = 10
    # STEP_LIST = [64, 128, 512]
    # STEP_LIST = [1024, 2048, 4096]
    STEP_LIST = [10, 50, 250, 500, 1000, 1500]

    rows = []

    sampled_seqs = random.sample(list(df_test.seq), k=min(N_SAMPLES, len(df_test)))

    for sample_id, seq in enumerate(sampled_seqs):
        print(f"\n=== sample_id={sample_id} | seq_len={len(seq)} ===", flush=True)

        # test convergency by mask token as baseline when step increasing
        for n_steps in STEP_LIST:
            print(f"running n_steps={n_steps}", flush=True)

            try:
                result = run_lig_mask(
                    seq,
                    model,
                    tokenizer,
                    n_steps=n_steps,
                    dim_reduction="sum",
                    baseline_token_id=NEUTRAL_TOKEN_ID,
                )

                rows.append({
                    "sample_id": sample_id,
                    "seq_len": len(seq),
                    "n_steps": n_steps,
                    "delta": result.convergence_delta,
                    "abs_delta": abs(result.convergence_delta),
                    "relative_delta": result.relative_delta,
                    "score_diff": result.score_diff,
                    "baseline_logit": result.baseline_logit,
                    "pred_prob": result.pred_prob,
                    "status": "ok",
                    "error": "",
                })

                print(
                    "sample_id=", sample_id,
                    "n_steps=", n_steps,
                    "delta=", result.convergence_delta,
                    "abs_delta=", abs(result.convergence_delta),
                    "relative_delta=", result.relative_delta,
                    "score_diff=", result.score_diff,
                    flush=True,
                )

            except Exception as e:
                rows.append({
                    "sample_id": sample_id,
                    "seq_len": len(seq),
                    "n_steps": n_steps,
                    "delta": np.nan,
                    "abs_delta": np.nan,
                    "relative_delta": np.nan,
                    "score_diff": np.nan,
                    "baseline_logit": np.nan,
                    "pred_prob": np.nan,
                    "status": "error",
                    "error": repr(e),
                })

                print(
                    f"[error] sample_id={sample_id}, n_steps={n_steps}: {repr(e)}",
                    flush=True,
                )
    delta_df = pd.DataFrame(rows)

    print("\n=== raw delta results ===")
    print(delta_df.to_string(index=False))

    ok_df = delta_df[delta_df["status"] == "ok"].copy()

    summary_df = (
        ok_df
        .groupby("n_steps")
        .agg(
            n=("relative_delta", "count"),

            delta_min=("delta", "min"),
            delta_median=("delta", "median"),
            delta_pr60=("delta", lambda x: np.percentile(x, 60)),
            delta_pr80=("delta", lambda x: np.percentile(x, 80)),
            delta_max=("delta", "max"),

            abs_delta_min=("abs_delta", "min"),
            abs_delta_median=("abs_delta", "median"),
            abs_delta_pr60=("abs_delta", lambda x: np.percentile(x, 60)),
            abs_delta_pr80=("abs_delta", lambda x: np.percentile(x, 80)),
            abs_delta_max=("abs_delta", 'max'),

            relative_delta_min=("relative_delta", "min"),
            relative_delta_median=("relative_delta", "median"),
            relative_delta_pr60=("relative_delta", lambda x: np.percentile(x, 60)),
            relative_delta_pr80=("relative_delta", lambda x: np.percentile(x, 80)),
            relative_delta_max=("relative_delta", 'max'),

            score_diff_min=("score_diff", "min"),
            score_diff_median=("score_diff", "median"),
            score_diff_pr60=("score_diff", lambda x: np.percentile(x, 60)),
            score_diff_pr80=("score_diff", lambda x: np.percentile(x, 80)),
            score_diff_max=("score_diff", "max"),
        )
        .reset_index()
    )

    print("\n=== delta convergence summary ===")
    print(summary_df.to_string(index=False))
if __name__ == '__main__':
    main()
