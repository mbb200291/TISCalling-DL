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
from main import _tokenizer, _PAD_ID

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
save_path = r'model.pt'
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

NT_COLOR = {"A": "#e74c3c", "U": "#3498db", "G": "#2ecc71", "C": "#f39c12", "N": "#95a5a6"}

def plot_attribution(
    result: IGResult,
    window: Optional[tuple] = None,
    figsize: tuple = (16, 4),
    ax_bar=None,
    ax_seq=None,
):
    seq   = result.sequence
    attrs = result.attributions.copy()
    if window is not None:
        s, e  = window
        seq   = seq[s:e]
        attrs = attrs[s:e]
    mx = np.abs(attrs).max()
    if mx > 0:
        attrs = attrs / mx
    pos    = np.arange(len(seq))
    colors = ["#e74c3c" if v > 0 else "#2563eb" for v in attrs]
    standalone = ax_bar is None
    if standalone:
        fig, (ax_bar, ax_seq) = plt.subplots(
            2, 1, figsize=figsize,
            gridspec_kw={"height_ratios": [4, 1]},
            facecolor="#0d1117",
        )
    for ax in (ax_bar, ax_seq):
        ax.set_facecolor("#0d1117")
    ax_bar.bar(pos, attrs, color=colors, width=0.85, linewidth=0)
    ax_bar.axhline(0, color="#30363d", lw=0.8)
    ax_bar.set_xlim(-0.5, len(seq) - 0.5)
    ax_bar.set_ylim(-1.2, 1.2)
    ax_bar.set_ylabel("Attribution (norm.)", color="#8b949e", fontsize=9)
    ax_bar.tick_params(colors="#8b949e")
    for sp in ax_bar.spines.values():
        sp.set_color("#30363d"); sp.set_linewidth(0.5)
    delta_str = f"Δ={result.convergence_delta:.4f}" if not np.isnan(result.convergence_delta) else "Δ=n/a"
    ax_bar.set_title(
        f"reduction_method: {result.dimension_reduction}  |  baseline: {result.baseline_type}  |  P(TIS)={result.pred_prob:.3f}  |  {delta_str}",
        color="#8b949e", fontsize=9, loc="right", pad=5,
    )
    ax_seq.set_facecolor("#0d1117")
    ax_seq.set_xlim(-0.5, len(seq) - 0.5)
    ax_seq.set_ylim(0, 1)
    ax_seq.axis("off")
    fs = max(5, min(11, 200 // len(seq)))
    for i, nt in enumerate(seq):
        ax_seq.text(
            i, 0.5, nt, ha="center", va="center",
            fontsize=fs, fontfamily="monospace", fontweight="bold",
            color=NT_COLOR.get(nt, "#95a5a6"),
        )
    if standalone:
        plt.tight_layout(h_pad=0)
        return fig


def save_ig_result(result: IGResult, path: str):
    """marshal as json"""
    data = {
        "sequence":          result.sequence,
        "attributions":      result.attributions.tolist(),
        "pred_prob":         result.pred_prob,
        "convergence_delta": result.convergence_delta if not np.isnan(result.convergence_delta) else None,
        "baseline_type":     result.baseline_type,
        "dim_reduction": result.dimension_reduction,
        "baseline_logit": result.baseline_logit,
        "score_diff": result.score_diff,
        "relative_delta": result.relative_delta,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        # json.dump(data, f, indent=2)
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
    print(f"saved → {path}")


def load_ig_result(path: str) -> IGResult:
    """load IGResult。"""
    with open(path) as f:
        data = json.load(f)
    return IGResult(
        sequence=          data["sequence"],
        attributions=      np.array(data["attributions"]),
        pred_prob=         data["pred_prob"],
        convergence_delta= data["convergence_delta"] if data["convergence_delta"] is not None else float("nan"),
        baseline_type=     data["baseline_type"],
        dim_reduction=data["dim_reduction"],
    )


def main():
    # import sys
    # ── example usage (requires real model) ──────────────────────────────────────
    NEUTRAL_TOKEN_ID = tokenizer.mask_token_id
    # NEUTRAL_TOKEN_ID = 0  # for pad token
    # NEUTRAL_TOKEN_ID = 10   # for N token
    dim_reduction = 'sum'
    # seq = sys.argv[1]
    seq = 'TATGAGTATAGATGAAGTATAGTAATAGTACTTAATATGTCGGATCCGTGCTATAAGCAATTTGGATTGTATGAAAAAATCTCTTAAATTTGATTGAAATATTATTCGTCCAATAATTATCAACTCAATACTAAATCGACCTATATCTAATTAATAATTTAACAAATTAATATTTTAAAAAATTGAAATATCAATAAGTCGAATCATTAAGCATAATTATCGGCCTATAAAAAAGATTTAGTCGTATTATCTGGAAAAGATTAAATAAATCCGGCGCCATATTCATAGTGATTTATGGCTGAAGCCCACACGTTATAACAACAACTACTCCATCAATGGAAGCTTCCATTTCTTGAATTTCTCAAACTCTTACTAAATTCAACTCCGGCGGTGGAACCCTTGTAATCTTCAACATATTTCGTTAATTAATTCTTATATACATATATATAAAGAAATTTGTTTTCTACTGCCGAAAGTTTCTTCTTCTCATCGGAGTAGATATGCCGAGCTTAATTGTCAAAGTTTACAGCTTACTCTTCAAGTATAACCTTAATCGCCGATTGCAATCACTAATCCAATCCCCAATTTCATACCCTTTTAACGGTGTCGTCTCACGCGCCGATGAATCGATTATCACTTCTAACCCTAGTTTCTCTACCGACGGTGTTGCAACTAAGGACCTGCATATTGATTCTTTGACTTGTCTATCTCTCAGGATTTACCTCCCTCAATCTGCACTTATTTCGTTGAGAAATTTGGAATCTGGTGAAGGGGTTTATGGGGGTTATGTACCGGGAAAAAATGGGAAAAATTGTAAGAAATTGCCGGTGATTTTGCAGTTTCATGGTGGTGCTTGGGTGACTGGGGGTATTGATACGGTTTCCAATGATGTTTTTTGTAGGAAATTGGCGAAATCTTGTGATGCTATTGTGATTGCTGTTGGGTATAGATTGGCACCGGAGAGTAGGTTTCCGGCTGCGTTTGAAGATGGGGTTGCGGCG'
    
    SAVE_DIR = "ig_data"
    import os; os.makedirs(SAVE_DIR, exist_ok=True)
    
    # sanity check
    sanity_check(seq, model, tokenizer,)
    
    # test convergency by mask token as baseline when step increasing
    for n_steps in [64, 128, 256, 512, 1024, 2048]:
        result = run_lig_mask(
            seq, model, tokenizer,
            n_steps=n_steps,
            dim_reduction="sum",
            baseline_token_id=NEUTRAL_TOKEN_ID,
        )
        print(
            n_steps,
            "delta=", result.convergence_delta,
            "relative_delta=", result.relative_delta,
            "score_diff=", result.score_diff,
            flush=True,
        )
        
    # # test convergency by di-nt shuffle step increasing
    # for n_steps in [64, 128, 256, 512, 1024, 2048]:
    #     result = run_lig_dinuc(
    #         seq, model, tokenizer,
    #         n_steps=n_steps,
    #         n_shuffles=10,
    #         dim_reduction='l2-norm',
    #     )
    #     print(
    #         n_steps,
    #         "delta=", result.convergence_delta,
    #         "relative_delta=", result.relative_delta,
    #         "score_diff=", result.score_diff,
    #         flush=True,
    #     )
        
    
    
    # df_test = pd.read_csv('df_test.csv')
    # df_test["seq"] = [
    #     centralize_transcript(hp["left_window"], hp["right_window"], s, p)
    #     for s, p in zip(df_test["seq"].values, df_test["position"].values)    
    # ]
    # c = 0
    # for s in df_test.seq:
    #     c += 1
    #     print(f"{c} -------------------------------", flush=True)        # if c == 3:
    #     #     break
    #     # lig_result = run_lig_dinuc(
    #     #     s, model, tokenizer, 
    #     #     n_shuffles=10,
    #     #     n_steps=500, # 10,
    #     #     dim_reduction='l2-norm'
    #     # )
    #     lig_result = run_lig_mask(
    #         s, model, tokenizer, NEUTRAL_TOKEN_ID,
    #         n_steps=1024,
    #         dim_reduction=dim_reduction
    #     )
        
    #     # print(f"P(TIS) = {lig_result.pred_prob:.4f}  |  Δ_conv = {lig_result.convergence_delta:.5f}")

    #     save_ig_result(lig_result, os.path.join(SAVE_DIR, "df_test_lig_results.jsonl"))

    # # fig, axes = plt.subplots(
    # #     2, 1, figsize=(16, 4),
    # #     gridspec_kw={"height_ratios": [4, 1]},
    # #     facecolor="#0d1117",
    # # )
    # # plot_attribution(lig_result, ax_bar=axes[0], ax_seq=axes[1])
    # # plt.tight_layout(h_pad=0)
    # # plt.savefig(os.path.join(SAVE_DIR, "lig_result.png"), dpi=150, bbox_inches="tight", facecolor="#0d1117")
    # # plt.show()
    # # print(f"saved → {save_path}")
    print(f"ig run complete")


if __name__ == '__main__':
    main()
