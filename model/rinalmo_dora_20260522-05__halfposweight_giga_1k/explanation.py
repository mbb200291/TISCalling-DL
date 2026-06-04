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
from main import _tokenizer

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


def build_baseline(input_ids: torch.Tensor, baseline_token_id: int) -> torch.Tensor:
    """
    Baseline A: replace all real tokens with [MASK].
    CLS (pos 0) and EOS (pos -1) are kept as-is — changing them
    can destabilise the model's positional encoding.
    """
    baseline = input_ids.clone()
    baseline[:, 1:-1] = baseline_token_id
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
    attention_mask = enc["attention_mask"].masked_fill(input_ids == _PAD_ID, 0)
    attention_mask.to(DEVICE)

    baseline_ids = build_baseline(input_ids, baseline_token_id)

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
        internal_batch_size=1,  # batch_size / n_steps
        return_convergence_delta=True,
    )  # attrs: (1, L, D)

    # dim reduction over embedding dim → per-token scalar, strip CLS/EOS
    if dim_reduction == 'l2-norm':
        attrs_per_token = attrs.norm(dim=-1).squeeze(0)[1:-1].detach().cpu().numpy()
    elif dim_reduction == 'sum':
        attrs_per_token = attrs.sum(dim=-1).squeeze(0)[1:-1].detach().cpu().numpy()   # sum over embedding dim
    else:
        raise ValueError(f"unknown: {dim_reduction}")

    with torch.no_grad():
        logit = forward_func(input_ids).item()
    prob = torch.sigmoid(torch.tensor(logit)).item()

    return IGResult(
        sequence=sequence,
        attributions=attrs_per_token,
        pred_prob=prob,
        convergence_delta=delta.item(),
        baseline_type="[MASK]",
        dimension_reduction=dim_reduction,
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
    """
    model.eval()

    enc = tokenizer(sequence, return_tensors="pt", padding=False, truncation=True)
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].masked_fill(input_ids == _PAD_ID, 0)
    attention_mask.to(DEVICE)

    forward_func = make_forward_func(model, attention_mask)
    lig = LayerIntegratedGradients(
        forward_func,
        model.backbone.base_model.model.embeddings.word_embeddings,
    )
    lpad = len(sequence) - len(sequence.lstrip('<pad>'))
    rpad = len(sequence) - len(sequence.lstrip('<pad>'))
    shuffled_seqs = map(lambda x: "<pad>" * lpad // 5 + x + "<pad>" * rpad // 5,
                        dinucleotide_shuffle(sequence[lpad:-rpad], n_shuffles=n_shuffles, seed=seed))
    all_attrs = []
    deltas = []

    for shuf_seq in shuffled_seqs:
        shuf_enc = tokenizer(
            shuf_seq, return_tensors="pt", padding=False, truncation=True
        )
        baseline_ids = shuf_enc["input_ids"].to(DEVICE)

        # Ensure same length as input (tokenization should be identical
        # for single-nt tokenizers; k-mer tokenizers may differ)
        if baseline_ids.shape != input_ids.shape:
            print(f"  [warn] baseline length mismatch — skipping this shuffle")
            continue

        attrs, delta = lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            n_steps=n_steps,
            method="gausslegendre",
            internal_batch_size=1,  # batch_size / n_steps
            return_convergence_delta=True,
        )  # (1, L, D)
        deltas.append(delta.item())
        if dim_reduction == 'l2-norm':
            attrs_per_token = attrs.norm(dim=-1).squeeze(0)[1:-1].detach().cpu().numpy()
        elif dim_reduction == 'sum':
            attrs_per_token = attrs.sum(dim=-1).squeeze(0)[1:-1].detach().cpu().numpy()   # sum over embedding dim
        else:
            raise ValueError(f"unknown: {dim_reduction}")

        all_attrs.append(attrs_per_token)
    
    print(f"delta mean: {np.mean(deltas):.4f}, mix: {np.min(np.abs(deltas)):.4f},  max: {np.max(np.abs(deltas)):.4f}")

    mean_attrs = np.mean(all_attrs, axis=0)   # (seq_len,)

    with torch.no_grad():
        logit = forward_func(input_ids).item()
    prob = torch.sigmoid(torch.tensor(logit)).item()

    return IGResult(
        sequence=sequence,
        attributions=mean_attrs,
        pred_prob=prob,
        convergence_delta=float("nan"),   # averaged baseline → no single delta
        baseline_type="dinuc-shuffle",
        dimension_reduction=dim_reduction,
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
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
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
    import sys
    # ── example usage (requires real model) ──────────────────────────────────────
    # seq = "GCCACCAUGGCCAAAGUGAAGAAGCAGAUCCUGCUG"
    # NEUTRAL_TOKEN_ID = 4  # for mask token
    # NEUTRAL_TOKEN_ID = 0  # for pad token
    NEUTRAL_TOKEN_ID = 10   # for N token
    seq = sys.argv[1]
    dim_reduction = 'sum'
    # lig_result = run_lig_mask(
    #     seq, model, tokenizer, NEUTRAL_TOKEN_ID,
    #     n_steps=1000,
    #     dim_reduction=dim_reduction
    # )
    
    lig_result = run_lig_dinuc(
        seq, model, tokenizer, 
        n_shuffles=10,
        n_steps=500, # 10,
        dim_reduction='l2-norm'
    )
    
    print(f"P(TIS) = {lig_result.pred_prob:.4f}  |  Δ_conv = {lig_result.convergence_delta:.5f}")

    SAVE_DIR = "ig_data"
    import os; os.makedirs(SAVE_DIR, exist_ok=True)

    fig, axes = plt.subplots(
        2, 1, figsize=(16, 4),
        gridspec_kw={"height_ratios": [4, 1]},
        facecolor="#0d1117",
    )
    plot_attribution(lig_result, ax_bar=axes[0], ax_seq=axes[1])
    plt.tight_layout(h_pad=0)
    plt.savefig(os.path.join(SAVE_DIR, "lig_result.png"), dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()
    save_ig_result(lig_result, os.path.join(SAVE_DIR, "lig_result.json"))
    print(f"saved → {save_path}")
    print(f"ig run complete")

    # print('------- debug -------')

    # with torch.no_grad():
    #     enc = tokenizer(seq, return_tensors="pt", padding=False, truncation=True)
    #     input_ids      = enc["input_ids"].to(DEVICE)
    #     attention_mask = enc["attention_mask"].to(DEVICE)
        
    #     baseline_ids = build_mask_baseline(input_ids, NEUTRAL_TOKEN_ID)
        
    #     forward_func = make_forward_func(model, attention_mask)
        
    #     f_input    = forward_func(input_ids).item()
    #     f_baseline = forward_func(baseline_ids).item()

    # print(f"f(input)    = {f_input:.4f}")
    # print(f"f(baseline) = {f_baseline:.4f}")
    # print(f"expected Δ  = {f_input - f_baseline:.4f}")


if __name__ == '__main__':
    main()
