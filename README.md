# TIS Prediction with RNA Language Models

This repository contains ongoing research based on the original [TISCalling project](https://github.com/yenmr/TISCalling). The original project predicts Translation Initiation Sites (TIS) using deliberate engineered sequence and transcript features with conventional machine-learning classifiers.

This work keeps the original dataset and baseline implementation as a reference, then extends the project with end-to-end sequence models based on the pretrained RNA language model RiNALMo.

## Relationship to the Original Project

The original TISCalling implementation and documentation are available here:

- [Original TISCalling repository](https://github.com/yenmr/TISCalling)
- [Original feature-extraction scripts](https://github.com/yenmr/TISCalling/tree/main/script)

The original method is retained in this repository under [`experiments/tiscalling_original_method/`](experiments/tiscalling_original_method/). It provides the baseline feature-generation, model-training, evaluation, and prediction code.

## Current Research Focus

The current research investigates whether a sequence-only deep learning model can predict TIS candidates with less feature engineering and better use of long-range sequence context.

The current pipeline explores:

- Transformer-based sequence classification
- RiNALMo pretrained RNA representations
- LoRA / DoRA parameter-efficient fine-tuning
- Fixed context windows centered on the candidate TIS
- Transcript-aware and sequence-similarity-aware data splitting
- Integrated Gradients for sequence-level interpretation

The current modeling roadmap is:

`RNN → Transformer → Pretrained RNA embeddings → RiNALMo fine-tuning → Sequence explanation`

## Dataset

The experiments use the original TIS candidate data and focus on four categories:

- 5′ UTR ATG/AUG
- 5′ UTR non-ATG
- CDS ATG/AUG
- CDS non-ATG

The working dataset contains:

| Category | Count |
|---|---:|
| True Positive | 4,394 |
| True Negative | 29,877 |
| Total | 34,271 |

A single transcript may contain multiple candidate TIS sites. To reduce sequence memorization, samples from the same transcript are kept within the same train, development, or test split. Some experiments additionally group sequences by similarity using MMseqs2.

## Model Design

The candidate TIS is placed at a fixed position within the input sequence:

```text
Left sequence ───── [Candidate TIS] ───── Right sequence
                         ↑ fixed position
```

The current experiments use RiNALMo as the pretrained RNA encoder and add a classification head for binary TIS prediction. When the sequence is shorter than the requested context window, padding is added at the sequence boundaries.

The experiments include RiNALMo model sizes from approximately 35M to 660M total parameters. The largest model used in the current results is RiNALMo Giga with a 1,024-nt context window.

## Current Results

The strongest recorded long-context result uses RiNALMo Giga with DoRA fine-tuning and a 1,024-nt context window:

| Model | Parameters | MCC | AUPR |
|---|---:|---:|---:|
| RiNALMo micro | 34.7M | 0.910 | 0.982 |
| RiNALMo mega | 150.7M | 0.924 | 0.984 |
| RiNALMo giga | 659.6M | **0.951** | **0.988** |

Category-level Balanced F1 for the Giga model is:

| Category | Balanced F1 |
|---|---:|
| 5′ UTR ATG/AUG | **0.948** |
| 5′ UTR non-ATG | **0.850** |
| CDS ATG/AUG | **0.951** |
| CDS non-ATG | **0.477** |

The main remaining weakness is CDS non-ATG classification. Overall AUPR is high, but the category-level results show that performance is not uniform across TIS types.

## Sequence Interpretation

Integrated Gradients is being evaluated to identify nucleotide-level contributions to model predictions. Current findings include:

- The `[MASK]` baseline requires a large number of integration steps to approach convergence.
- In one Giga-1k experiment, the relative delta reached approximately 0.004 at 2,048 steps.
- Only 4 of 109 samples had relative delta < 0.05; 48 had relative delta < 0.5.
- Stronger contributions appear near the TIS when the full sequence is used as the baseline.
- A stable, generalizable sequence motif has not yet been identified.

## Repository Structure

```text
.
├── experiments/
│   ├── tiscalling_original_method/   # Original feature-based baseline
│   ├── prev_experiments/              # Earlier model experiments
│   └── rinalmo_dora_*/                # RiNALMo + LoRA/DoRA experiments
├── performance.xlsx                   # Experiment results and metrics
├── progress_report.md                 # Research progress notes
├── report.pptx                        # Progress presentation
├── requirements.txt                   # Python dependencies
└── LICENSE                            # MIT license
```

## Reproducing the Experiments

Create the Python environment used by the project:

```bash
conda create -n TIScalling python=3.10
conda activate TIScalling
pip install -r requirements.txt
```

The original feature-based baseline is located at:

```bash
cd experiments/tiscalling_original_method
```

The newer experiments are organized as dated directories under `experiments/`. Most experiment directories contain shell scripts for training and evaluation, together with the corresponding Python entry points and output logs. These experiments may require access to pretrained RiNALMo weights and the computing environment used for the original runs.

## Evaluation

The main metrics used in the current work are:

- **AUPR:** primary metric for imbalanced TIS classification
- **MCC:** supplementary measure of balanced classification quality
- **Balanced F1:** category-level performance measure

For a compact summary of the current experiments, see [`performance.xlsx`](performance.xlsx) and [`progress_report.md`](progress_report.md).

## Current Limitations and Next Steps

- Investigate the causes of poor CDS non-ATG performance.
- Perform broader cross-validation across context-window and split strategies.
- Improve Integrated Gradients baseline convergence.
- Evaluate additional sequence-replacement and explanation methods.
- Search for reproducible sequence motifs after attribution results become stable.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
