# TIS Identification Progress Report

_This document is an AI-generated summary based on existing reports and data._

## 1. Research Objective

- Predict Translation Initiation Sites (TIS) directly from mRNA sequences
- Reduce manual feature engineering with an end-to-end deep learning model
- Investigate long-range cis-element regulation and the value of RNA foundation models

**Overall roadmap**

`RNN → Transformer → Pretrained RNA embeddings → RiNALMo fine-tuning → Sequence explanation (partialy) → Sequence generation (not yet)`

---

## 2. Dataset and Training Objective

| Category | Count | Description |
|---|---:|---|
| True Positive | 4,394 | TIS identified from ribosome profiling data and confirmed by the intersection of two biological replicates |
| True Negative | 29,877 | Candidate sites upstream of the most downstream TP TIS within the same transcript, but not labeled as TP |
| Total | 34,271 | A single transcript may contain multiple TIS candidates |

**Training objective**

- Train one model across UTR/CDS and cognate/near-cognate candidate sites
- Use AUPR as the primary evaluation metric
- Keep samples from the same transcript within the same split to reduce the risk of sequence memorization

---

## 3. Input Representation and Model Design

### Fixed context window around the candidate site

```text
Left sequence ───── [Candidate TIS] ───── Right sequence
                         ↑ fixed position
```

- Use sequence classification rather than token classification
- Place the candidate TIS at a fixed central position
- Add padding at either end when the available sequence is too short
- Prediction head: Transformer encoder + classification layer + sigmoid

### Pretrained model

- RiNALMo: approximately 650M parameters
- Pretrained on approximately 36M non-coding RNA sequences from RNAcentral
- Fine-tuned with parameter-efficient LoRA / DoRA

---

## 4. Evaluation Design

### Group splitting

- Keep each transcript entirely within the train, development, or test split
- Also cluster sequences by similarity with MMseqs2 to test whether performance depends on gene identity
- Main experiments use transcript-aware splitting; selected experiments use 4-fold or 5-fold cross-validation

### Evaluation metrics

- **AUPR:** primary metric; more stable for imbalanced positive and negative classes
- **MCC:** supplementary measure of overall classification quality
- **Balanced F1:** evaluates performance across different TIS categories

---

## 5. Main Performance Results

Best long-context experiments using RiNALMo + DoRA with a 1,024-nt window (500 nt on each side):

| Model size | Parameters | MCC | AUPR |
|---|---:|---:|---:|
| RiNALMo micro | 34.7M | 0.910 | 0.982 |
| RiNALMo mega | 150.7M | 0.924 | 0.984 |
| RiNALMo giga | 659.6M | **0.951** | **0.988** |

**Key observations**

- Larger models and longer context windows generally improve overall performance
- RiNALMo achieves comparable or better performance than traditional TISCalling models across most categories
- Overall AUPR is close to 0.99, but performance still varies substantially between categories

---

## 6. Performance by TIS Category

Balanced F1 for the best Giga model:

| Category | Balanced F1 |
|---|---:|
| 5′ UTR ATG/AUG | **0.948** |
| 5′ UTR non-ATG | **0.850** |
| CDS ATG/AUG | **0.951** |
| CDS non-ATG | **0.477** |

**Main challenge**

- CDS non-ATG is clearly the weakest category and the highest-priority area for improvement
- Further investigation is needed into sample distribution, label quality, class weighting, and candidate-site definitions
- Overall AUPR alone can hide important category-level weaknesses

---

## 7. Effects of Context and Data Splitting

The context-selection experiments suggest that:

- A 500-nt window (300 nt left, 200 nt right) achieved approximately MCC 0.905 and AUPR 0.977 in cross-validation
- A 1,000-nt window with the 659M-parameter model achieved approximately MCC 0.951 and AUPR 0.988
- Splitting by gene ID and splitting by sequence similarity produced no significant performance difference

**Current direction**

> Use a larger model with a longer context window as the primary modeling direction, and use AUPR as the most stable comparison metric.

---

## 8. Sequence Explanation with Integrated Gradients

### Current progress

- Use Integrated Gradients to estimate each nucleotide/token's contribution to the prediction
- Analyze saliency maps across the full sequence and within the TIS-flanking region
- Test `[MASK]`, `[PAD]`, N-token, and shuffle-based baselines

### Main findings

- The `[MASK]` baseline requires a very large number of steps to approach convergence
- For RiNALMo Giga-1k with the mask baseline, the relative delta reached approximately 0.004 at 2,048 steps
- Among 109 samples, only 4 had relative delta < 0.05, and 48 had relative delta < 0.5
- With the full sequence as the baseline, the strongest contributions were concentrated in several nucleotides near the TIS
- No stable, generalizable sequence pattern has been identified yet

---

## 9. Conclusions and Next Steps

### Conclusions

1. End-to-end sequence models can achieve results comparable to or better than traditional feature-engineering approaches for TIS prediction.
2. RiNALMo fine-tuning shows promising value for RNA foundation-model-based TIS classification.
3. Larger models and longer context windows improve overall predictive performance.
4. CDS non-ATG remains the most significant performance bottleneck.
5. Integrated Gradients analysis is currently limited by baseline convergence issues.

### Next steps

- Diagnose the data and loss-weighting issues affecting CDS non-ATG
- Perform comprehensive cross-validation across context-window and splitting strategies
- Evaluate additional explanation and replacement methods
- Retrain models to improve the interpretability of padding and mask baselines
- Once attribution is stable, characterize sequence motifs at highly influential positions

---

## Data Sources

- `report.pptx`: research motivation, model design, evaluation workflow, and Integrated Gradients analysis
- `performance.xlsx`: experiment records, MCC, AUPR, and category-level Balanced F1 scores

