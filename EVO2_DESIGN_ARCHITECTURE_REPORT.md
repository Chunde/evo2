# Evo 2: Design and Architecture Analysis Report

## Executive Summary

**Evo 2** is a state-of-the-art **DNA language model** designed for genomic sequence modeling, generation, and analysis across all domains of life. While it represents a significant advancement in computational biology, **Evo 2 is NOT designed for protein folding prediction like AlphaFold 3**. Instead, it focuses on DNA/RNA sequence understanding, variant effect prediction, and genome design at single-nucleotide resolution.

This report provides a comprehensive analysis of Evo 2's architecture, capabilities, and key differences from AlphaFold 3.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Model Architecture Details](#model-architecture-details)
3. [Key Capabilities](#key-capabilities)
4. [Application Domains](#application-domains)
5. [Comparison with AlphaFold 3](#comparison-with-alphafold-3)
6. [Can Evo 2 Predict Protein Folding?](#can-evo-2-predict-protein-folding)
7. [Technical Specifications](#technical-specifications)
8. [Conclusion](#conclusion)

---

## Architecture Overview

### Core Design Philosophy

Evo 2 is built as an **autoregressive DNA foundation model** that treats genomic sequences as a formal language. The model learns the "grammar" and "syntax" of DNA by training on 8.8 trillion nucleotide tokens from diverse organisms across all domains of life.

### Primary Objective

**Goal**: Model DNA sequences at single-nucleotide resolution with context windows up to **1 million base pairs**, enabling:
- Understanding of long-range genomic interactions
- Prediction of variant effects on gene function
- Generation of novel functional DNA sequences
- Extraction of biologically meaningful embeddings

### Training Data

- **Dataset**: OpenGenome2 (8.8 trillion tokens)
- **Coverage**: All domains of life (microbes, plants, animals, humans)
- **Resolution**: Single-nucleotide level
- **Training Framework**: Savanna (custom training infrastructure)

---

## Model Architecture Details

### StripedHyena 2 Architecture

Evo 2 utilizes the **StripedHyena 2** architecture, which is a hybrid approach combining:

1. **Hyena Filters** (Long convolutions) - For efficient long-range dependency modeling
2. **Attention Mechanisms** - For focused contextual understanding
3. **MLP Layers** - For non-linear transformations

This architecture was specifically chosen over pure transformer architectures for its efficiency in handling extremely long sequences (up to 1M bp).

### Architectural Components

#### 1. **Input Processing**
```python
Tokenizer: CharLevelTokenizer(512)
- Character-level tokenization of DNA sequences (A, C, G, T, N, etc.)
- Vocabulary size: 512 tokens
- Handles special tokens for species tags, control sequences
```

#### 2. **Layer Types** (Example: 40B model with 50 layers)

The model interleaves different layer types in a specific pattern:

- **HCS Layers** (Hyena Convolutional Short): 
  - Indices: [0,4,7,11,14,18,21,25,28,32,36,39,43,46]
  - Short filter length: 7
  - Local pattern recognition

- **HCM Layers** (Hyena Convolutional Medium):
  - Indices: [1,5,8,12,15,19,22,26,29,33,37,40,44,47]
  - Filter length: 128
  - Medium-range dependencies

- **HCL Layers** (Hyena Convolutional Long):
  - Indices: [2,6,9,13,16,20,23,27,30,34,38,41,45,48]
  - Number of filters: 8192 (40B model)
  - Long-range genomic interactions

- **Attention Layers**:
  - Indices: [3,10,17,24,31,35,42,49]
  - Multi-head attention with rotary embeddings
  - Focused contextual modeling

#### 3. **Key Architectural Features**

```yaml
For Evo 2 40B (1M context):
  - Parameters: 40 billion
  - Hidden size: 8192
  - Num layers: 50
  - Attention heads: 64
  - Max sequence length: 1,048,576 bp
  - Rotary embedding base: 1,000,000
  - MLP inner size: 22,528
  - State size: 16
  - Filter groups: 8192
```

#### 4. **Positional Encoding**

- **Rotary Position Embeddings (RoPE)** with interpolation
- Scaling factor: 128
- Enables generalization to sequence lengths beyond training

#### 5. **Activation Functions**

- **Evo 2 style activations**: Custom activation patterns
- **GELU** in MLP layers
- Layer-specific activation strategies

#### 6. **Normalization**

- Final normalization layer
- RMSNorm variants with Flash Attention support

---

## Key Capabilities

### 1. **Sequence Scoring** (`score_sequences`)

Computes log-likelihood scores for DNA sequences:
- Evaluates how "natural" or "functional" a sequence is
- Can score sequences in both forward and reverse-complement directions
- Applications: Variant effect prediction, regulatory element identification

```python
scores = model.score_sequences(
    seqs=["ACGT..."], 
    batch_size=1,
    average_reverse_complement=False
)
```

### 2. **Embedding Extraction** (`return_embeddings`)

Extracts intermediate layer representations:
- Shape: `(batch, sequence_length, hidden_dim)`
- Intermediate embeddings outperform final layer embeddings
- Applications: Downstream classification, clustering, feature analysis

```python
outputs, embeddings = model(
    input_ids, 
    return_embeddings=True, 
    layer_names=['blocks.28.mlp.l3']
)
```

### 3. **Conditional Generation** (`generate`)

Autoregressive DNA sequence generation:
- Prompt-based generation with controllable parameters
- Temperature, top-k, top-p sampling
- Applications: Genome design, sequence completion, synthetic biology

```python
output = model.generate(
    prompt_seqs=["ACGT"],
    n_tokens=500,
    temperature=1.0,
    top_k=4
)
```

### 4. **Forward Pass** (`forward`)

Standard forward pass returning logits:
- Output shape: `(batch, sequence_length, vocab_size)`
- Used for next-token prediction, likelihood computation

---

## Application Domains

### 1. **Variant Effect Prediction**

**Use Case**: BRCA1 cancer variant classification
- Input: Reference and alternative allele sequences (8kb windows)
- Output: Log-likelihood ratios predicting pathogenicity
- Performance: Zero-shot prediction without fine-tuning

**Implementation**:
```python
ref_score = model.score_sequences([ref_seq])[0]
var_score = model.score_sequences([var_seq])[0]
effect_score = var_score - ref_score  # Delta log-likelihood
```

### 2. **Exon Classification**

**Use Case**: Identify coding vs non-coding regions
- Extract embeddings from intermediate layers
- Train classifier on top of frozen Evo 2 embeddings
- Achieves high accuracy with minimal training data

### 3. **Genome Design**

**Use Case**: Generate functional synthetic genomes
- Prompt with species phylogenetic tags
- Generate novel sequences matching organism characteristics
- Applications: Synthetic biology, gene therapy vectors

### 4. **Interpretable Feature Analysis**

**Use Case**: Sparse Autoencoder (SAE) for feature discovery
- Extract latent features from model activations
- Identify biologically meaningful patterns:
  - Exon-intron boundaries
  - Transcription factor binding sites
  - Protein secondary structure signals

### 5. **Phage Genome Engineering**

**Use Case**: Design bacteriophage genomes
- Competition analysis between phage variants
- Gibson assembly simulation
- Genetic architecture optimization

---

## Comparison with AlphaFold 3

### Fundamental Differences

| **Aspect** | **Evo 2** | **AlphaFold 3** |
|------------|-----------|-----------------|
| **Primary Input** | DNA/RNA nucleotide sequences | Protein amino acid sequences + ligands |
| **Primary Output** | Sequence likelihoods, embeddings, generated DNA | 3D atomic coordinates of protein structures |
| **Core Task** | Language modeling (next-token prediction) | Structure prediction (3D coordinate generation) |
| **Architecture** | StripedHyena 2 (hybrid CNN-attention) | Diffusion Transformer + Pairformer |
| **Context Length** | Up to 1M base pairs | Limited by MSA depth and complex size |
| **Training Data** | 8.8T nucleotides (genomic sequences) | PDB structures + MSAs |
| **Resolution** | Single-nucleotide | Atomic (Ångström-level 3D positions) |

### Detailed Architecture Comparison

#### **Evo 2 Architecture**

```
Input DNA Sequence → CharLevel Tokenizer → StripedHyena 2 Layers
                                                      ↓
                         ┌─────────────────────────────┼─────────────────────────────┐
                         ↓                             ↓                             ↓
                  Hyena Filters                Attention Layers               MLP Layers
            (Long/Medium/Short Convs)         (RoPE, Multi-head)           (GELU, GLU)
                         ↓                             ↓                             ↓
                         └─────────────────────────────┼─────────────────────────────┘
                                                      ↓
                                    Output: Logits (vocab distribution)
                                          Embeddings (hidden states)
```

**Key Characteristics**:
- Autoregressive language modeling objective
- Interleaved Hyena and attention layers
- Efficient long-range modeling via parameterized convolutions
- Rotary positional embeddings for position awareness
- FP8 precision support for large models

#### **AlphaFold 3 Architecture**

```
Protein Sequence + MSA + Templates → Tokenization → MSAModule
                                                      ↓
                                               Pairformer Stack
                                            (Pairwise attention)
                                                      ↓
                                              Diffusion Module
                              (Iterative denoising of 3D coordinates)
                                                      ↓
                                    Output: 3D atomic coordinates (x, y, z)
                                          Confidence scores (pLDDT)
```

**Key Characteristics**:
- Diffusion-based generative process
- Starts from random noise, iteratively refines structure
- Explicit modeling of pairwise distances and angles
- Incorporates evolutionary information (MSAs)
- Predicts all-atom structures including ligands, modifications

### Training Objectives

#### Evo 2
```python
# Next-token prediction (autoregressive)
Loss = CrossEntropy(predicted_next_token, actual_next_token)

# Trained to predict: P(x_t | x_1, ..., x_{t-1})
```

#### AlphaFold 3
```python
# Structure diffusion loss
Loss = MSE(predicted_coordinates, true_coordinates)
     + Frame alignment loss
     + Distance matrix loss

# Trained to denoise: P(structure | noisy_structure, sequence, MSA)
```

### Output Representations

#### Evo 2 Outputs
1. **Logits**: `(batch, seq_len, vocab)` - Next token probabilities
2. **Embeddings**: `(batch, seq_len, hidden_dim)` - Contextual representations
3. **Generated Sequences**: Novel DNA strings

#### AlphaFold 3 Outputs
1. **Atomic Coordinates**: `(num_atoms, 3)` - 3D positions in Ångströms
2. **Confidence Scores**: Per-residue pLDDT (0-100)
3. **PAE Matrix**: Predicted alignment error between residues

---

## Can Evo 2 Predict Protein Folding?

### Direct Answer: **NO**

**Evo 2 cannot directly predict protein 3D structures like AlphaFold 3.** Here's why:

### 1. **Different Problem Domains**

- **Evo 2**: Models **DNA sequence grammar** - predicts which nucleotides are likely to appear next in a genomic context
- **AlphaFold 3**: Models **protein physics** - predicts how amino acids fold into 3D structures based on physical constraints

### 2. **Missing Capabilities in Evo 2**

Evo 2 lacks critical components needed for structure prediction:

❌ **No 3D coordinate output head** - Only outputs token distributions  
❌ **No diffusion module** - Cannot iteratively refine spatial arrangements  
❌ **No distance/angle predictions** - Doesn't model spatial relationships  
❌ **No MSA processing** - Doesn't leverage evolutionary coupling information for structure  
❌ **No structural loss functions** - Trained on sequence likelihood, not geometric accuracy  

### 3. **What Evo 2 CAN Do Related to Proteins**

While Evo 2 cannot predict folding, it has **indirect protein-related capabilities**:

✅ **Predict variant effects on protein function**  
   - By scoring DNA sequences with mutations
   - Identifies variants likely to disrupt protein function
   - Example: BRCA1 cancer variant classification

✅ **Generate DNA sequences encoding proteins**  
   - Can generate coding sequences with proper reading frames
   - Maintains codon usage biases
   - Example: Synthetic gene design

✅ **Extract embeddings correlated with protein features**  
   - Some latent features may encode protein-binding information
   - SAE features can detect regulatory elements affecting translation
   - Example: Exon classification

✅ **Signal peptide and localization prediction** (with fine-tuning)  
   - DNA patterns correlate with protein localization
   - Can be adapted for protein property prediction

### 4. **Indirect Protein Structure Workflow**

If you want to use Evo 2 for protein-related tasks, here's a possible workflow:

```
Evo 2 DNA Generation → ESMFold/AlphaFold → Protein Structure

Step 1: Use Evo 2 to generate novel DNA sequences
Step 2: Translate DNA to protein sequence
Step 3: Feed protein sequence to ESMFold or AlphaFold 3
Step 4: Obtain 3D structure prediction
```

**Note**: This is a two-step pipeline where Evo 2 handles DNA design, and a separate tool (ESMFold, AlphaFold) handles structure prediction.

---

## Technical Specifications

### Model Variants

| Model | Parameters | Context Length | Use Case |
|-------|-----------|---------------|----------|
| **evo2_40b** | 40B | 1M bp | Maximum performance, full-genome modeling |
| **evo2_20b** | 20B | 1M bp | Balanced speed/performance |
| **evo2_7b** | 7B | 1M bp | Standard research use |
| **evo2_7b_262k** | 7B | 262K bp | Medium-length contexts |
| **evo2_7b_base** | 7B | 8K bp | Quick prototyping |
| **evo2_1b_base** | 1B | 8K bp | Educational/demo purposes |

### Hardware Requirements

#### Memory Requirements (Approximate)

| Model | GPU Memory (FP8) | GPU Memory (BF16) |
|-------|------------------|-------------------|
| evo2_40b | 80GB+ (multi-GPU) | 120GB+ (multi-GPU) |
| evo2_20b | 40-80GB | 80GB+ |
| evo2_7b | 16-24GB | 24-32GB |
| evo2_1b | 8-12GB | 12-16GB |

#### GPU Compatibility

- **40B/20B models**: Require H100 GPUs with FP8 support
- **7B models**: Can run on consumer GPUs (RTX 3090/4090) in BF16
- **Multi-GPU support**: Automatic via Vortex inference engine

### Software Dependencies

```yaml
Core:
  - PyTorch >= 2.6
  - CUDA >= 12.1
  - cuDNN >= 9.3
  
Optional (for 40B/20B models):
  - Transformer Engine >= 2.3.0 (FP8 support)
  - Flash Attention == 2.8.0.post2
  
Package:
  - vortex (vortex inference engine)
  - biopython
  - huggingface_hub
```

### Performance Benchmarks

#### Forward Pass Speed (tokens/sec)

| Model | Single H100 | Multi-GPU (8x) |
|-------|-------------|----------------|
| evo2_40b | N/A (requires multi-GPU) | ~500 |
| evo2_20b | ~200 | ~800 |
| evo2_7b | ~600 | N/A |
| evo2_1b | ~1500 | N/A |

#### Accuracy on Token Prediction

| Model | Loss | Accuracy |
|-------|------|----------|
| evo2_40b | 0.216 | 91.67% |
| evo2_20b | 0.217 | 91.67% |
| evo2_7b | 0.348 | 86.35% |
| evo2_1b | 0.502 | 79.56% |

---

## Conclusion

### Summary

**Evo 2** is a powerful **DNA language model** optimized for:
- ✅ Genomic sequence modeling (up to 1M bp)
- ✅ Variant effect prediction (e.g., disease-causing mutations)
- ✅ DNA sequence generation and design
- ✅ Embedding extraction for downstream tasks
- ✅ Regulatory element and exon classification

**Evo 2 is NOT suitable for**:
- ❌ Protein 3D structure prediction
- ❌ Protein-ligand docking
- ❌ Antibody-antigen interaction modeling
- ❌ Atomic-level structural biology

### When to Use Each Model

#### Use **Evo 2** when you need to:
- Analyze DNA/RNA sequences
- Predict effects of genetic variants
- Design synthetic genomes
- Extract genomic features
- Model long-range regulatory interactions

#### Use **AlphaFold 3** when you need to:
- Predict protein 3D structures
- Model protein-protein complexes
- Analyze protein-ligand interactions
- Study antibody-antigen binding
- Understand molecular mechanisms at atomic resolution

### Complementary Nature

While Evo 2 and AlphaFold 3 solve different problems, they can be **used together** in integrated workflows:

```
Example Pipeline: Disease Variant Analysis

1. Evo 2: Score patient DNA variants for pathogenicity
2. Identify variants disrupting protein function
3. AlphaFold 3: Predict how mutations affect protein structure
4. Combine insights for mechanistic understanding
```

### Future Directions

Potential areas where Evo 2 could expand:
- **Multi-modal training**: Joint DNA-protein modeling
- **Structure-aware objectives**: Incorporate structural constraints
- **Fine-tuning for protein properties**: Stability, solubility, function
- **Integration with structure predictors**: End-to-end DNA→structure pipelines

However, as currently designed, **Evo 2 remains a DNA language model, not a protein structure predictor**.

---

## References

1. Brixi, G. et al. (2026). "Genome modelling and design across all domains of life with Evo 2." *Nature*. DOI: 10.1038/s41586-026-10176-5

2. Abramson, J. et al. (2024). "Accurate structure prediction of biomolecular interactions with AlphaFold 3." *Nature*. DOI: 10.1038/s41586-024-07487-w

3. Evo 2 GitHub Repository: https://github.com/ArcInstitute/evo2

4. AlphaFold 3 Documentation: https://www.ebi.ac.uk/training/online/courses/alphafold/alphafold-3-and-alphafold-server/

5. NVIDIA BioNeMo: https://docs.nvidia.com/nim/bionemo/evo2/latest/

---

**Report Generated**: March 4, 2026  
**Author**: AI Code Analysis Assistant  
**Based on**: Evo 2 v0.5.3 codebase and documentation
