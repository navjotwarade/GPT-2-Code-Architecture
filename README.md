# GPT-2 From Scratch — Attention & Transformer Implementation

This repository contains a code-only Jupyter notebook that builds the main components of a GPT-2-style language model from the ground up using PyTorch.

## Notebook

- `Techdose_GPT2_Attention_code_only.ipynb`

## What This Notebook Covers

The notebook follows the implementation flow from raw text to a GPT-style Transformer:

1. **Text preprocessing**
   - Read raw text from `the-verdict.txt`
   - Basic regular-expression tokenization
   - Vocabulary creation
   - Token-to-ID and ID-to-token conversion
   - Unknown and end-of-text tokens

2. **GPT-2 BPE tokenization**
   - `tiktoken`
   - GPT-2 tokenizer
   - Encoding and decoding text
   - Token counts

3. **Dataset and DataLoader**
   - Context windows
   - Input/target token pairs
   - Sliding windows using `stride`
   - PyTorch `Dataset` and `DataLoader`

4. **Token and positional embeddings**
   - Token embedding matrix
   - Positional embeddings
   - Combining token and positional embeddings

5. **Self-attention**
   - Attention scores
   - Dot-product attention
   - Softmax attention weights
   - Context vectors
   - Query, Key, and Value projections
   - Scaled dot-product attention

6. **Causal attention**
   - Lower-triangular masking
   - Preventing access to future tokens
   - Attention dropout
   - Batched causal attention

7. **Multi-head attention**
   - Multiple attention heads
   - Head dimension
   - Reshaping and transposing tensors
   - Concatenating head outputs
   - Output projection

8. **Transformer building blocks**
   - Layer normalization
   - GELU activation
   - Feed-forward network
   - Residual/shortcut connections
   - Transformer block

9. **GPT-2-style model**
   - GPT-2 124M configuration
   - Token embeddings
   - Positional embeddings
   - Transformer blocks
   - Final LayerNorm
   - Vocabulary projection/output head

10. **Text generation**
    - Autoregressive token generation
    - Greedy next-token selection
    - Converting generated token IDs back to text

11. **Language-model loss**
    - Log probabilities
    - Cross-entropy loss
    - Perplexity

12. **Training**
    - Training/validation split
    - Training and validation DataLoaders
    - Loss calculation
    - AdamW optimizer
    - Training loop
    - Periodic evaluation
    - Text generation during training
    - Training/validation loss visualization

## GPT-2 Configuration

The notebook uses a GPT-2-style 124M configuration:

| Parameter | Value |
|---|---:|
| Vocabulary size | 50,257 |
| Context length | 1,024 |
| Embedding dimension | 768 |
| Attention heads | 12 |
| Transformer layers | 12 |
| Dropout | 0.1 |
| QKV bias | False |

Some later notebook sections use a shortened context length of `256` for experimentation/training.

## Architecture

The implemented flow is approximately:

```text
Text
  ↓
GPT-2 Tokenizer
  ↓
Token IDs
  ↓
Token Embeddings + Positional Embeddings
  ↓
Dropout
  ↓
Transformer Blocks
  ├── LayerNorm
  ├── Multi-Head Causal Self-Attention
  ├── Residual Connection
  ├── LayerNorm
  ├── Feed-Forward Network + GELU
  └── Residual Connection
  ↓
Final LayerNorm
  ↓
Linear Output Head
  ↓
Logits
  ↓
Softmax
  ↓
Next Token
```

## Requirements

Python 3.x with the following packages:

```bash
pip install torch tiktoken numpy matplotlib
```

## Running the Notebook

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
Techdose_GPT2_Attention_code_only.ipynb
```

Run the cells sequentially because several cells depend on variables, classes, and models created earlier in the notebook.

## Dataset

The notebook uses:

```text
the-verdict.txt
```

If the file is not present, one section downloads it from the `rasbt/LLMs-from-scratch` GitHub repository.

## Main Classes Implemented

The notebook includes implementations of:

```text
TokenizerV1
TokenizerV2
GPTDatasetV1
SelfAttention_v1
SelfAttention_v2
CausalAttention
MultiHeadAttentionWrapper
MultiHeadAttention
LayerNorm
GELU
FeedForward
TransformerBlock
GPTModel
```

It also contains simplified/dummy versions of GPT components for inspecting tensor flow and model structure.

## Key Tensor Flow

A typical GPT input follows:

```text
(batch_size, sequence_length)
        ↓
Token Embedding
        ↓
(batch_size, sequence_length, embedding_dimension)
        ↓
Positional Embedding
        ↓
Transformer Blocks
        ↓
(batch_size, sequence_length, embedding_dimension)
        ↓
Linear Output Head
        ↓
(batch_size, sequence_length, vocabulary_size)
```

## Learning Goal

The notebook is intended to make the internal mechanics of GPT-style Transformers explicit by implementing the major components directly in PyTorch rather than treating the Transformer as a black box.

The emphasis is on understanding:

- How text becomes token IDs
- How token IDs become embeddings
- How Q/K/V projections are created
- How attention scores are calculated
- Why attention is scaled by `sqrt(d_k)`
- How causal masking works
- How multiple attention heads work
- How residual connections and LayerNorm are used
- How Transformer blocks are assembled
- How logits become next-token probabilities
- How cross-entropy loss is calculated
- How a GPT-style model can be trained and used for text generation

## Reference

The notebook uses the GPT-2 tokenizer through `tiktoken` and follows a from-scratch implementation approach based around GPT-style Transformer architecture.
