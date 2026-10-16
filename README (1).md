# GPT-2 for Text Classification

This project demonstrates how a pretrained GPT-style decoder model can be adapted for a **binary text-classification task**, such as sentiment analysis.

## Architecture

![GPT-2 architecture adapted for binary classification](./assets/gpt2-text-classification.png)

The diagram shows the main stages:

1. **Tokenization** — converts input text into token IDs.
2. **Token embeddings** — maps each token ID to a learned vector.
3. **Positional embeddings** — adds information about each token's position in the sequence.
4. **GPT transformer blocks (12× in the illustrated model)** — use masked multi-head self-attention and feed-forward layers to build contextual representations. Residual connections, layer normalization, and dropout support training.
5. **Classification head** — replaces the original language-model output layer with a small linear layer that maps the hidden representation (768 dimensions in the illustrated GPT-2 Small configuration) to two class scores: **positive** and **negative**.

### Language modeling vs. classification

The original language-model head maps each 768-dimensional hidden representation to a score for every vocabulary token (50,257 in the diagram). That lets the model predict the next token.

For binary classification, the vocabulary-sized output head is replaced with a 2-unit classification head. The two outputs are **logits** (raw class scores), not probabilities by themselves. A softmax can convert them into class probabilities.

> The diagram is illustrative. A complete implementation must also define how token-level representations are reduced to one sequence-level representation (for example, by selecting the final non-padding token) and how labels are mapped to class IDs.

## How classification works

Given an example such as:

> “The movie was excellent and moving.”

The model processes the token sequence and produces two scores:

- `negative` — score for the negative class
- `positive` — score for the positive class

The predicted class is usually the class with the larger logit. During training, the model's logits are compared with the true class label using a classification loss such as cross-entropy.

## Key concepts

| Component | Purpose |
|---|---|
| Token embedding | Represents each token as a learned vector |
| Positional embedding | Encodes token order |
| Masked multi-head attention | Lets each token attend to earlier tokens without seeing future tokens |
| Feed-forward network | Further transforms each token's contextual representation |
| Residual connection | Adds a block's input to its output to help information and gradients flow |
| Layer normalization | Stabilizes the scale of activations |
| Dropout | Regularization used during training |
| Linear classification head | Maps the chosen sequence representation to two class logits |

## Typical workflow

1. Prepare labeled text examples.
2. Tokenize the text using the tokenizer associated with the pretrained GPT-2 model.
3. Fine-tune the model on the classification dataset.
4. Evaluate on held-out data using metrics such as accuracy, precision, recall, F1 score, and a confusion matrix.
5. Inspect errors and check for class imbalance or data leakage.

## Model details in the diagram

- Transformer blocks: **12**
- Hidden size: **768**
- Original vocabulary size: **50,257**
- Binary classification output size: **2**

These dimensions correspond to the GPT-2 Small configuration represented in the diagram; other GPT-2 variants have different dimensions.

## Important implementation notes

- GPT-2 uses **causal (masked) self-attention**; it is not inherently a retrieval-augmented generation (RAG) system.
- A pretrained GPT-2 language model can be fine-tuned for classification. The original language-model head is not the same as the classification head.
- If batches contain padding, ensure the implementation selects the correct final non-padding token or otherwise pools the sequence representation appropriately.
- Use a consistent label mapping in training, validation, and inference.
- Keep a separate test set for a fair final evaluation.

## Repository structure

```text
.
├── README.md
├── assets/
│   └── gpt2-text-classification.png
├── data/
├── notebooks/
└── src/
```

Adjust the structure above to match the actual files in your repository.

## References

- [OpenAI GPT-2](https://openai.com/index/better-language-models/)
- [Hugging Face GPT-2 documentation](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [Hugging Face text-classification task guide](https://huggingface.co/docs/transformers/tasks/sequence_classification)

## License

Add a license file and update this section to reflect the license you choose.
