# Transformer Encoder Architecture (from Scratch)

This repository contains a NumPy-based implementation of a Transformer Encoder module from scratch — inspired by the original Transformer architecture described in ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

---

## 🚀 Project Highlights

- ✅ **Word Tokenization and Embedding**
- ✅ **Positional Encoding (Sinusoidal)**
- ✅ **Scaled Dot-Product Attention**
- ✅ **Multi-Head Self Attention**
- ✅ **Residual Connections & Layer Normalization**
- ✅ **Position-wise Feed-Forward Network**
- ✅ **Stacked Encoder Layers**

This implementation is **minimal**, **educational**, and built **without using PyTorch or TensorFlow** — just **NumPy**.

---

## 🧠 Why This Project?

The aim of this project is to **understand the inner workings** of the Transformer encoder block by manually implementing each component step-by-step — reinforcing your understanding of:

- Self-Attention mechanism  
- Positional encoding  
- Multi-head attention  
- Feed-forward layers  
- Residual connections and layer normalization

---

## 📁 Core Components

| Component                        | Description                                                         |
| -------------------------------- | ------------------------------------------------------------------- |
| **Tokenization**                 | Splits the sentence into lowercase word tokens                      |
| **Word Embedding**               | Randomly initializes word vectors                                   |
| **Positional Encoding**          | Adds sinusoidal patterns to capture position                        |
| **Q, K, V Matrices**             | Projects input embeddings to Query, Key, and Value vectors          |
| **Scaled Dot-Product Attention** | Calculates attention weights and combines Value vectors accordingly |
| **Multi-Head Attention**         | Runs multiple parallel attention heads                              |
| **Projection Layer**             | Concatenates all heads and applies a linear transformation          |
| **Residual + LayerNorm**         | Applies skip connections and normalizes                             |
| **Feed-Forward Network**         | Two-layer MLP with ReLU activation in between                       |
| **Encoder Stack**                | Repeats the encoder block for N layers (default = 8)                |

## 🙌 Acknowledgments
Attention Is All You Need (Vaswani et al.)

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

Blogs and tutorials from the deep learning community

## Connect With Me
If you're interested in deep learning, NLP, or building projects like this — let's connect!

📧 vickyvk5156@gmail.com

