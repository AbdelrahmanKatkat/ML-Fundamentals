# Chapter 7: Advanced Text Modeling

## 1. Topic Modeling (Unsupervised)
Topic modeling discovers hidden "topics" in a collection of documents without labels.

### Latent Dirichlet Allocation (LDA)
LDA assumes:
1.  Each document is a mixture of topics.
2.  Each topic is a mixture of words.

**Intuition**:
*   Doc A contains "CPU", "GPU", "RAM". LDA infers "Hardware" topic.
*   Doc B contains "Nurse", "Medicine". LDA infers "Health" topic.
*   Doc C contains "CPU", "Nurse". LDA infers 50% Hardware, 50% Health.

**Mathematical Note**: It uses Dirichlet distributions (priors) to generate this probability.

---

## 2. Sequence-to-Sequence (Seq2Seq)
Used for Translation, Summarization.
*   **Encoder**: Reads input text into a context vector.
*   **Decoder**: Generates output text from the context vector.
*   **Issue**: Fixed context vector size limits long sentences (Information Bottleneck).
*   **Solution**: **Attention Mechanism** (Learn what to focus on).

---

## 3. The Transformer Revolution
Transformers (BERT, GPT) replaced RNNs/LSTMs.
*   **Self-Attention**: The model looks at *all* words in the sentence at the same time and calculates how much each word relates to every other word.
    *   Example: "The animal didn't cross the street because **it** was too tired."
    *   Self-attention links "**it**" strongly to "**animal**".

### BERT (Bidirectional Encoder Representations from Transformers)
*   **Bi-directional**: Reads left-to-right and right-to-left.
*   **Use Case**: Understanding tasks (Classification, NER, QA).

### GPT (Generative Pre-trained Transformer)
*   **Auto-regressive**: Predicts the next word.
*   **Use Case**: Generation (Writing text, Code).

---
