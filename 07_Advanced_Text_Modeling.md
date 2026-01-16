# Chapter 7: Advanced Text Modeling

Simple counting (TF-IDF) can only take us so far. To truly "understand" language, machines need to grasp context, relationships, and hidden themes. This is where **Advanced Text Modeling** comes in.

---

## 1. Topic Modeling: The Generative Story
Imagine you have a stack of 1,000 unlabeled news articles. You want to group them into "Sports", "Politics", and "Tech" automatically. We use **Latent Dirichlet Allocation (LDA)**.

### The Intuition: "Reverse Engineering"
LDA assumes every document was written using a simple process:
1.  **Pick Topics**: I want this article to be 70% Politics and 30% Tech.
2.  **Pick Words**: 
    *   For the "Politics" part, I'll pick words like *Election, Vote, Law*.
    *   For the "Tech" part, I'll pick words like *Software, Algorithm, Silicon*.

**The Model's Job**: It looks at the final article and tries to **reverse engineer** which topics were chosen.
*   **Mathematical Note**: It uses the **Dirichlet Distribution**, which acts like a "preference" setting. One setting might prefer articles to have only ONE topic, while another prefers a mix.

### When to Use LDA?
*   **Document Clustering**: Grouping similar documents without labels.
*   **Exploratory Analysis**: Understanding what topics exist in a large corpus.
*   **Recommendation Systems**: Finding similar articles based on topic overlap.

### Practical Tips for LDA
1.  **Number of Topics**: Use **perplexity** or **coherence score** to find the optimal number of topics. Start with 5-10 and experiment.
2.  **Preprocessing**: Remove stop words and rare words (appearing in <5 documents).
3.  **Interpretability**: Manually review the top words for each topic to ensure they make sense.

---

## 2. Sequence Models: The Memory Problem
Before Transformers, we used **Recurrent Neural Networks (RNNs)** and **LSTMs** for sequence tasks.

### RNN: The Forgetful Reader
*   **How it works**: Reads text word by word, left to right, maintaining a "memory" of what it's seen.
*   **The Problem**: By the time it reaches word 100, it has forgotten word 1. This is called the **vanishing gradient problem**.

### LSTM: The Notebook
*   **How it works**: Like an RNN, but with a "notebook" (cell state) to store important information long-term.
*   **Gates**: 
    *   **Forget Gate**: Decides what to erase from the notebook.
    *   **Input Gate**: Decides what new information to write.
    *   **Output Gate**: Decides what to read from the notebook.

**Use Case**: LSTMs work well for tasks like sentiment analysis, named entity recognition, and short text generation.

---

## 3. Seq2Seq & The Attention Mechanism
In tasks like translation (English $\to$ Spanish), we use an **Encoder-Decoder** model.

### The Problem: The Information Bottleneck
Imagine someone reads a 500-page book and has to summarize the *entire* meaning into a single sentence. Then, they give that sentence to a second person who has to rewrite the whole book in another language. 
*   **The Issue**: The single sentence (Context Vector) is too small. Details get lost.

### The Solution: The Attention "Spotlight"
Instead of forcing the meaning into one vector, **Attention** allows the Decoder to "look back" at the original text while it is writing.
*   **Analogy**: When translating the word "Bank," the model shines a spotlight on surrounding words like "River" or "Money" to know which meaning to use.

### Attention Score Calculation
For each word the decoder generates, it calculates an **attention score** for every word in the input:
1.  **Score**: How relevant is each input word to the current output word?
2.  **Softmax**: Convert scores to probabilities (they sum to 1).
3.  **Weighted Sum**: Multiply each input word's embedding by its attention probability and sum them up.

**Result**: The decoder gets a context-aware representation that focuses on the most relevant input words.

---

## 4. Transformers: Parallel Reading
Before 2017, machines read text like a human: word by word, left to right. This was slow and forgot the beginning of long sentences. **Transformers** changed this by reading the entire paragraph **all at once**.

### Self-Attention: The Filing System Math
To understand a word, the Transformer uses three vectors:
1.  **Query ($Q$)**: "What am I looking for?" (The current word's intent).
2.  **Key ($K$)**: "What info do I have?" (The labels of all other words).
3.  **Value ($V$)**: "The actual content."

**The Math**:
$$ \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V $$
*   **Intuition**: We multiply $Q$ and $K$ to see how much they "match." If they match well, the model pays more "Attention" to that word's **Value ($V$)**.

### BERT vs. GPT: The Two "Personalities"
*   **BERT (The Reader)**: Reads bidirectionally (Left $\leftrightarrow$ Right). Good at understanding what a sentence *means*. Use for: *Sentiment Analysis, Question Answering.*
*   **GPT (The Writer)**: Reads only what came before (Left $\to$ Right). Optimized for predicting the *next word*. Use for: *Chatbots, Writing assistance.*

### Why Transformers Dominate
1.  **Parallelization**: Unlike RNNs/LSTMs (which read sequentially), Transformers process all words simultaneously. This makes training **much faster** on GPUs.
2.  **Long-Range Dependencies**: Attention allows the model to connect word 1 with word 100 directly, without the vanishing gradient problem.
3.  **Pre-training**: Models like BERT and GPT are pre-trained on billions of words, then **fine-tuned** on specific tasks with small datasets.

---

## 5. Tips and Tricks for Advanced Text Modeling
1.  **Start with pre-trained models**: Don't train BERT or GPT from scratch. Use Hugging Face's `transformers` library to load pre-trained models.
2.  **Fine-tuning vs Feature Extraction**:
    *   **Fine-tuning**: Update all layers of the pre-trained model on your task (better performance, slower).
    *   **Feature Extraction**: Freeze the pre-trained layers and only train a small classifier on top (faster, less data needed).
3.  **Computational Cost**: Transformers are **expensive**. BERT-base has 110M parameters. Use smaller models (DistilBERT, ALBERT) for production.
4.  **Sequence Length Limits**: BERT has a max sequence length of 512 tokens. For longer documents, use **Longformer** or **BigBird**.
5.  **Domain Adaptation**: If your text is very different from Wikipedia/news (e.g., medical, legal), consider **domain-specific pre-training** (e.g., BioBERT for medical text).
6.  **Hyperparameter Tuning**: For fine-tuning, start with:
    *   Learning rate: 2e-5 to 5e-5
    *   Batch size: 16 or 32
    *   Epochs: 3-5
7.  **Avoid Overfitting**: Use **early stopping** and monitor validation loss. Transformers overfit easily on small datasets (<1000 samples).
8.  **Use Mixed Precision Training**: Enable FP16 (16-bit floating point) to reduce memory usage and speed up training.
9.  **Batch Size vs Sequence Length Trade-off**: Longer sequences require more memory. Reduce batch size if you run out of GPU memory.
10. **Evaluate on Multiple Metrics**: For classification, use Accuracy, F1, Precision, and Recall. For generation, use BLEU, ROUGE, or human evaluation.

---

## 6. Model Selection Guide
| Task | Recommended Model | Why? |
| :--- | :--- | :--- |
| **Sentiment Analysis** | BERT, RoBERTa | Bidirectional context captures nuanced sentiment |
| **Text Classification** | DistilBERT, ALBERT | Faster and smaller than BERT, good for production |
| **Question Answering** | BERT, ELECTRA | Designed for understanding context |
| **Text Generation** | GPT-2, GPT-3 | Autoregressive, predicts next word |
| **Machine Translation** | T5, mBART | Seq2Seq with attention |
| **Summarization** | BART, T5 | Encoder-decoder architecture |
| **Named Entity Recognition** | BERT, SpaCy Transformers | Token-level classification |
| **Topic Modeling** | LDA, BERTopic | LDA for interpretability, BERTopic for accuracy |

---

## 7. Real-World Example: Fine-Tuning BERT for Sentiment Analysis
A company wants to classify product reviews as Positive, Neutral, or Negative.

### Step 1: Data Preparation
*   5,000 labeled reviews.
*   Split: 80% train, 10% validation, 10% test.

### Step 2: Load Pre-trained BERT
```python
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
```

### Step 3: Tokenize Text
```python
inputs = tokenizer(reviews, padding=True, truncation=True, max_length=512, return_tensors="pt")
```

### Step 4: Fine-Tune
*   Learning rate: 2e-5
*   Batch size: 16
*   Epochs: 3
*   Optimizer: AdamW

### Step 5: Evaluate
*   Test Accuracy: 92%
*   F1 Score: 0.91

### Step 6: Deploy
*   Save the fine-tuned model.
*   Use a REST API (FastAPI) to serve predictions.

### Computational Cost
*   Training time: 2 hours on a single GPU (NVIDIA T4).
*   Inference: 50ms per review.

---

## 8. The Future of NLP
*   **Larger Models**: GPT-4, PaLM, and beyond (100B+ parameters).
*   **Multimodal Models**: CLIP, DALL-E (combining text and images).
*   **Efficient Transformers**: Reformer, Linformer (reducing O(n²) complexity).
*   **Few-Shot Learning**: GPT-3 can perform tasks with just a few examples (no fine-tuning).
*   **Ethical Considerations**: Bias in language models, misinformation, and responsible AI.

---
