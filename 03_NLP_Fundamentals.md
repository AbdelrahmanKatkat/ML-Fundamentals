# Chapter 3: Natural Language Processing (NLP) Fundamentals

## 1. Language Theory for ML
Text is **unstructured data**. To a computer, "I love AI" is just a stream of bytes. To perform ML, we must bridge the gap between **Linguistics** (Syntax, Semantics) and **Mathematics** (Vectors, Matrices).

*   **Syntax**: The grammatical structure.
*   **Semantics**: The meaning. "I'm feeling blue" (Sad) vs "The sky is blue" (Color).
*   **Context**: "Bank" (Finance) vs "Bank" (River). Transformer models (BERT/GPT) excel here because they look at all words at once.

---

## 2. Text Preprocessing Pipeline
Since text is noisy, we clean it before vectorization.

1.  **Cleaning**: Lowercase, remove HTML tags, remove special chars.
2.  **Tokenization**: Splitting text into units (tokens).
    *   "Hello World" $\to$ `["Hello", "World"]`
3.  **Stop Words Removal**: Removing common words ("the", "is", "and") that carry little unique info.
    *   *Caution*: In Sentiment Analysis, "not" is a stop word but is crucial ("not good"). Don't blindly remove.
4.  **Normalization**:
    *   **Stemming**: Chopping ends. `Running` $\to$ `Run`. Fast, but crude (`Better` $\to$ `Better`? No, `Bat`?).
    *   **Lemmatization**: Dictionary-based reduction. `Better` $\to$ `Good`. Slower, but accurate.

---

## 3. Feature Extraction (Vectorization)
We need to turn text into numbers.

### Bag of Words (BoW)
Count frequency of each word.
*   Vector size = Vocabulary size (can be huge).
*   **Issue**: "This movie is not good" and "This movie is good" might look similar if "not" is ignored or overwhelmed by other counts. Loses order.

### TF-IDF (Term Frequency - Inverse Document Frequency)
Used to penalize words that appear everywhere ("the") and boost words that are unique to a document.

#### Mathematical Formulas
1.  **TF (Term Frequency)**: How often word $t$ appears in doc $d$.
    $$ TF(t, d) = \frac{\text{count}(t \text{ in } d)}{\text{total words in } d} $$
2.  **IDF (Inverse Document Frequency)**: How rare is the word across all docs $D$?
    $$ IDF(t) = \log \left( \frac{N}{DF(t)} \right) $$
    Where $N$ is total docs, $DF(t)$ is number of docs containing $t$.
3.  **Score**:
    $$ \text{TF-IDF} = TF \times IDF $$

#### Numerical Intuition Example
Corpus:
*   Doc A: "The cat sat"
*   Doc B: "The dog sat"

Calculate TF-IDF for **"cat"** in Doc A.
1.  **TF("cat", A)**: "cat" appears 1 time. Total words = 3.
    $$ TF = 1/3 \approx 0.33 $$
2.  **IDF("cat")**:
    *   $N = 2$ docs.
    *   "cat" appears in 1 doc (Doc A).
    *   $$ IDF = \log(2/1) = \log(2) \approx 0.301 $$
3.  **TF-IDF**: $0.33 \times 0.301 \approx 0.099$

Calculate TF-IDF for **"The"** in Doc A.
1.  **TF**: $1/3 \approx 0.33$.
2.  **IDF**: "The" appears in 2 docs.
    $$ IDF = \log(2/2) = \log(1) = 0 $$
3.  **TF-IDF**: $0.33 \times 0 = 0$.
**Result**: The word "cat" has a score, "The" has 0. The model learns to focus on "cat".

---

## 4. Word Embeddings (Word2Vec)
TF-IDF creates sparse vectors. Embeddings create **dense** vectors where similar words are close in space.
*   **Concept**: "$King - Man + Woman \approx Queen$"
*   These are learned by training a neural network to predict a word given its context (CBOW) or context given a word (Skip-gram).

---
