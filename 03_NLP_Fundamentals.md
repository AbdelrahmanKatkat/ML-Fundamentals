# Chapter 3: Natural Language Processing (NLP) Fundamentals

Computers are great at math, but they are terrible at reading books. To a computer, a sentence is just a long list of characters. **Natural Language Processing (NLP)** is the set of tools we use to turn human language into math that a machine can understand.

---

## 1. The Bridge: Language to Math
To bridge the gap between "Hello" and `[0.1, 0.4, -0.2]`, we look at three things:

1.  **Syntax (The Structure)**: The "Rules" of grammar. Like the blueprint of a house.
2.  **Semantics (The Meaning)**: What is actually being said. 
    *   Example: *"I'm feeling blue"* (Sadness) vs. *"The sky is blue"* (Color).
3.  **Pragmatics (The Context)**: How the situation changes the meaning.
    *   Example: Telling someone *"Great job!"* when they just broke a plate is sarcasm. Traditional ML struggles here; modern LLMs (Transformers) excel at it.

---

## 2. Preprocessing: Cleaning the Noise
Before we can calculate anything, we need to tidy up the text. Think of this as **filtering your data**.

### A. Tokenization (Slicing the Bread)
Breaking a sentence into small pieces called "Tokens" (usually words).
*   *"I love AI"* $\to$ `["I", "love", "AI"]`

### B. Stop Words (The Generic Filter)
Words like "the", "is", and "at" appear in almost every English sentence but carry very little unique information. We often remove them to save space and focus on the important words.
> [!CAUTION]
> In **Sentiment Analysis**, "not" is a stop word but it's vital! If you remove it, "I am not happy" becomes "I am happy," and your model fails.

### C. Normalization (Finding the Roots)
Words have many variations (run, running, ran). To a computer, these are three different words. We want to treat them as one.
*   **Stemming (The Axe)**: Chopping off the end of words. `Studying` $\to$ `Studi`. It's fast but messy.
*   **Lemmatization (The Surgeon)**: Looking up the dictionary "root" of a word. `Better` $\to$ `Good`. It's accurate but slower.

### D. Handling Special Characters and Emojis
In real-world text (especially social media), you'll encounter:
*   **Emojis**: 😊, 🔥, 💔
*   **Hashtags**: #MachineLearning
*   **Mentions**: @username
*   **URLs**: https://example.com

**Strategies**:
1.  **Remove**: For formal text (news articles, academic papers).
2.  **Keep**: For sentiment analysis, emojis carry strong emotional signals.
3.  **Replace**: Convert emojis to text (e.g., 😊 → "happy_face").

### E. Text Normalization
Standardize variations:
*   **Contractions**: "gonna" → "going to", "won't" → "will not"
*   **Slang**: "u" → "you", "ur" → "your"
*   **Spelling Correction**: "recieve" → "receive"

**Practical Tip**: Use libraries like `textblob` or `autocorrect` for automatic correction, but be careful—they can introduce errors.

---

## 3. Vectorization: Turning Words into Scores
How do we turn words into numbers?

### A. Bag of Words (BoW)
Imagine a literal bucket for every word in the dictionary. Every time a word appears in a document, you throw a stone into its bucket.
*   **Intuition**: Documents with similar "buckets" are likely about the same thing.
*   **The Flaw**: It ignores word order. "Dog bites man" and "Man bites dog" have the same buckets.

### B. TF-IDF (The "Uniqueness" Score)
This is the most popular way to find which words **actually matter** in a document.

#### The Intuition
*   **TF (Term Frequency)**: Does the word appear a lot in *this* document? (More is good).
*   **IDF (Inverse Document Frequency)**: Does the word appear in *every* document? (If yes, it's generic and boring—punish its score).

#### The Math
1.  **Term Frequency (TF)**:
    $$ TF(t, d) = \frac{\text{Count of word } t \text{ in doc } d}{\text{Total words in doc } d} $$
2.  **Inverse Document Frequency (IDF)**:
    $$ IDF(t) = \log \left( \frac{\text{Total number of documents}}{\text{Number of documents with word } t} \right) $$
3.  **Final Score**:
    $$ \text{TF-IDF} = TF \times IDF $$

#### Numerical Example
Imagine two documents:
*   **Doc 1**: "The blue cat"
*   **Doc 2**: "The red cat"

**Score for "blue" in Doc 1**:
1.  **TF**: 1 "blue" / 3 words = $0.33$
2.  **IDF**: $\log(2 \text{ docs} / 1 \text{ doc with blue}) = \log(2) \approx 0.3$
3.  **Result**: $0.33 \times 0.3 = \mathbf{0.099}$

**Score for "The" in Doc 1**:
1.  **TF**: 1 "The" / 3 words = $0.33$
2.  **IDF**: $\log(2 \text{ docs} / 2 \text{ docs with 'the'}) = \log(1) = \mathbf{0}$
3.  **Result**: $0.33 \times 0 = \mathbf{0}$
*The model learns that "The" is useless, but "blue" is a key feature.*

### C. N-Grams: Capturing Context
**Bag of Words** ignores word order. **N-Grams** capture sequences of words.

*   **Unigram**: Single words ("I", "love", "AI")
*   **Bigram**: Pairs of words ("I love", "love AI")
*   **Trigram**: Triplets ("I love AI")

**Example**:
*   Sentence: "I love machine learning"
*   Bigrams: ["I love", "love machine", "machine learning"]

**Use Case**: "New York" is a bigram. If you split it into unigrams ("New", "York"), you lose the meaning.

**Trade-off**: Higher n-grams capture more context but create exponentially more features (curse of dimensionality).

---

## 4. Embeddings: Maps of Meaning
Modern NLP uses **Word Embeddings** (like Word2Vec). Instead of just counting words, we map them into a multi-dimensional space.

*   **Intuition**: Similar words are stored "near" each other in the machine's brain.
*   **The Magic**: You can perform math on meanings:
    $$ \text{King} - \text{Man} + \text{Woman} \approx \text{Queen} $$

### Types of Embeddings
1.  **Word2Vec**: Learns embeddings by predicting context words (CBOW) or target words (Skip-gram).
2.  **GloVe**: Learns embeddings from word co-occurrence statistics.
3.  **FastText**: Extends Word2Vec by learning embeddings for character n-grams (handles misspellings better).
4.  **Contextual Embeddings (BERT, GPT)**: Modern embeddings that change based on context. "Bank" in "river bank" vs "money bank" gets different embeddings.

**Practical Tip**: For most tasks, use **pre-trained embeddings** (e.g., GloVe, Word2Vec trained on Wikipedia) instead of training from scratch.

---

## 5. Tips and Tricks for NLP
1.  **Always lowercase**: Unless case matters (e.g., "US" vs "us"), convert everything to lowercase to reduce vocabulary size.
2.  **Remove or keep stop words strategically**: For topic modeling, remove them. For sentiment analysis, keep "not", "no", "never".
3.  **Stemming vs Lemmatization**: Use stemming for speed (e.g., search engines). Use lemmatization for accuracy (e.g., sentiment analysis).
4.  **Handle imbalanced classes**: In sentiment analysis, if 90% of reviews are positive, use SMOTE or class weights.
5.  **Use TF-IDF for traditional ML**: For Logistic Regression, SVM, or Naive Bayes, TF-IDF works great.
6.  **Use embeddings for deep learning**: For LSTMs, Transformers, or neural networks, use Word2Vec or BERT embeddings.
7.  **Limit vocabulary size**: Keep only the top 10,000 most frequent words to reduce dimensionality.
8.  **Use n-grams for phrases**: Bigrams and trigrams capture phrases like "New York" or "not good".
9.  **Beware of data leakage**: Fit TF-IDF vectorizers on training data only, then transform test data.
10. **Visualize embeddings**: Use t-SNE or PCA to plot word embeddings in 2D and see if similar words cluster together.

---

## 6. Real-World Example: Sentiment Analysis Pipeline
A company wants to classify customer reviews as Positive or Negative.

### Step 1: Data Collection
*   10,000 reviews from Amazon.
*   Labels: 60% Positive, 40% Negative.

### Step 2: Preprocessing
1.  **Lowercase**: "Great Product!" → "great product!"
2.  **Remove URLs and mentions**: "Check out https://example.com @user" → "check out"
3.  **Tokenization**: "great product!" → ["great", "product", "!"]
4.  **Remove punctuation**: ["great", "product"]
5.  **Remove stop words**: Keep "not" for sentiment.
6.  **Lemmatization**: ["great", "product"]

### Step 3: Vectorization
*   Use **TF-IDF** with bigrams to capture phrases like "not good".
*   Limit vocabulary to top 5,000 words.

### Step 4: Handle Imbalance
*   Use **class weights** in Logistic Regression to give more importance to Negative reviews.

### Step 5: Train Model
*   Use **Logistic Regression** (fast and interpretable).
*   Accuracy: 85%, F1 Score: 0.83.

### Step 6: Interpret Results
*   Top positive words: "excellent", "love", "amazing".
*   Top negative words: "terrible", "waste", "disappointed".

### Step 7: Deploy
*   Save the TF-IDF vectorizer and model.
*   Apply the same preprocessing pipeline to new reviews.

---

## 7. Advanced NLP Concepts (Preview)
For deeper NLP tasks, you'll need:
*   **Sequence Models**: LSTMs, GRUs for tasks like machine translation.
*   **Attention Mechanisms**: Focus on important words (covered in Chapter 7).
*   **Transformers**: BERT, GPT for state-of-the-art performance (covered in Chapter 7).
*   **Named Entity Recognition (NER)**: Extracting names, dates, locations from text.
*   **Topic Modeling**: LDA for discovering hidden topics in documents (covered in Chapter 7).

---
