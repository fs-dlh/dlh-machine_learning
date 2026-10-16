# NLP Cheat Sheet — Attention, Transformers, BERT, GPT & More

*A one-page, plain-English reference. Formulas kept to one line. Diagrams are ASCII.*

---

## 1. What is the attention mechanism?

**Idea:** While processing one word, the model looks at *all* other words and focuses on the ones that matter.

**Example**
- "Ball" is related to "blue" and "holding", not to "boy".
- "The cat drank the milk because **it** was hungry." → *it* = cat
- "The cat drank the milk because **it** was sweet."   → *it* = milk

Attention lets the model link *it* to the right word, even if words are far apart.

**The formula (one line):**
$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V
$$
- **Q** (Query): "what I'm looking for" (the current word)
- **K** (Key): "what I contain" (every word in the sentence)
- **V** (Value): "what I contribute" (the meaning passed on)
- softmax turns scores into weights that sum to 1.

**Multi-head attention:** several attention "spotlights" at once — one watches grammar, another watches meaning, etc.

```
Word:  it  ─────────────┐
                       ├─► head 1 → "cat"
Sentence: The cat drank the milk because it was hungry.
                       ├─► head 2 → "hungry"
```

---

## 2. How to apply attention to RNNs

**Problem with RNNs:** They read word-by-word (sequentially). To predict word *t*, they must finish word *t−1*. Slow, and they forget distant words.

**Fix:** Add an attention layer on top of the RNN's hidden states. Instead of passing only the last hidden state to the decoder, the decoder looks back at *all* encoder hidden states and weights which ones are useful for each output word.

```
RNN encoder:  h1  h2  h3  h4  h5   (all hidden states)
                         ▲
Decoder output word ────┴─ attention weights decide which hi to use
```

Result: faster, better long-range memory. (Transformers later replaced RNNs entirely with self-attention.)

---

## 3. What is a Transformer?

A neural network for text that uses **only attention** — no recurrence, no convolutions. Introduced in *"Attention Is All You Need"* (2017). Reads the whole sentence in parallel (fast) and connects any two words regardless of distance.

**Two stacks:**
- **Encoder:** understands the input (e.g., English sentence).
- **Decoder:** produces the output (e.g., French translation).

**Why it's better than RNNs:** parallel training, no vanishing long-range memory, context-aware word meanings.

```
Transformer = Embedding + Positional Encoding + Encoder Stack + Decoder Stack + Output
```

---

## 4. How to create an encoder-decoder Transformer model

**Steps**
1. **Tokenize** the input sentence into words/sub-words.
2. **Embed** each token into a vector; add **positional encoding** (so order isn't lost).
3. **Encoder stack** (repeat N times): Multi-head self-attention → Add & Norm → Feed-forward → Add & Norm.
4. **Decoder stack** (repeat N times): Masked multi-head self-attention → Encoder-Decoder attention → Add & Norm → Feed-forward → Add & Norm.
   - *Masked* = decoder can't peek at future words.
   - *Encoder-Decoder* attention = decoder queries the encoder's Keys/Values.
5. **Output layer:** Linear → Softmax over vocabulary to pick the next word.
6. **Train** with teacher forcing (feed the correct previous word, not the model's guess).

```
Input  → Embed + PosEnc → [Encoder ×N] ──────────────┐
                                                        ├─► Enc-Dec Attn ─┐
Output → Embed + PosEnc → [Decoder ×N] ─► Linear → Softmax ─► Prediction
```

---

## 5. What is GPT?

**GPT** = Generative Pre-trained Transformer (OpenAI). It is **decoder-only**.

- Trained to predict the **next word** given the previous words (autoregressive, left-to-right).
- Pre-trained on huge text, then fine-tuned for tasks.
- Good at: text generation, completion, chat.

```
GPT = Decoder stack only, causal (masked) self-attention, next-token prediction
```

---

## 6. What is BERT?

**BERT** = Bidirectional Encoder Representations from Transformers (Google). It is **encoder-only**.

- Trained **bidirectionally** (sees left and right context at once).
- Two self-supervised pre-training tasks:
  1. **Masked Language Modeling (MLM):** hide some words, predict them.
  2. **Next Sentence Prediction (NSP):** does sentence B follow sentence A?
- Good at: understanding, classification, question answering.

```
BERT = Encoder stack only, bidirectional, MLM + NSP pre-training
```

---

## 7. What is self-supervised learning?

**Idea:** Create labels *from the data itself* (no expensive human labeling), then train like a supervised task. The pretext task is usually a means to learn good representations.

**Examples**
- Rotate an image randomly → train model to predict the rotation.
- Hide words in a sentence → train model to predict them (BERT's MLM).
- Predict the next word (language modeling).

**Why:** unlabeled data is abundant; labels are expensive. Learn useful features from unlabeled data, then fine-tune with a little labeled data.

```
Self-supervised = make up a task from the data → learn representations → use downstream
```

---

## 8. How to use BERT for specific NLP tasks

**Fine-tuning:** take pre-trained BERT and add a small task-specific head on top, then train on your labeled data.

| Task | What BERT does | Head added |
|------|----------------|------------|
| Sentiment / classification | Read sentence, output label | Linear layer → class |
| Named Entity Recognition | Tag each word (person, place…) | Linear per token |
| Question Answering (SQuAD) | Find answer span in passage | Start + end position |
| Textual entailment | Pair of sentences → entail/contradict/neutral | Linear → 3 classes |

**Recipe:** Load BERT → add head → fine-tune on task data → infer.

---

## 9. What is SQuAD? GLUE?

**SQuAD** (Stanford Question Answering Dataset): A reading-comprehension benchmark. Given a passage and a question, predict the span of text in the passage that answers the question. Used to train/evaluate BERT-style QA models.

**GLUE** (General Language Understanding Evaluation): A benchmark of **9 diverse NLP tasks** (sentiment, entailment, paraphrase, etc.) used to compare models like BERT. One score summarizes general language understanding.

```
SQuAD = question answering (span prediction)
GLUE  = 9-task general language understanding leaderboard
```

---

## Quick comparison

| Model | Parts | Direction | Best for |
|-------|-------|-----------|----------|
| RNN (+attention) | Recurrent layers | Sequential | Sequences (older) |
| Transformer | Encoder + Decoder | Bidirectional (enc) / causal (dec) | Translation, seq-to-seq |
| BERT | Encoder only | Bidirectional | Understanding, classification |
| GPT | Decoder only | Left-to-right (causal) | Generation, chat |

---

*Tip: Attention = "focus on related words." Transformer = "attention only, no recurrence." BERT = "encoder, understand." GPT = "decoder, generate."*
