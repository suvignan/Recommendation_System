# Learning plan: redo everything myself (after the project ends)

**When:** after the project is finished (after Day 16+ deployment), not before.
**Where:** a new branch `learning/redo`, **never merged into `main`**.
**Data:** a sample of the same H&M data: **10-20% of customers** (all their transactions), so every step runs in minutes.
**Rule:** I write **every line myself**. Claude only explains the concept, the names and a toy example, then reviews my
code and explains my bugs. Fresher style: simple loops and pandas, one step per line, comments in my own words.
**Goal:** be able to explain every line to an interviewer, and honestly say "I wrote this".

Tick each step when I can do it **without looking** at the original notebook.

---

## Part 1: the notebooks

### 01 Load and window (Day 1)
- [ ] Look at the columns only (pyarrow schema, no full load)
- [ ] Decide which columns to keep, and why
- [ ] Load transactions normally, measure the memory
- [ ] Make the dtypes smaller (int32, float32, int8), measure again
- [ ] Basic counts (rows, customers, articles, dates)
- [ ] Look at the most recent weeks; compare a few window sizes
- [ ] Choose the window (12 weeks) and write why

### 02 EDA (Day 2)
- [ ] Purchases per customer, per article, per day
- [ ] Missing values in customers and articles
- [ ] Duplicates: what counts as one (same day + customer + article)

### 03 Split (Days 3-4)
- [ ] Week number = whole weeks since the start date
- [ ] Weekly purchase chart
- [ ] Time-based split: train weeks 0-9, validation week 10, test week 11
- [ ] The answer key: one row per validation customer with what they bought
- [ ] Cold customers and cold articles in validation
- [ ] Have customers already bought these articles? (the repurchase idea)
- [ ] Save the splits, reload and prove they're right

### 04 Features (Day 5)
- [ ] Missing values in customer columns, and what missing means
- [ ] Vocabularies: value -> number, 0 = unknown
- [ ] Age buckets with pd.cut
- [ ] Which feature goes in which tower

### 05 Baselines and ALS (Days 6-8)
- [ ] Recall@10 and NDCG@10 on paper (the 0.6667 / 0.6257 example)
- [ ] Write `recall_at_k` and `ndcg_at_k`, check them on the paper example
- [ ] Global top-10 and last-week top-10; score both
- [ ] Warm vs cold scores
- [ ] Build the sparse matrix (sorted mappings, row 0 empty); the single-cell check
- [ ] Train ALS; look inside the model with one customer
- [ ] Score ALS with a fallback; filter or no filter?
- [ ] Repurchase baseline: own purchases, newest first, then fill

### 06 Two-tower (Days 8-10): the hard one, go slowly
- [ ] Customer features as ONE function
- [ ] Article features and price scaling
- [ ] The training pairs table, and its checks
- [ ] Batches with tf.data
- [ ] The towers; parameter counts; one batch through both
- [ ] In-batch softmax: "predicted minus true"
- [ ] Duplicates in a batch (false negatives)
- [ ] **The logQ correction** (weak spot: redo on paper first)
- [ ] My own train_step
- [ ] **Overfit one small batch** (weak spot: what it proves)
- [ ] Validation recall@10; time one epoch; the full run; save and reload
- [ ] ID dropout for cold start; why row 0 never learns

### 07 Blend and cold path (Day 10)
- [ ] Parity checks: prove this notebook matches 05 and 06
- [ ] Blend: own purchases first, then a filler (popularity / ALS / two-tower)
- [ ] Cold path: last-week popularity per age bucket, filled from the global list
- [ ] Freeze the final system (before the test week)

### 08 Test day (Day 11)
- [ ] Write the test-day plan before opening anything
- [ ] Checksums of every frozen file
- [ ] The inputs as functions of the history (weeks 0-9 vs 0-10)
- [ ] Retrain ALS on weeks 0-10 with the frozen settings
- [ ] Generate all test lists once, then score; the results table
- [ ] Serving cost: real-time vs precomputed

### 09 Tie rule (Day 14)
- [ ] Why ties in file order are a problem
- [ ] The explicit rule (smaller article_id first); measure what changed; re-validate

---

## Part 2: the pipeline (only if time permits)

Same order, one stage at a time, each with its config, artifact and one test:
- [ ] Logger, exception, constants, `read_yaml`, `schema.yaml` vs `params.yaml`
- [ ] Ingestion (window, ids, dedup, week) and validation (11 checks, report first, then fail)
- [ ] Splitter; list builder; model trainer (vectors must match the notebook)
- [ ] Evaluation + gate (parity, lift over popularity, cold rule) + the warning
- [ ] Recommendation table + serving files (`.npy` + manifest)
- [ ] Each stage from the command line; `dvc.yaml`; predict what reruns
- [ ] The fixture, the smoke test, CI
- [ ] The API (`app/main.py`) and its tests
