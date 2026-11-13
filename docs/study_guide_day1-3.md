# Study guide: Days 1–3 (H&M recommender)

How to use this: read one day, then close it and try the **self-quiz** at the end of that day out loud. If you can't answer, re-read only that part.

---

## The big picture (read first)

We are building a system that answers: **"Which 10 articles will this customer buy next week?"**

| Day | Notebook | One-line purpose |
|---|---|---|
| 1 | `01_load_and_window.ipynb` | Load the data cheaply and keep only the recent 12 weeks |
| 2 | `02_eda.ipynb` | Describe how unusual recommendation data is, with numbers |
| 3 | `03_split.ipynb` | Clean, split by time into train / validation / test, build the answer key |

The data has 3 files:
- **transactions**: who bought what, when (one row = one item bought)
- **customers**: one row per customer (age, club status…)
- **articles**: one row per product (type, colour, department…)

---

# Day 1: Load and choose the window

## 1. Look at columns before loading
`pq.read_schema(path)` reads only the column names and types, not the rows, so it's instant even for 31 million rows.
**Why:** decide what you need *before* you spend memory.

## 2. Column decisions

| File | Kept | Dropped, and why |
|---|---|---|
| Transactions | all 5: `t_dat, customer_id, article_id, price, sales_channel_id` | nothing; price and channel are cheap (4 bytes, 1 byte) |
| Customers | `customer_id, age, club_member_status, fashion_news_frequency, FN, Active` | `postal_code`: 352,899 distinct values; with age it can identify a person, a GDPR risk |
| Articles | `article_id, product_code`, category name columns | `*_no` columns (duplicates of `*_name`), `prod_name` (≈ product_code), `detail_desc` (free text = NLP project, out of V1) |

**Trap the mentor pointed out:** `price` and `sales_channel_id` describe a **transaction**, not an **article**. When you recommend something nobody has bought yet, there is no "price paid". To use price as a feature, turn it into something known in advance, like **the article's median price in the training weeks**.

**Why keep `product_code`:** many articles share one product code (same T-shirt, different colours). Later you'll check that the model doesn't recommend 5 colours of one T-shirt.

## 3. Making the data small (dtypes)

| Column | Before | After | Why |
|---|---|---|---|
| `customer_id` | 64-character text | `customer_join_idx` int32 | text × 31M rows is huge; a number is 4 bytes |
| `article_id` | int64 | int32 | max ≈ 959M fits in int32 (max ≈ 2.1 billion) |
| `price` | float64 | float32 | half the memory, enough precision |
| `sales_channel_id` | int64 | int8 | only values 1 and 2 |
| `t_dat` | text | datetime | the time split depends on real dates |

**Memory: 3,456 MB → 637 MB (5.4× smaller).**

### How the customer number was made
```python
customers['customer_join_idx'] = range(len(customers))      # 0, 1, 2, ...
id_to_number = dict(zip(customers['customer_id'], customers['customer_join_idx']))
transactions['customer_id'] = transactions['customer_id'].map(id_to_number)
```
- `dict(zip(a, b))` builds a lookup: `{'00000dba…': 0, '0000423b…': 1, …}`
- `.map(dict)` replaces every text id with its number
- a customer not in the dictionary becomes **NaN**, so `isna().sum() == 0` proves every transaction customer exists in the customers file

### ⭐ Panel question: "Why int32 and not `category`?"
> A `category` column numbers its values **separately in each table**, so the same customer could be code 5 in transactions and 912 in customers, and a join would silently match the wrong people. I built **one** int32 index from the customers file, so every table uses the same number for the same customer. int32 instead of int64 because 1.4M customers fits easily and it halves memory.

### ⭐ Join key ≠ model vocabulary
`customer_join_idx` is **only for joining tables**. On Day 4 the model gets its **own** customer mapping, built from **training weeks only**, with an "unknown" bucket. If the model's mapping included customers who only appear in validation/test, that's **leakage** (they exist only because of future data).

## 4. Counts
- Dates: **2018-09-20 → 2020-09-22** (2 years)
- **31,788,324** rows · **1,362,281** customers · **104,547** articles
- Every transaction article exists in the articles file; every transaction customer exists in the customers file

## 5. The window: last 12 weeks (2020-07-01 → 2020-09-22)
- **Split room:** 10 train weeks + 1 validation + 1 test
- **Season:** summer → early autumn, same as the test week
- **Skips the clearance sale:** late-June weeks had ~1.8× normal volume
- **Fresh catalogue:** ~41k articles selling vs 105k in the catalogue
- **Hardware:** 3,386,963 rows, 68 MB, fast on a 16 GB laptop
- ⚠️ **It's a hypothesis:** on Day 9, compare 12 vs 16 weeks on validation

### ⭐ Interview: "Why train on only recent weeks, and what do you lose?"
> Fashion data is **non-stationary**: trends, stock and season change fast, so recent weeks look much more like the week we'll serve than purchases from two years ago. The cost is history: inactive customers become cold-start and we lose long-term taste and yearly seasonality. And window length is a hyperparameter I validate, not a guess.

## 6. DVC
`dvc add data/RAW` stores a **hash** of the raw files in a small pointer file `data/RAW.dvc`. Git tracks the pointer, DVC tracks the data. Panel line: *"my raw data had a hash from day one."*

### Day 1 self-quiz
1. Why read the schema before loading?
2. Why drop `postal_code`? Give the number.
3. Why int32 and not `category` for customers?
4. What's the difference between the join key and the model vocabulary?
5. Why can't transaction `price` be used as a feature directly?
6. Memory before and after?
7. Give 4 reasons for the 12-week window.

---

# Day 2: EDA, how unusual recommender data is

## 1. Purchases per customer and per article
```python
customer_count = transactions.groupby('customer_join_idx').size()
customer_count.quantile([0.5, 0.9, 0.95, 0.99])
```
- `groupby(...).size()` = rows per group = purchases per customer
- `.quantile(0.5)` = median ("half are at or below this")

| | 50th | 90th | 95th | 99th | Max |
|---|---|---|---|---|---|
| Customers | **4** | 15 | 21 | 39 | 316 |
| Articles | **12** | 226 | 412 | 970 | 5,673 |

**Why median, not mean:** one customer bought 316 items and pulls the mean (6.9) up. The median (4) describes the typical customer honestly.

**What it means:** the median customer bought only **4 items in 12 weeks**, too little to learn a good per-customer ID vector, so **side features** (age, club status) must help (Day 7).

## 2. Log-log popularity plot
1. Sort article counts biggest → smallest
2. Rank 1, 2, 3…
3. Plot rank vs count with **both axes on log scale**

**Why log:** on normal axes you see one spike and a flat line. On log axes each step is ×10, so big and small numbers both get space.

**What we saw:** roughly a **straight line for the top ~1,000 articles** (power-law-like), then a **steep drop** after ~rank 3,000.

**Why it matters:** popularity is very uneven, so the popularity baseline is strong, and **in-batch negatives over-sample popular items** (Day 8: logQ correction).

## 3. Sparsity and concentration
```
sparsity = 1 - unique (customer, article) pairs / (customers × articles)
         = 1 - 2,992,254 / (489,553 × 40,772) = 99.985%
```
- Drop duplicate pairs first: buying the same thing twice is still one cell.

**Top 1% of articles (407) = 18.3% of all purchases.**

### ⭐ The key idea: an empty cell is NOT a dislike
> A customer who didn't buy an article usually **never saw it**. That's **implicit feedback**: we only see what people did, never what they disliked. So "negatives" can't be read from the data; they must be **sampled** (negative sampling).

## 4. Same-day duplicates and repeats

**Same-day duplicates** (no quantity column, so 3 socks = 3 identical rows):
```python
is_duplicate = transactions.duplicated(subset=['t_dat', 'customer_join_idx', 'article_id'])
is_duplicate.sum()    # 301,088
is_duplicate.mean()   # 8.89%
```
- True counts as 1, so `.sum()` = how many and `.mean()` = what fraction
- **Decision:** count as **one** positive. Buying 2 is quantity, not stronger liking.

**Different-day repeats:** 2.95% of customer–article pairs were bought on 2+ days.

⚠️ **Mentor correction:** 2.95% is the **wrong number** for the Day 6 "filter already-bought items?" decision. The right question is: *of the articles bought in the validation week, what share had that customer already bought in training?* That's the situation the filter faces. In H&M this is known to be **high** ("recommend what they bought recently" was a strong trick in the Kaggle competition). **Still to compute** (see Open items).

### Bugs we hit (good lessons)
| Bug | Lesson |
|---|---|
| `.sum().mean()` gave 301,088 | chaining: `.mean()` ran on the single number from `.sum()`. Store in a variable, then call each separately |
| read 0.0295 as 0.029% | a fraction × 100 = percent. Use `f"{x:.2%}"` to let Python do it |
| plot showed customer histogram | answer the question asked: article **rank vs count** |
| table built but not shown | a notebook only shows the **last bare line** of a cell |

### ⭐ Interview: "What does a long tail do to a model, and what do you do about it?"
> The model sees popular items thousands of times and tail items a handful, so it learns that bestsellers are always safe and recommends the same few to everyone (popularity bias). In my data 1% of articles are 18% of purchases. I'd use side features so rare items borrow signal from similar ones, a logQ correction for how often popular items are sampled as negatives, and track coverage alongside recall.

### Day 2 self-quiz
1. Median customer purchases? Why median, not mean?
2. Why plot on log-log? What shape did you see?
3. Sparsity formula and value? Why drop duplicate pairs first?
4. Why is an empty cell not a dislike? What's that called?
5. Top-1% share?
6. Duplicate rate, and your decision?
7. Why is 2.95% the wrong number for the filter decision?

---

# Day 3: Time-based split

## 1. Which cleaning is safe before the split?

| Safe before split (row-local) | NOT safe (statistics over many rows) |
|---|---|
| fixing dtypes | article popularity |
| dropping same-day duplicates | median price |
| removing broken rows | filling missing age with the average |
| | building ID vocabularies |

**Rule:** looks at one row (or identical-key rows) → safe now. Computes a number from many rows → do it **after** the split, **on train only**.

### ⭐ Two-sentence answer
> Dropping same-day duplicates is **row-local**: each row's fate depends only on rows with the same customer, article and date, so nothing from future weeks reaches training. Article popularity is counted over many rows, so computing it before the split would include validation and test purchases, which is **leakage**.

## 2. Dedup and assertions
```python
clean = transactions.drop_duplicates(subset=['t_dat', 'customer_join_idx', 'article_id'], keep='first')
```
- Price and channel are **not** in the rule: one choice at two prices is still one choice.
- Expected 3,386,963 − 301,088 = **3,085,875**, and got exactly that.

**Assertions, not prints:** a print needs you to notice; an assert **stops** the notebook. Always add a message saying what was expected:
```python
assert rows_after == 3_085_875, f'expected 3,085,875 rows, got {rows_after:,}'
```
Checks written: dtypes (first!), row count, first/last date, Wednesday/Tuesday, no missing keys, no duplicates left. These become the **DataValidation** step on Day 11.

**Dtype trap:** `t_dat` is `datetime64[us]`, so checking `== 'datetime64[ns]'` fails. Use `pd.api.types.is_datetime64_any_dtype(...)`.

## 3. Week number (your own definition)
```python
clean['week'] = (clean['t_dat'] - start_date).dt.days // 7
```
- date − start = a time gap; `.dt.days` turns it into a number; `// 7` = whole weeks
- weeks **0–9 = train, 10 = validation, 11 = test**
- **Why not pandas' default week?** It ends on Sunday, so every bucket would mix two of your Wed–Tue weeks, with no error.

## 4. Weekly table and chart
```python
weekly = clean.groupby('week').agg(
    start_date=('t_dat', 'min'),
    purchases=('article_id', 'size'),
    active_customers=('customer_join_idx', 'nunique'),
    active_articles=('article_id', 'nunique'),
)
```
Pattern: `new_name=('column', 'what to do')`.

**Conservation checks:** weekly purchases sum to 3,085,875 ✅; every week has exactly 7 dates ✅.

**What the chart showed:**
- Weeks 10–11 are the quietest but follow a gentle downward trend, not a jump.
- Week 9 is normal (no spike), so the "last week's top 10" baseline is fair.
- Active articles fall 21.5k → 18k, so the catalogue is turning over (cold-start).
- If the test week were a sale, the **popularity baseline** would benefit most (everyone buys the same discounted items).

## 5. Split
```python
train = clean[clean['week'] <= 9].copy()
val   = clean[clean['week'] == 10].copy()
test  = clean[clean['week'] == 11].copy()
```
| Table | Rows | Dates |
|---|---|---|
| Train | 2,638,103 | 2020-07-01 → 2020-09-08 |
| Val | 230,980 | 2020-09-09 → 2020-09-15 |
| Test | 216,792 | 2020-09-16 → 2020-09-22 |

- `=` stores, `==` compares
- Sum check = 3,085,875 (nothing lost or counted twice)
- Date check: train ends **before** val starts (no time overlap = no leakage)
- **Test is opened only once, at the very end.**

## 6. Answer key
```python
val_answers = val.groupby('customer_join_idx')['article_id'].unique()
```
- One row per customer = the **set** of articles they bought in week 10
- **72,019** customers (matches the weekly table)
- Repeats within the week count **once**: one recommendation slot can only be one hit
- Average **3.16** articles per customer (estimate rows/customers = 3.21; lower because 3,070 repeats were removed)

**Why it matters:** with ~3 purchases, even a perfect top-10 list gets ~3 hits, so **recall@10 always looks "low"**, and that's normal.

### ⭐ Interview: "What is the correct answer for one customer, and why a group?"
> It's the set of articles the customer actually bought in the evaluation week. It's a group because people buy several items a week (about 3 here) and the model returns a ranked list, so we score how many recommended items fall in that set, unlike fraud where each row has one label.

### Day 3 self-quiz
1. Row-local vs statistics: give 2 examples of each.
2. Why aren't price and channel in the duplicate rule?
3. Why assertions instead of prints? Why check dtypes first?
4. How is the week number calculated? Why not pandas' default week?
5. Train/val/test row counts and dates?
6. What is the answer key? Why count repeats once?
7. Why does recall@10 look low in recommenders?

---

# Open items (not done yet)
1. **"Already bought before" rate:** of validation-week purchases, what share had that customer already bought in train? Decides the Day 6 filter. Tool: `val_pairs.merge(train_pairs, how='left', indicator=True)`, then `(result['_merge'] == 'both').mean()`.
2. **Day 4:** build the **model** customer vocabulary from train weeks only, with an unknown bucket. Name it differently from `customer_join_idx`.
3. **Git:** commit Days 1–3; re-record `Data/RAW.dvc` as `data/RAW.dvc` (case rename).

# The 10 numbers to know by heart
| | |
|---|---|
| Full data | 31.8M rows, 1.36M customers, 105k articles, 2 years |
| Memory | 3,456 → 637 MB |
| 12-week window | 3.39M rows → 3.09M after dedup |
| Median customer | 4 purchases in 12 weeks |
| Sparsity | 99.985% |
| Top 1% articles | 18.3% of purchases |
| Same-day duplicates | 8.89% |
| Train / val / test | 2.64M / 231k / 217k rows |
| Val customers | 72,019 |
| Articles per val customer | 3.16 |
