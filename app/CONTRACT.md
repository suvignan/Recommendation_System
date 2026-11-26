# Recommendation API: contract

Written **before** the code (Day 16, Step 2). The API only **looks up** lists the weekly pipeline already built and
checked (`artifacts/recommendation_table/serving/`). It never trains, scores or reads the raw data.

---

## `GET /recommend`

### Request

| Parameter | Required | Type | Meaning |
|---|---|---|---|
| `customer_id` | yes | string | H&M's customer id: **exactly 64 hexadecimal characters** (`0-9`, `a-f`; upper case is accepted and lower-cased) |
| `age` | no | number | the customer's age, used **only** when the customer has no stored age (see the rules) |

### Response `200`

```json
{
  "customer_id": "00000dbacae5abe5e23885899a1fa44253a17956c6d1c3d25f88aa139fdfc657",
  "source": "personal",
  "articles": [568601043, 859416011, 795440001, 751471001, 448509014, 762846027, 806388001, 924243001, 918522001, 909370001],
  "table_version": "3f9a1c0b7d2e"
}
```

- `articles`: always **exactly 10 distinct article ids**, best first.
- `source`: **where the list came from**, so "why did this customer get these?" is answered in the response itself:
  `personal` (their own history), `age_bucket` (popular last week with their age group), or `global` (popular last week overall).
- `table_version`: the first 12 characters of the serving manifest's table checksum: which weekly build answered.
- Header **`X-Lookup-Ms`**: the time the lookup took inside the app (milliseconds, 3 decimals).

### The `source` rule (checked in this order)

1. **Personal:** the customer has a purchase history in the table → their own list → `personal`.
2. **Known, no history:** the customer is in the customer file but never bought anything →
   their **stored** age's bucket list → `age_bucket`.
   If their stored age is missing or out of range (bucket 0), use the `age` parameter if it gives a bucket.
3. **Unknown customer, valid `age`:** that age's bucket list → `age_bucket`.
4. **Otherwise** → the global list → `global`.

The **bucket 0 list is the global list**, so whenever the bucket turns out to be 0 the response says `global`, never
`age_bucket`: `source` describes the list actually served, not the path taken to find it.

Ages are turned into buckets by **the same function as the pipeline** (`age_to_bucket`), with the same edges
(`params.yaml` → `features.age_edges`: 16, 20, 25, 30, 40, 50, 60, 100; each bucket is `[left, right)`).

### Decisions on bad input

| Input | Result | Why |
|---|---|---|
| `customer_id` missing | **422** | required parameter |
| `customer_id` not 64 hex characters (wrong length, `g`, spaces...) | **422** `{"detail": "customer_id must be 64 hexadecimal characters"}` | Every real H&M id has this shape, so a malformed one is a **bug in the caller** (a truncated id, the wrong field). Quietly answering with the global list would hide that bug forever while the customer silently gets worse recommendations. A clear error gets it fixed. A frontend that wants a fallback can catch the 422 and show its own default. |
| well-formed `customer_id` we've never seen | **200**, rules 3-4 | A new customer is normal, not an error: they get the best list we have. |
| `age` not a number | **422** | FastAPI validates the type |
| `age < 0` or `age > 120` | **422** `{"detail": "age must be between 0 and 120"}` | Impossible ages are a caller bug (e.g. a birth year sent as an age). |
| `age` 0-15 or 100-120 (e.g. `age=7`) | **200**, treated as **no usable age** → `global` (or the stored age, rule 2) | Possible ages, but outside the buckets the model was built on. The pipeline does exactly the same: `age_to_bucket` puts them in bucket 0, so the API and the pipeline never disagree. |
| `age=150` | **422** | impossible (> 120) |

### Response `503` (not ready)

If the serving files didn't load or **any checksum didn't match**, `/recommend` refuses to answer:
`503 {"detail": "not ready: <reason>"}`. It never serves a half-written, swapped or corrupted file.

---

## `POST /recommend` (the form)

The same rules and the same answer as `GET /recommend`, for a form where a visitor fills in what they know.
The body is JSON; **both fields are optional**:

```json
{"customer_id": "00000dbacae5...c657", "age": 25}
```

| Body | Result |
|---|---|
| `{"customer_id": "..."}` | exactly what `GET /recommend?customer_id=...` returns |
| `{"age": 25}` (a new visitor) | their age group's list -> `age_bucket` (`customer_id` is `null` in the response) |
| `{}` (an empty form) | the global list -> `global` |
| a malformed id, `age` < 0 or > 120, `age` not a number | **422**, the same decisions as GET |

**Why both GET and POST:** GET is for quick testing (paste a URL in a browser). POST is for a real form: the id
travels in the **body**, so it doesn't end up in URLs and access logs, and a new field later is one more line in
the body, not a new URL. Both call the same `serve()` function, so they can never disagree.
A visitor without an id is logged as the hash of `"anonymous"`.

---

## `GET /health`

**Liveness:** `200 {"status": "alive"}` as long as the process is running, even when not ready. It does no work, so
it always answers fast. (A failing health check means "restart me".)

## `GET /ready`

**Readiness:** `200 {"status": "ready", "table_version": "..."}` only if **every** serving file loaded **and every
SHA-256 matched** `serving_manifest.json`. Otherwise `503 {"status": "not ready", "reason": "..."}`.
Render (or any load balancer) sends traffic only while this passes.

## `GET /version`

`200` with the serving manifest: `table_version` (full checksum), per-file `sha256`, `git_commit`,
`history_end_week`, `built_at`. Answers even when not ready, so you can see **which** build failed its checks.

---

## The serving files (written by the pipeline, read by the API)

All in `artifacts/recommendation_table/serving/` (written by the table stage, next to its Parquet files):

| File | dtype, shape | Meaning |
|---|---|---|
| `recs.npy` | int32, (customers with history + 1, 10) | row *r* = one customer's list; **row 0 unused** (all zeros) |
| `bucket_lists.npy` | int32, (8, 10) | row *b* = age bucket *b*'s list; row 0 = the global list |
| `lookup_ids.npy` | uint64, (all known customers,) | 64-bit customer ids, **strictly increasing** (sorted, no duplicates) |
| `lookup_rec_row.npy` | int32, same length | that customer's row in `recs.npy`; **0 = no history** |
| `lookup_bucket.npy` | int8, same length | the stored age bucket (0 = unknown) |
| `serving_manifest.json` | JSON | `sha256` of every file above, `table_version`, `git_commit`, `history_end_week`, `built_at` |

**The 64-bit id:** the last 16 hex characters of `customer_id`, read as an unsigned 64-bit number. One function
(`recsys.utils.shared.customer_id_to_uint64`) does this for **both** the pipeline and the API. The pipeline checks the
sorted ids are strictly increasing, which also proves there are no collisions.

`table_version` = SHA-256 of the per-file checksums (sorted by file name, joined), so it changes if **any** file changes.

**Why a separate "rec row", not the pipeline's `customer_join_idx`:** two reasons.
1. `customer_join_idx` **0 is a real customer** (file order, Day 1), so "0 = no history" would make that customer
   invisible.
2. Indexing `recs.npy` by `customer_join_idx` would need a row for all 1,371,980 customers (about 55 MB, mostly empty).
   A dense row number for just the 489,553 customers with history is 489,554 x 10 x 4 bytes = **about 20 MB**.

So the rec rows are 1..N in the order of `customer_join_idx`, and row 0 = "no history". The join index itself never
reaches the API: it only needs to know which row to read.
