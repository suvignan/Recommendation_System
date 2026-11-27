"""The recommendation API: looks up the lists the weekly pipeline built. Rules: app/CONTRACT.md."""
import hashlib
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import BackgroundTasks, Body, FastAPI, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.request_log import RequestLogger
from recsys.utils.shared import age_bucket_for, customer_id_to_uint64, sha256_of

PROJECT = Path(__file__).resolve().parents[1]                 # app/main.py -> the project folder
DEFAULT_FOLDER = PROJECT / "artifacts" / "recommendation_table" / "serving"
SERVING_FOLDER = Path(os.environ.get("SERVING_DIR", DEFAULT_FOLDER))   # tests point this at the fixture's files

FILES = ["recs.npy", "bucket_lists.npy", "lookup_ids.npy", "lookup_rec_row.npy", "lookup_bucket.npy",
         "article_names.json"]
VALID_ID = re.compile(r"^[0-9a-f]{64}$")                       # exactly 64 characters, each 0-9 or a-f

logger = logging.getLogger("uvicorn.error")                    # uvicorn's own logger, so lines show in its output

class ServingState:
    def __init__(self):
        self.ready = False
        self.reason = "not loaded yet"
        self.manifest = None
        self.recs = None
        self.bucket_lists = None
        self.lookup_ids = None
        self.lookup_rec_row = None
        self.lookup_bucket = None
        self.article_names = None

    def load(self,folder):
        manifest_path = folder / "serving_manifest.json"
        if not manifest_path.exists():
            self.reason = f"no serving_manifest.json in {folder}"
            return
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        for name in FILES:
            path = folder/name 
            if not path.exists():
                self.reason = f"missing {name} in {folder}"
                return
            
            if sha256_of(path) != self.manifest['sha256'][name]:
                self.reason = f"checksum mismatch: {name} (half-written, swapped or corrupted)"
                return
        
        self.recs = np.load(folder/"recs.npy")
        self.bucket_lists = np.load(folder/"bucket_lists.npy")
        self.lookup_ids = np.load(folder/"lookup_ids.npy")
        self.lookup_rec_row = np.load(folder/"lookup_rec_row.npy")
        self.lookup_bucket = np.load(folder/"lookup_bucket.npy")
        self.article_names = json.loads((folder/"article_names.json").read_text(encoding="utf-8"))
        self.ready = True
        self.reason =""
state = ServingState()
request_log = RequestLogger.from_environment()       # off unless RECSYS_DATABASE_URL + RECSYS_HASH_KEY are set

@asynccontextmanager
async def lifespan(app):
    state.load(SERVING_FOLDER)                                 # runs ONCE when the server starts
    if state.ready:
        megabytes = (state.recs.nbytes + state.bucket_lists.nbytes + state.lookup_ids.nbytes
                     + state.lookup_rec_row.nbytes + state.lookup_bucket.nbytes) / 1024 ** 2
        logger.info(f"Serving table {state.manifest['table_version'][:12]} "
                    f"(history to week {state.manifest['history_end_week']}), {megabytes:.1f} MB loaded")
    else:
        logger.error(f"NOT READY: {state.reason}")
    request_log.setup()                                        # create our log table if needed (or stay off)
    yield                                                      # the server answers requests while we're here


app = FastAPI(title="H&M recommendations", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "alive"}                                 # no work: answers even when not ready


@app.get("/ready")
def ready():
    if not state.ready:
        return JSONResponse(status_code=503, content={"status": "not ready", "reason": state.reason})
    return {"status": "ready", "table_version": state.manifest["table_version"][:12]}


@app.get("/version")
def version():
    if state.manifest is None:
        return JSONResponse(status_code=503, content={"status": "no manifest", "reason": state.reason})
    return state.manifest

def find_customer(customer_id):
    """The customer's position in the lookup arrays, or None if we've never seen them (or got no id)."""
    if customer_id is None:                                     # POST without an id: a new visitor
        return None
    number = np.uint64(customer_id_to_uint64(customer_id))
    position = int(np.searchsorted(state.lookup_ids, number))   # where it is, OR where it WOULD go
    if position < len(state.lookup_ids) and state.lookup_ids[position] == number:
        return position
    return None                                                 # not there: a new customer


def pick_list(customer_id, age):
    """The contract's source rules, in order. Returns (articles, source)."""
    edges = state.manifest["age_edges"]
    position = find_customer(customer_id)

    if position is not None:
        # Rule 1: they have a history -> their own list
        rec_row = int(state.lookup_rec_row[position])
        if rec_row > 0:
            return state.recs[rec_row].tolist(), "personal"
        # Rule 2: known, no history -> the bucket of their STORED age
        bucket = int(state.lookup_bucket[position])
        if bucket > 0:
            return state.bucket_lists[bucket].tolist(), "age_bucket"

    # Rule 3: the age parameter (unknown customer, or a known one with no stored age)
    if age is not None:
        bucket = age_bucket_for(age, edges)
        if bucket > 0:
            return state.bucket_lists[bucket].tolist(), "age_bucket"

    # Rule 4: nothing else -> the global list (bucket 0)
    return state.bucket_lists[0].tolist(), "global"

def serve(customer_id, age, response, background_tasks):
    """ONE copy of the work, used by both GET and POST /recommend. customer_id may be None (POST, new visitor)."""
    # 1. Check the input (422 = the caller sent something wrong)
    if customer_id is not None:
        customer_id = customer_id.lower()
        if not VALID_ID.match(customer_id):
            raise HTTPException(status_code=422, detail="customer_id must be 64 hexadecimal characters")
    if age is not None and (age < 0 or age > 120):
        raise HTTPException(status_code=422, detail="age must be between 0 and 120")

    # 2. Never serve unchecked files (503 = we can't answer right now)
    if not state.ready:
        raise HTTPException(status_code=503, detail=f"not ready: {state.reason}")

    # 3. The lookup itself, timed
    start = time.perf_counter()
    articles, source = pick_list(customer_id, age)
    lookup_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Lookup-Ms"] = f"{lookup_ms:.3f}"

    # 4. Log it to Neon AFTER the response is sent, so a slow database never slows the customer down
    #    (a visitor without an id is logged as "anonymous", hashed like any other id)
    background_tasks.add_task(request_log.log, customer_id or "anonymous", source, articles,
                              state.manifest["table_version"][:12], lookup_ms)

    return {
        "customer_id": customer_id,
        "source": source,
        "articles": articles,
        # the same articles, readable: what a person testing the API actually wants to see
        "items": [{"article_id": article, "name": state.article_names[str(article)]} for article in articles],
        "table_version": state.manifest["table_version"][:12],
    }


# Real H&M ids for the /docs dropdown, one per source. Change a few letters (0-9, a-f) = a brand-new customer.
EXAMPLE_IDS = {
    "personal": {"summary": "bought recently: their own list",
                 "value": "00000dbacae5abe5e23885899a1fa44253a17956c6d1c3d25f88aa139fdfc657"},
    "age_bucket": {"summary": "no recent purchases: their age group's list",
                   "value": "00005ca1c9ed5f5146b52ac8639a40ca9d57aeff4d1bd2c5feb1ca5dff07c43e"},
    "global": {"summary": "no recent purchases, no usable age: the global list",
               "value": "000064249685c11552da43ef22a5030f35a147f723d5b02ddd9fd22452b1f5a6"},
}


@app.get("/recommend")
def recommend_get(response: Response, background_tasks: BackgroundTasks,
                  customer_id: str = Query(openapi_examples=EXAMPLE_IDS,
                                           description="64 hex characters. Edit a few letters to try a new customer."),
                  age: float | None = None):
    """Quick testing: /recommend?customer_id=...&age=25 (the id is in the URL, so it shows up in access logs)."""
    return serve(customer_id, age, response, background_tasks)


class RecommendRequest(BaseModel):
    """The JSON body of POST /recommend: what a visitor fills in. Both fields are optional."""
    customer_id: int | str | None = None    # your own number id (1234245), a real H&M id (64 hex text), or nothing
    age: float | None = None                # their age, or nothing


# The /docs dropdown for POST (without it, /docs pre-fills "customer_id": "string", which is a 422)
EXAMPLE_BODIES = {
    "new_visitor": {"summary": "new visitor: only an age", "value": {"age": 25}},
    "anonymous": {"summary": "anonymous: nothing at all", "value": {}},
    "number_id": {"summary": "your own number id: turned into a hex id you can reuse in GET",
                  "value": {"customer_id": 1234245, "age": 25}},
}
for name, example in EXAMPLE_IDS.items():          # the same three real ids as GET
    EXAMPLE_BODIES[name] = {"summary": example["summary"], "value": {"customer_id": example["value"]}}


@app.post("/recommend")
def recommend_post(response: Response, background_tasks: BackgroundTasks,
                   body: RecommendRequest = Body(openapi_examples=EXAMPLE_BODIES)):
    """A form: {"customer_id": "...", "age": 25}. The id travels in the body, not the URL (not in access logs)."""
    customer_id = body.customer_id
    # A number id (1234245) becomes 64 hex characters: its SHA-256. Same number -> same hex id, so the id in the
    # response also works in GET. Text must be a real H&M id (64 hex): serve() gives a 422 for anything else ("a123").
    if isinstance(customer_id, int):
        if customer_id < 0:
            raise HTTPException(status_code=422, detail="customer_id must be a positive number or 64 hex characters")
        customer_id = hashlib.sha256(str(customer_id).encode()).hexdigest()
    return serve(customer_id, body.age, response, background_tasks)

