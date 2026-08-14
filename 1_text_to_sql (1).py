"""
1. TEXT-TO-SQL WORKFLOW
End-to-end pipeline: schema retrieval -> SQL generation -> safe execution
-> self-correction retry loop.

    question -> retrieve relevant tables -> LLM writes SQL -> run it
              -> on DB error, feed error back to LLM and retry (bounded)

Requires: pip install anthropic
Set ANTHROPIC_API_KEY in your environment before running.
"""
import os
import re
import math
import sqlite3
from collections import Counter
from anthropic import Anthropic

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
client = Anthropic()

# -------------------------------------------------------------------
# Schema catalog: business-context descriptions used for retrieval.
# In production this is generated once from your DB and hand-annotated.
# -------------------------------------------------------------------
SCHEMA_CATALOG = [
    {
        "name": "customers",
        "description": "One row per customer: contact info and country.",
        "columns": ["customer_id", "name", "email", "country", "signup_date"],
    },
    {
        "name": "orders",
        "description": "One row per order: date, status, which customer.",
        "columns": ["order_id", "customer_id", "order_date", "status"],
    },
    {
        "name": "order_items",
        "description": "Line items per order: product, quantity, price paid.",
        "columns": ["order_item_id", "order_id", "product_id", "quantity", "unit_price_paid"],
    },
    {
        "name": "products",
        "description": "Product catalog: name, category, current price.",
        "columns": ["product_id", "name", "category_id", "unit_price"],
    },
]


# -------------------------------------------------------------------
# Retrieval: dependency-free TF-IDF + cosine similarity over table
# descriptions, so only relevant tables are put into the prompt.
# -------------------------------------------------------------------
def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def retrieve_tables(question, top_k=3):
    docs = [_tokenize(t["name"] + " " + t["description"] + " " + " ".join(t["columns"]))
            for t in SCHEMA_CATALOG]
    df = Counter()
    for d in docs:
        for term in set(d):
            df[term] += 1
    n = len(docs)

    def tfidf(tokens):
        tf = Counter(tokens)
        return {t: c * (math.log((1 + n) / (1 + df[t])) + 1) for t, c in tf.items()}

    def cosine(a, b):
        common = set(a) & set(b)
        dot = sum(a[t] * b[t] for t in common)
        na = math.sqrt(sum(v * v for v in a.values())) or 1e-9
        nb = math.sqrt(sum(v * v for v in b.values())) or 1e-9
        return dot / (na * nb)

    q_vec = tfidf(_tokenize(question))
    scored = [(cosine(q_vec, tfidf(d)), t) for d, t in zip(docs, SCHEMA_CATALOG)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:top_k]]


def schema_context(tables):
    return "\n".join(f"TABLE {t['name']} ({', '.join(t['columns'])}) -- {t['description']}"
                      for t in tables)


# -------------------------------------------------------------------
# Generation: LLM writes SQL wrapped in <sql> tags for reliable parsing.
# -------------------------------------------------------------------
def generate_sql(question, schema_ctx, prior_sql=None, prior_error=None):
    error_note = ""
    if prior_error:
        error_note = (f"\nPrevious attempt: {prior_sql}\n"
                       f"It failed with: {prior_error}\nFix it.\n")

    prompt = (
        "You write SQLite-compatible SQL. Only use the columns listed below. "
        "Never invent columns. Return exactly one SELECT statement wrapped in "
        "<sql></sql> tags.\n\n"
        f"Schema:\n{schema_ctx}\n\nQuestion: {question}{error_note}"
    )
    resp = client.messages.create(
        model=MODEL, max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    match = re.search(r"<sql>(.*?)</sql>", text, re.DOTALL)
    if not match:
        raise ValueError(f"No SQL found in model output:\n{text}")
    return match.group(1).strip()


# -------------------------------------------------------------------
# Execution: read-only guardrail, runs against SQLite.
# -------------------------------------------------------------------
def execute_sql(db_path, sql):
    if not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE):
        return {"ok": False, "error": "Only SELECT statements are permitted."}
    if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|PRAGMA)\b", sql, re.IGNORECASE):
        return {"ok": False, "error": "Only SELECT statements are permitted."}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql)
        rows = [dict(r) for r in cur.fetchmany(200)]
        return {"ok": True, "rows": rows}
    except sqlite3.Error as e:
        return {"ok": False, "error": str(e)}
    finally:
        conn.close()


# -------------------------------------------------------------------
# Orchestration: retrieve -> generate -> execute -> retry on error.
# -------------------------------------------------------------------
def text_to_sql(question, db_path, max_retries=2):
    tables = retrieve_tables(question)
    ctx = schema_context(tables)

    prior_sql, prior_error = None, None
    for attempt in range(1, max_retries + 2):
        sql = generate_sql(question, ctx, prior_sql, prior_error)
        result = execute_sql(db_path, sql)
        if result["ok"]:
            return {"sql": sql, "rows": result["rows"], "attempts": attempt}
        prior_sql, prior_error = sql, result["error"]

    return {"sql": prior_sql, "error": prior_error, "attempts": attempt}


if __name__ == "__main__":
    # Point db_path at your real database; see db_setup.py for a demo DB.
    result = text_to_sql(
        "Which customers have never had a cancelled order?",
        db_path="sample.db",
    )
    print(result)
