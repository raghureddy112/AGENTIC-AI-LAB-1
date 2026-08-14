"""
4. SQL AGENT WITH TOOL USE (ReAct-style)
An agent that reasons step-by-step and decides which tool to call next,
observes the result, and repeats until it can answer -- the classic
ReAct (Reason + Act) loop:

    Thought -> Action -> Observation -> Thought -> ... -> Final Answer

Rather than hand-parsing "Thought:/Action:" text (fragile), this uses
Claude's native tool-calling: the model's reasoning happens internally
and it emits structured tool_use blocks, which we execute and feed
back as tool_result blocks. This IS the ReAct pattern -- just with a
reliable, structured Action/Observation channel instead of free text.

Tools given to the agent:
    - list_tables()               discover what's in the database
    - get_table_schema(table)     inspect columns before writing SQL
    - run_sql(query)               execute a read-only SELECT

Requires: pip install anthropic
Set ANTHROPIC_API_KEY in your environment before running.
"""
import os
import re
import json
import sqlite3
from anthropic import Anthropic

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
client = Anthropic()

SYSTEM_PROMPT = """You are a database analyst agent. You answer questions about
a SQLite database by using the tools available to you.

Always work step by step:
1. Use list_tables to see what tables exist (if you don't already know).
2. Use get_table_schema on the relevant table(s) before writing SQL, so you
   never guess at column names.
3. Use run_sql to execute a single read-only SELECT statement.
4. Once you have enough information, give a final plain-English answer.

Never call run_sql with anything other than a SELECT statement."""


# -------------------------------------------------------------------
# Tool implementations (the "Act" half of ReAct)
# -------------------------------------------------------------------
def list_tables(db_path: str) -> list:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def get_table_schema(db_path: str, table: str) -> list:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
        return [{"name": r[1], "type": r[2]} for r in rows]
    finally:
        conn.close()


def run_sql(db_path: str, query: str) -> dict:
    if not re.match(r"^\s*SELECT\b", query, re.IGNORECASE):
        return {"error": "Only SELECT statements are permitted."}
    if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|PRAGMA)\b",
                 query, re.IGNORECASE):
        return {"error": "Only SELECT statements are permitted."}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(query)
        rows = [dict(r) for r in cur.fetchmany(100)]
        return {"rows": rows}
    except sqlite3.Error as e:
        return {"error": str(e)}
    finally:
        conn.close()


# -------------------------------------------------------------------
# Tool schema definitions given to the model
# -------------------------------------------------------------------
TOOLS = [
    {
        "name": "list_tables",
        "description": "List all tables in the database.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_table_schema",
        "description": "Get column names and types for a specific table.",
        "input_schema": {
            "type": "object",
            "properties": {"table": {"type": "string", "description": "Table name"}},
            "required": ["table"],
        },
    },
    {
        "name": "run_sql",
        "description": "Execute a single read-only SELECT statement and return rows.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A SELECT statement"}},
            "required": ["query"],
        },
    },
]


def _dispatch_tool(db_path: str, name: str, tool_input: dict):
    if name == "list_tables":
        return list_tables(db_path)
    if name == "get_table_schema":
        return get_table_schema(db_path, tool_input["table"])
    if name == "run_sql":
        return run_sql(db_path, tool_input["query"])
    return {"error": f"Unknown tool: {name}"}


# -------------------------------------------------------------------
# The ReAct loop
# -------------------------------------------------------------------
def run_sql_agent(question: str, db_path: str, max_steps: int = 6) -> dict:
    messages = [{"role": "user", "content": question}]
    trace = []  # records each Thought/Action/Observation for inspection

    for step in range(max_steps):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Any plain-text reasoning the model produced this turn ("Thought")
        thoughts = [b.text for b in response.content if b.type == "text"]
        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if thoughts:
            trace.append({"thought": " ".join(thoughts)})

        if response.stop_reason != "tool_use":
            # Model gave its final answer -- loop ends.
            final_answer = " ".join(thoughts).strip()
            return {"answer": final_answer, "steps": step + 1, "trace": trace}

        # Execute every requested tool call ("Action" -> "Observation")
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for call in tool_uses:
            observation = _dispatch_tool(db_path, call.name, call.input)
            trace.append({
                "action": call.name,
                "action_input": call.input,
                "observation": observation,
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": json.dumps(observation),
            })
        messages.append({"role": "user", "content": tool_results})

    return {"answer": None, "steps": max_steps, "trace": trace,
            "error": "Max steps reached without a final answer."}


if __name__ == "__main__":
    result = run_sql_agent(
        "Which country do our customers most commonly come from, "
        "and how many customers is that?",
        db_path="sample.db",
    )
    print("Answer:", result["answer"])
    print("\n--- Trace ---")
    for entry in result["trace"]:
        print(entry)
