"""
3. PROMPT CHAINING FOR SUMMARIZATION
A three-step prompt pipeline where each step's output feeds the next:

    raw text -> extract key points -> structured summary -> final summary

Each step is a separate, focused LLM call rather than one big prompt.
This tends to produce more reliable output than asking for everything
at once, and each intermediate result is inspectable/debuggable on
its own.

Requires: pip install anthropic
Set ANTHROPIC_API_KEY in your environment before running.

(Swappable backend: replace `ask_llm` with a call to a local model via
Ollama -- e.g. `ollama.chat(model="qwen3:8b", messages=[...])` -- and
every function below works unchanged, since they only depend on
ask_llm's (prompt -> text) signature.)
"""
import os
from anthropic import Anthropic

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
client = Anthropic()


def ask_llm(prompt: str) -> str:
    resp = client.messages.create(
        model=MODEL, max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


# -------------------------------------------------------------------
# Step 1: Extract important information
# -------------------------------------------------------------------
def extract_key_points(text: str) -> str:
    prompt = f"""You are an information extraction system.
Read the following text and extract the most important information.
Return:
1. Main topic
2. Important facts
3. Important numbers or dates
4. Important conclusions
Do not add information that is not present in the text.

TEXT:
{text}"""
    return ask_llm(prompt)


# -------------------------------------------------------------------
# Step 2: Convert key points into a structured summary
# -------------------------------------------------------------------
def create_structured_summary(key_points: str) -> str:
    prompt = f"""You are a professional summarization system.
Using the extracted information below, create a structured summary.
Use these sections:
- Overview
- Key Points
- Important Details
- Conclusion
Keep the information accurate and remove unnecessary repetition.

EXTRACTED INFORMATION:
{key_points}"""
    return ask_llm(prompt)


# -------------------------------------------------------------------
# Step 3: Generate final concise summary
# -------------------------------------------------------------------
def create_final_summary(structured_summary: str) -> str:
    prompt = f"""You are a final summarization assistant.
Convert the structured summary below into a concise, easy-to-understand
final summary.
Requirements:
- 1 short introductory paragraph
- 4 to 6 bullet points
- Mention important facts and numbers
- Do not introduce new information
- Keep the language simple

STRUCTURED SUMMARY:
{structured_summary}"""
    return ask_llm(prompt)


# -------------------------------------------------------------------
# Main prompt-chaining pipeline
# -------------------------------------------------------------------
def summarize(text: str) -> str:
    key_points = extract_key_points(text)
    structured_summary = create_structured_summary(key_points)
    final_summary = create_final_summary(structured_summary)

    return final_summary


if __name__ == "__main__":
    text = """
    Artificial intelligence is rapidly transforming software development.
    Modern AI coding assistants can generate code, explain programming
    concepts, detect bugs and help developers understand large codebases.
    Large language models such as Qwen, Llama and GPT can be integrated
    into development environments and applications. Local models are
    particularly useful when privacy is important because data can remain
    on the user's computer.

    However, AI-generated code still requires human review. Developers
    must verify security, performance and correctness before deploying
    AI-generated solutions.

    Organizations are increasingly combining AI with traditional software
    engineering practices. This creates new roles such as AI engineer,
    LLM engineer and AI platform engineer. Knowledge of Python, APIs,
    databases, cloud platforms and DevOps is becoming increasingly useful
    for these roles.
    """
    print(summarize(text))
