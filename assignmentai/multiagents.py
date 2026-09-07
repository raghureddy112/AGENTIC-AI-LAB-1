import os
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# LOAD API KEY
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env")
    exit()


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-3.7-flash"


# ============================================================
# HELPER FUNCTION
# ============================================================

def ask_gemini(prompt, use_search=False):

    try:

        if use_search:

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[
                        types.Tool(
                            google_search=types.GoogleSearch()
                        )
                    ]
                )
            )

        else:

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )

        return response.text

    except Exception as e:

        return f"ERROR: {e}"


# ============================================================
# AGENT 1: RESEARCH AGENT
# ============================================================

def research_agent(task):

    print("\n")
    print("=" * 70)
    print("AGENT 1: RESEARCH AGENT")
    print("=" * 70)

    print("Researching the topic...")

    prompt = f"""
You are the RESEARCH AGENT in a multi-agent AI system.

The user has given this task:

{task}

Your responsibility is to research the topic using reliable
publicly available information.

Instructions:

1. Understand exactly what the user is asking.
2. Search the internet for relevant information.
3. Prefer reliable sources.
4. Prefer:
   - Research papers
   - Government sources
   - Universities
   - Official documentation
   - Reputable organizations
5. Collect important facts.
6. Identify important statistics, concepts, technologies,
   advantages, disadvantages, and examples when relevant.
7. Do not make up information.
8. Mention the sources used.

Return your research in a structured format.

Include:

TOPIC
KEY FINDINGS
IMPORTANT FACTS
TECHNICAL INFORMATION
EXAMPLES
ADVANTAGES
DISADVANTAGES
SOURCES

USER TASK:

{task}
"""

    result = ask_gemini(prompt, use_search=True)

    print("\nResearch completed.")

    return result


# ============================================================
# AGENT 2: ANALYST AGENT
# ============================================================

def analyst_agent(task, research):

    print("\n")
    print("=" * 70)
    print("AGENT 2: ANALYST AGENT")
    print("=" * 70)

    print("Analyzing research findings...")

    prompt = f"""
You are the ANALYST AGENT in a multi-agent AI system.

The original user task is:

{task}

The Research Agent produced the following research:

---------------- RESEARCH ----------------

{research}

--------------------------------------------

Your responsibility is to analyze the research.

Perform the following:

1. Identify the most important findings.
2. Remove irrelevant information.
3. Identify relationships between the findings.
4. Compare alternatives when appropriate.
5. Identify important advantages and disadvantages.
6. Identify risks or limitations.
7. Determine the most important conclusions.
8. Check whether the research actually answers the user's task.
9. Do not invent information that is not supported by the research.

Return:

ANALYSIS
KEY INSIGHTS
COMPARISON
RISKS / LIMITATIONS
IMPORTANT CONCLUSIONS
RECOMMENDATION
"""

    result = ask_gemini(prompt)

    print("\nAnalysis completed.")

    return result


# ============================================================
# AGENT 3: REPORT AGENT
# ============================================================

def report_agent(task, research, analysis):

    print("\n")
    print("=" * 70)
    print("AGENT 3: REPORT AGENT")
    print("=" * 70)

    print("Generating final report...")

    prompt = f"""
You are the REPORT AGENT in a multi-agent AI system.

Your job is to create a professional final report using the
work completed by the Research Agent and Analyst Agent.

ORIGINAL USER TASK:

{task}


================ RESEARCH =================

{research}


================ ANALYSIS =================

{analysis}


============================================

Create a clear, professional and easy-to-understand report.

Use this structure:

==================================================
FINAL REPORT
==================================================

1. EXECUTIVE SUMMARY

Give a short summary of the topic and major findings.

2. INTRODUCTION

Explain the topic and why it is important.

3. KEY FINDINGS

Explain the most important information discovered
by the Research Agent.

4. ANALYSIS

Explain the Analyst Agent's conclusions.

5. ADVANTAGES

List important advantages where applicable.

6. DISADVANTAGES / LIMITATIONS

List important limitations.

7. RISKS

Explain relevant risks.

8. RECOMMENDATIONS

Provide practical recommendations.

9. CONCLUSION

Give a concise final conclusion.

10. SOURCES

List the sources identified by the Research Agent.

IMPORTANT:

- Do not invent sources.
- Do not invent facts.
- Keep the report directly related to the user's task.
- Use simple but professional language.
"""

    result = ask_gemini(prompt)

    print("\nReport generated.")

    return result


# ============================================================
# MULTI-AGENT WORKFLOW
# ============================================================

def run_multi_agent_system(task):

    print("\n")
    print("=" * 70)
    print("       MULTI-AGENT AI RESEARCH SYSTEM")
    print("=" * 70)

    print("\nUSER TASK:")
    print(task)

    # --------------------------------------------------------
    # STEP 1: RESEARCH AGENT
    # --------------------------------------------------------

    research = research_agent(task)

    # --------------------------------------------------------
    # STEP 2: ANALYST AGENT
    # --------------------------------------------------------

    analysis = analyst_agent(
        task,
        research
    )

    # --------------------------------------------------------
    # STEP 3: REPORT AGENT
    # --------------------------------------------------------

    final_report = report_agent(
        task,
        research,
        analysis
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("                    FINAL RESULT")
    print("=" * 70)

    print(final_report)

    print("\n")
    print("=" * 70)
    print("              MULTI-AGENT WORKFLOW COMPLETE")
    print("=" * 70)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("       MULTI-AGENT RESEARCH & REPORT SYSTEM")
    print("=" * 70)

    print("""
Three specialized agents work together:

    Agent 1 → Research Agent
    Agent 2 → Analyst Agent
    Agent 3 → Report Agent

The agents collaborate automatically.

Type EXIT to stop.
""")

    while True:

        task = input("\nEnter your task: ").strip()

        if task.lower() == "exit":

            print("\nSystem stopped.")
            break

        if not task:

            print("Please enter a task.")
            continue

        run_multi_agent_system(task)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()