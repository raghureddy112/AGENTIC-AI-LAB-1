import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# AI RESEARCH AGENT - GEMINI ONLY
# ============================================================

load_dotenv()


# ============================================================
# API KEY
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing. "
        "Please add GOOGLE_API_KEY to your .env file."
    )


# ============================================================
# INITIALIZE GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY
)


# ============================================================
# CONVERT GEMINI RESPONSE TO STRING
# ============================================================

def response_to_text(response):

    content = response.content

    # Normal string response
    if isinstance(content, str):
        return content

    # Gemini may return a list
    if isinstance(content, list):

        text = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":
                    text.append(
                        item.get("text", "")
                    )

            elif isinstance(item, str):

                text.append(item)

        return "\n".join(text)

    return str(content)


# ============================================================
# STEP 1 - RESEARCH TOPIC
# ============================================================

def research_topic(topic):

    print("\n🧠 Researching topic using Gemini...")

    prompt = f"""
You are an AI research assistant.

Research Topic:
{topic}

Provide a detailed research analysis about this topic.

Include:

1. Main findings
2. Important facts
3. Major concepts
4. Current trends, if known
5. Different viewpoints
6. Advantages
7. Disadvantages
8. Challenges
9. Future scope
10. Conclusion

Rules:

- Use your available knowledge.
- Do not invent facts.
- If you are uncertain about something, clearly mention it.
- Keep the explanation accurate.
- Use simple and understandable language.
- Organize the information clearly.
"""

    try:

        response = llm.invoke(prompt)

        return response_to_text(response)

    except Exception as e:

        print("\n❌ Gemini error while researching:")
        print(e)

        return ""


# ============================================================
# STEP 2 - GENERATE STRUCTURED REPORT
# ============================================================

def generate_report(topic, research):

    print("\n📝 Generating research report...")

    prompt = f"""
You are a professional research report generator.

Research Topic:
{topic}

Research Information:
{research}

Create a professional structured research report.

Use exactly this structure:

# RESEARCH REPORT

## Topic
{topic}

## 1. Introduction

Explain the research topic and why it is important.

## 2. Key Findings

Present the major findings using bullet points.

## 3. Detailed Analysis

Explain the research findings in detail.

## 4. Trends and Observations

Explain important trends and observations.

## 5. Advantages

Explain the major advantages.

## 6. Disadvantages

Explain the major disadvantages.

## 7. Challenges and Limitations

Discuss challenges, limitations, uncertainties,
and areas where information may be incomplete.

## 8. Future Scope

Explain possible future developments.

## 9. Conclusion

Provide a clear conclusion based only on
the research information.

Rules:

- Do not invent information.
- Do not create fake references.
- Keep the report professional.
- Use clear and simple language.
- Use headings and bullet points where appropriate.
"""

    try:

        response = llm.invoke(prompt)

        return response_to_text(response)

    except Exception as e:

        print("\n❌ Gemini error while generating report:")
        print(e)

        return ""


# ============================================================
# STEP 3 - SAVE REPORT
# ============================================================

def save_report(report):

    filename = "research_report.txt"

    try:

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(report)

        print(
            f"\n✅ Report saved successfully: {filename}"
        )

    except Exception as e:

        print("\n❌ Error while saving report:")
        print(e)


# ============================================================
# STEP 4 - DISPLAY REPORT
# ============================================================

def display_report(report):

    print("\n")
    print("========================================")
    print("             FINAL REPORT")
    print("========================================")

    print(report)


# ============================================================
# STEP 5 - MAIN RESEARCH AGENT
# ============================================================

def research_agent(topic):

    print("\n========================================")
    print("          AI RESEARCH AGENT")
    print("========================================")

    print(f"\nResearch Topic: {topic}")

    # --------------------------------------------------------
    # RESEARCH
    # --------------------------------------------------------

    research = research_topic(topic)

    if not research:

        print("\n❌ Could not generate research.")

        return

    print("\n========================================")
    print("        RESEARCH INFORMATION")
    print("========================================")

    print(research)

    # --------------------------------------------------------
    # GENERATE REPORT
    # --------------------------------------------------------

    report = generate_report(
        topic,
        research
    )

    if not report:

        print("\n❌ Could not generate report.")

        return

    # --------------------------------------------------------
    # DISPLAY REPORT
    # --------------------------------------------------------

    display_report(report)

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    save_report(report)


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("       WELCOME TO AI RESEARCH AGENT")
    print("========================================")

    topic = input(
        "\nEnter your research topic: "
    ).strip()

    if not topic:

        print(
            "\n❌ Please enter a research topic."
        )

    else:

        research_agent(topic)