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
    print("ERROR: GEMINI_API_KEY was not found.")
    print("Please check your .env file.")
    exit()


# ============================================================
# CREATE GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=API_KEY)


# ============================================================
# MODEL
# ============================================================

MODEL = "gemini-3.8-flash"


# ============================================================
# RESEARCH FUNCTION
# ============================================================

def research_question(question):

    print("\n" + "=" * 70)
    print("RESEARCHING...")
    print("=" * 70)

    prompt = f"""
You are a Public PDF Research Agent.

The user has asked the following question:

{question}

Your job is to research the question using publicly available
information on the internet.

IMPORTANT INSTRUCTIONS:

1. Search the public internet for information related to the question.

2. PRIORITIZE publicly available PDF documents whenever possible.

3. Give preference to reliable sources such as:
   - Research papers
   - Academic papers
   - IEEE publications
   - ACM publications
   - arXiv papers
   - University publications
   - Government reports
   - Official technical reports
   - Official documentation

4. Do not rely only on your internal knowledge.

5. Use the information retrieved from the web to construct the answer.

6. If relevant PDF documents are available, use them as important
   sources for the answer.

7. Do not invent facts, papers, authors, URLs, or citations.

8. Clearly explain the answer in simple language.

9. If the question requires comparison, provide a clear comparison.

10. If the question is technical, explain the technical concepts
    step-by-step.

11. At the end of the answer, provide a section called:

    SOURCES

12. List the important sources used to answer the question.

13. Include the source title and URL when available.

14. If no suitable PDF can be found, use other reliable web sources
    and clearly mention that a suitable PDF was not found.

USER QUESTION:

{question}
"""

    try:

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

        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)

        print(response.text)

        print("\n" + "=" * 70)
        print("RESEARCH FINISHED")
        print("=" * 70)

    except Exception as e:

        print("\n" + "=" * 70)
        print("ERROR")
        print("=" * 70)

        print(e)

        print("\nPossible causes:")
        print("1. Invalid API key")
        print("2. API key is disabled")
        print("3. API access is not enabled")
        print("4. Gemini model is not available for your project")
        print("5. You have exceeded your API quota")
        print("6. Internet connection problem")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("        PUBLIC PDF RESEARCH AGENT")
    print("=" * 70)

    print("""
This agent:
    - Accepts your question
    - Searches the public internet
    - Prioritizes publicly available PDFs
    - Uses Gemini to analyze the information
    - Generates an answer
    - Shows sources

Type 'exit' to stop the program.
""")

    while True:

        question = input("\nEnter your question: ").strip()

        if question.lower() == "exit":
            print("\nAgent stopped.")
            break

        if not question:
            print("Please enter a question.")
            continue

        research_question(question)


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()