from google import genai
from dotenv import load_dotenv
import os

# Load API Key
load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("=" * 60)
print("        Prompt Chaining using Gemini")
print("=" * 60)

# User Input
topic = input("\nEnter a topic: ")

try:
    # ---------------- STEP 1 ----------------
    print("\nGenerating Explanation...\n")

    explanation = client.models.generate_content(
        model="gemini-flash-latest",
        contents=f"""
Explain the topic "{topic}" in only 5-6 simple sentences.
Use easy English.
"""
    ).text

    # ---------------- STEP 2 ----------------
    print("Generating Summary...\n")

    summary = client.models.generate_content(
        model="gemini-flash-latest",
        contents=f"""
Summarize the following explanation in only 3 short bullet points.

Explanation:
{explanation}
"""
    ).text

    # ---------------- STEP 3 ----------------
    print("Extracting Key Points...\n")

    key_points = client.models.generate_content(
        model="gemini-flash-latest",
        contents=f"""
From the summary below, extract only 5 important key points.

Summary:
{summary}
"""
    ).text


    # ---------------- OUTPUT ----------------
    print("\n" + "=" * 60)
    print("EXPLANATION")
    print("=" * 60)
    print(explanation)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summary)

    print("\n" + "=" * 60)
    print("KEY POINTS")
    print("=" * 60)
    print(key_points)

    print("\n" + "=" * 60)
    print("QUESTIONS")
    print("=" * 60)
    print(questions)

except Exception as e:
    print("\nError:", e)