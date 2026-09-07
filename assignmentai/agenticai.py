from google import genai
from dotenv import load_dotenv
import os

# Load API Key
load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("=" * 60)
print("Simple AI Agent")
print("=" * 60)

task = input("\nEnter your task: ")

try:
    # Step 1: Planning
    print("\nPlanning...\n")

    plan = client.models.generate_content(
        model="gemini-flash-latest",
        contents=f"""
You are an AI agent.

Create a clear execution plan for this task:
{task}

Return exactly 4 numbered steps.
"""
    ).text

    print(plan)


    # Final Output
    print("\n" + "=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)
    print(result)

except Exception as e:
    print("Error:", e)