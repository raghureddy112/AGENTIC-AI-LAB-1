from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("=" * 50)
print("LLM Workflow using Gemini")
print("=" * 50)

while True:
    user_input = input("\nEnter your question (type 'exit' to quit): ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=user_input
    )

    print("\nGemini Response:")
    print(response.text)