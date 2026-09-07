import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent


# ============================================================
# 1. LOAD GEMINI API CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found.\n"
        "Please add it to your .env file."
    )


# ============================================================
# 2. CREATE SQLITE DATABASE
# ============================================================

DATABASE_URL = "sqlite:///college.db"

engine = create_engine(DATABASE_URL)


# ============================================================
# 3. CREATE DATABASE TABLE
# ============================================================

with engine.begin() as connection:

    connection.execute(text("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            marks INTEGER NOT NULL
        )
    """))


# ============================================================
# 4. INSERT SAMPLE DATA
# ============================================================

with engine.begin() as connection:

    # Remove previous data so that duplicate records
    # are not created every time the program runs.
    connection.execute(
        text("DELETE FROM students")
    )

    connection.execute(text("""
        INSERT INTO students
        (id, name, department, marks)
        VALUES
        (1, 'Ramya', 'CSE', 92),
        (2, 'Anu', 'CSE', 85),
        (3, 'Ravi', 'ECE', 76),
        (4, 'Priya', 'CSE', 90),
        (5, 'Kiran', 'ECE', 68),
        (6, 'Rahul', 'CSE', 78),
        (7, 'Sneha', 'ECE', 88),
        (8, 'Arjun', 'CSE', 95)
    """))


print("\nDatabase created successfully!")


# ============================================================
# 5. CONNECT LANGCHAIN TO SQLITE
# ============================================================

db = SQLDatabase(engine)

print("\nAvailable tables:")

for table in db.get_usable_table_names():
    print("-", table)


# ============================================================
# 6. CREATE GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model=MODEL,
    google_api_key=API_KEY,
    temperature=0
)


# ============================================================
# 7. CREATE SQL AGENT
# ============================================================

agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="tool-calling",
    verbose=True,
    handle_parsing_errors=True
)


# ============================================================
# 8. FUNCTION TO ASK QUESTIONS
# ============================================================

def ask_question(question):

    print("\n")
    print("=" * 70)

    print("USER QUESTION:")
    print(question)

    print("=" * 70)

    try:

        response = agent.invoke(
            {
                "input": question
            }
        )

        print("\nFINAL ANSWER:")
        print(response["output"])

    except Exception as error:

        print("\nERROR:")
        print(error)

    print("=" * 70)


# ============================================================
# 9. INTERACTIVE SQL AGENT
# ============================================================

print("\n")
print("=" * 70)
print("              SQL AGENT WITH TOOL USE")
print("=" * 70)

print("\nGemini Model:", MODEL)

print("\nThe agent can answer questions about the student database.")

print("\nExample questions:")
print("1. Which student has the highest marks?")
print("2. Show all CSE students.")
print("3. What is the average mark of CSE students?")
print("4. How many ECE students are there?")
print("5. Who scored more than 85?")
print("6. Show the top 3 students.")
print("7. Which department has the highest average marks?")

print("\nType 'exit' to stop the program.")

print("\n")


# ============================================================
# 10. MAIN LOOP
# ============================================================

while True:

    question = input("You: ")

    if question.lower().strip() == "exit":

        print("\nSQL Agent stopped.")
        break

    if not question.strip():

        print("Please enter a question.")
        continue

    ask_question(question)