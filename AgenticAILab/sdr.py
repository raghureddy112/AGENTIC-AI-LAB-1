import os
import json

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3-flash-preview"
)

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found.\n"
        "Please check your .env file."
    )


# ============================================================
# 2. CREATE GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model=MODEL,
    google_api_key=API_KEY,
    temperature=0.3
)


# ============================================================
# 3. HELPER FUNCTION
# ============================================================

def ask_llm(prompt):
    """
    Sends a prompt to Gemini and returns clean text.
    """

    response = llm.invoke(prompt)

    content = response.content

    # Gemini may return a string
    if isinstance(content, str):
        return content.strip()

    # Gemini may return a list of content blocks
    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":
                    text_parts.append(
                        item.get("text", "")
                    )

            elif isinstance(item, str):

                text_parts.append(item)

        return "\n".join(text_parts).strip()

    return str(content).strip()


# ============================================================
# 4. LEAD GENERATION AGENT
# ============================================================

def lead_generation_agent(industry, location, product):
    """
    Agent 1:
    Generates potential leads based on industry,
    location and product.
    """

    print("\n")
    print("=" * 70)
    print("LEAD GENERATION AGENT")
    print("=" * 70)

    prompt = f"""
You are a Lead Generation AI Agent.

Your task is to generate 5 realistic SAMPLE business leads.

Target industry:
{industry}

Target location:
{location}

Product or service being sold:
{product}

IMPORTANT:
This is an educational demonstration.
Do NOT claim that these are verified real companies.
Create clearly labeled SAMPLE companies.

For each lead provide:

1. Company name
2. Industry
3. Location
4. Estimated employees
5. Potential business need
6. Estimated budget level
7. Contact role

Return ONLY valid JSON.

Format:

[
    {{
        "company": "Sample Company 1",
        "industry": "...",
        "location": "...",
        "employees": 100,
        "need": "...",
        "budget": "High",
        "contact_role": "IT Manager"
    }}
]
"""

    result = ask_llm(prompt)

    # Remove markdown code fences if Gemini adds them
    result = result.replace("```json", "")
    result = result.replace("```", "")
    result = result.strip()

    try:

        leads = json.loads(result)

    except json.JSONDecodeError:

        print("\nCould not parse JSON.")
        print(result)

        return []

    print("\nGenerated Leads:\n")

    for index, lead in enumerate(leads, start=1):

        print(
            f"{index}. {lead.get('company', 'Unknown')} | "
            f"{lead.get('industry', 'Unknown')} | "
            f"{lead.get('location', 'Unknown')}"
        )

    return leads


# ============================================================
# 5. LEAD QUALIFICATION AGENT
# ============================================================

def lead_qualification_agent(leads, product):
    """
    Agent 2:
    Evaluates each lead and assigns a score.
    """

    print("\n")
    print("=" * 70)
    print("LEAD QUALIFICATION AGENT")
    print("=" * 70)

    qualified_leads = []

    for lead in leads:

        prompt = f"""
You are a Lead Qualification AI Agent.

Evaluate the following business lead for a product/service.

Product:
{product}

Lead:
Company: {lead.get("company")}
Industry: {lead.get("industry")}
Location: {lead.get("location")}
Employees: {lead.get("employees")}
Need: {lead.get("need")}
Budget: {lead.get("budget")}
Contact Role: {lead.get("contact_role")}

Score the lead from 0 to 100.

Use these criteria:

Industry relevance: 25 points
Company size: 20 points
Business need: 25 points
Budget: 20 points
Contact relevance: 10 points

Classification:

80-100 = HOT
60-79 = WARM
0-59 = COLD

Return ONLY valid JSON.

Format:

{{
    "score": 85,
    "status": "HOT",
    "reason": "Short explanation"
}}
"""

        result = ask_llm(prompt)

        result = result.replace("```json", "")
        result = result.replace("```", "")
        result = result.strip()

        try:

            qualification = json.loads(result)

        except json.JSONDecodeError:

            qualification = {
                "score": 0,
                "status": "COLD",
                "reason": "Could not evaluate lead."
            }

        lead["score"] = qualification.get(
            "score",
            0
        )

        lead["status"] = qualification.get(
            "status",
            "COLD"
        )

        lead["reason"] = qualification.get(
            "reason",
            ""
        )

        print("\n")
        print("Company:", lead["company"])
        print("Score:", lead["score"])
        print("Status:", lead["status"])
        print("Reason:", lead["reason"])

        # Only HOT and WARM leads continue
        if lead["status"] in ["HOT", "WARM"]:

            qualified_leads.append(lead)

    return qualified_leads


# ============================================================
# 6. EMAIL GENERATION AGENT
# ============================================================

def email_generation_agent(lead, product):
    """
    Agent 3:
    Generates a personalized sales email.
    """

    prompt = f"""
You are an Email Generation AI Agent.

Create a professional and personalized sales outreach email.

Product/service:
{product}

Lead information:

Company: {lead.get("company")}
Industry: {lead.get("industry")}
Location: {lead.get("location")}
Employees: {lead.get("employees")}
Business Need: {lead.get("need")}
Budget: {lead.get("budget")}
Contact Role: {lead.get("contact_role")}
Lead Score: {lead.get("score")}
Lead Status: {lead.get("status")}

Create:

1. Email subject
2. Professional email body

Requirements:

- Keep it concise.
- Personalize it for the company.
- Mention the likely business need.
- Explain how the product could help.
- Include a simple call to action.
- Do not make unsupported claims.
- Do not pretend that previous communication occurred.

Return in this format:

SUBJECT:
<subject>

EMAIL:
<email body>
"""

    return ask_llm(prompt)


# ============================================================
# 7. ORCHESTRATOR
# ============================================================

def run_sdr_system(industry, location, product):
    """
    Orchestrates all three agents.
    """

    print("\n")
    print("=" * 70)
    print("             MULTI-AGENT SDR SYSTEM")
    print("=" * 70)

    print("\nTarget Industry :", industry)
    print("Target Location :", location)
    print("Product/Service :", product)

    # --------------------------------------------------------
    # Agent 1: Generate Leads
    # --------------------------------------------------------

    leads = lead_generation_agent(
        industry,
        location,
        product
    )

    if not leads:

        print("\nNo leads were generated.")
        return

    # --------------------------------------------------------
    # Agent 2: Qualify Leads
    # --------------------------------------------------------

    qualified_leads = lead_qualification_agent(
        leads,
        product
    )

    if not qualified_leads:

        print("\nNo qualified leads found.")
        return

    # --------------------------------------------------------
    # Agent 3: Generate Emails
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("EMAIL GENERATION AGENT")
    print("=" * 70)

    for lead in qualified_leads:

        print("\n")
        print("-" * 70)

        print(
            f"Generating email for: "
            f"{lead['company']}"
        )

        email = email_generation_agent(
            lead,
            product
        )

        print("\nCompany:", lead["company"])
        print("Lead Score:", lead["score"])
        print("Lead Status:", lead["status"])

        print("\nPERSONALIZED EMAIL")
        print("-" * 50)

        print(email)

        print("-" * 70)


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("          AI-POWERED MULTI-AGENT SDR")
    print("=" * 70)

    print("""
This system contains three AI agents:

1. Lead Generation Agent
2. Lead Qualification Agent
3. Email Generation Agent

The Orchestrator coordinates the complete workflow.
""")

    print("=" * 70)

    # --------------------------------------------------------
    # Get user input
    # --------------------------------------------------------

    industry = input(
        "\nEnter target industry: "
    ).strip()

    location = input(
        "Enter target location: "
    ).strip()

    product = input(
        "Enter product/service: "
    ).strip()

    if not industry or not location or not product:

        print(
            "\nPlease provide all three inputs."
        )

        return

    # --------------------------------------------------------
    # Run Multi-Agent SDR
    # --------------------------------------------------------

    run_sdr_system(
        industry,
        location,
        product
    )


# ============================================================
# 9. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()