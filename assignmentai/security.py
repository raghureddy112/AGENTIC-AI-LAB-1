import os
from dotenv import load_dotenv
from google import genai


# ============================================================
# LOAD GEMINI API KEY
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("ERROR: GEMINI_API_KEY not found.")
    print("Make sure your .env file contains:")
    print("GEMINI_API_KEY=your_api_key")
    exit()


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-3.7-flash"


# ============================================================
# SECURITY LOG ANALYSIS AGENT
# ============================================================

def analyze_security_log(logs):

    prompt = f"""
You are an expert Cybersecurity Log Analysis Agent.

Analyze the following security logs or security alerts.

================ SECURITY LOGS ================

{logs}

================================================

Perform the following tasks:

1. IDENTIFY POTENTIAL THREATS
   - Determine whether the logs indicate suspicious or malicious activity.
   - Identify the likely attack or security event.
   - Examples include:
     * Brute-force attack
     * Port scanning
     * Malware activity
     * SQL injection
     * Privilege escalation
     * Unauthorized access
     * DDoS activity
     * Phishing
     * Data exfiltration
     * Suspicious login
     * Account compromise
     * Reconnaissance
     * Lateral movement

2. EXTRACT IMPORTANT EVIDENCE
   Identify important information from the logs such as:
   - Source IP
   - Destination IP
   - Username
   - Timestamp
   - Port
   - Protocol
   - Number of attempts
   - Error messages
   - Suspicious commands
   - Other indicators of compromise

3. CLASSIFY SEVERITY

Use exactly one of:

   CRITICAL
   HIGH
   MEDIUM
   LOW
   INFORMATIONAL

Explain why the selected severity is appropriate.

4. DETERMINE CONFIDENCE

Give a confidence level:

   HIGH
   MEDIUM
   LOW

Explain why.

5. ATTACK ANALYSIS

Explain:
   - What happened?
   - Why is it suspicious?
   - What attack technique may be involved?
   - What could happen if the activity continues?

6. MITIGATION

Provide practical defensive actions.

Separate them into:

   IMMEDIATE ACTIONS
   SHORT-TERM ACTIONS
   LONG-TERM ACTIONS

Examples:
   - Block malicious IP
   - Disable compromised account
   - Reset credentials
   - Enable MFA
   - Isolate affected host
   - Review firewall rules
   - Patch vulnerable software
   - Increase monitoring
   - Check related logs
   - Search for indicators of compromise

7. SOC RECOMMENDATION

Give a short recommendation for what a security analyst
should do next.

8. FALSE POSITIVE ANALYSIS

Explain whether this could potentially be a false positive
and what additional evidence should be checked.

IMPORTANT:
- Do not invent information that does not appear in the logs.
- Clearly distinguish observed evidence from assumptions.
- If the logs are insufficient to determine the threat,
  explicitly say so.
- Do not claim that an attack is confirmed unless the evidence
  supports that conclusion.

Return the result using this structure:

==================================================
SECURITY ALERT ANALYSIS
==================================================

THREAT:
...

THREAT TYPE:
...

SEVERITY:
...

CONFIDENCE:
...

EVIDENCE:
- ...
- ...
- ...

ANALYSIS:
...

POTENTIAL IMPACT:
...

IMMEDIATE ACTIONS:
1. ...
2. ...
3. ...

SHORT-TERM ACTIONS:
1. ...
2. ...
3. ...

LONG-TERM ACTIONS:
1. ...
2. ...
3. ...

FALSE POSITIVE POSSIBILITY:
...

SOC RECOMMENDATION:
...

==================================================
"""

    try:

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        return response.text

    except Exception as e:

        return f"""
ERROR WHILE ANALYZING LOGS:

{e}
"""


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)
    print("          CYBERSECURITY LOG ANALYSIS AGENT")
    print("=" * 70)

    print("""
This agent analyzes security logs and alerts.

It can:
- Identify potential threats
- Classify severity
- Extract security evidence
- Estimate confidence
- Explain potential impact
- Suggest mitigation
- Identify possible false positives

Paste your security logs below.

Type END on a new line when finished.
Type EXIT to stop.
""")

    while True:

        print("\n" + "-" * 70)

        first_line = input("Paste security log / alert: ").strip()

        if first_line.lower() == "exit":
            print("\nAgent stopped.")
            break

        if not first_line:
            print("Please enter a log or alert.")
            continue

        # ----------------------------------------------------
        # MULTI-LINE LOG INPUT
        # ----------------------------------------------------

        log_lines = [first_line]

        while True:

            line = input()

            if line.strip().upper() == "END":
                break

            log_lines.append(line)

        logs = "\n".join(log_lines)

        # ----------------------------------------------------
        # ANALYZE
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("ANALYZING SECURITY EVENT...")
        print("=" * 70)

        result = analyze_security_log(logs)

        print("\n")
        print(result)


# ============================================================
# START AGENT
# ============================================================

if __name__ == "__main__":
    main()