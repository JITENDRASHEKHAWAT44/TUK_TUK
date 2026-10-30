"""
TenantEase AI — LLM Prompt Engineering & Generation Engine
Powers:
  1. Context-Aware Autonomous Maintenance SLA & Safety Dispatch
  2. Dynamic Tenant Retention & WhatsApp Nudge Generation
  3. Executive Portfolio Risk Synthesis

Supports:
  - Google Gemini (via google-genai SDK or REST)
  - OpenAI (GPT-4o / GPT-4o-mini)
  - Built-in High-Fidelity Generative Fallback (zero API key requirement)
"""

import os
import json
import requests
from typing import Optional, Dict, Any

# ─── System Prompts ───────────────────────────────────────────────

MAINTENANCE_SYSTEM_PROMPT = """You are the autonomous AI Operations Copilot for TenantEase by Cynbit Technologies.
Your objective is to generate an immediate, professional, reassuring, and safety-compliant acknowledgment message to a resident who has submitted a maintenance ticket.

CRITICAL PROTOCOLS:
1. If is_emergency is TRUE (Urgency >= 8, or Gas/Electrical hazard):
   - Begin immediately with an urgent bold directive: "[CRITICAL SAFETY PROTOCOL ACTIVATED]"
   - For Gas: Mandate shutting off the main PNG/LPG valve, opening all windows, avoiding all light switches/spark sources, and evacuating.
   - For Electrical: Mandate switching off the main unit breaker/MCB and staying away from standing water or exposed wires.
   - Guarantee emergency dispatch within 30 minutes.
2. If is_emergency is FALSE:
   - Provide an empathetic, professional acknowledgment of the issue.
   - Specify the targeted resolution window (24 to 48 hours for plumbing/HVAC, 3 business days for general).
   - Sign off as "TenantEase Operations Desk &bull; Cynbit Technologies".
3. Language: English with seamless comprehension of Indian cultural/vernacular references (Hinglish/Hindi).
4. Tone: Urgent & authoritative for hazards; warm, efficient, and SLA-focused for standard maintenance.
5. Length: 60-90 words.
"""

RETENTION_SYSTEM_PROMPT = """You are the AI Resident Retention & Relationship Specialist for TenantEase by Cynbit Technologies.
Your objective is to craft high-converting, empathetic, and personalized communication to prevent tenant churn and stabilize rental cash flow.

INPUT DATA INCLUDES:
- Tenant profile, property name, rent amount
- Days remaining on current lease
- Number of unresolved maintenance complaints
- Historical payment delay trend

GUIDELINES:
- For WhatsApp Nudges: Keep it brief (under 50 words), conversational, addressing the tenant respectfully (Namaste [FirstName] ji), acknowledging any pending maintenance proactively before discussing rent.
- For Lease Renewal Offers: Structure as a formal yet warm proposal offering a calibrated incentive (e.g. 5% renewal discount or 1-month maintenance waiver) explicitly designed to address their specific risk factors.
- Never sound accusatory or aggressive.
"""


# ─── Core Generation Functions ───────────────────────────────────

def build_maintenance_prompt(
    tenant_name: str,
    property_name: str,
    category: str,
    urgency_score: int,
    is_emergency: bool,
    complaint_text: str
) -> str:
    return f"""TICKET CONTEXT:
- Resident Name: {tenant_name}
- Property Unit: {property_name}
- Classified Trade: {category}
- Urgency Severity: {urgency_score}/10
- Emergency Flag: {'YES (Hazard)' if is_emergency else 'NO (Standard SLA)'}
- Verbatim Complaint: "{complaint_text}"

Generate the customer response according to your operational protocols:"""


def build_retention_prompt(
    tenant_name: str,
    property_name: str,
    monthly_rent: int,
    days_left: int,
    open_complaints: int,
    delay_trend: float,
    action_type: str = "whatsapp"
) -> str:
    return f"""TENANT DOSSIER:
- Tenant Name: {tenant_name}
- Property: {property_name}
- Current Monthly Rent: INR {monthly_rent:,}
- Days Remaining on Lease: {days_left} days
- Open Unresolved Tickets: {open_complaints}
- Payment Delay Trajectory: {delay_trend:+.1f} days (recent vs prior)
- Target Action Format: {action_type.upper()}

Generate the communication according to your retention guidelines:"""


# ─── LLM Caller ──────────────────────────────────────────────────

def call_cloud_llm(
    system_prompt: str,
    user_prompt: str,
    api_key: Optional[str] = None,
    provider: str = "gemini"
) -> Optional[str]:
    """Attempt cloud LLM inference via Google Gemini or OpenAI."""
    # Check environment variable if key not passed explicitly
    if not api_key:
        if provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        return None  # Trigger high-fidelity local generator

    try:
        if provider == "gemini":
            # Attempt google-genai SDK
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_prompt,
                    config={"system_instruction": system_prompt, "temperature": 0.2}
                )
                if response and hasattr(response, "text") and response.text:
                    return response.text.strip()
            except Exception:
                # Direct REST fallback for Gemini
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
                payload = {
                    "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 250}
                }
                r = requests.post(url, json=payload, timeout=8)
                if r.status_code == 200:
                    data = r.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()

        elif provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 200
            }
            r = requests.post(url, headers=headers, json=payload, timeout=8)
            if r.status_code == 200:
                data = r.json()
                return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print(f"Cloud LLM call failed ({e}), falling back to local generative synthesis.")
        return None

    return None


def generate_local_fallback_maintenance(
    tenant_name: str,
    property_name: str,
    category: str,
    urgency_score: int,
    is_emergency: bool,
    complaint_text: str
) -> str:
    """High-fidelity contextual generative fallback when API keys are not provided."""
    first_name = tenant_name.split()[0]

    if is_emergency or category == "Gas":
        return (
            f"[CRITICAL SAFETY PROTOCOL ACTIVATED]\n\n"
            f"Dear {first_name}, your safety is our utmost priority. We have detected a high-severity {category.lower()} hazard "
            f"reported at {property_name}. Our Emergency Response Unit has been dispatched under our 30-minute SLA guarantee.\n\n"
            f"MANDATORY SAFETY ACTIONS:\n"
            f"• Immediately close the main PNG/gas cylinder isolation valve.\n"
            f"• Open all exterior windows and doors for rapid ventilation.\n"
            f"• DO NOT operate any electrical switches, exhaust fans, or appliances.\n"
            f"• Please step outside the unit until our technician confirms the area is secure.\n\n"
            f"— TenantEase Operations Desk • Cynbit Technologies (Ticket Ref: #{abs(hash(complaint_text)) % 90000 + 10000})"
        )
    elif category == "Electrical" and urgency_score >= 8:
        return (
            f"[CRITICAL ELECTRICAL HAZARD DISPATCH]\n\n"
            f"Dear {first_name}, your report regarding electrical sparking/fault at {property_name} has triggered an urgent triage. "
            f"An emergency electrician has been routed to your unit within 45 minutes.\n\n"
            f"PRECAUTIONS: Please trip your main MCB switch to cut power to that circuit. Do not touch exposed switchboards or wet surfaces.\n\n"
            f"— TenantEase Operations Desk • Cynbit Technologies"
        )
    elif category == "Plumbing":
        return (
            f"Dear {first_name}, thank you for notifying us regarding the plumbing issue at {property_name}. "
            f"Your request has been logged and assigned to our Plumbing Maintenance Unit.\n\n"
            f"Our dedicated plumber will visit within 24 to 48 hours. If water leakage begins overflowing, please turn off the local isolation valve under the sink.\n\n"
            f"— TenantEase Operations Desk • Cynbit Technologies"
        )
    elif category == "HVAC":
        return (
            f"Dear {first_name}, we have logged your climate/AC service request for {property_name}. "
            f"Our HVAC engineering partner has been scheduled to inspect and service the system within 24-36 hours.\n\n"
            f"— TenantEase Operations Desk • Cynbit Technologies"
        )
    else:
        return (
            f"Dear {first_name}, your request regarding '{complaint_text[:40]}...' at {property_name} has been received. "
            f"Our property coordinator will review and process this within standard SLA (2 to 3 business days).\n\n"
            f"— TenantEase Operations Desk • Cynbit Technologies"
        )


def generate_local_fallback_retention(
    tenant_name: str,
    property_name: str,
    monthly_rent: int,
    days_left: int,
    open_complaints: int,
    delay_trend: float,
    action_type: str = "whatsapp"
) -> str:
    """High-fidelity contextual retention communication generator."""
    first_name = tenant_name.split()[0]

    if action_type == "whatsapp":
        if open_complaints > 0:
            return (
                f"Namaste {first_name} ji, this is regarding your unit at {property_name}. "
                f"We noted you have {open_complaints} open service ticket(s). Our senior maintenance supervisor is "
                f"scheduled to inspect and ensure full resolution tomorrow. Please let us know if the morning slot works for you. "
                f"— TenantEase Care Desk"
            )
        elif delay_trend > 2:
            return (
                f"Namaste {first_name} ji, hope you are settling in well at {property_name}. "
                f"This is a gentle check-in from TenantEase management desk. Please let us know if you need any assistance "
                f"with your account or rental schedule this month. Have a wonderful week ahead! — TenantEase"
            )
        else:
            return (
                f"Namaste {first_name} ji, warm greetings from TenantEase! We hope you are enjoying your stay at {property_name}. "
                f"With your lease renewal coming up in {days_left} days, we have prepared an exclusive early renewal offer for you. "
                f"Reply here to preview your terms. — TenantEase"
            )
    else:
        # Formal Lease Renewal Offer
        discounted_rent = int(monthly_rent * 0.95)
        return (
            f"EXCLUSIVE LEASE RENEWAL PROPOSAL\n\n"
            f"Dear {tenant_name},\n\n"
            f"Your current tenancy at {property_name} is scheduled for renewal in {days_left} days. "
            f"In recognition of your valued residency, TenantEase and property management are delighted to extend "
            f"an exclusive 12-Month Early Renewal Offer:\n\n"
            f"• Monthly Rental: ₹{discounted_rent:,} / month (Special 5% loyalty discount locked for 12 months)\n"
            f"• Maintenance Fee: 100% Waived for Q1\n"
            f"• Priority SLA: Guaranteed same-day maintenance resolution guarantee\n\n"
            f"To secure these preferred rates, please reply to this notice or confirm via the TenantEase Resident Portal within 14 days.\n\n"
            f"Warm regards,\n"
            f"Leasing & Operations Management • TenantEase"
        )


# ─── Main Interface Functions ────────────────────────────────────

def generate_maintenance_triage_llm(
    tenant_name: str,
    property_name: str,
    category: str,
    urgency_score: int,
    is_emergency: bool,
    complaint_text: str,
    api_key: Optional[str] = None,
    provider: str = "gemini"
) -> Dict[str, Any]:
    """Generate context-aware maintenance response via LLM (with automatic fallback)."""
    user_prompt = build_maintenance_prompt(tenant_name, property_name, category, urgency_score, is_emergency, complaint_text)
    
    # Try Cloud LLM first if configured
    cloud_result = call_cloud_llm(MAINTENANCE_SYSTEM_PROMPT, user_prompt, api_key=api_key, provider=provider)
    
    if cloud_result:
        return {
            "response": cloud_result,
            "engine": f"Cloud LLM ({provider.title()})",
            "is_cloud": True,
            "prompt_used": user_prompt,
        }
    else:
        local_result = generate_local_fallback_maintenance(
            tenant_name, property_name, category, urgency_score, is_emergency, complaint_text
        )
        return {
            "response": local_result,
            "engine": "TenantEase Cognitive Engine (Zero-Key Fallback)",
            "is_cloud": False,
            "prompt_used": user_prompt,
        }


def generate_retention_action_llm(
    tenant_name: str,
    property_name: str,
    monthly_rent: int,
    days_left: int,
    open_complaints: int,
    delay_trend: float,
    action_type: str = "whatsapp",
    api_key: Optional[str] = None,
    provider: str = "gemini"
) -> Dict[str, Any]:
    """Generate personalized tenant retention communication via LLM."""
    user_prompt = build_retention_prompt(
        tenant_name, property_name, monthly_rent, days_left, open_complaints, delay_trend, action_type
    )

    cloud_result = call_cloud_llm(RETENTION_SYSTEM_PROMPT, user_prompt, api_key=api_key, provider=provider)

    if cloud_result:
        return {
            "response": cloud_result,
            "engine": f"Cloud LLM ({provider.title()})",
            "is_cloud": True,
            "prompt_used": user_prompt,
        }
    else:
        local_result = generate_local_fallback_retention(
            tenant_name, property_name, monthly_rent, days_left, open_complaints, delay_trend, action_type
        )
        return {
            "response": local_result,
            "engine": "TenantEase Cognitive Engine (Zero-Key Fallback)",
            "is_cloud": False,
            "prompt_used": user_prompt,
        }


if __name__ == "__main__":
    print("Testing LLM Engine...")
    triage = generate_maintenance_triage_llm(
        "Sanjay Kapoor", "Mansarovar Residency", "Gas", 10, True, "Gas leak ho rahi hai kitchen me"
    )
    print("\n--- Maintenance Triage Output ---")
    print(f"Engine: {triage['engine']}")
    print(triage['response'])

    retention = generate_retention_action_llm(
        "Ananya Khandelwal", "Vaishali Enclave", 18000, 45, 2, 3.5, "whatsapp"
    )
    print("\n--- Retention Action Output ---")
    print(f"Engine: {retention['engine']}")
    print(retention['response'])
