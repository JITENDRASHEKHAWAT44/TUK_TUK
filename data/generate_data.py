"""
Synthetic Data Generator for TenantEase AI PoC
Generates realistic tenant, property, payment, and maintenance data.
"""

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

np.random.seed(42)
random.seed(42)

# Indian Names & Contact Generator (zero dependency)
FIRST_NAMES = [
    "Aarav", "Aditi", "Amit", "Ananya", "Arjun", "Deepak", "Devansh", "Divya",
    "Gaurav", "Harsh", "Ishaan", "Kavya", "Manish", "Meera", "Mohit", "Neha",
    "Nikhil", "Pooja", "Pranav", "Priya", "Rahul", "Rakesh", "Riya", "Rohan",
    "Rohit", "Sakshi", "Sanjay", "Shreya", "Siddharth", "Sneha", "Suresh", "Tanvi",
    "Utkarsh", "Varun", "Vikas", "Vikram", "Yash", "Zoya", "Aniket", "Bhavna"
]

LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Mehta", "Singh", "Jain", "Agarwal", "Choudhary",
    "Mishra", "Patel", "Reddy", "Nair", "Saxena", "Bhatia", "Malhotra", "Kapoor",
    "Rathore", "Shekhawat", "Khandelwal", "Mittal", "Goyal", "Yadav", "Trivedi"
]

def generate_indian_name():
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    return f"{first} {last}"

def generate_indian_phone():
    prefix = random.choice(["98", "97", "99", "94", "80", "88", "70", "75"])
    rest = "".join([str(random.randint(0, 9)) for _ in range(8)])
    return f"+91-{prefix}{rest}"

def generate_email(name):
    clean = name.lower().replace(" ", ".")
    domain = random.choice(["gmail.com", "outlook.com", "yahoo.in", "cynbit.in", "tcs.com", "infosys.com"])
    return f"{clean}{random.randint(10, 99)}@{domain}"


# ─── Configuration ───────────────────────────────────────────────
NUM_PROPERTIES = 25
NUM_TENANTS = 80
PAYMENT_MONTHS = 12  # 12 months of history
NUM_MAINTENANCE_REQUESTS = 200

JAIPUR_AREAS = [
    "Mansarovar", "Vaishali Nagar", "Malviya Nagar", "Jagatpura",
    "Pratap Nagar", "Tonk Road", "C-Scheme", "Raja Park",
    "Sodala", "Sanganer", "Sitapura", "Vidhyadhar Nagar",
    "Bani Park", "Durgapura", "Ajmer Road"
]

PROPERTY_TYPES = ["1BHK", "2BHK", "3BHK", "Studio", "1RK"]

RENT_RANGES = {
    "1RK": (5000, 8000),
    "Studio": (6000, 10000),
    "1BHK": (8000, 15000),
    "2BHK": (12000, 22000),
    "3BHK": (18000, 35000),
}

MAINTENANCE_CATEGORIES = {
    "Plumbing": [
        "Bathroom me paani nahi aa raha hai",
        "Kitchen sink leaking badly need help",
        "Toilet flush not working since morning",
        "Geyser se paani nahi aa raha",
        "Water pipe burst in bathroom flooding",
        "Tap se paani drip ho raha hai continuously",
        "Drainage blocked in bathroom, bad smell",
        "Washbasin pipe broken water everywhere",
        "Hot water not coming in shower",
        "Overhead tank overflowing wasting water",
        "Toilet seat is broken need replacement",
        "Water pressure very low in kitchen tap",
        "Bathroom floor drain is completely blocked",
        "Pipe fitting is loose and water is spraying",
        "Commode is cracked and leaking from base",
    ],
    "Electrical": [
        "Power cut in my flat only neighbors have light",
        "Switch board me spark aa rahi hai very dangerous",
        "Ceiling fan not working making grinding noise",
        "Bijli ka bill bahut zyada aa raha hai check meter",
        "MCB tripping again and again every few minutes",
        "Light fitting fell from ceiling needs urgent fix",
        "Inverter not charging battery properly",
        "Exhaust fan stopped working in kitchen",
        "Doorbell not working please fix",
        "Tube light flickering continuously in bedroom",
        "One room has no power supply socket dead",
        "Wiring is exposed in the hallway dangerous",
        "Plug point is sparking when I connect charger",
        "All lights went off suddenly fuse problem",
        "Meter reading seems wrong bill too high",
    ],
    "HVAC": [
        "AC not cooling at all room is very hot",
        "AC se paani gir raha hai floor pe",
        "AC making loud rattling noise cant sleep",
        "Heater not working and its very cold",
        "AC remote not working need new one",
        "AC compressor is making clicking sound",
        "Split AC outdoor unit is vibrating badly",
        "Window AC is blowing warm air only",
        "AC is leaking water inside the room",
        "Cooler motor has stopped working",
        "AC gas needs refilling not cooling properly",
        "Room heater tripping the MCB every time",
        "AC smell is very bad musty odor",
        "Thermostat not responding to temperature changes",
        "Ceiling fan speed controller not working",
    ],
    "Structural": [
        "Wall me crack aa gayi hai damp bhi hai",
        "Ceiling se paint utar raha hai peeling off",
        "Door lock broken cant lock my flat",
        "Window glass cracked during storm",
        "Seepage in wall during rain very bad",
        "Floor tiles broken in living room",
        "Balcony railing is loose and shaking",
        "Main door hinge broken door not closing",
        "Bathroom tiles lifting from floor",
        "Plaster falling from ceiling in bedroom",
        "Kitchen platform slab is cracked",
        "Window frame is rusted and stuck",
        "Terrace waterproofing has failed leaking below",
        "Staircase railing broken in common area",
        "Wall paint is peeling due to moisture",
    ],
    "Gas": [
        "Gas leak ho rahi hai kitchen me bahut smell aa rahi hai",
        "Gas pipeline se leak hai please come urgently",
        "Gas stove ignition not working have to use matchstick",
        "PNG meter showing wrong reading bill too high",
        "Gas regulator making hissing sound seems dangerous",
        "Strong gas smell in kitchen even when stove is off",
        "Gas pipe connection is loose at the joint",
        "Gas burner flame is yellow instead of blue",
        "Smell of gas coming from pipeline in wall",
        "Gas supply suddenly stopped not coming at all",
    ],
    "Pest": [
        "Cockroach infestation in kitchen bahut zyada hai",
        "Termite damage in wooden cupboard doors",
        "Rats have entered the flat from somewhere",
        "Mosquito problem is unbearable need solution",
        "Bed bugs in mattress cant sleep at night",
        "Ants colony in bathroom wall near basin",
        "Pigeons nesting in balcony making mess",
        "Lizards everywhere in the flat too many",
        "Spiders making webs in all corners",
        "Bees have made hive near window dangerous",
    ],
    "General": [
        "Need rent receipt for last 3 months for IT filing",
        "Parking space issue neighbor taking my spot",
        "Society maintenance charge seems too high",
        "Need NOC for address proof for passport",
        "Lift not working in building for 3 days",
        "Common area lights not working at night dark",
        "Water tanker schedule is irregular",
        "Security guard is not present at night",
        "Garbage collection not happening daily",
        "CCTV cameras in lobby are not working",
        "Need duplicate key for main door",
        "Intercom system not functioning",
        "Nameplate needs to be changed on door",
        "Water purifier RO service is needed",
        "WiFi router placement needs to be changed",
    ],
}

URGENCY_MAP = {
    "Gas": 10,
    "Electrical": 7,
    "Plumbing": 6,
    "HVAC": 4,
    "Structural": 5,
    "Pest": 3,
    "General": 2,
}


def generate_properties():
    """Generate property data."""
    properties = []
    for i in range(1, NUM_PROPERTIES + 1):
        prop_type = random.choice(PROPERTY_TYPES)
        area = random.choice(JAIPUR_AREAS)
        rent_low, rent_high = RENT_RANGES[prop_type]
        base_rent = random.randint(rent_low // 500, rent_high // 500) * 500

        properties.append({
            "property_id": f"PROP-{i:03d}",
            "property_name": f"{random.choice(LAST_NAMES)} {random.choice(['Residency', 'Apartments', 'Tower', 'Enclave', 'Villa'])}",
            "area": area,
            "city": "Jaipur",
            "property_type": prop_type,
            "floor": random.randint(0, 8),
            "total_floors": random.randint(3, 12),
            "furnishing": random.choice(["Unfurnished", "Semi-Furnished", "Fully-Furnished"]),
            "building_age_years": random.randint(1, 25),
            "monthly_rent": base_rent,
            "deposit": base_rent * random.choice([2, 3, 6, 10]),
        })
    return pd.DataFrame(properties)


def generate_tenants(properties_df):
    """Generate tenant data with behavioral profiles."""
    tenants = []
    available_props = list(properties_df["property_id"].values)
    random.shuffle(available_props)

    for i in range(1, NUM_TENANTS + 1):
        # Assign behavioral profile (determines payment/churn patterns)
        profile = random.choices(
            ["reliable", "moderate", "risky", "churner"],
            weights=[0.40, 0.30, 0.20, 0.10],
            k=1
        )[0]

        days_ago = random.randint(90, 720)
        lease_start = (datetime.now() - timedelta(days=days_ago)).date()
        lease_months = random.choice([6, 11, 12, 24])
        lease_end = lease_start + timedelta(days=lease_months * 30)

        prop_id = available_props[i % len(available_props)]
        prop = properties_df[properties_df["property_id"] == prop_id].iloc[0]
        tenant_name = generate_indian_name()

        tenants.append({
            "tenant_id": f"TNT-{i:03d}",
            "tenant_name": tenant_name,
            "phone": generate_indian_phone(),
            "email": generate_email(tenant_name),
            "property_id": prop_id,
            "monthly_rent": prop["monthly_rent"],
            "lease_start": lease_start,
            "lease_end": lease_end,
            "lease_duration_months": lease_months,
            "deposit_paid": prop["deposit"],
            "profile": profile,  # hidden field — drives behavior
            "age": random.randint(22, 55),
            "occupation": random.choice(["IT Professional", "Business Owner", "Student", "Government Employee", "Freelancer", "Doctor", "Teacher", "Startup Founder"]),
        })
    return pd.DataFrame(tenants)


def generate_payments(tenants_df):
    """Generate 12 months of payment history per tenant."""
    payments = []
    base_date = datetime(2025, 11, 1)  # Start from Nov 2025

    for _, tenant in tenants_df.iterrows():
        profile = tenant["profile"]

        for month_offset in range(PAYMENT_MONTHS):
            due_date = base_date + timedelta(days=30 * month_offset)
            rent = tenant["monthly_rent"]

            # Determine delay based on profile
            if profile == "reliable":
                delay = max(0, int(np.random.normal(0, 1)))
                paid = True
            elif profile == "moderate":
                delay = max(0, int(np.random.normal(3, 3)))
                paid = random.random() > 0.03
            elif profile == "risky":
                delay = max(0, int(np.random.normal(8, 5)))
                paid = random.random() > 0.10
                # Risky tenants tend to get worse over time
                delay += month_offset // 3
            else:  # churner
                delay = max(0, int(np.random.normal(5, 4)))
                paid = random.random() > 0.05
                # Churners get progressively worse
                if month_offset > 8:
                    delay += random.randint(3, 10)

            if not paid:
                delay = None  # Unpaid
                payment_date = None
                amount_paid = 0
            else:
                payment_date = due_date + timedelta(days=delay)
                # Partial payments sometimes
                if profile in ["risky", "churner"] and random.random() < 0.15:
                    amount_paid = rent * random.choice([0.5, 0.75, 0.8])
                else:
                    amount_paid = rent

            payments.append({
                "payment_id": f"PAY-{len(payments)+1:05d}",
                "tenant_id": tenant["tenant_id"],
                "property_id": tenant["property_id"],
                "due_date": due_date.strftime("%Y-%m-%d"),
                "payment_date": payment_date.strftime("%Y-%m-%d") if payment_date else None,
                "amount_due": rent,
                "amount_paid": amount_paid,
                "delay_days": delay if delay is not None else -1,
                "is_paid": paid,
                "is_late": delay > 3 if delay is not None else True,
                "month": due_date.strftime("%Y-%m"),
            })

    return pd.DataFrame(payments)


def generate_maintenance_requests(tenants_df):
    """Generate maintenance request data."""
    requests = []

    for i in range(NUM_MAINTENANCE_REQUESTS):
        tenant = tenants_df.sample(1).iloc[0]
        category = random.choices(
            list(MAINTENANCE_CATEGORIES.keys()),
            weights=[20, 15, 15, 12, 5, 13, 20],
            k=1
        )[0]

        complaint_text = random.choice(MAINTENANCE_CATEGORIES[category])

        # Add some noise — sometimes tenants describe multiple issues
        if random.random() < 0.2:
            other_cat = random.choice(list(MAINTENANCE_CATEGORIES.keys()))
            extra = random.choice(MAINTENANCE_CATEGORIES[other_cat])
            complaint_text += f". Also, {extra.lower()}"

        base_urgency = URGENCY_MAP[category]
        # Emergency keywords boost urgency
        emergency_words = ["urgent", "dangerous", "emergency", "flooding", "gas leak", "fire", "spark", "burst"]
        has_emergency = any(word in complaint_text.lower() for word in emergency_words)
        urgency = min(10, base_urgency + (2 if has_emergency else 0) + random.randint(-1, 1))

        days_ago = random.randint(1, 365)
        created_date = (datetime.now() - timedelta(days=days_ago)).date()
        resolved = random.random() < 0.75
        resolution_days = random.randint(0, 14) if resolved else None

        # Risky/churner tenants file more complaints
        if tenant["profile"] in ["risky", "churner"] and random.random() < 0.4:
            resolved = False
            resolution_days = None

        requests.append({
            "request_id": f"MNT-{i+1:04d}",
            "tenant_id": tenant["tenant_id"],
            "property_id": tenant["property_id"],
            "category": category,
            "complaint_text": complaint_text,
            "urgency_score": urgency,
            "created_date": created_date.strftime("%Y-%m-%d"),
            "is_resolved": resolved,
            "resolution_days": resolution_days,
            "is_emergency": has_emergency or category == "Gas",
        })

    return pd.DataFrame(requests)


def compute_tenant_features(tenants_df, payments_df, maintenance_df):
    """
    Compute aggregated features per tenant for ML models.
    This is the feature engineering step.
    """
    features = []

    for _, tenant in tenants_df.iterrows():
        tid = tenant["tenant_id"]

        # Payment features
        t_payments = payments_df[payments_df["tenant_id"] == tid]
        total_payments = len(t_payments)
        paid_payments = t_payments[t_payments["is_paid"] == True]
        late_payments = t_payments[t_payments["is_late"] == True]

        avg_delay = paid_payments["delay_days"].mean() if len(paid_payments) > 0 else 0
        max_delay = paid_payments["delay_days"].max() if len(paid_payments) > 0 else 0
        late_rate = len(late_payments) / total_payments if total_payments > 0 else 0
        payment_rate = len(paid_payments) / total_payments if total_payments > 0 else 0

        # Recent trend (last 3 months vs earlier)
        recent = t_payments.tail(3)
        earlier = t_payments.head(max(1, total_payments - 3))
        recent_avg_delay = recent[recent["is_paid"] == True]["delay_days"].mean() if len(recent[recent["is_paid"] == True]) > 0 else 0
        earlier_avg_delay = earlier[earlier["is_paid"] == True]["delay_days"].mean() if len(earlier[earlier["is_paid"] == True]) > 0 else 0
        delay_trend = recent_avg_delay - earlier_avg_delay  # positive = getting worse

        # Partial payment count
        partial_payments = t_payments[(t_payments["is_paid"] == True) & (t_payments["amount_paid"] < t_payments["amount_due"])]
        partial_rate = len(partial_payments) / total_payments if total_payments > 0 else 0

        # Maintenance features
        t_maint = maintenance_df[maintenance_df["tenant_id"] == tid]
        total_complaints = len(t_maint)
        unresolved_complaints = len(t_maint[t_maint["is_resolved"] == False])
        emergency_complaints = len(t_maint[t_maint["is_emergency"] == True])
        avg_urgency = t_maint["urgency_score"].mean() if len(t_maint) > 0 else 0

        # Lease features
        today = datetime.now().date()
        lease_end = pd.to_datetime(tenant["lease_end"]).date()
        days_until_lease_end = (lease_end - today).days
        lease_duration = tenant["lease_duration_months"]
        tenure_months = max(1, (today - pd.to_datetime(tenant["lease_start"]).date()).days // 30)

        # Churn label — based on profile + behavior signals
        churn_probability = 0.0
        if tenant["profile"] == "churner":
            churn_probability = 0.85 + random.uniform(-0.1, 0.1)
        elif tenant["profile"] == "risky":
            churn_probability = 0.55 + random.uniform(-0.15, 0.15)
        elif tenant["profile"] == "moderate":
            churn_probability = 0.25 + random.uniform(-0.1, 0.1)
        else:
            churn_probability = 0.08 + random.uniform(-0.05, 0.08)

        # Adjust churn based on complaints
        churn_probability += unresolved_complaints * 0.03
        churn_probability = min(1.0, max(0.0, churn_probability))
        will_churn = 1 if churn_probability > 0.5 else 0

        # Late payment prediction label (next month)
        if tenant["profile"] == "reliable":
            will_pay_late = 1 if random.random() < 0.08 else 0
        elif tenant["profile"] == "moderate":
            will_pay_late = 1 if random.random() < 0.35 else 0
        elif tenant["profile"] == "risky":
            will_pay_late = 1 if random.random() < 0.72 else 0
        else:
            will_pay_late = 1 if random.random() < 0.60 else 0

        features.append({
            "tenant_id": tid,
            "tenant_name": tenant["tenant_name"],
            "property_id": tenant["property_id"],
            "monthly_rent": tenant["monthly_rent"],
            "occupation": tenant["occupation"],
            "age": tenant["age"],
            # Payment features
            "avg_delay_days": round(avg_delay, 1),
            "max_delay_days": int(max_delay),
            "late_payment_rate": round(late_rate, 3),
            "payment_rate": round(payment_rate, 3),
            "delay_trend": round(delay_trend, 1),
            "partial_payment_rate": round(partial_rate, 3),
            # Maintenance features
            "total_complaints": total_complaints,
            "unresolved_complaints": unresolved_complaints,
            "emergency_complaints": emergency_complaints,
            "avg_complaint_urgency": round(avg_urgency, 1),
            # Lease features
            "days_until_lease_end": days_until_lease_end,
            "lease_duration_months": lease_duration,
            "tenure_months": tenure_months,
            # Labels
            "will_pay_late_next_month": will_pay_late,
            "will_churn": will_churn,
            "churn_probability": round(churn_probability, 3),
        })

    return pd.DataFrame(features)


def generate_all_data(output_dir="data"):
    """Generate all datasets and save to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    print("Generating properties...")
    properties = generate_properties()
    properties.to_csv(os.path.join(output_dir, "properties.csv"), index=False)
    print(f"  -> {len(properties)} properties")

    print("Generating tenants...")
    tenants = generate_tenants(properties)
    tenants.to_csv(os.path.join(output_dir, "tenants.csv"), index=False)
    print(f"  -> {len(tenants)} tenants")

    print("Generating payment history...")
    payments = generate_payments(tenants)
    payments.to_csv(os.path.join(output_dir, "payments.csv"), index=False)
    print(f"  -> {len(payments)} payment records")

    print("Generating maintenance requests...")
    maintenance = generate_maintenance_requests(tenants)
    maintenance.to_csv(os.path.join(output_dir, "maintenance_requests.csv"), index=False)
    print(f"  -> {len(maintenance)} maintenance requests")

    print("Computing tenant features...")
    tenant_features = compute_tenant_features(tenants, payments, maintenance)
    tenant_features.to_csv(os.path.join(output_dir, "tenant_features.csv"), index=False)
    print(f"  -> {len(tenant_features)} tenant feature rows")

    print("\n[OK] All data generated successfully!")
    return properties, tenants, payments, maintenance, tenant_features


if __name__ == "__main__":
    generate_all_data()
