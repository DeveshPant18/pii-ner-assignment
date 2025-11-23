import json
import random
from faker import Faker
from num2words import num2words

fake = Faker()

def noisify_text(text, p_num=0.4):
    text = str(text).lower()
    text = text.replace(".", " dot ").replace("@", " at ").replace("-", " ").replace(":", " ")
    
    words = []
    for word in text.split():
        if word.isdigit() and random.random() < p_num:
            try:
                # Keep years as digits (1990-2030)
                if len(word) == 4 and (1900 < int(word) < 2030):
                    pass 
                else:
                    word = num2words(int(word)).replace("-", " ").replace(",", "")
            except:
                pass
        words.append(word)
    
    cleaned = "".join([c for c in " ".join(words) if c.isalnum() or c.isspace()])
    return " ".join(cleaned.split())

def generate_entry(uid):
    profile = fake.profile()
    
    name = profile['name']
    email = profile['mail']
    phone = fake.phone_number()
    card = fake.credit_card_number()
    date = str(fake.date()) 
    city = fake.city()
    location = fake.street_address() + " " + fake.secondary_address()

    n_name = noisify_text(name, p_num=0.0) 
    n_email = noisify_text(email, p_num=0.3)
    n_phone = noisify_text(phone, p_num=0.5)
    n_card = noisify_text(card, p_num=0.5)
    n_date = noisify_text(date, p_num=0.5)
    n_city = noisify_text(city, p_num=0.0)
    n_loc = noisify_text(location, p_num=0.4)

    templates = [
        # --- EXISTING MIXED ---
        (f"my name is {n_name} and my email is {n_email}", [("PERSON_NAME", n_name), ("EMAIL", n_email)]),
        (f"call me at {n_phone} regarding the issue", [("PHONE", n_phone)]),
        (f"i was born on {n_date} in {n_city}", [("DATE", n_date), ("CITY", n_city)]),
        (f"verify account {n_email}", [("EMAIL", n_email)]),
        (f"schedule for {n_date} is confirmed", [("DATE", n_date)]),
        (f"meet me at {n_loc} for the delivery", [("LOCATION", n_loc)]),
        (f"delivery to {n_loc} in {n_city}", [("LOCATION", n_loc), ("CITY", n_city)]),
        
        # --- NAMES ---
        (f"hello this is {n_name} speaking", [("PERSON_NAME", n_name)]),
        (f"account holder is {n_name}", [("PERSON_NAME", n_name)]),
        
        # --- PHONE (Boosted) ---
        (f"reach me at {n_phone}", [("PHONE", n_phone)]),
        (f"my number is {n_phone}", [("PHONE", n_phone)]),
        
        # --- CREDIT CARD (HEAVILY BOOSTED) ---
        (f"my credit card is {n_card}", [("CREDIT_CARD", n_card)]),
        (f"payment of {n_card} declined", [("CREDIT_CARD", n_card)]),
        (f"charging card number {n_card}", [("CREDIT_CARD", n_card)]),
        (f"card {n_card} has expired", [("CREDIT_CARD", n_card)]),
        (f"use card {n_card} for this", [("CREDIT_CARD", n_card)])
    ]
    
    final_text, entity_list = random.choice(templates)
    
    final_text = final_text.lower()
    final_entities = []
    for label, val in entity_list:
        val = val.lower()
        start = final_text.find(val)
        if start != -1:
            final_entities.append({
                "start": start,
                "end": start + len(val),
                "label": label
            })
            
    return {
        "id": f"utt_{uid}",
        "text": final_text,
        "entities": final_entities
    }

print("Generating final boosted data...")
with open("data/train.jsonl", "w", encoding="utf-8") as f:
    for i in range(1200):
        f.write(json.dumps(generate_entry(i)) + "\n")

with open("data/dev.jsonl", "w", encoding="utf-8") as f:
    for i in range(200):
        f.write(json.dumps(generate_entry(i+1200)) + "\n")
print("Done.")