# !pip install sentence-transformers supabase python-dotenv tqdm

import json
import os
from typing import Dict, Any
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from tqdm import tqdm
import torch
# CONFIGURATION
SUPABASE_URL = "https://cyupadrdftndslrvmays.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImN5dXBhZHJkZnRuZHNscnZtYXlzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI1Mjg0ODAsImV4cCI6MjA3ODEwNDQ4MH0.rFqsYXQGsuFmtBEjfqiIlNEVT5sHlVXo8-d_S84wzxo"
INPUT_FILE = "/content/drive/MyDrive/players_finals.jsonl"  
if "YOUR_SUPABASE" in SUPABASE_URL:
    print("WARNING: Please replace SUPABASE_URL and SUPABASE_KEY with your actual credentials!")
def make_player_document(p: Dict[str, Any]) -> str:
    """Generates a rich text biography for the player."""
    identity = p.get('identity', {})
    stats = p.get('stats', {})
    
    name = p.get('name', 'Unknown')
    full_name = identity.get('full_name', name)
    nationality = identity.get('nationality', 'Unknown')
    birth_year = identity.get('birth_year', 'Unknown')
    position = identity.get('position', 'Unknown')
    height = identity.get('height', 'Unknown')
    
    current_club = p.get('current_club', 'Unknown')
    league = p.get('current_league', 'Unknown')
    
    goals = stats.get('goals', 0)
    assists = stats.get('assists', 0)
    matches = stats.get('matches_played', 0)
    
    doc = f"""
Player: {name}
Full Name: {full_name}
Nationality: {nationality}
Position: {position}
Born: {birth_year}
Height: {height}
Current Club: {current_club}
League: {league}
Career Stats:
- Matches: {matches}
- Goals: {goals}
- Assists: {assists}
    """.strip()
    
    return doc
def prepare_record(player_data: Dict[str, Any]) -> Dict[str, Any]:
    current_league = player_data.get("current_league")
    nationality = player_data.get("identity", {}).get("nationality") if player_data.get("identity") else None
    position = player_data.get("identity", {}).get("position") if player_data.get("identity") else None
    metadata = player_data.copy()
    metadata.pop("current_league", None)
    if "identity" in metadata:
        if "nationality" in metadata["identity"]: metadata["identity"].pop("nationality", None)
        if "position" in metadata["identity"]: metadata["identity"].pop("position", None)
    document = make_player_document(player_data)
    record = {
        "player_id": player_data["entity_id"],  
        "name": player_data.get("name", "unknown"),
        "current_league": current_league,
        "nationality": nationality,
        "birth_year": player_data.get('identity', 'unknown').get('birth_year','unknown'),
        "position": position,
        "metadata": metadata,
        "document": document, 
        "current_team_id": player_data.get("current_club_id", "unknown")  
    }
    return record
def main():
    print(f"Loading {INPUT_FILE}...")
    records = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records)} raw records.")
    print("Generating biographies...")
    prepared_records = [prepare_record(r) for r in records]
    print("Loading Embedding Model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    print(f"Loading model: {model_name} (768 dimensions, Multilingual)...")
    model = SentenceTransformer(model_name, device=device)
    documents = [r['document'] for r in prepared_records]
    
    print("Generating embeddings...")
    embeddings = model.encode(documents, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    for i, record in enumerate(prepared_records):
        record['embedding'] = embeddings[i].tolist()
    print("Connecting to Supabase...")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    batch_size = 100
    total = len(prepared_records)
    print(f"Upserting {total} records to 'players' table...")
    
    for i in range(0, total, batch_size):
        batch = prepared_records[i:i + batch_size]
        try:
            supabase.table("players").upsert(batch).execute()
            print(f"Batch {i//batch_size + 1} done.")
        except Exception as e:
            print(f"Error batch {i//batch_size + 1}: {e}")
    print("Data is live on Supabase.")
if __name__ == "__main__":
    main()