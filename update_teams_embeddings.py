import json
import os
import logging
from typing import List
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client 

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


try:
    import dotenv
    dotenv.load_dotenv(dotenv_path='/content/drive/MyDrive/.env')
except ImportError:
    pass


BATCH_SIZE = 50

INPUT_FILE = '/content/drive/MyDrive/team_complete_metadata_for_supabase.jsonl'

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

def gen_team_bio(team_data: dict) -> str:
    """
    Re-using the logic from upload_teams_to_supabase.py
    to ensure consistency.
    """
    meta = team_data.get("metadata", {})
    identity = meta.get("identity", {})
    venue = meta.get("venue", {})
    current_league_raw = meta.get("current_league")

    if isinstance(current_league_raw, dict):
        current_league = current_league_raw
    else:
        current_league = {"name": current_league_raw or "unknown"}

    name = identity.get("full_name", team_data.get("name", "unknown"))
    country = identity.get("country", "unknown")
    founded_year = identity.get("founded_year", "unknown")
    city = venue.get("city", "unknown")
    stadium = venue.get("stadium_name", "unknown")
    league = current_league.get("name", "unknown")

    bio = f"{name} is a football club from {city}, {country}. Founded in {founded_year}, they compete in {league} and play their home matches at {stadium}."
    return bio[:200] # Keeping the truncation as per original script

def main():
    # 1. Load Model
    logger.info(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 2. Load Data
    logger.info(f"Reading data from {INPUT_FILE}...")
    records = []
    if not os.path.exists(INPUT_FILE):
        logger.error(f"File not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} teams.")

    # 3. Process & Embed
    logger.info("Generating embeddings...")
    
    updates = []
    bios = []
    
    for record in records:
        # Re-generate bio
        bio = gen_team_bio(record)
        bios.append(bio)
        
        # Prepare upsert payload
        updates.append({
            "team_id": record.get("team_id"),
            "name": record.get("name", "Unknown"), # Fix: Add name to satisfy NOT NULL constraint
            "document": bio, 
            # embedding added later
        })

    # Generate embeddings
    embeddings = model.encode(bios, batch_size=32, show_progress_bar=True)
    
    # Attach embeddings
    for i, update in enumerate(updates):
        update["embedding"] = embeddings[i].tolist()

    # 4. Upsert to Supabase
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        logger.error("Missing Supabase credentials. Check .env file.")
        return

    supabase = create_client(supabase_url, supabase_key)
    
    logger.info(f"Upserting {len(updates)} records to 'teams' table...")
    
    for i in range(0, len(updates), BATCH_SIZE):
        batch = updates[i : i + BATCH_SIZE]
        try:
            supabase.table("teams").upsert(batch).execute()
            logger.info(f"Batch {i//BATCH_SIZE + 1} done.")
        except Exception as e:
            logger.error(f"Error upserting batch: {e}")

    print("✅ All team embeddings updated successfully!")

if __name__ == "__main__":
    main()
