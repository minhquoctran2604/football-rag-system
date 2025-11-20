import json
import os
from pathlib import Path
from src.utils.supabase_client import create_client, Client
import dotenv
from src.utils.gemini_client import GeminiClient
import logging
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

gemini = GeminiClient()
dotenv.load_dotenv()

BATCH_SIZE = 100
EMBEDDING_BATCH_SIZE = 10


def gen_team_bio(team_data: dict) -> str:
    # Fix: access metadata nested structure from JSONL
    meta = team_data.get("metadata", {})
    identity = meta.get("identity", {})
    venue = meta.get("venue", {})
    current_league_raw = meta.get("current_league")

    # Parse current_league: string → dict or keep as-is
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
    return bio[:200]


def prepare_before_upsert(team_data: dict) -> dict:
    meta = team_data.get("metadata", {})
    identity = meta.get("identity", {})
    current_league_raw = meta.get("current_league")

    if isinstance(current_league_raw, dict):
        current_league = current_league_raw
    else:
        current_league = {"name": current_league_raw or "unknown"}

    return {
        "team_id": team_data.get("team_id", "unknown"),
        "name": team_data.get("name") or identity.get("full_name", "unknown"),
        "country": identity.get("country"),
        "founded_year": identity.get("founded_year"),
        "current_league": current_league.get("name", "unknown"),
        "current_league_id": current_league.get("league_id", "unknown"),
        "metadata": meta,  # store metadata object
    }


def generate_embeddings_batch(team_bios: List[str]) -> List[List[float]]:
    embeddings = []
    for bio in team_bios:
        try:
            embedding = gemini.get_embedding(bio)
            embeddings.append(embedding)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            embeddings.append([0.0] * 768)
    return embeddings


def upsert_to_supabase(
    supabase_client: Client,
    records: list,
    table_name: str,
    batch_size: int = BATCH_SIZE,
):

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]

        try:
            response = supabase_client.table(table_name).upsert(batch).execute()
            logger.info(
                f"Successfully upserted batch {i//batch_size + 1}, count: {len(batch)}"
            )
        except Exception as e:
            logger.error(f"Error upserting batch {i//batch_size + 1}: {str(e)}")


def main():

    file_path = (
        r"D:\Document\RAG_Football\data\teams\team_complete_metadata_for_supabase.jsonl"
    )

    logger.info("Reading team data...")
    raw_teams = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            raw_teams.append(json.loads(line))

    logger.info(f"Loaded {len(raw_teams)} raw teams for processing")

    logger.info("Generating team bios in batches...")
    team_bios = [gen_team_bio(team) for team in raw_teams]

    logger.info("Generating embeddings in batches...")
    all_embeddings = []
    for i in range(0, len(team_bios), EMBEDDING_BATCH_SIZE):
        bio_batch = team_bios[i : i + EMBEDDING_BATCH_SIZE]
        logger.info(f"Processing embedding batch {i//EMBEDDING_BATCH_SIZE + 1}")
        embeddings_batch = generate_embeddings_batch(bio_batch)
        all_embeddings.extend(embeddings_batch)

    teams_data = []
    for i, raw_team in enumerate(raw_teams):
        prepared = prepare_before_upsert(raw_team)
        prepared["embedding"] = all_embeddings[i]
        teams_data.append(prepared)

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        logger.error("Supabase URL or key not found in environment variables")
        return

    supabase = create_client(supabase_url, supabase_key)

    logger.info("Starting upsert to Supabase...")
    try:
        upsert_to_supabase(supabase, teams_data, "teams")
    except Exception as e:
        logger.error(f"Error during upsert operation: {str(e)}")
        raise
    finally:
        import time

        time.sleep(1)

    logger.info("Upsert completed")


if __name__ == "__main__":
    main()
