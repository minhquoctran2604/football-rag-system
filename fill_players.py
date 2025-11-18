import json
import os
from typing import Dict, Any
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

def load_jsonl_file(file_path: str):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def prepare_record_for_upsert(player_data: Dict[str, Any]) -> Dict[str, Any]:
    current_league = player_data.get("current_league")
    nationality = player_data.get("identity", {}).get("nationality") if player_data.get("identity") else None
    position = player_data.get("identity", {}).get("position") if player_data.get("identity") else None

    metadata = player_data.copy()

    metadata.pop("current_league", None)
    if "identity" in metadata and "nationality" in metadata["identity"]:
        metadata["identity"].pop("nationality", None)
    if "identity" in metadata and "position" in metadata["identity"]:
        metadata["identity"].pop("position", None)

    record = {
        "player_id": player_data["entity_id"],  
        "name": player_data.get("name", "unknown"),
        "current_league": current_league,
        "nationality": nationality,
        "birth_year": player_data.get('identity', 'unknown').get('birth_year','unknown'),
        "position": position,
        "metadata": metadata,
        "document": player_data.get('biography', 'unknown'), 
        "current_team_id": player_data.get("current_club_id", "unknown")  
    }

    return record

def upsert_to_supabase(supabase_client: Client, records: list, table_name: str):
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]

        try:
            response = supabase_client.table(table_name).upsert(batch).execute()
            print(f"Đã upsert thành công batch {i//batch_size + 1}, số lượng: {len(batch)}")
        except Exception as e:
            print(f"Lỗi khi upsert batch {i//batch_size + 1}: {str(e)}")

def main():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("Please set SUPABASE_URL and SUPABASE_KEY environment variables")

    supabase: Client = create_client(url, key)

    jsonl_file_path = "./data/players/players_finals.jsonl"

    print("Loading data from JSONL file...")
    jsonl_data = load_jsonl_file(jsonl_file_path)

    print(f"Preparing {len(jsonl_data)} records...")
    records_to_upsert = []

    for player_data in jsonl_data:
        record = prepare_record_for_upsert(player_data)
        records_to_upsert.append(record)

    table_name = "players"

    print(f"Upserting {len(records_to_upsert)} records into table {table_name}...")
    upsert_to_supabase(supabase, records_to_upsert, table_name)

    print("Data upsert process completed!")

if __name__ == "__main__":
    main()