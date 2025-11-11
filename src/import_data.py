import json
import os
from supabase import create_client
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

def make_player_document(p):
    identity = p.get('identity', {})
    stats = p.get('season_stats', {})
    
    doc = f"""
Name: {p.get('name', 'Unknown')}
Position: {identity.get('position', 'N/A')}
Nationality: {identity.get('nationality', 'N/A')}
Birth Year: {identity.get('birth_year', 'N/A')}
Height: {identity.get('height_cm', 'N/A')} cm
Preferred Foot: {identity.get('preferred_foot', 'N/A')}

Current Club: {p.get('current_club', 'N/A')}
Current League: {p.get('current_league', 'N/A')}
Season: {p.get('current_season', 'N/A')}

Season Stats:
- Matches: {stats.get('matches', 0)}
- Goals: {stats.get('goals', 0)}
- Assists: {stats.get('assists', 0)}
- Minutes: {stats.get('minutes', 0)}

Biography: {p.get('biography', 'N/A')}
    """.strip()
    
    return doc

def make_team_document(t):
    identity = t.get('identity', {})
    venue = t.get('venue', {})
    stats = t.get('season_stats', {})
    league = t.get('current_league', {})
    
    doc = f"""
Team: {t.get('name', 'Unknown')}
Full Name: {identity.get('full_name', 'N/A')}
Country: {identity.get('country', 'N/A')}
City: {identity.get('city', 'N/A')}
Founded: {identity.get('founded_year', 'N/A')}

Stadium: {venue.get('stadium_name', 'N/A')}
Capacity: {venue.get('capacity', 'N/A')}

League: {league.get('name', league) if isinstance(league, dict) else league}
Season: {t.get('current_season', 'N/A')}

Season Stats:
- Rank: {stats.get('rank', 'N/A')}
- Points: {stats.get('points', 0)}
- Wins: {stats.get('wins', 0)}
- Draws: {stats.get('draws', 0)}
- Losses: {stats.get('losses', 0)}
- Goals For: {stats.get('goals_for', 0)}
- Goals Against: {stats.get('goals_against', 0)}
    """.strip()
    
    return doc

def load_json(filepath):
    # load jsonl
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                data.append(json.loads(line))
    return data

def insert_teams(filepath):
    """Load teams và insert vào Supabase"""
    print(" Loading teams...")
    teams = load_json(filepath)
    print(f"   Found {len(teams)} teams")
    
    # Transform: thêm document field
    for team in teams:
        team['document'] = make_team_document(team)
        
        # Đổi entity_id -> team_id nếu cần
        if 'entity_id' in team and 'team_id' not in team:
            team['team_id'] = team.pop('entity_id')
        
        # Chỉ giữ các field có trong schema
        allowed_fields = {
            'team_id', 'name', 'metadata', 'document', 'embedding', 'created_at', 'updated_at'
        }
        filtered_team = {k: v for k, v in team.items() if k in allowed_fields}
        # Đảm bảo team_id có
        if 'team_id' in team and 'team_id' not in filtered_team:
            filtered_team['team_id'] = team['team_id']
        # Update team reference
        team.clear()
        team.update(filtered_team)
    
    # Insert in batches
    print(f" Inserting {len(teams)} teams...")
    for i in range(0, len(teams), 100):
        batch = teams[i:i+100]
        supabase.table("teams").upsert(batch).execute()
        print(f"   {min(i+100, len(teams))}/{len(teams)}")
    
    print("Teams inserted!")


def insert_players(filepath):
    """Load players và insert vào Supabase"""
    print(" Loading players...")
    players = load_json(filepath)
    print(f"Found {len(players)} players")
    
    # Transform: thêm document field
    for player in players:
        player['document'] = make_player_document(player)
        
        # Đổi entity_id -> player_id nếu cần
        if 'entity_id' in player and 'player_id' not in player:
            player['player_id'] = player.pop('entity_id')
        
        # Chỉ giữ các field có trong schema
        allowed_fields = {
            'player_id', 'name', 'current_team_id', 'metadata', 'document', 'embedding', 'created_at', 'updated_at'
        }
        filtered_player = {k: v for k, v in player.items() if k in allowed_fields}
        # Update player reference
        player.clear()
        player.update(filtered_player)

    # Insert in batches
    print(f"Inserting {len(players)} players...")
    for i in range(0, len(players), 100):
        batch = players[i:i+100]
        supabase.table("players").upsert(batch).execute()
        print(f"   {min(i+100, len(players))}/{len(players)}")
    
    print("Players inserted!")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 LOADING DATA TO SUPABASE")
    print("=" * 60)
    print()
    
    # Insert teams first
    try:
        insert_teams("data\\teams\\team_metadata_2023_2024.jsonl")
    except FileNotFoundError:
        print("⚠️  data\\teams\\team_metadata_2023_2024.jsonl not found, skipping...")
    except Exception as e:  
        print(f"❌ Error: {e}")
    
    # Insert players
    try:
        insert_players("data\\players\\players_complete_metadata.jsonl")
    except FileNotFoundError:
        print("⚠️  data\\players\\players_complete_metadata.jsonl not found, skipping...")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("=" * 60)
    print("COMPLETED!")
    print("=" * 60)