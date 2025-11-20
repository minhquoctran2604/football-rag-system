
import json
import os
from datetime import datetime
from pathlib import Path
import re
import unicodedata

def slug(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')

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

def transform_teams_for_supabase(input_file, output_file):
    
    print(f"Loading teams from {input_file}...")
    
    teams = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: 
                teams.append(json.loads(line))
    
    print(f"Loaded {len(teams)} teams")
    
    transformed_teams = []
    for team in teams:
        document = make_team_document(team)
        
        team_id = team.get('canonical_team_id') or \
                f"team_{slug(team.get('name', 'unknown'))}_{team.get('identity', {}).get('founded_year', 'unknown')}"
        
        metadata = team.copy()
        for field in ['team_id', 'name', 'metadata', 'document', 'embedding', 'created_at', 'updated_at']:
            metadata.pop(field, None)
        
        transformed_team = {
            'team_id': team_id,
            'name': team.get('name', ''),
            'metadata': metadata,
            'document': document,
            'embedding': None,  
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        transformed_teams.append(transformed_team)
    
    print(f"Transformed {len(transformed_teams)} teams for Supabase")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for team in transformed_teams:
            f.write(json.dumps(team, ensure_ascii=False) + '\n')
    
    print(f"Written transformed teams to {output_file}")
    
    print("\nSample of transformed team data:")
    print(json.dumps(transformed_teams[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    input_path = Path("data/teams/team_complete_metadata_2023_2024.jsonl")
    output_path = Path("data/teams/team_complete_metadata_for_supabase.jsonl")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    transform_teams_for_supabase(input_path, output_path)