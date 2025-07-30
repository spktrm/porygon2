#!/usr/bin/env python3
"""
Read a list of PokéPaste URLs from a .txt file, extract teams with
pokepastes_scraper, verify each team has 6 Pokémon, and save all valid
teams to a JSON file—now in parallel with ThreadPoolExecutor.

Usage:
    python build_teams_json.py pokepastes.txt teams.json [-j 32]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import requests
from pokepastes_scraper import team_from_url

# ---------------------------------------------------------------------------


def url_is_alive(url: str, timeout: int = 10) -> bool:
    """Return True if an HTTP GET returns status‑code 200."""
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        return r.status_code == 200
    except requests.RequestException:
        return False


def extract_team(url: str) -> Optional[Dict]:
    """
    Return a team (dict) only if it has exactly 6 Pokémon, otherwise None.
    Any exception from team_from_url is caught and treated as invalid.
    """
    try:
        team = team_from_url(url)  # may raise if URL bad / not a team
        if len(team.members) == 6:
            data = team.to_dict()  # convert to dict
            data["source_url"] = url  # keep provenance
            return data
    except Exception as e:
        print(f"[warn] {url}: {e}", file=sys.stderr)
    return None


def process_url(url: str) -> Optional[Dict]:
    """
    Wrapper executed in each thread:
    1. Check URL liveness.
    2. Extract & validate team.
    Returns the team dict or None.
    """
    if not url_is_alive(url):
        print(f"[skip] Dead link: {url}", file=sys.stderr)
        return None

    team_dict = extract_team(url)
    if team_dict is None:
        print(f"[skip] Invalid team @ {url}", file=sys.stderr)
        return None

    print(f"[ok ]  {url}")
    return team_dict


def build_dataset(txt_path: Path, max_workers: int = 20) -> List[Dict]:
    """
    Read all URLs, dispatch them to a thread pool, de‑duplicate identical teams,
    and return the ordered list of unique team dicts.
    """
    with open(txt_path, encoding="utf-8") as fh:
        urls = [line.strip() for line in fh if line.strip()]

    seen_hashes: set[str] = set()
    dataset: List[Dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as tp:
        # submit returns futures; we iterate them as they complete
        future_to_url = {tp.submit(process_url, url): url for url in urls}
        for fut in as_completed(future_to_url):
            team_dict = fut.result()
            if team_dict is None:
                continue

            h = json.dumps(team_dict, sort_keys=True)
            if h in seen_hashes:
                print(f"[dup]  {team_dict['source_url']}", file=sys.stderr)
                continue

            seen_hashes.add(h)
            dataset.append(team_dict)

    return dataset


def main():
    txt_path = "scrape/pokepastes.txt"
    out_path = "scrape/cleaned_teams.json"

    teams = build_dataset(txt_path, max_workers=32)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(teams, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
