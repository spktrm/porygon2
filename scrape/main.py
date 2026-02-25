import asyncio
import re
import json
import sys
import os
import argparse
import time
from typing import List, Set, Optional, Tuple
from urllib.parse import urljoin, urlparse
from collections import deque

import aiohttp
from bs4 import BeautifulSoup

# --- Constants ---
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
TIMEOUT = aiohttp.ClientTimeout(total=15)
# The strict prefix required for all internal links
STRICT_PREFIX = "https://www.smogon.com/forums/threads"


async def get_soup(
    session: aiohttp.ClientSession, url: str
) -> Optional[Tuple[str, BeautifulSoup]]:
    """Fetches a URL and returns the URL and its BeautifulSoup object."""
    try:
        async with session.get(url) as response:
            if response.status != 200:
                return None
            html = await response.text()
            return url, BeautifulSoup(html, "html.parser")
    except Exception:
        return None


async def main(args: argparse.Namespace):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, args.output)

    # The user input is the root, but we filter all queue additions by STRICT_PREFIX
    visited_pages: Set[str] = set()
    to_visit = deque([(args.url, 0)])
    pokepaste_links: Set[str] = set()

    total_pages_scanned = 0
    start_time = time.time()

    connector = aiohttp.TCPConnector(limit_per_host=args.jobs)
    async with aiohttp.ClientSession(
        headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, connector=connector
    ) as session:

        while to_visit:
            batch = []
            while to_visit and len(batch) < args.jobs:
                url, depth = to_visit.popleft()
                # Clean URL of fragments to avoid redundant visits
                clean_url = url.split("#")[0].rstrip("/")
                if clean_url not in visited_pages:
                    visited_pages.add(clean_url)
                    batch.append((clean_url, depth))

            if not batch:
                continue

            tasks = [get_soup(session, u) for u, d in batch]
            depth_map = {u: d for u, d in batch}

            for task in asyncio.as_completed(tasks):
                result = await task
                if not result:
                    continue

                current_url, soup = result
                current_depth = depth_map[current_url]
                total_pages_scanned += 1

                # 1. Extract Pokepaste links (from posts)
                found_pastes = soup.find_all(
                    "a", href=re.compile(r"pokepast\.es/[0-9a-f]+")
                )
                for a in found_pastes:
                    match = re.search(r"https?://pokepast\.es/[0-9a-f]+", a["href"])
                    if match:
                        pokepaste_links.add(match.group(0))

                # 2. Extract internal links with strict Smogon prefix
                if current_depth < args.depth:
                    for a in soup.find_all("a", href=True):
                        full_url = (
                            urljoin(current_url, a["href"]).split("#")[0].rstrip("/")
                        )

                        # STRICT ENFORCEMENT: Link must start with the specific threads path
                        if (
                            full_url.startswith(STRICT_PREFIX)
                            and full_url not in visited_pages
                        ):
                            to_visit.append((full_url, current_depth + 1))

                # Live logging
                sys.stdout.write(
                    f"\r[Smogon Scrape] Depth: {current_depth}/{args.depth} | Pages: {total_pages_scanned} | Q: {len(to_visit)} | Teams: {len(pokepaste_links)} "
                )
                sys.stdout.flush()

        final_list = sorted(list(pokepaste_links))
        duration = time.time() - start_time

        print(f"\n\n{'='*40}\nCRAWL FINISHED\n{'='*40}")
        print(f"Total Pages Scanned:  {total_pages_scanned}")
        print(f"Total Unique Teams:   {len(final_list)}")
        print(f"Execution Time:       {duration:.2f}s")
        print(f"Output:               {output_path}")

        try:
            with open(output_path, "w") as f:
                json.dump(final_list, f, indent=2)
        except Exception as e:
            print(f"Error saving: {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursive Smogon Thread Scraper")
    parser.add_argument("url", help="Start URL (e.g., a Smogon subforum or thread)")
    parser.add_argument(
        "output", nargs="?", default="smogon_pastes.json", help="Output filename"
    )
    parser.add_argument("-j", "--jobs", type=int, default=10, help="Concurrency")
    parser.add_argument(
        "-d", "--depth", type=int, default=2, help="Max depth (default: 2)"
    )

    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted. Progress saved to JSON.")
        sys.exit(0)
