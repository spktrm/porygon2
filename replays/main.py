import asyncio
import functools
import json
import os
from typing import Any, Dict, List, Optional, Set

import aiohttp
from aiohttp import ClientSession
from tqdm import tqdm

# Global rate limit semaphore (adjust this number as appropriate for your environment)
RATE_LIMIT = 100
rate_limiter = asyncio.Semaphore(RATE_LIMIT)


async def aprint(*args, **kwargs):
    """Asynchronously print to stdout using the default print function in a thread executor."""
    loop = asyncio.get_running_loop()
    # Use functools.partial to pre-bind the arguments to print
    partial_print = functools.partial(print, *args, **kwargs)
    await loop.run_in_executor(None, partial_print)


async def fetch_json(session: ClientSession, url: str) -> Optional[Dict[str, Any]]:
    """Fetch JSON from a given URL, respecting the rate limit and handling errors gracefully."""
    async with rate_limiter:
        for attempt in range(3):  # try up to 3 times
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        # Non-200, log and return None to skip
                        await aprint(
                            f"Warning: Received status {response.status} from {url}"
                        )
                        return None
                    return await response.json()
            except aiohttp.ClientError as e:
                await aprint(f"ClientError on attempt {attempt + 1} for URL {url}: {e}")
                await asyncio.sleep(1)
            except aiohttp.ContentTypeError as e:
                # Response isn't JSON or is malformed
                await aprint(
                    f"ContentTypeError on attempt {attempt + 1} for URL {url}: {e}"
                )
                await asyncio.sleep(1)
        # All attempts failed
        return None


async def get_leaderboard(
    session: ClientSession, format_id: str
) -> Optional[Dict[str, Any]]:
    """Fetch the leaderboard for a given format."""
    url = f"https://pokemonshowdown.com/ladder/{format_id}.json"
    return await fetch_json(session, url)


async def get(
    session: ClientSession, base_url: str, suffix: str = ""
) -> Optional[Dict[str, Any]]:
    """Generic GET request for JSON data."""
    url = base_url + suffix
    return await fetch_json(session, url)


async def get_battles_ids(
    session: ClientSession,
    user: Optional[str] = None,
    format_id: Optional[str] = None,
    page: int = 1,
    rating_threshold: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Recursively fetch battle IDs for a user/format combination starting from a page,
    filtering by rating_threshold if provided."""
    query_params = {
        "user": user,
        "format": format_id,
        "page": page,
    }
    # Build suffix for query
    suffix_parts = [f"{k}={v}" for k, v in query_params.items() if v is not None]
    suffix = "?" + "&".join(suffix_parts) if suffix_parts else ""

    res = await get(session, "https://replay.pokemonshowdown.com/search.json", suffix)
    if not res or not isinstance(res, list):
        return []

    # If more than 50 results returned, assume there's a next page
    next_page = []
    if len(res) > 50:
        next_page = await get_battles_ids(
            session, user, format_id, page + 1, rating_threshold
        )

    final = res + next_page

    # Filter by rating threshold if provided
    if rating_threshold is not None:
        final = [b for b in final if (b.get("rating") or 0) >= rating_threshold]

    return final


class Search:
    def __init__(self, session: ClientSession, format_id: str, limit: int = 10000):
        self.session = session
        self.format_id = format_id
        self.game_ids: Set[str] = set()
        self.player_ids: Set[str] = set()
        self.limit = limit
        self.is_broken = False

    async def get_battle_ids(self, player_queue: asyncio.Queue):
        """Worker coroutine to fetch battles from the player queue."""
        while not self.is_broken:
            if player_queue.empty():
                break
            player_id = await player_queue.get()
            if player_id in self.player_ids:
                continue

            self.player_ids.add(player_id)

            # Fetch battles with a rating threshold of 1000 to reduce irrelevant matches
            battles = await get_battles_ids(
                self.session,
                user=player_id,
                format_id=self.format_id,
                rating_threshold=1000,
            )
            for battle in battles:
                if len(self.game_ids) >= self.limit:
                    self.is_broken = True
                    break
                battle_str = json.dumps(battle, sort_keys=True)
                if battle_str not in self.game_ids:
                    self.game_ids.add(battle_str)
                    # Add new players discovered from this battle to the queue
                    for battle_player_id in battle.get("players", []):
                        if battle_player_id not in self.player_ids:
                            await player_queue.put(battle_player_id)

            print(
                f"Num Games: {len(self.game_ids):,} | Frontier Size: {player_queue.qsize():,} ",
                end="\r",
            )
            if self.is_broken:
                break

    def get_results(self) -> List[Dict[str, Any]]:
        if not self.game_ids:
            raise ValueError("No results!")
        results = [json.loads(item) for item in self.game_ids]
        self.game_ids.clear()
        return results


class Counter:
    """Counter class to track the number of requests made with `count` method."""

    def __init__(self, session: ClientSession, total: int):
        self._count = 0
        self.bar = tqdm(desc="Downloading replays", total=total)
        self.session = session

    async def count(self, base_url: str, suffix: str) -> Optional[Dict[str, Any]]:
        res = await get(self.session, base_url, suffix)
        self._count += 1
        self.bar.update(1)
        return res


async def main():
    format_id = "gen3ou"

    async with aiohttp.ClientSession() as session:
        leaderboard = await get_leaderboard(session, format_id)
        if not leaderboard or "toplist" not in leaderboard:
            print("No leaderboard data found.")
            return

        player_queue = asyncio.Queue()

        # Initialize the search with users from the leaderboard
        initlist = [user["userid"] for user in leaderboard["toplist"]]
        for player in initlist:
            await player_queue.put(player)

        search = Search(session, format_id, limit=5000)

        # Use a moderate number of workers
        workers = 10
        await asyncio.gather(
            *[search.get_battle_ids(player_queue) for _ in range(workers)]
        )
        results = search.get_results()

        # Now fetch the replay data for each game
        c = Counter(session, len(results))

        # Gather replays with rate limiting
        replays = []
        chunk_replays = await asyncio.gather(
            *[
                c.count("https://replay.pokemonshowdown.com/", game["id"] + ".json")
                for game in results
            ]
        )
        replays = [r for r in chunk_replays if r is not None]

        # Save the replays to disk
        root_dir = "replays/data/"
        format_dir = os.path.join(root_dir, format_id)
        os.makedirs(format_dir, exist_ok=True)

        for replay in replays:
            if "id" not in replay:
                continue
            replay_id = replay["id"]
            replay_path = os.path.join(format_dir, replay_id + ".json")
            with open(replay_path, "w", encoding="utf-8") as f:
                json.dump(replay, f, ensure_ascii=False, indent=2)

        print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
