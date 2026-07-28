import argparse
import asyncio
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from aiohttp import ClientSession
from tqdm import tqdm

SEARCH_URL = "https://replay.pokemonshowdown.com/search.json"
REPLAY_URL = "https://replay.pokemonshowdown.com/{replay_id}.json"
LADDER_URL = "https://pokemonshowdown.com/ladder/{format_id}.json"
ROOT_DIR = "replays/data/"
USER_AGENT = "porygon2-replay-downloader (https://github.com/spktrm/porygon2)"

# Results per search page. search.json returns up to 51 rows; the 51st only
# signals that another page exists (see pokemon-showdown-client/WEB-API.md).
PAGE_SIZE = 50

MAX_ATTEMPTS = 5
BACKOFF_BASE = 1.0
BACKOFF_CAP = 60.0
DEFAULT_429_COOLDOWN = 15.0


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter."""
    return min(BACKOFF_CAP, BACKOFF_BASE * 2**attempt) * (0.5 + random.random())


class RateLimitedClient:
    """GET-JSON client with global request spacing, bounded concurrency,
    retries with backoff, and a shared cooldown that pauses *all* requests
    whenever the server returns 429."""

    def __init__(self, session: ClientSession, concurrency: int, max_rps: float):
        self.session = session
        self._sem = asyncio.Semaphore(concurrency)
        self._min_interval = 1.0 / max_rps if max_rps > 0 else 0.0
        self._slot_lock = asyncio.Lock()
        self._next_slot = 0.0
        self._cooldown_until = 0.0
        self._consecutive_429 = 0

    async def _wait_turn(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            async with self._slot_lock:
                start = max(loop.time(), self._next_slot, self._cooldown_until)
                self._next_slot = start + self._min_interval
            delay = start - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            # A 429 may have landed while we slept; re-check the cooldown.
            if loop.time() >= self._cooldown_until:
                return

    def _enter_cooldown(self, seconds: float) -> None:
        loop = asyncio.get_running_loop()
        self._cooldown_until = max(self._cooldown_until, loop.time() + seconds)

    @staticmethod
    def _retry_after(resp: aiohttp.ClientResponse) -> float | None:
        value = resp.headers.get("Retry-After")
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            return None  # HTTP-date form; fall back to our own backoff

    async def get_json(
        self, url: str, params: dict[str, Any] | None = None
    ) -> Any | None:
        for attempt in range(MAX_ATTEMPTS):
            await self._wait_turn()
            try:
                async with self._sem:
                    async with self.session.get(url, params=params) as resp:
                        if resp.status == 200:
                            self._consecutive_429 = 0
                            return await resp.json(content_type=None)
                        if resp.status == 404:
                            return None
                        if resp.status == 429:
                            self._consecutive_429 += 1
                            wait = self._retry_after(resp) or (
                                DEFAULT_429_COOLDOWN * 2 ** (self._consecutive_429 - 1)
                            )
                            wait = min(wait, 300.0)
                            self._enter_cooldown(wait)
                            tqdm.write(f"429 from server, cooling down {wait:.0f}s")
                            continue
                        if resp.status >= 500:
                            await asyncio.sleep(
                                self._retry_after(resp) or _backoff(attempt)
                            )
                            continue
                        tqdm.write(f"Warning: status {resp.status} from {resp.url}")
                        return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                tqdm.write(
                    f"{type(e).__name__} on attempt {attempt + 1} for {url}: {e}"
                )
                await asyncio.sleep(_backoff(attempt))
        tqdm.write(f"Giving up on {url} after {MAX_ATTEMPTS} attempts")
        return None


@dataclass
class Stats:
    pages: int = 0
    players: int = 0
    enqueued: int = 0
    skipped_existing: int = 0
    skipped_rating: int = 0
    downloaded: int = 0
    failed: int = 0
    seen_ids: set[str] = field(default_factory=set)


@dataclass
class Filters:
    limit: int
    min_rating: int
    before: int | None
    after: int | None
    existing_ids: set[str]
    # End a crawl stream after this many consecutive already-downloaded
    # results — it has paged back into previously-synced territory, so
    # incremental runs only spend search requests on genuinely new games.
    # 0 disables (full re-scan).
    stop_after_existing: int = 100


async def _crawl_search(
    client: RateLimitedClient,
    queue: "asyncio.Queue[dict[str, Any] | None]",
    stats: Stats,
    base_params: dict[str, Any],
    filters: Filters,
) -> bool:
    """Paginate one search query newest-to-oldest via the `before` cursor,
    enqueueing battles that pass the filters. Returns True once the global
    limit is reached. `page=` pagination is poorly supported upstream, so it
    is not used."""
    cursor = filters.before
    existing_streak = 0
    while True:
        params = dict(base_params)
        if cursor is not None:
            params["before"] = cursor
        res = await client.get_json(SEARCH_URL, params=params)
        if not isinstance(res, list) or not res:
            return False
        stats.pages += 1

        page = res[:PAGE_SIZE]
        has_more = len(res) > PAGE_SIZE
        for battle in page:
            uploadtime = battle.get("uploadtime")
            if (
                filters.after is not None
                and uploadtime is not None
                and uploadtime < filters.after
            ):
                return False  # results are newest-first; the rest is older
            battle_id = battle.get("id")
            if not battle_id or battle_id in stats.seen_ids:
                continue
            stats.seen_ids.add(battle_id)
            if battle.get("private"):
                continue
            if (battle.get("rating") or 0) < filters.min_rating:
                stats.skipped_rating += 1
                continue
            if battle_id in filters.existing_ids:
                stats.skipped_existing += 1
                existing_streak += 1
                if (
                    filters.stop_after_existing
                    and existing_streak >= filters.stop_after_existing
                ):
                    return False  # reached previously-synced territory
                continue
            await queue.put(battle)
            stats.enqueued += 1
            existing_streak = 0
            if stats.enqueued >= filters.limit:
                return True

        if not has_more:
            return False
        next_cursor = page[-1].get("uploadtime")
        if next_cursor is None or next_cursor == cursor:
            return False  # avoid looping on a degenerate cursor
        cursor = next_cursor


async def produce_from_ladder(
    client: RateLimitedClient,
    queue: "asyncio.Queue[dict[str, Any] | None]",
    stats: Stats,
    format_id: str,
    filters: Filters,
) -> None:
    """Quality-first search: walk the current ladder toplist in rank order and
    fetch each player's replays (newest first), so the strongest players'
    games are collected before anything else."""
    ladder = await client.get_json(LADDER_URL.format(format_id=format_id))
    toplist = (ladder or {}).get("toplist") or []
    if not toplist:
        tqdm.write("No ladder data; falling back to format-wide sweep.")
        await produce_from_format(client, queue, stats, format_id, filters)
        return
    for user in toplist:
        userid = user.get("userid")
        if not userid:
            continue
        stats.players += 1
        done = await _crawl_search(
            client,
            queue,
            stats,
            {"format": format_id, "user": userid},
            filters,
        )
        if done:
            return
    tqdm.write(
        f"Exhausted the ladder toplist ({stats.players} players) at "
        f"{stats.enqueued}/{filters.limit} replays."
    )


async def produce_from_format(
    client: RateLimitedClient,
    queue: "asyncio.Queue[dict[str, Any] | None]",
    stats: Stats,
    format_id: str,
    filters: Filters,
) -> None:
    """Sweep every public replay of the format, newest first."""
    await _crawl_search(client, queue, stats, {"format": format_id}, filters)


def save_replay(format_dir: str, replay: dict[str, Any]) -> None:
    """Atomic write so an interrupted run never leaves truncated JSON."""
    replay_path = os.path.join(format_dir, replay["id"] + ".json")
    tmp_path = replay_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(replay, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, replay_path)


async def download_worker(
    client: RateLimitedClient,
    queue: "asyncio.Queue[dict[str, Any] | None]",
    stats: Stats,
    format_dir: str,
    bar: tqdm,
) -> None:
    while True:
        battle = await queue.get()
        try:
            if battle is None:
                return
            replay = await client.get_json(REPLAY_URL.format(replay_id=battle["id"]))
            if replay and "id" in replay:
                save_replay(format_dir, replay)
                stats.downloaded += 1
            else:
                stats.failed += 1
            bar.update(1)
        finally:
            queue.task_done()


def load_existing_ids(format_dir: str) -> set[str]:
    existing_ids: set[str] = set()
    if os.path.exists(format_dir):
        for filename in os.listdir(format_dir):
            if filename.endswith(".json"):
                existing_ids.add(filename[:-5])
    return existing_ids


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Pokemon Showdown replays for a format"
    )
    parser.add_argument("format_id", type=str, help="Format id, e.g. gen9randombattle")
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=10000,
        help="Maximum number of new replays to download (default: 10000)",
    )
    parser.add_argument(
        "--source",
        choices=["ladder", "format"],
        default="ladder",
        help=(
            "'ladder' (default) crawls the current top-ladder players in rank "
            "order so the highest-quality games come first; 'format' sweeps "
            "all public replays of the format newest-first"
        ),
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=1000,
        help="Skip games whose search-result rating is below this (default: 1000)",
    )
    parser.add_argument(
        "--before",
        type=int,
        default=None,
        help="Only include games uploaded before this uploadtime (seconds since epoch)",
    )
    parser.add_argument(
        "--after",
        type=int,
        default=None,
        help="Only include games uploaded after this uploadtime (seconds since epoch)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Maximum in-flight requests (default: 8)",
    )
    parser.add_argument(
        "--max-rps",
        type=float,
        default=10.0,
        help="Maximum requests per second across all workers (default: 10)",
    )
    parser.add_argument(
        "--stop-after-existing",
        type=int,
        default=100,
        help=(
            "End a crawl stream after this many consecutive already-"
            "downloaded results (default: 100, i.e. two full pages of known "
            "games means that stream is synced; 0 forces a full re-scan)"
        ),
    )
    args = parser.parse_args()

    format_dir = os.path.join(ROOT_DIR, args.format_id)
    existing_ids = load_existing_ids(format_dir)
    if existing_ids:
        print(f"Found {len(existing_ids)} existing replays in {format_dir}.")
    os.makedirs(format_dir, exist_ok=True)

    filters = Filters(
        limit=args.limit,
        min_rating=args.min_rating,
        before=args.before,
        after=args.after,
        existing_ids=existing_ids,
        stop_after_existing=args.stop_after_existing,
    )
    stats = Stats()
    queue: "asyncio.Queue[dict[str, Any] | None]" = asyncio.Queue(maxsize=200)

    timeout = aiohttp.ClientTimeout(total=60, connect=10, sock_read=30)
    headers = {"User-Agent": USER_AGENT}

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        client = RateLimitedClient(session, args.concurrency, args.max_rps)
        with tqdm(desc="Downloading replays", total=args.limit, unit="replay") as bar:
            workers = [
                asyncio.create_task(
                    download_worker(client, queue, stats, format_dir, bar)
                )
                for _ in range(args.concurrency)
            ]
            producer = (
                produce_from_ladder if args.source == "ladder" else produce_from_format
            )
            await producer(client, queue, stats, args.format_id, filters)
            for _ in workers:
                await queue.put(None)
            await asyncio.gather(*workers)
            bar.total = stats.downloaded + stats.failed
            bar.refresh()

    print(
        f"Done! downloaded={stats.downloaded} failed={stats.failed} "
        f"skipped_existing={stats.skipped_existing} "
        f"skipped_low_rating={stats.skipped_rating} "
        f"search_pages={stats.pages}"
        + (f" players_crawled={stats.players}" if args.source == "ladder" else "")
    )


if __name__ == "__main__":
    asyncio.run(main())