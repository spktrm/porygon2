import asyncio
import json
import os

import aiohttp
from tqdm import tqdm


async def get_leaderboard(format_id: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://pokemonshowdown.com/ladder/{format_id}.json"
        ) as response:
            return await response.json()


async def get(base_url: str, suffix: str = ""):
    async with aiohttp.ClientSession() as session:
        url = base_url + suffix
        async with session.get(url) as response:
            while True:
                try:
                    return await response.json()
                except:
                    content = await response.read()
                    print(content.decode())
                else:
                    break


async def get_battles_ids(
    user: str = None, format_id: str = None, page: int = 1, rating_threshold: int = None
):
    suffix = "?" + "&".join(
        [
            f"{key}={value}"
            for key, value in {"user": user, "format": format_id, "page": page}.items()
            if value is not None
        ]
    )

    try:
        res = await get("https://replay.pokemonshowdown.com/search.json", suffix)
    except:
        return []

    next_page = []
    if len(res) > 50:
        next_page = await get_battles_ids(user, format_id, page + 1, rating_threshold)

    final = res + next_page
    if final:
        return [b for b in final if (b.get("rating") or 0) >= rating_threshold]
    else:
        return []


class Search:
    def __init__(self, format_id: str, limit: int = 10000):
        self.format_id = format_id
        self.game_ids = set()
        self.player_ids = set()
        self.results = None
        self.limit = limit
        self.is_broken = False

    async def get_battle_ids(self, player_queue: asyncio.Queue):
        while not player_queue.empty() or not self.is_broken:
            player_id = await player_queue.get()
            self.player_ids.add(player_id)

            battles = await get_battles_ids(
                player_id, self.format_id, rating_threshold=1000
            )
            for battle in battles:
                if len(self.game_ids) >= self.limit:
                    self.is_broken = True
                    break

                self.game_ids.add(json.dumps(battle))

                for battle_player_id in battle["players"]:
                    if battle_player_id not in self.player_ids:
                        await player_queue.put(battle_player_id)

            print(
                f"Num Games: {len(self.game_ids):,}",
                f"Fronter Size: {player_queue.qsize():,}",
                " " * 100,
                end="\r",
            )
            if self.is_broken:
                break

    def get_results(self):
        self.results = list(self.game_ids)
        results = self.results
        if results is None:
            raise ValueError("No results!")
        self.results = None
        return results


class Counter:
    def __init__(self):
        self._count = 0

    async def count(self, *args, **kwargs):
        res = await get(*args, **kwargs)
        print(self._count)
        self._count += 1
        return res


async def main():
    format_id = "gen3ou"

    leaderboard = await get_leaderboard(format_id)

    player_queue = asyncio.Queue()

    initlist = [user["userid"] for user in leaderboard["toplist"]]
    for player in initlist:
        await player_queue.put(player)

    search = Search(format_id, limit=500)
    await asyncio.gather(*[search.get_battle_ids(player_queue) for _ in range(64)])
    results = search.get_results()
    results = [json.loads(res) for res in tqdm(results, desc="loading results")]

    c = Counter()
    replays = []

    replays = await asyncio.gather(
        *[
            c.count("https://replay.pokemonshowdown.com/", game["id"] + ".json")
            for game in results
        ]
    )

    root_dir = "replays/data/"
    format_dir = os.path.join(root_dir, format_id)
    if not os.path.exists(format_dir):
        os.mkdir(format_dir)

    for replay in replays:
        replay_id = replay["id"]
        replay_path = os.path.join(format_dir, replay_id + ".json")
        with open(replay_path, "w") as f:
            json.dump(replay, f)


if __name__ == "__main__":
    asyncio.run(main())
