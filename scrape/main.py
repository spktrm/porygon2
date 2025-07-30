#!/usr/bin/env python3
"""
Multithreaded crawler that gathers up to N distinct
https://pokepast.es/<16‑hex> links, then stops.

Usage examples
--------------
# crawl one page, stop after 30 PokéPastes
python crawl_pokepastes.py --max 30 \
  https://www.smogon.com/forums/threads/adv-ou-sample-teams.3687813/

# default limit (100) with the hard‑coded START_URLS
python crawl_pokepastes.py
"""

import re
import sys
import time
import queue
import argparse
import threading
import urllib.parse as up
from collections import OrderedDict

import requests
from bs4 import BeautifulSoup

###############################################################################
THREADS = 64  # worker threads (adjust as needed)
TIMEOUT = 15  # HTTP timeout (s)
START_URLS = ["https://www.smogon.com/forums/forums/adv/"]
DEFAULT_MAX = 2000  # default “stop after N PokéPastes”
###############################################################################

POKEPASTE_RE = re.compile(r"https://pokepast\.es/[0-9a-fA-F]{16}")


class CrawlState:
    """Holds shared objects used by all threads."""

    def __init__(self, max_links: int):
        self.queue = queue.Queue()
        self.visited = set()
        self.pokepaste = set()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.max_links = max_links


def enqueue_url(state: CrawlState, url: str) -> None:
    """Queue a URL unless it’s been visited or we’re done."""
    if state.stop_event.is_set():
        return
    with state.lock:
        if url not in state.visited:
            state.visited.add(url)
            state.queue.put(url)


def extract_links(base_url: str, html: str):
    """Return (pokepastes, same‑domain links) from page HTML."""
    pokes = set(POKEPASTE_RE.findall(html))

    parsed = up.urlparse(base_url)
    domain = f"{parsed.scheme}://{parsed.netloc}" + "/forums/threads"
    same_dom = set()

    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = up.urljoin(base_url, a["href"]).split("#", 1)[0]
        if href.startswith(domain) and "adv" in base_url:
            same_dom.add(href)

    return pokes, same_dom


def worker(state: CrawlState, session: requests.Session, tid: int):
    """
    Worker exits only when:
      1. stop_event is set  *and*
      2. the queue has been emptied
    This guarantees every en‑queued item gets a matching task_done().
    """
    while True:
        # If we’ve been told to stop but the queue is empty, we’re done
        if state.stop_event.is_set() and state.queue.empty():
            break

        try:
            url = state.queue.get(timeout=1)
        except queue.Empty:
            continue  # try again (check stop_event at top of loop)

        try:
            r = session.get(url, timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"[Thread {tid}] !! Error fetching {url}: {e}", file=sys.stderr)
            state.queue.task_done()
            continue

        new_pokes, new_links = extract_links(url, r.text)

        with state.lock:
            state.pokepaste.update(new_pokes)
            print(len(state.pokepaste), state.queue.qsize())
            if len(state.pokepaste) >= state.max_links:
                state.stop_event.set()  # signal everyone to wrap up
                new_links = []  # don’t enqueue more work

        for link in new_links:
            enqueue_url(state, link)

        state.queue.task_done()


def crawl(start_urls, max_links=DEFAULT_MAX, threads=THREADS):
    state = CrawlState(max_links)
    session = requests.Session()

    for url in start_urls:
        enqueue_url(state, url)

    workers = [
        threading.Thread(target=worker, args=(state, session, i), daemon=True)
        for i in range(threads)
    ]
    for w in workers:
        w.start()

    start = time.time()
    # Wait until we either hit the limit OR the queue empties
    while not state.stop_event.is_set() and any(w.is_alive() for w in workers):
        try:
            state.queue.join()
        except KeyboardInterrupt:
            state.stop_event.set()
            break

    elapsed = time.time() - start

    ordered = list(OrderedDict.fromkeys(state.pokepaste))[:max_links]

    for w in workers:
        w.join()

    ordered = list(OrderedDict.fromkeys(state.pokepaste))[:max_links]
    print(f"\nCollected {len(ordered)} PokéPastes:\n")
    for url in ordered:
        print(url)

    # Optional: save to file
    with open("scrape/pokepastes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(ordered) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl for PokéPaste links.")
    parser.add_argument("urls", nargs="*", help="Seed pages to start from")
    parser.add_argument(
        "--max",
        "-n",
        type=int,
        default=DEFAULT_MAX,
        help="Stop after collecting N PokéPastes (default: %(default)s)",
    )
    args = parser.parse_args()

    seeds = args.urls if args.urls else START_URLS
    if not seeds:
        sys.exit("No start URLs provided.")

    crawl(seeds, max_links=args.max)
