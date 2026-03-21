"""
generate_urls.py - Wikipedia URL Sampling Script
Conversational AI Assignment 2 - Group 122

Generates fixed_url.json (200 URLs) and random_url.json (300 URLs)
using hash-based deterministic sampling from Wikipedia.

Usage:
    python generate_urls.py --fixed 200 --random 300 --group GROUP_122
    python generate_urls.py --skip-fixed --random 300 --group GROUP_122
    python generate_urls.py --fixed 200 --skip-random --group GROUP_122
"""

import argparse
import json
import random
import re
import time
import uuid
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_BASE_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "ConvAI-Assignment2-Group122/1.0 (Educational; Contact: group122@bits-pilani.ac.in)"

DOMAIN_KEYWORDS = {
    "people": [
        "birth", "death", "born", "people", "politician", "actor", "actress",
        "singer", "writer", "artist", "scientist", "player", "biography",
    ],
    "places": [
        "city", "country", "town", "village", "geography", "location",
        "region", "state", "province", "district", "mountain", "river",
    ],
    "events": [
        "war", "battle", "revolution", "event", "history", "incident",
        "disaster", "election", "championship", "tournament",
    ],
    "science": [
        "science", "biology", "chemistry", "physics", "mathematics",
        "medicine", "astronomy", "geology",
    ],
    "technology": [
        "technology", "software", "computer", "internet", "programming",
        "engineering", "device", "invention",
    ],
    "arts": [
        "art", "music", "film", "movie", "album", "song", "book", "novel",
        "painting", "literature", "theatre", "dance",
    ],
    "sports": [
        "sport", "football", "basketball", "cricket", "tennis", "olympics",
        "athlete", "team", "league", "match",
    ],
    "organization": [
        "company", "organization", "corporation", "university", "institution",
        "foundation", "agency", "association",
    ],
}

# 200 hardcoded (url, domain) tuples from the Part 1 notebook (cell 17).
FIXED_URLS = [
    # PEOPLE (50)
    ("https://en.wikipedia.org/wiki/Albert_Einstein", "people"),
    ("https://en.wikipedia.org/wiki/Marie_Curie", "people"),
    ("https://en.wikipedia.org/wiki/Isaac_Newton", "people"),
    ("https://en.wikipedia.org/wiki/Charles_Darwin", "people"),
    ("https://en.wikipedia.org/wiki/Nikola_Tesla", "people"),
    ("https://en.wikipedia.org/wiki/Stephen_Hawking", "people"),
    ("https://en.wikipedia.org/wiki/Galileo_Galilei", "people"),
    ("https://en.wikipedia.org/wiki/Richard_Feynman", "people"),
    ("https://en.wikipedia.org/wiki/Ada_Lovelace", "people"),
    ("https://en.wikipedia.org/wiki/Alan_Turing", "people"),
    ("https://en.wikipedia.org/wiki/Abraham_Lincoln", "people"),
    ("https://en.wikipedia.org/wiki/Winston_Churchill", "people"),
    ("https://en.wikipedia.org/wiki/Mahatma_Gandhi", "people"),
    ("https://en.wikipedia.org/wiki/Nelson_Mandela", "people"),
    ("https://en.wikipedia.org/wiki/Martin_Luther_King_Jr.", "people"),
    ("https://en.wikipedia.org/wiki/Napoleon", "people"),
    ("https://en.wikipedia.org/wiki/Cleopatra", "people"),
    ("https://en.wikipedia.org/wiki/Julius_Caesar", "people"),
    ("https://en.wikipedia.org/wiki/Queen_Victoria", "people"),
    ("https://en.wikipedia.org/wiki/Alexander_the_Great", "people"),
    ("https://en.wikipedia.org/wiki/Leonardo_da_Vinci", "people"),
    ("https://en.wikipedia.org/wiki/Vincent_van_Gogh", "people"),
    ("https://en.wikipedia.org/wiki/Pablo_Picasso", "people"),
    ("https://en.wikipedia.org/wiki/Michelangelo", "people"),
    ("https://en.wikipedia.org/wiki/William_Shakespeare", "people"),
    ("https://en.wikipedia.org/wiki/Wolfgang_Amadeus_Mozart", "people"),
    ("https://en.wikipedia.org/wiki/Ludwig_van_Beethoven", "people"),
    ("https://en.wikipedia.org/wiki/Johann_Sebastian_Bach", "people"),
    ("https://en.wikipedia.org/wiki/Frida_Kahlo", "people"),
    ("https://en.wikipedia.org/wiki/Claude_Monet", "people"),
    ("https://en.wikipedia.org/wiki/Elon_Musk", "people"),
    ("https://en.wikipedia.org/wiki/Steve_Jobs", "people"),
    ("https://en.wikipedia.org/wiki/Bill_Gates", "people"),
    ("https://en.wikipedia.org/wiki/Jeff_Bezos", "people"),
    ("https://en.wikipedia.org/wiki/Mark_Zuckerberg", "people"),
    ("https://en.wikipedia.org/wiki/Oprah_Winfrey", "people"),
    ("https://en.wikipedia.org/wiki/Barack_Obama", "people"),
    ("https://en.wikipedia.org/wiki/Angela_Merkel", "people"),
    ("https://en.wikipedia.org/wiki/Malala_Yousafzai", "people"),
    ("https://en.wikipedia.org/wiki/Greta_Thunberg", "people"),
    ("https://en.wikipedia.org/wiki/Michael_Jordan", "people"),
    ("https://en.wikipedia.org/wiki/Lionel_Messi", "people"),
    ("https://en.wikipedia.org/wiki/Cristiano_Ronaldo", "people"),
    ("https://en.wikipedia.org/wiki/Serena_Williams", "people"),
    ("https://en.wikipedia.org/wiki/Usain_Bolt", "people"),
    ("https://en.wikipedia.org/wiki/Muhammad_Ali", "people"),
    ("https://en.wikipedia.org/wiki/Roger_Federer", "people"),
    ("https://en.wikipedia.org/wiki/LeBron_James", "people"),
    ("https://en.wikipedia.org/wiki/Tiger_Woods", "people"),
    ("https://en.wikipedia.org/wiki/Michael_Phelps", "people"),
    # PLACES (30)
    ("https://en.wikipedia.org/wiki/Grand_Canyon", "places"),
    ("https://en.wikipedia.org/wiki/Mount_Everest", "places"),
    ("https://en.wikipedia.org/wiki/Niagara_Falls", "places"),
    ("https://en.wikipedia.org/wiki/Great_Barrier_Reef", "places"),
    ("https://en.wikipedia.org/wiki/Amazon_rainforest", "places"),
    ("https://en.wikipedia.org/wiki/Sahara", "places"),
    ("https://en.wikipedia.org/wiki/Victoria_Falls", "places"),
    ("https://en.wikipedia.org/wiki/Yellowstone_National_Park", "places"),
    ("https://en.wikipedia.org/wiki/Tokyo", "places"),
    ("https://en.wikipedia.org/wiki/New_York_City", "places"),
    ("https://en.wikipedia.org/wiki/London", "places"),
    ("https://en.wikipedia.org/wiki/Paris", "places"),
    ("https://en.wikipedia.org/wiki/Rome", "places"),
    ("https://en.wikipedia.org/wiki/Sydney", "places"),
    ("https://en.wikipedia.org/wiki/Dubai", "places"),
    ("https://en.wikipedia.org/wiki/Singapore", "places"),
    ("https://en.wikipedia.org/wiki/Hong_Kong", "places"),
    ("https://en.wikipedia.org/wiki/Berlin", "places"),
    ("https://en.wikipedia.org/wiki/United_States", "places"),
    ("https://en.wikipedia.org/wiki/China", "places"),
    ("https://en.wikipedia.org/wiki/India", "places"),
    ("https://en.wikipedia.org/wiki/Japan", "places"),
    ("https://en.wikipedia.org/wiki/Brazil", "places"),
    ("https://en.wikipedia.org/wiki/Australia", "places"),
    ("https://en.wikipedia.org/wiki/Canada", "places"),
    ("https://en.wikipedia.org/wiki/Germany", "places"),
    ("https://en.wikipedia.org/wiki/France", "places"),
    ("https://en.wikipedia.org/wiki/United_Kingdom", "places"),
    ("https://en.wikipedia.org/wiki/Italy", "places"),
    ("https://en.wikipedia.org/wiki/Russia", "places"),
    # EVENTS (25)
    ("https://en.wikipedia.org/wiki/World_War_II", "events"),
    ("https://en.wikipedia.org/wiki/World_War_I", "events"),
    ("https://en.wikipedia.org/wiki/French_Revolution", "events"),
    ("https://en.wikipedia.org/wiki/American_Revolution", "events"),
    ("https://en.wikipedia.org/wiki/Industrial_Revolution", "events"),
    ("https://en.wikipedia.org/wiki/Renaissance", "events"),
    ("https://en.wikipedia.org/wiki/Cold_War", "events"),
    ("https://en.wikipedia.org/wiki/Moon_landing", "events"),
    ("https://en.wikipedia.org/wiki/Fall_of_the_Berlin_Wall", "events"),
    ("https://en.wikipedia.org/wiki/September_11_attacks", "events"),
    ("https://en.wikipedia.org/wiki/Black_Death", "events"),
    ("https://en.wikipedia.org/wiki/Reformation", "events"),
    ("https://en.wikipedia.org/wiki/Russian_Revolution", "events"),
    ("https://en.wikipedia.org/wiki/American_Civil_War", "events"),
    ("https://en.wikipedia.org/wiki/Hiroshima_and_Nagasaki_atomic_bombings", "events"),
    ("https://en.wikipedia.org/wiki/COVID-19_pandemic", "events"),
    ("https://en.wikipedia.org/wiki/Great_Depression", "events"),
    ("https://en.wikipedia.org/wiki/Chernobyl_disaster", "events"),
    ("https://en.wikipedia.org/wiki/Titanic", "events"),
    ("https://en.wikipedia.org/wiki/Discovery_of_the_Americas", "events"),
    ("https://en.wikipedia.org/wiki/Ancient_Egypt", "events"),
    ("https://en.wikipedia.org/wiki/Roman_Empire", "events"),
    ("https://en.wikipedia.org/wiki/Ancient_Greece", "events"),
    ("https://en.wikipedia.org/wiki/Byzantine_Empire", "events"),
    ("https://en.wikipedia.org/wiki/Ottoman_Empire", "events"),
    # TECHNOLOGY (25)
    ("https://en.wikipedia.org/wiki/Artificial_intelligence", "technology"),
    ("https://en.wikipedia.org/wiki/Internet", "technology"),
    ("https://en.wikipedia.org/wiki/Computer", "technology"),
    ("https://en.wikipedia.org/wiki/Smartphone", "technology"),
    ("https://en.wikipedia.org/wiki/World_Wide_Web", "technology"),
    ("https://en.wikipedia.org/wiki/Machine_learning", "technology"),
    ("https://en.wikipedia.org/wiki/Blockchain", "technology"),
    ("https://en.wikipedia.org/wiki/Cryptocurrency", "technology"),
    ("https://en.wikipedia.org/wiki/Bitcoin", "technology"),
    ("https://en.wikipedia.org/wiki/Electric_vehicle", "technology"),
    ("https://en.wikipedia.org/wiki/Robotics", "technology"),
    ("https://en.wikipedia.org/wiki/3D_printing", "technology"),
    ("https://en.wikipedia.org/wiki/Virtual_reality", "technology"),
    ("https://en.wikipedia.org/wiki/Augmented_reality", "technology"),
    ("https://en.wikipedia.org/wiki/Cloud_computing", "technology"),
    ("https://en.wikipedia.org/wiki/5G", "technology"),
    ("https://en.wikipedia.org/wiki/Quantum_computing", "technology"),
    ("https://en.wikipedia.org/wiki/Self-driving_car", "technology"),
    ("https://en.wikipedia.org/wiki/SpaceX", "technology"),
    ("https://en.wikipedia.org/wiki/Tesla,_Inc.", "technology"),
    ("https://en.wikipedia.org/wiki/Apple_Inc.", "technology"),
    ("https://en.wikipedia.org/wiki/Google", "technology"),
    ("https://en.wikipedia.org/wiki/Microsoft", "technology"),
    ("https://en.wikipedia.org/wiki/Amazon_(company)", "technology"),
    ("https://en.wikipedia.org/wiki/Facebook", "technology"),
    # SCIENCE (25)
    ("https://en.wikipedia.org/wiki/Climate_change", "science"),
    ("https://en.wikipedia.org/wiki/DNA", "science"),
    ("https://en.wikipedia.org/wiki/Evolution", "science"),
    ("https://en.wikipedia.org/wiki/Black_hole", "science"),
    ("https://en.wikipedia.org/wiki/Big_Bang", "science"),
    ("https://en.wikipedia.org/wiki/Quantum_mechanics", "science"),
    ("https://en.wikipedia.org/wiki/Theory_of_relativity", "science"),
    ("https://en.wikipedia.org/wiki/Photosynthesis", "science"),
    ("https://en.wikipedia.org/wiki/Human_brain", "science"),
    ("https://en.wikipedia.org/wiki/Solar_System", "science"),
    ("https://en.wikipedia.org/wiki/Milky_Way", "science"),
    ("https://en.wikipedia.org/wiki/Periodic_table", "science"),
    ("https://en.wikipedia.org/wiki/Atom", "science"),
    ("https://en.wikipedia.org/wiki/Cell_(biology)", "science"),
    ("https://en.wikipedia.org/wiki/Genetics", "science"),
    ("https://en.wikipedia.org/wiki/Vaccination", "science"),
    ("https://en.wikipedia.org/wiki/Antibiotic", "science"),
    ("https://en.wikipedia.org/wiki/Nuclear_power", "science"),
    ("https://en.wikipedia.org/wiki/Renewable_energy", "science"),
    ("https://en.wikipedia.org/wiki/Global_warming", "science"),
    ("https://en.wikipedia.org/wiki/Biodiversity", "science"),
    ("https://en.wikipedia.org/wiki/Ecosystem", "science"),
    ("https://en.wikipedia.org/wiki/Oceanography", "science"),
    ("https://en.wikipedia.org/wiki/Astronomy", "science"),
    ("https://en.wikipedia.org/wiki/Neuroscience", "science"),
    # ARTS (20)
    ("https://en.wikipedia.org/wiki/Mona_Lisa", "arts"),
    ("https://en.wikipedia.org/wiki/Starry_Night", "arts"),
    ("https://en.wikipedia.org/wiki/The_Last_Supper_(Leonardo)", "arts"),
    ("https://en.wikipedia.org/wiki/Sistine_Chapel_ceiling", "arts"),
    ("https://en.wikipedia.org/wiki/The_Scream", "arts"),
    ("https://en.wikipedia.org/wiki/Girl_with_a_Pearl_Earring", "arts"),
    ("https://en.wikipedia.org/wiki/The_Birth_of_Venus", "arts"),
    ("https://en.wikipedia.org/wiki/Guernica_(Picasso)", "arts"),
    ("https://en.wikipedia.org/wiki/The_Persistence_of_Memory", "arts"),
    ("https://en.wikipedia.org/wiki/The_Great_Wave_off_Kanagawa", "arts"),
    ("https://en.wikipedia.org/wiki/Statue_of_Liberty", "arts"),
    ("https://en.wikipedia.org/wiki/Eiffel_Tower", "arts"),
    ("https://en.wikipedia.org/wiki/Great_Wall_of_China", "arts"),
    ("https://en.wikipedia.org/wiki/Taj_Mahal", "arts"),
    ("https://en.wikipedia.org/wiki/Colosseum", "arts"),
    ("https://en.wikipedia.org/wiki/Machu_Picchu", "arts"),
    ("https://en.wikipedia.org/wiki/Petra", "arts"),
    ("https://en.wikipedia.org/wiki/Louvre", "arts"),
    ("https://en.wikipedia.org/wiki/Metropolitan_Museum_of_Art", "arts"),
    ("https://en.wikipedia.org/wiki/British_Museum", "arts"),
    # SPORTS (15)
    ("https://en.wikipedia.org/wiki/Olympic_Games", "sports"),
    ("https://en.wikipedia.org/wiki/FIFA_World_Cup", "sports"),
    ("https://en.wikipedia.org/wiki/Super_Bowl", "sports"),
    ("https://en.wikipedia.org/wiki/NBA", "sports"),
    ("https://en.wikipedia.org/wiki/UEFA_Champions_League", "sports"),
    ("https://en.wikipedia.org/wiki/Wimbledon_Championships", "sports"),
    ("https://en.wikipedia.org/wiki/Tour_de_France", "sports"),
    ("https://en.wikipedia.org/wiki/Formula_One", "sports"),
    ("https://en.wikipedia.org/wiki/Cricket_World_Cup", "sports"),
    ("https://en.wikipedia.org/wiki/Rugby_World_Cup", "sports"),
    ("https://en.wikipedia.org/wiki/Association_football", "sports"),
    ("https://en.wikipedia.org/wiki/Basketball", "sports"),
    ("https://en.wikipedia.org/wiki/Tennis", "sports"),
    ("https://en.wikipedia.org/wiki/Golf", "sports"),
    ("https://en.wikipedia.org/wiki/Swimming_(sport)", "sports"),
    # ORGANIZATION (10)
    ("https://en.wikipedia.org/wiki/United_Nations", "organization"),
    ("https://en.wikipedia.org/wiki/World_Health_Organization", "organization"),
    ("https://en.wikipedia.org/wiki/NASA", "organization"),
    ("https://en.wikipedia.org/wiki/European_Union", "organization"),
    ("https://en.wikipedia.org/wiki/NATO", "organization"),
    ("https://en.wikipedia.org/wiki/Red_Cross", "organization"),
    ("https://en.wikipedia.org/wiki/Greenpeace", "organization"),
    ("https://en.wikipedia.org/wiki/Amnesty_International", "organization"),
    ("https://en.wikipedia.org/wiki/World_Bank", "organization"),
    ("https://en.wikipedia.org/wiki/International_Monetary_Fund", "organization"),
]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    """Create a requests session with proper headers."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def api_request(session: requests.Session, params: dict, max_retries: int = 3) -> dict:
    """
    Make a MediaWiki API request with retry and exponential backoff.

    Returns the parsed JSON response, or an empty dict on failure.
    """
    params.setdefault("format", "json")
    for attempt in range(max_retries):
        try:
            resp = session.get(API_BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            wait = 2 ** attempt
            print(f"  [retry {attempt + 1}/{max_retries}] {exc} — waiting {wait}s")
            time.sleep(wait)
    return {}


def get_random_article_titles(count: int, session: requests.Session) -> list[str]:
    """
    Fetch `count` random article titles via the MediaWiki API (list=random).
    Returns a list of title strings.
    """
    titles = []
    batch_size = 50  # API max for list=random
    while len(titles) < count:
        n = min(batch_size, count - len(titles))
        data = api_request(session, {
            "action": "query",
            "list": "random",
            "rnnamespace": "0",  # main namespace only
            "rnlimit": str(n),
        })
        for page in data.get("query", {}).get("random", []):
            titles.append(page["title"])
        time.sleep(1)
    return titles


def get_article_data_batch(
    titles: list[str], session: requests.Session
) -> dict[str, dict]:
    """
    Batch-fetch extracts and categories for a list of titles.

    Returns {title: {"extract": str, "categories": [str]}} for each title.
    Processes up to 20 titles per API call (MediaWiki limit for extracts).
    """
    results: dict[str, dict] = {}
    batch_size = 20
    for i in range(0, len(titles), batch_size):
        batch = titles[i : i + batch_size]
        data = api_request(session, {
            "action": "query",
            "titles": "|".join(batch),
            "prop": "extracts|categories",
            "explaintext": "1",
            "exintro": "0",
            "cllimit": "20",
            "redirects": "1",
        })
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            title = page.get("title", "")
            extract = page.get("extract", "")
            cats = [c["title"] for c in page.get("categories", [])]
            results[title] = {"extract": extract, "categories": cats}
        time.sleep(1)
    return results


# ---------------------------------------------------------------------------
# Domain classification
# ---------------------------------------------------------------------------

def classify_domain(categories: list[str], title: str) -> str:
    """
    Classify a Wikipedia article into a domain using category keywords.
    Ported from the notebook's detect_domain() but works on API-returned
    category strings instead of scraping HTML.
    """
    combined = " ".join(categories).lower() + " " + title.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return domain
    return "other"


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

def extract_content_api(extract_text: str) -> str:
    """
    Clean the plaintext extract returned by the MediaWiki API.
    Removes reference markers and normalises whitespace.
    """
    text = re.sub(r"\[\d+\]", "", extract_text)
    text = " ".join(text.split())
    return text.strip()


def extract_content_bs4(url: str, session: requests.Session) -> str:
    """
    Fallback scraper: fetch the Wikipedia page HTML and extract paragraph text.
    Ported from the notebook's extract_text_from_wikipedia().
    """
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        content_div = soup.find("div", {"id": "mw-content-text"})
        if not content_div:
            return ""
        paragraphs = content_div.find_all("p")
        parts = []
        for p in paragraphs:
            text = p.get_text()
            text = re.sub(r"\[\d+\]", "", text)
            text = " ".join(text.split())
            if text:
                parts.append(text)
        return " ".join(parts)
    except Exception:
        return ""


def _title_from_url(url: str) -> str:
    """Extract the article title from a Wikipedia URL."""
    # e.g. https://en.wikipedia.org/wiki/Albert_Einstein -> Albert_Einstein
    prefix = "https://en.wikipedia.org/wiki/"
    if url.startswith(prefix):
        return requests.utils.unquote(url[len(prefix):]).replace("_", " ")
    return ""


# ---------------------------------------------------------------------------
# Pipeline: fixed dataset
# ---------------------------------------------------------------------------

def generate_fixed_dataset(
    count: int,
    group_id: str,
    session: requests.Session,
    min_words: int = 200,
) -> list[dict]:
    """
    Build the fixed URL dataset from the hardcoded FIXED_URLS list.
    Uses deterministic seed hash(group_id).
    Content is fetched via the MediaWiki API in batches.
    """
    random.seed(hash(group_id))
    urls_to_fetch = FIXED_URLS[:count]

    print(f"\n=== Generating Fixed Dataset ({len(urls_to_fetch)} URLs) ===")

    # Build title -> (url, domain) mapping
    title_map: dict[str, tuple[str, str]] = {}
    for url, domain in urls_to_fetch:
        title = _title_from_url(url)
        title_map[title] = (url, domain)

    titles = list(title_map.keys())
    print(f"Fetching article content for {len(titles)} titles...")
    article_data = get_article_data_batch(titles, session)

    dataset = []
    failed_titles = []

    for title in tqdm(titles, desc="Processing fixed URLs"):
        url, domain = title_map[title]
        data = article_data.get(title, {})
        content = extract_content_api(data.get("extract", ""))

        # Fallback to BS4 if API extract is too short
        if len(content.split()) < min_words:
            print(f"  API extract too short for '{title}', trying BS4 fallback...")
            content = extract_content_bs4(url, session)
            time.sleep(1)

        if len(content.split()) < min_words:
            print(f"  WARNING: '{title}' has only {len(content.split())} words (min {min_words})")
            failed_titles.append(title)

        dataset.append({
            "id": str(uuid.uuid4()),
            "url": url,
            "domain": domain,
            "content": content,
        })

    if failed_titles:
        print(f"\n  {len(failed_titles)} articles below {min_words}-word minimum:")
        for t in failed_titles:
            print(f"    - {t}")

    print(f"Fixed dataset: {len(dataset)} articles generated.")
    return dataset


# ---------------------------------------------------------------------------
# Pipeline: random dataset
# ---------------------------------------------------------------------------

def generate_random_dataset(
    count: int,
    group_id: str,
    fixed_urls: set[str],
    session: requests.Session,
    min_words: int = 200,
) -> list[dict]:
    """
    Build the random URL dataset via the MediaWiki API list=random.
    Uses timestamp-based seed so the set differs each run.
    Excludes any URLs that are already in the fixed set.
    """
    random.seed(hash(group_id + datetime.now().isoformat()))

    print(f"\n=== Generating Random Dataset ({count} URLs) ===")

    dataset = []
    seen_urls: set[str] = set(fixed_urls)
    stall_count = 0
    max_stalls = 10  # give up after 10 consecutive batches with zero new articles

    with tqdm(total=count, desc="Collecting random articles") as pbar:
        while len(dataset) < count:
            # Fetch a batch of random titles
            needed = count - len(dataset)
            fetch_count = min(50, max(needed * 2, 50))  # over-fetch to account for filtering
            titles = get_random_article_titles(fetch_count, session)

            # Batch-fetch their content and categories
            article_data = get_article_data_batch(titles, session)

            added_this_batch = 0
            for title, data in article_data.items():
                if len(dataset) >= count:
                    break

                content = extract_content_api(data.get("extract", ""))
                word_count = len(content.split())

                if word_count < min_words:
                    continue

                # Reconstruct URL from title
                url_title = title.replace(" ", "_")
                url = f"https://en.wikipedia.org/wiki/{requests.utils.quote(url_title, safe='(),_')}"

                if url in seen_urls:
                    continue

                domain = classify_domain(data.get("categories", []), title)

                seen_urls.add(url)
                dataset.append({
                    "id": str(uuid.uuid4()),
                    "url": url,
                    "domain": domain,
                    "content": content,
                })
                added_this_batch += 1
                pbar.update(1)

            if added_this_batch == 0:
                stall_count += 1
                if stall_count >= max_stalls:
                    print(f"\nWARNING: Stalled after {max_stalls} batches with no new articles. "
                          f"Got {len(dataset)}/{count}.")
                    break
            else:
                stall_count = 0

    print(f"Random dataset: {len(dataset)} articles generated.")
    return dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Wikipedia URL datasets for Conversational AI Assignment 2"
    )
    parser.add_argument("--fixed", type=int, default=200,
                        help="Number of fixed URLs to generate (default: 200)")
    parser.add_argument("--random", type=int, default=300,
                        help="Number of random URLs to generate (default: 300)")
    parser.add_argument("--group", type=str, default="GROUP_122",
                        help="Group ID for deterministic seeding (default: GROUP_122)")
    parser.add_argument("--min-words", type=int, default=200,
                        help="Minimum word count per article (default: 200)")
    parser.add_argument("--skip-fixed", action="store_true",
                        help="Skip generating fixed_url.json")
    parser.add_argument("--skip-random", action="store_true",
                        help="Skip generating random_url.json")
    args = parser.parse_args()

    session = _make_session()

    fixed_url_set: set[str] = set()

    # --- Fixed dataset ---
    if not args.skip_fixed:
        fixed_data = generate_fixed_dataset(
            count=args.fixed,
            group_id=args.group,
            session=session,
            min_words=args.min_words,
        )
        fixed_url_set = {entry["url"] for entry in fixed_data}

        with open("fixed_url.json", "w", encoding="utf-8") as f:
            json.dump(fixed_data, f, ensure_ascii=False, indent=2)
        print(f"Wrote fixed_url.json ({len(fixed_data)} entries)")
    else:
        # Load existing fixed URLs to exclude from random set
        try:
            with open("fixed_url.json", "r", encoding="utf-8") as f:
                existing = json.load(f)
            fixed_url_set = {entry["url"] for entry in existing}
            print(f"Loaded {len(fixed_url_set)} existing fixed URLs for deduplication")
        except FileNotFoundError:
            print("WARNING: fixed_url.json not found; random set won't deduplicate against it")

    # --- Random dataset ---
    if not args.skip_random:
        random_data = generate_random_dataset(
            count=args.random,
            group_id=args.group,
            fixed_urls=fixed_url_set,
            session=session,
            min_words=args.min_words,
        )

        with open("random_url.json", "w", encoding="utf-8") as f:
            json.dump(random_data, f, ensure_ascii=False, indent=2)
        print(f"Wrote random_url.json ({len(random_data)} entries)")

    print("\nDone!")


if __name__ == "__main__":
    main()
