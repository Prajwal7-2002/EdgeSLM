"""Download real documents into a folder ready for src/scan.py.

    python src/data/download_real_docs.py --sec-user-agent "Your Name you@example.com"
    python src/data/download_real_docs.py --sources enron handbook --per-source 15

Sources:
  enron     Enron email corpus (CMU, 443 MB download, cached). Real business
            email, including passwords, salaries and deals. Real people: keep
            the files local and never commit or republish them.
  handbook  GitLab Handbook pages (MIT / CC BY-SA 4.0): real company
            procedures, the closest public stand-in for INTERNAL documents.
  sec       SEC EDGAR press releases (US government, public domain). Needs
            --sec-user-agent with your name and email: the SEC blocks
            requests without one.

Writes <out>/<source>/<file> plus <out>/labels.csv with an empty your_label
column. Fill it in from the label definitions BEFORE running the classifier,
then compare with:

    python src/scan.py real_docs --labels real_docs/labels.csv

The script never assigns labels. "selected_because" only records which
keyword pulled an email in, so you can see how the sample was drawn.
"""

import argparse
import csv
import json
import random
import re
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


ENRON_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
HANDBOOK_PROJECT = "gitlab-com%2Fcontent-sites%2Fhandbook"
HANDBOOK_API = f"https://gitlab.com/api/v4/projects/{HANDBOOK_PROJECT}/repository/tree"
HANDBOOK_RAW = "https://gitlab.com/gitlab-com/content-sites/handbook/-/raw/main/"
HANDBOOK_SECTIONS = (
    "content/handbook/security",
    "content/handbook/engineering/infrastructure",
    "content/handbook/people-group",
    "content/handbook/finance",
    "content/handbook/support",
)
SEC_SEARCH = "https://efts.sec.gov/LATEST/search-index"
CACHE = Path("data/downloads")

# Keyword groups used to draw a varied email sample. They decide which emails
# get pulled, not what label they get.
ENRON_BUCKETS = {
    "password-or-login": re.compile(r"\b(password|passwd|login|user ?id|pin number)\b", re.I),
    "pay-or-personal": re.compile(r"\b(salary|bonus|compensation|termination|severance|medical leave)\b", re.I),
    "deal-or-contract": re.compile(r"\b(confidential|term sheet|draft contract|merger|acquisition)\b", re.I),
    "routine-logistics": re.compile(r"\b(meeting|conference room|schedule|lunch|reminder)\b", re.I),
    "announcement": re.compile(r"\b(press release|announce[sd]?|newsletter)\b", re.I),
}
MAX_EMAIL_BYTES = 20_000
RESERVOIR = 300

USER_AGENT = "EdgeSLM document-classifier evaluation (research)"


def fetch(url, user_agent=USER_AGENT, attempts=3):
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))


def download_file(url, destination):
    if destination.exists():
        print(f"  using cached {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"  downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as handle:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(1 << 20):
            handle.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 2**20:,.0f} / {total / 2**20:,.0f} MB", end="", flush=True)
    print()
    partial.rename(destination)
    return destination


def enron(out, count, rng):
    archive = download_file(ENRON_URL, CACHE / "enron_mail_20150507.tar.gz")
    per_bucket = max(1, count // len(ENRON_BUCKETS))

    # Reservoir-sample every keyword group across the whole archive, so the
    # emails come from many mailboxes instead of the first few in the tarball.
    reservoirs = defaultdict(list)
    seen = Counter()
    print("  scanning all ~500k emails (a few minutes)...")
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if not member.isfile() or member.size > MAX_EMAIL_BYTES:
                continue
            parts = member.name.split("/")
            mailbox = parts[1] if len(parts) > 2 else "unknown"
            data = tar.extractfile(member).read()
            body = data.decode("latin-1", errors="replace")
            for bucket, pattern in ENRON_BUCKETS.items():
                if pattern.search(body):
                    seen[bucket] += 1
                    reservoir = reservoirs[bucket]
                    if len(reservoir) < RESERVOIR:
                        reservoir.append((mailbox, data))
                    else:
                        slot = rng.randrange(seen[bucket])
                        if slot < RESERVOIR:
                            reservoir[slot] = (mailbox, data)
                    break

    rows = []
    folder = out / "enron"
    folder.mkdir(parents=True, exist_ok=True)
    for bucket, candidates in reservoirs.items():
        rng.shuffle(candidates)
        used_mailboxes = set()
        chosen = []
        for mailbox, data in candidates:
            if mailbox not in used_mailboxes:
                used_mailboxes.add(mailbox)
                chosen.append((mailbox, data))
            if len(chosen) == per_bucket:
                break
        for index, (mailbox, data) in enumerate(chosen):
            # Enron file names end in a dot ("1."), which Windows cannot store,
            # so every email is saved under a new name as .eml.
            path = folder / f"{bucket}_{mailbox}_{index:02d}.eml"
            path.write_bytes(data)
            rows.append({"file": path, "source": "enron", "selected_because": bucket})
    return rows


def handbook(out, count, rng):
    pages = []
    for section in HANDBOOK_SECTIONS:
        query = urllib.parse.urlencode({"path": section, "recursive": "true", "per_page": 100})
        entries = json.loads(fetch(f"{HANDBOOK_API}?{query}"))
        pages += [e["path"] for e in entries if e["type"] == "blob" and e["path"].endswith(".md")]
    rng.shuffle(pages)

    rows = []
    folder = out / "handbook"
    folder.mkdir(parents=True, exist_ok=True)
    for page in pages[:count]:
        name = page.removeprefix("content/handbook/").replace("/", "__")
        path = folder / name
        path.write_bytes(fetch(HANDBOOK_RAW + urllib.parse.quote(page)))
        rows.append({"file": path, "source": "handbook", "selected_because": page.split("/")[2]})
        time.sleep(0.2)
    return rows


def sec(out, count, rng, user_agent):
    if not user_agent:
        print("  skipped: pass --sec-user-agent \"Your Name you@example.com\" (required by the SEC)")
        return []
    query = urllib.parse.urlencode({"q": '"press release"', "forms": "8-K"})
    hits = json.loads(fetch(f"{SEC_SEARCH}?{query}", user_agent))["hits"]["hits"]
    rng.shuffle(hits)

    rows = []
    folder = out / "sec"
    folder.mkdir(parents=True, exist_ok=True)
    for hit in hits:
        if len(rows) >= count:
            break
        accession, filename = hit["_id"].split(":", 1)
        if not filename.lower().endswith((".htm", ".html", ".txt")):
            continue
        cik = int(hit["_source"]["ciks"][0])
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{filename}"
        company = re.sub(r"[^A-Za-z0-9]+", "-", hit["_source"]["display_names"][0].split("(")[0]).strip("-")
        path = folder / f"{company}_{filename}"
        path.write_bytes(fetch(url, user_agent))
        rows.append({"file": path, "source": "sec", "selected_because": "8-K press release"})
        time.sleep(0.2)  # the SEC allows at most 10 requests per second
    return rows


def write_labels(out, rows):
    """Add new files to labels.csv, keeping any labels already filled in."""
    path = out / "labels.csv"
    existing = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = {row["file"]: row for row in csv.DictReader(handle)}
    for row in rows:
        key = row["file"].relative_to(out).as_posix()
        existing.setdefault(key, {
            "file": key,
            "source": row["source"],
            "selected_because": row["selected_because"],
            "your_label": "",
            "notes": "",
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "source", "selected_because", "your_label", "notes"])
        writer.writeheader()
        writer.writerows(existing.values())
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("real_docs"))
    parser.add_argument("--sources", nargs="*", default=["enron", "handbook", "sec"],
                        choices=["enron", "handbook", "sec"])
    parser.add_argument("--per-source", type=int, default=20)
    parser.add_argument("--sec-user-agent", default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for source in args.sources:
        print(f"[{source}]")
        try:
            if source == "enron":
                rows += enron(args.out, args.per_source, rng)
            elif source == "handbook":
                rows += handbook(args.out, args.per_source, rng)
            else:
                rows += sec(args.out, args.per_source, rng, args.sec_user_agent)
        except Exception as error:
            print(f"  failed: {error}", file=sys.stderr)
        print(f"  {sum(r['source'] == source for r in rows)} files")

    if not rows:
        raise SystemExit("nothing downloaded")
    labels = write_labels(args.out, rows)
    print(f"\n{len(rows)} files in {args.out}/. Label them in {labels} (PUBLIC, INTERNAL, "
          "CONFIDENTIAL or RESTRICTED) before running src/scan.py.")


if __name__ == "__main__":
    main()
