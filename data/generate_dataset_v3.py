"""Generate the v3 training set for EdgeSLM.

v1: Generated template sentences with self-naming labels ("The document contains credentials...").
v2: Taught concepts (placeholders vs live secrets, security vocabulary in public docs),
    but accidentally created a massive spurious correlation:
    - INTERNAL: Sterile process pages with ZERO people names, ZERO emails, ZERO signatures.
    - CONFIDENTIAL: Filled with people names and contact numbers.
    Result: On real documents (e.g. Enron emails), the model flagged 19 of 26 ordinary INTERNAL
    emails as CONFIDENTIAL because real emails contain coworker names and email headers!

v3 fixes this root cause:
- INTERNAL: Heavily augmented with realistic coworker communications (meeting scheduling,
  shift swaps, 1:1 invites, PR reviews, travel logistics, IT tickets, forwarded articles)
  with people names, realistic email headers (From, To, Subject, Date) and signatures.
- PUBLIC: Adds author bylines, conference speakers, and open-source contributor credits.
- CONFIDENTIAL: Specifically focuses on actual sensitive material (compensation reviews,
  whistleblower reports, undisclosed valuations, customer churn, acquisition term sheets).
- RESTRICTED: Keeps live secrets, tokens, connection strings, and credential leaks.

Usage:
    uv run python data/generate_dataset_v3.py
"""

import json
import random
import string
from pathlib import Path

SEED = 42
PER_CLASS = 500
VALIDATION_FRACTION = 0.1
OUTPUT_DIR = Path("data/v3")

rng = random.Random(SEED)


def pick(options):
    return rng.choice(options)


# ------------------------------------------------------------------
# Shared slot pools
# ------------------------------------------------------------------

PEOPLE = [
    "Ananya Iyer", "Vikram Shetty", "Fatima Sheikh", "Daniel Okafor",
    "Lena Fischer", "Kiran Desai", "Sofia Martins", "Rahul Bhat", "Grace Liu",
    "Tomasz Nowak", "Aisha Bello", "Nikhil Joshi", "Elena Petrova",
    "Marcus Reid", "Divya Hegde", "Omar Haddad", "Chen Wei", "Isabel Ruiz",
    "Suresh Gowda", "Hannah Becker", "John Arnold", "Sally Beck",
    "Mark Haedicke", "Richard Sanders", "Louise Kitchen", "Vince Kaminski",
]
FIRST_NAMES = [name.split()[0] for name in PEOPLE]

CUSTOMERS = [
    "Bluepeak Insurance", "Orion Retail", "Sahyadri Foods", "Tidewater Freight",
    "Cobalt Health", "Maple Credit Union", "Vantage Telecom", "Indus Textiles",
    "Greenfield Schools", "Polaris Airlines", "Redwood Clinics", "Summit Energy",
    "Lakeshore Hotels", "Quartz Media", "Ironbridge Steel", "Delta Power", "Cargill Energy",
]

SERVICES = [
    "the reporting API", "the ingest worker", "the analytics warehouse",
    "the notification service", "the search cluster", "the mail relay",
    "the backup job", "the ledger service", "the scheduler",
    "the export service", "the image pipeline", "the file store",
    "the metrics collector", "the queue consumer", "the sync daemon",
]

TEAMS = [
    "platform", "finance", "support", "sales", "design", "data", "security",
    "people operations", "mobile", "infrastructure", "marketing", "legal",
]

ROLES = [
    "Staff SRE", "Engineering Manager", "Lead Architect", "Product Manager",
    "DevOps Engineer", "Data Scientist", "Support Engineer", "Account Executive",
]

DOMAINS = ["enron.com", "company.internal", "corp.net", "techops.io", "cloudscale.org"]

FEATURES = [
    "scheduled reports", "a dark theme", "bulk user import", "audit trails",
    "webhook retries", "custom dashboards", "a Kannada interface",
    "role-based permissions", "usage alerts", "SAML login", "a CLI tool",
    "a Python SDK",
]

CITIES = ["Pune", "Mysuru", "Hyderabad", "Chennai", "Kochi", "Dharwad", "Jaipur", "Houston", "London"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
WORDS = ["falcon", "amber", "cedar", "orbit", "beacon", "summit", "drift", "harbor", "matrix"]
ENV_NAMES = ["SERVICE_TOKEN", "APP_SECRET", "INGEST_KEY", "WEBHOOK_SECRET", "DB_PASSWORD", "SMTP_PASS"]

NEUTRAL_TAILS = [
    "", "", "",
    " Ping me if anything is unclear.",
    " Thanks.",
    " Best regards.",
    " Let me know what you think.",
    " See the attached page for details.",
]


def tail():
    return pick(NEUTRAL_TAILS)


def digits(n):
    return "".join(rng.choices(string.digits, k=n))


def hex_string(n):
    return "".join(rng.choices("0123456789abcdef", k=n))


def token():
    prefix = pick(["sk_live_", "ghp_", "xoxb-", "glpat-", "AKIA"])
    return prefix + "".join(rng.choices(string.ascii_letters + string.digits, k=28))


def placeholder(name):
    return pick([f"<{name}>", f"${{{name.upper()}}}", f"YOUR_{name.upper()}"])


def password():
    return "".join([
        pick(["Copper!", "Winter#", "Falcon$", "Anchor%", "Summit@", "Amber*"]),
        digits(4),
    ])


def username():
    return pick(["svc_ingest", "deployer", "admin", "reports_bot", "etl_worker", "backup_user"])


def host():
    return f"{pick(WORDS)}-{pick(['prod', 'staging', 'db', 'internal'])}.{pick(DOMAINS)}"


def date():
    return f"{pick(MONTHS)} {rng.randint(1, 28)}, 2026"


def money():
    return pick([
        f"{rng.randint(2, 95)} lakh rupees",
        f"${rng.randint(1, 40)}.{rng.randint(0, 9)} million",
        f"${rng.randint(120, 950)},000",
    ])


def percent():
    return f"{rng.randint(4, 35)}%"


def phone():
    return f"+1 ({rng.randint(200, 900)}) {rng.randint(200, 900)}-{digits(4)}"


def email_header(sender, recipient, subject):
    s_user = sender.lower().replace(" ", ".")
    r_user = recipient.lower().replace(" ", ".")
    domain = pick(DOMAINS)
    return (
        f"From: {sender} <{s_user}@{domain}>\n"
        f"To: {recipient} <{r_user}@{domain}>\n"
        f"Date: {date()}\n"
        f"Subject: {subject}\n\n"
    )


def email_signature(person):
    first = person.split()[0]
    return f"\n\nThanks,\n{first}\n--\n{person}\n{pick(ROLES)} · {pick(TEAMS).capitalize()} Team\nPhone: {phone()}"


# ------------------------------------------------------------------
# 1. RESTRICTED: Live secrets, credentials, auth bypasses
# ------------------------------------------------------------------

def restricted():
    kind = rng.randint(0, 8)
    service = pick(SERVICES)
    p1 = pick(PEOPLE)
    p2 = pick(PEOPLE)

    if kind == 0:
        return pick([
            f"{service} runs as {username()} / {password()}.{tail()}",
            f"For {service} write access use {username()} with secret {password()}.{tail()}",
            f"{email_header(p1, p2, f'Credentials for {service}')}Hey {p2.split()[0]},\nHere is the login for {service}: user '{username()}', password '{password()}'.{email_signature(p1)}",
        ])
    if kind == 1:
        scheme = pick(["postgresql", "mysql", "mongodb+srv", "redis", "amqps"])
        return pick([
            f"{scheme}://{username()}:{password()}@{host()}/{pick(['ledger', 'events', 'files'])}{tail()}",
            f"{email_header(p1, p2, 'Updated staging DB string')}Connection string for the new DB instance:\n{scheme}://{username()}:{password()}@{host()}/app_db{email_signature(p1)}",
        ])
    if kind == 2:
        return pick([
            f"{service} accepts this in the X-Api-Key header: {token()}{tail()}",
            f"curl -H 'Authorization: Bearer {token()}' https://{host()}/v2/export",
            f"{email_header(p1, p2, 'Production API Token')}Here is the token for {service}: {token()}{email_signature(p1)}",
        ])
    if kind == 3:
        block = pick(["RSA PRIVATE KEY", "OPENSSH PRIVATE KEY", "EC PRIVATE KEY"])
        body = "".join(rng.choices(string.ascii_letters + string.digits, k=48))
        return f"Host key for {service}:\n-----BEGIN {block}-----\n{body}\n-----END {block}-----{tail()}"
    if kind == 4:
        return pick([
            f"The root cloud account console sign-in is: {password()}.{tail()}",
            f"One-time backup codes for {pick(TEAMS)} root account: " + " ".join(digits(6) for _ in range(4)) + ".",
        ])
    if kind == 5:
        control, how = pick([
            ("the deploy approval", "tag the commit with [hotfix] and the pipeline skips review"),
            ("the VPN requirement", "the staging admin panel is still open on port 8443 from anywhere"),
            ("the rate limiter", "send requests with the internal health-check user agent"),
            ("the data export limit", "call the v0 export route, it was never locked down"),
        ])
        return f"If {control} slows you down, {how}.{tail()}"
    if kind == 6:
        return f"The {pick(TEAMS)} team's master vault key is {token()}; it unlocks all secret entries.{tail()}"
    if kind == 7:
        return f"Payment card for cloud compute: {digits(4)} {digits(4)} {digits(4)} {digits(4)}, expiry {rng.randint(1, 12):02d}/{rng.randint(27, 31)}, CVV {digits(3)}.{tail()}"
    return f"SNMP community string for core switches: {pick(WORDS)}{digits(4)}.{tail()}"


# ------------------------------------------------------------------
# 2. PUBLIC: Tutorials, public docs, press releases, blog authors
# ------------------------------------------------------------------

def public():
    kind = rng.randint(0, 8)
    feature = pick(FEATURES)
    person = pick(PEOPLE)
    customer = pick(CUSTOMERS)

    if kind == 0:
        return pick([
            f"Quickstart: install the SDK, set your token with API_KEY={placeholder('token')} and run the client.",
            f"Docs: authenticate by passing your key in the header: X-Api-Key: {placeholder('token')}.",
            f"Tutorial: always read secrets from environment variables, for example DB_PASSWORD={placeholder('password')}.",
        ])
    if kind == 1:
        return pick([
            f"API Reference: endpoints are limited to {rng.randint(3, 30) * 10} requests per second per token.",
            f"Release Announcement: version {rng.randint(2, 6)}.{rng.randint(0, 9)} is live today with {feature}.",
        ])
    if kind == 2:
        return pick([
            f"Blog post by {person}: 'Scaling {service_name(feature)} with Zero-Downtime Deploys'. Read on our engineering blog.",
            f"Conference keynote: {person} ({pick(ROLES)}) presents at CloudCon {pick(CITIES)} on {feature}.",
        ])
    if kind == 3:
        return pick([
            f"Bug bounty: we award up to {money()} for critical security findings such as auth bypasses.",
            f"Security advisory: patch available for desktop client CVE-2026-1049. Update recommended immediately.",
        ])
    if kind == 4:
        return pick([
            f"Press release: {customer} selects our platform to power their {feature} infrastructure across {pick(CITIES)}.",
            f"Customer story: '{feature} reduced our processing overhead by 40%', says {customer} VP of Tech.",
        ])
    if kind == 5:
        return pick([
            f"Terms of service: service availability SLA is 99.{rng.randint(5, 9)}% monthly uptime.",
            f"Community guidelines: respect other members, do not post PII, and keep discussions technical.",
        ])
    if kind == 6:
        return pick([
            f"Open source: the {feature} client library is available under the Apache 2.0 license on GitHub.",
            f"Contributor spotlight: special thanks to community contributor {person} for testing {feature}.",
        ])
    if kind == 7:
        return pick([
            f"Newsletter digest: highlights from {pick(MONTHS)}, customer webinars, and our new office in {pick(CITIES)}.",
            f"Documentation FAQ: common questions regarding single-sign-on and {feature} configuration.",
        ])
    return f"Public developer roadmap: {feature} is scheduled for Q3 public beta. Vote on our feedback forum."


def service_name(feat):
    return feat.replace("a ", "").replace("an ", "")


# ------------------------------------------------------------------
# 3. INTERNAL: Policies, handbook, AND REAL COWORKER EMAILS
# ------------------------------------------------------------------

def internal():
    kind = rng.randint(0, 11)
    service = pick(SERVICES)
    team = pick(TEAMS)
    p1 = pick(PEOPLE)
    p2 = pick(PEOPLE)
    customer = pick(CUSTOMERS)

    # Handbook & policy styles (from v2)
    if kind == 0:
        return pick([
            f"How to read a secret from vault: run 'vault read secret/{team}/<name>'. Your lead approves access.{tail()}",
            f"Local dev setup for {service}: copy .env.example to .env and fill in {placeholder('api_key')} from vault.{tail()}",
        ])
    if kind == 1:
        return pick([
            f"On-call rotation for {service} changes weekly on Wednesdays. Update the pager schedule if swapping shifts.{tail()}",
            f"Postmortem guidelines: document timeline, root cause, and follow-up tickets. Blameless culture.{tail()}",
        ])
    if kind == 2:
        return pick([
            f"Code review standards: pull requests should be scoped under 400 lines and reviewed within 24 hours.{tail()}",
            f"Naming convention for {service}: snake_case for tables and kebab-case for message queues.{tail()}",
        ])
    if kind == 3:
        return pick([
            f"Staging database for {service} is refreshed every Sunday. Do not store permanent test accounts there.{tail()}",
            f"Upcoming maintenance: {service} will undergo routine index rebuilding on Saturday night.{tail()}",
        ])

    # Realistic Coworker Emails & Meeting Notes (The missing piece that breaks the shortcut!)
    if kind == 4:
        return (
            f"{email_header(p1, p2, f'Rescheduling {service} sync')}"
            f"Hi {p2.split()[0]},\n\n"
            f"Could we push our 1:1 sync on {service} to tomorrow at 3:00 PM? "
            f"I have a conflict with the {team} sprint review this morning."
            f"{email_signature(p1)}"
        )
    if kind == 5:
        return (
            f"{email_header(p1, p2, f'Feedback on {service} pull request')}"
            f"Hey {p2.split()[0]},\n\n"
            f"Left a few minor comments on your PR for {service}. The retry logic looks solid overall. "
            f"Let's merge once the CI pipeline finishes running."
            f"{email_signature(p1)}"
        )
    if kind == 6:
        return (
            f"{email_header(p1, 'team-all', f'Staffing & site visit for {customer} deployment')}"
            f"Hi team,\n\n"
            f"Please note that {p1} and {p2} will be visiting the {customer} facility in {pick(CITIES)} next week "
            f"to assist with the initial network cutover. They will be reachable on cell during business hours."
            f"{email_signature(p1)}"
        )
    if kind == 7:
        return (
            f"{email_header(p1, p2, 'Shift coverage request')}"
            f"Hey {p2.split()[0]},\n\n"
            f"Would you be able to take my primary on-call rotation next Tuesday? "
            f"Happy to take your weekend shift on {date()} in exchange."
            f"{email_signature(p1)}"
        )
    if kind == 8:
        return (
            f"IT Helpdesk Request #{digits(4)}:\n"
            f"User: {p1} ({team.capitalize()} Team)\n"
            f"Request: Standard issue development laptop setup with Ubuntu and local Docker daemon.\n"
            f"Status: Approved by {p2.split()[0]}. Hardware shipping to {pick(CITIES)} office.{tail()}"
        )
    if kind == 9:
        return (
            f"{email_header(p1, 'engineering-leads', 'FYI: Industry trading and regulatory updates')}"
            f"Sharing a useful summary on recent energy market scheduling changes and FERC reporting rules: "
            f"https://energy-regulations-digest.org/articles/2026-summary. Good background context for our sprint planning."
            f"{email_signature(p1)}"
        )
    if kind == 10:
        return (
            f"{email_header(p1, 'all-staff', 'Office notice: Facilities maintenance')}"
            f"Hello all,\n\n"
            f"The facilities team will be performing HVAC maintenance on the 3rd floor of the {pick(CITIES)} office "
            f"this Friday afternoon. Please use the meeting rooms on the 2nd floor if needed."
            f"{email_signature(p1)}"
        )
    return (
        f"{email_header(p1, p2, 'Lunch / coffee catch up')}"
        f"Hi {p2.split()[0]},\n\n"
        f"Let's grab a quick coffee in the cafeteria at 2pm to catch up on the {service} milestones."
        f"{email_signature(p1)}"
    )


# ------------------------------------------------------------------
# 4. CONFIDENTIAL: Sensitive HR, compensation, term sheets, legal
# ------------------------------------------------------------------

def confidential():
    kind = rng.randint(0, 7)
    person = pick(PEOPLE)
    p2 = pick(PEOPLE)
    customer = pick(CUSTOMERS)
    team = pick(TEAMS)

    if kind == 0:
        return pick([
            f"Executive bonus allocation for the {team} leadership: {person} receives {money()}, the remainder held for retention.{tail()}",
            f"{email_header(p2, person, 'Strictly Confidential: Annual Compensation Adjustment')}{person.split()[0]},\n\nYour revised base compensation for the upcoming fiscal year has been approved at {money()}. Please keep this confidential.{email_signature(p2)}",
        ])
    if kind == 1:
        return pick([
            f"Work visa renewal for {person} in {pick(CITIES)} was refused by immigration authorities; legal counsel drafting appeal.{tail()}",
            f"Disciplinary review notes: investigation into {person}'s unauthorized disclosure of customer records.{tail()}",
            f"Whistleblower report received {date()} regarding accounting irregularities at {customer}'s subsidiary.{tail()}",
        ])
    if kind == 2:
        return pick([
            f"Negotiation strategy for {customer}: open at full rate card, authorize sales VP to discount up to {percent()} if signed by quarter end.{tail()}",
            f"Customer risk memo: {customer}'s platform traffic dropped {percent()} following their executive turnover.{tail()}",
            f"Signed purchase agreement with {customer}: {money()} over {rng.randint(2, 5)} years with custom indemnity clauses.{tail()}",
        ])
    if kind == 3:
        return pick([
            f"Series C term sheet draft: {money()} investment at a {money()} pre-money valuation. Embargoed until formal announcement.{tail()}",
            f"Acquisition evaluation memo: initial diligence on {customer}'s IP portfolio, valuation capped at {money()}.{tail()}",
        ])
    if kind == 4:
        return pick([
            f"Emergency contact details and personal home addresses for all employees in the {team} department.{tail()}",
            f"Export of {rng.randint(50, 400)} churned customer records including direct phone numbers and account spend figures.{tail()}",
        ])
    if kind == 5:
        return pick([
            f"Legal settlement discussion notes: outside counsel recommends offering {money()} to resolve pending patent dispute.{tail()}",
            f"Internal audit finding: unrecorded liabilities of {money()} identified in {pick(CITIES)} regional accounts.{tail()}",
        ])
    if kind == 6:
        return pick([
            f"Performance improvement plan for {person}: 30-day objectives and severance terms if targets are not met.{tail()}",
            f"Promotion committee closed deliberations: {person} not recommended for advancement; private feedback scheduled.{tail()}",
        ])
    return f"Profitability analysis by client: accounts for {customer} are operating at a negative {percent()} margin under current discount.{tail()}"


# ------------------------------------------------------------------
# Main generation loop
# ------------------------------------------------------------------

GENERATORS = {
    "PUBLIC": public,
    "INTERNAL": internal,
    "CONFIDENTIAL": confidential,
    "RESTRICTED": restricted,
}


def main():
    rows = []
    seen = set()
    for label, generate in GENERATORS.items():
        produced = 0
        attempts = 0
        while produced < PER_CLASS:
            attempts += 1
            if attempts > PER_CLASS * 50:
                raise RuntimeError(f"{label}: could not produce {PER_CLASS} unique documents")
            text = generate().strip()
            if text in seen:
                continue
            seen.add(text)
            rows.append({"text": text, "label": label})
            produced += 1

    rng.shuffle(rows)
    cut = int(len(rows) * VALIDATION_FRACTION)
    splits = {"validation": rows[:cut], "train": rows[cut:]}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        path = OUTPUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in split:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path}: {len(split)} rows")


if __name__ == "__main__":
    main()
