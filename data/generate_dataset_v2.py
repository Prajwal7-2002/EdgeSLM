"""Generate the v2 training set.

The v1 generator (generate_dataset.py) produces one-sentence documents that name
their own sensitivity ("The document contains production administrator
credentials for ..."). A model trained on it learns to spot the words
"credentials" and "confidential", not the ideas behind the labels. v2 targets
the ideas instead:

- RESTRICTED documents carry a live secret (a password, token, connection
  string, physical access code, key block) or an instruction for getting around
  a security control. Many never use a word like "password".
- PUBLIC documents freely use security vocabulary, because docs and tutorials
  talk about tokens and keys all the time, usually with placeholders such as
  <your-token> where a real value would go.
- INTERNAL documents describe how staff work, including security tooling,
  without containing a secret or anything about a specific person or deal.
- CONFIDENTIAL documents are about specific people, customers, deals or
  undisclosed plans and numbers.

Placeholders versus live values is the main contrast taught here: the same
config snippet is PUBLIC or INTERNAL with <placeholders> and RESTRICTED with a
real value filled in.

The frozen contrast test set (data/contrast/contrast_test.jsonl) was written
before this file by the same author. To keep it a fair test, the scenarios
below avoid the situations it uses (no expense deadlines, earnings-call
embargoes, door codes, URL-parameter bypasses and so on), and
src/evaluation/audit_dataset.py --contrast reports the nearest training
document for every test document so any remaining overlap is visible.

    python data/generate_dataset_v2.py
"""

import json
import random
import string
from pathlib import Path


SEED = 7
PER_CLASS = 450
VALIDATION_FRACTION = 0.1
OUTPUT_DIR = Path("data/v2")

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
    "Suresh Gowda", "Hannah Becker",
]
FIRST_NAMES = [name.split()[0] for name in PEOPLE]

CUSTOMERS = [
    "Bluepeak Insurance", "Orion Retail", "Sahyadri Foods", "Tidewater Freight",
    "Cobalt Health", "Maple Credit Union", "Vantage Telecom", "Indus Textiles",
    "Greenfield Schools", "Polaris Airlines", "Redwood Clinics", "Summit Energy",
    "Lakeshore Hotels", "Quartz Media", "Ironbridge Steel",
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

FEATURES = [
    "scheduled reports", "a dark theme", "bulk user import", "audit trails",
    "webhook retries", "custom dashboards", "a Kannada interface",
    "role-based permissions", "usage alerts", "SAML login", "a CLI tool",
    "a Python SDK",
]

CITIES = ["Pune", "Mysuru", "Hyderabad", "Chennai", "Kochi", "Dharwad", "Jaipur"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
ENV_NAMES = ["SERVICE_TOKEN", "APP_SECRET", "INGEST_KEY", "WEBHOOK_SECRET", "DB_PASSWORD", "SMTP_PASS"]

# Neutral closing lines shared by every class, so a trailing sentence carries
# no label signal of its own.
NEUTRAL_TAILS = [
    "", "", "", "",
    " Ping me if anything is unclear.",
    " Thanks.",
    " Updated after the last review.",
    " See the linked page for more.",
]


def tail():
    return pick(NEUTRAL_TAILS)


def money():
    return pick([
        f"{rng.randint(2, 95)} lakh rupees",
        f"${rng.randint(1, 40)}.{rng.randint(0, 9)} million",
        f"${rng.randint(120, 950)},000",
    ])


def percent():
    return f"{rng.randint(3, 48)}%"


def date():
    return f"{rng.randint(1, 28)} {pick(MONTHS)}"


# ------------------------------------------------------------------
# Secret material. Formats are random and deliberately avoid real
# provider prefixes so nothing here trips secret scanners.
# ------------------------------------------------------------------

WORDS = ["falcon", "river", "copper", "lantern", "monsoon", "orbit", "pepper",
         "granite", "willow", "ember", "cedar", "comet", "tulip", "basalt"]


def password():
    return pick([
        f"{pick(WORDS).capitalize()}{pick('!$%&@')}{rng.randint(10, 9999)}",
        f"{pick(WORDS)}.{pick(WORDS)}.{rng.randint(10, 99)}",
        "".join(rng.choice(string.ascii_letters + string.digits + "!@$") for _ in range(rng.randint(10, 16))),
    ])


def token():
    body = "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(rng.randint(24, 40)))
    return pick(["", "", "tk_", "svc-", "ops_"]) + body


def hex_string(n):
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def base64ish(n):
    return "".join(rng.choice(string.ascii_letters + string.digits + "+/") for _ in range(n))


def username():
    return pick(["etl", "svc_ingest", "reporting", "deployer", "backup_ro",
                 "sa", "app_rw", "metrics"]) + pick(["", str(rng.randint(1, 9))])


def host():
    return pick([
        f"172.{rng.randint(16, 31)}.{rng.randint(0, 255)}.{rng.randint(2, 254)}",
        f"{pick(['ledger', 'search', 'queue', 'mail'])}{rng.randint(1, 9)}.corp.example",
    ])


def digits(n):
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


def placeholder(kind):
    return pick({
        "token": ["<your-token>", "YOUR_API_TOKEN", "${TOKEN}", "<token>", "xxxxxxxx"],
        "password": ["<password>", "********", "YOUR_PASSWORD", "<db-password>"],
        "user": ["<username>", "YOUR_USER", "<user>"],
    }[kind])


def config_snippet(live):
    """The same snippet shape with either a live value or a placeholder."""
    name = pick(ENV_NAMES)
    value = (password() if "PASS" in name else token()) if live else placeholder(
        "password" if "PASS" in name else "token"
    )
    return pick([
        f"export {name}={value}",
        f"{name}: {value}",
        f"{pick(SERVICES)}.yaml:\n  user: {username() if live else placeholder('user')}\n  secret: {value}",
    ])


# ------------------------------------------------------------------
# RESTRICTED
# ------------------------------------------------------------------

LEAD_INS = [
    "", "", "FYI ", "Re the ingest outage: ", "{first}, ",
    "Copying this from the old notes: ", "Temporary until the new setup lands: ",
]


def lead_in():
    return pick(LEAD_INS).format(first=pick(FIRST_NAMES))


def restricted():
    kind = rng.randint(0, 8)
    service = pick(SERVICES)
    if kind == 0:
        return pick([
            f"{lead_in()}{service} runs as {username()} / {password()}.{tail()}",
            f"{lead_in()}for {service} use {username()} and {password()}, it has write access.{tail()}",
        ])
    if kind == 1:
        return f"{lead_in()}{config_snippet(live=True)}"
    if kind == 2:
        return pick([
            f"{lead_in()}{service} accepts this in the X-Api-Key header: {token()}{tail()}",
            f"{lead_in()}the webhook signing value for {service} is {hex_string(40)}.{tail()}",
            f"curl -H 'Authorization: Bearer {token()}' https://{host()}/v2/export",
        ])
    if kind == 3:
        scheme = pick(["postgresql", "mysql", "mongodb+srv", "redis", "amqps", "ftp"])
        return (
            f"{lead_in()}{scheme}://{username()}:{password()}@{host()}/"
            f"{pick(['ledger', 'events', 'files', 'reports'])}{tail()}"
        )
    if kind == 4:
        block = pick(["RSA PRIVATE KEY", "OPENSSH PRIVATE KEY", "EC PRIVATE KEY"])
        return pick([
            f"{lead_in()}{service} host key, needed on the new box: -----BEGIN {block}----- {base64ish(rng.randint(40, 70))}...",
            f"{lead_in()}keystore for {service}, store pass {password()}, alias {pick(WORDS)}.{tail()}",
        ])
    if kind == 5:
        return pick([
            f"{lead_in()}the cloud console root account still has its original sign-in: {password()}.{tail()}",
            f"{lead_in()}one-time codes for the {pick(TEAMS)} admin account: " + " ".join(digits(6) for _ in range(rng.randint(3, 5))) + ".",
            f"{lead_in()}{pick(FIRST_NAMES)}'s session cookie, if you need to act as them: sid={base64ish(32)}{tail()}",
        ])
    if kind == 6:
        control, how = pick([
            ("the deploy approval", "tag the commit with [hotfix] and the pipeline skips review"),
            ("the VPN requirement", "the staging admin panel is still open on port 8443 from anywhere"),
            ("the rate limiter", "send requests with the internal health-check user agent"),
            ("the data export limit", "call the v0 export route, it was never locked down"),
            ("the second approver", "create the change from the automation account and approve it yourself"),
            ("the access logs", "use the maintenance shell, its commands aren't recorded"),
        ])
        return f"{lead_in()}if {control} slows you down, {how}.{tail()}"
    if kind == 7:
        return f"{lead_in()}the {pick(TEAMS)} team's shared vault master value is {token()}; it unlocks every entry.{tail()}"
    return pick([
        f"{lead_in()}payment card for cloud spend: {digits(4)} {digits(4)} {digits(4)} {digits(4)}, expiry {rng.randint(1, 12):02d}/{rng.randint(27, 31)}, CVV {digits(3)}.{tail()}",
        f"{lead_in()}SNMP community string for the core switches: {pick(WORDS)}{digits(3)}.{tail()}",
    ])


# ------------------------------------------------------------------
# PUBLIC
# ------------------------------------------------------------------

def public():
    kind = rng.randint(0, 8)
    feature = pick(FEATURES)
    if kind == 0:
        return pick([
            f"Quickstart: install the SDK, then set your key with {config_snippet(live=False)} and run the example.",
            f"Docs: authenticate by sending your token in the X-Api-Key header, for example X-Api-Key: {placeholder('token')}.",
            f"Tutorial: store credentials in environment variables, never in code. Example: {config_snippet(live=False)}",
        ])
    if kind == 1:
        return pick([
            f"API reference: requests are limited to {rng.randint(5, 50) * 10} per minute per token. Responses include a Retry-After header.",
            f"Webhook docs: each delivery is signed with your endpoint's signing secret so you can verify it came from us.",
        ])
    if kind == 2:
        return pick([
            f"What's new: {feature} is live for every workspace from today.",
            f"Version {rng.randint(2, 9)}.{rng.randint(0, 20)} adds {feature} and fixes a sign-in bug on older browsers.",
        ])
    if kind == 3:
        return pick([
            f"Bug bounty: we pay up to {money()} for critical findings such as authentication bypasses or leaked keys. See the rules before testing.",
            "Security advisory: an issue in our desktop client could expose session tokens on shared machines. Update to the latest version; no action is needed on the server side.",
        ])
    if kind == 4:
        return pick([
            f"Our terms of service promise 99.{rng.randint(5, 9)}% monthly uptime, with service credits if we miss it.",
            "Community rules: be respectful, don't post personal data, and never share your account details in the forum.",
        ])
    if kind == 5:
        return pick([
            f"Call for speakers: our user conference in {pick(CITIES)} takes talks until {date()}.",
            f"Newsletter: this month we cover {feature}, a customer webinar, and our new {pick(CITIES)} office.",
        ])
    if kind == 6:
        customer = pick(CUSTOMERS)
        return pick([
            f"Testimonial from {customer}: '{feature} saved our team hours every week.' Quoted with permission.",
            f"Partner directory: {customer} is a certified implementation partner in {pick(CITIES)}.",
        ])
    if kind == 7:
        return pick([
            f"Open-source notice: the {feature} module is released under the Apache 2.0 license on our public repository.",
            f"App store description: manage your team, track usage and get alerts, now with {feature}.",
        ])
    return pick([
        f"Public roadmap: next quarter we plan to ship {feature}. Vote for features on our feedback board.",
        f"We published an anonymised dataset of {rng.randint(10, 90)}k support conversations for researchers, licensed CC BY 4.0.",
    ])


# ------------------------------------------------------------------
# INTERNAL
# ------------------------------------------------------------------

def internal():
    kind = rng.randint(0, 8)
    service = pick(SERVICES)
    team = pick(TEAMS)
    if kind == 0:
        return pick([
            f"How to read a secret from the vault: run vault read secret/{team}/<name>. Your team lead grants the policy. {tail()}".strip(),
            f"Local setup for {service}: copy .env.example to .env and fill in {config_snippet(live=False)} with values from the vault.",
        ])
    if kind == 1:
        return pick([
            f"On-call for {service} rotates weekly on Wednesdays. Swap shifts with a teammate and update the pager schedule.{tail()}",
            f"Postmortem template: summary, timeline, impact, root cause, action items. Blameless, and due within five working days.{tail()}",
        ])
    if kind == 2:
        return pick([
            f"Code review guidelines: keep pull requests under 400 lines and reply to comments within one working day.{tail()}",
            f"Naming conventions for {service}: kebab-case for queues, snake_case for tables, and a team prefix on every metric.{tail()}",
        ])
    if kind == 3:
        return pick([
            f"The staging data for {service} is reset every Sunday night. Don't rely on test accounts lasting longer than a week.{tail()}",
            f"We're moving {service} to the new scheduler next sprint. Expect a short pause in nightly jobs.{tail()}",
        ])
    if kind == 4:
        return pick([
            f"Hackathon logistics: teams of up to four, demos on Friday at 4pm, food in the {pick(CITIES)} office pantry.{tail()}",
            f"Desk moves: the {team} team moves to the second floor over the weekend. Pack personal items by Friday.{tail()}",
        ])
    if kind == 5:
        return pick([
            f"Training calendar: the {team} team runs a SQL basics session every other Tuesday. Sign up on the learning page.{tail()}",
            f"Dependency upgrades for {service} happen in the first week of each month; open one pull request per library.{tail()}",
        ])
    if kind == 6:
        return pick([
            f"Security tooling: the secret scanner runs on every push. If it flags a value, rotate it in the vault and mark the alert resolved.{tail()}",
            f"Service accounts are requested through the platform team's form. Their secrets are issued only through the vault.{tail()}",
        ])
    if kind == 7:
        return pick([
            f"Data retention procedure: logs from {service} are kept 30 days, then archived automatically.{tail()}",
            f"How we file IT issues: use the helpdesk form, pick a category, and attach a screenshot.{tail()}",
        ])
    return pick([
        f"Style guide: write docs in plain English, use sentence case for headings, and link to the source code.{tail()}",
        f"The {team} team's weekly planning happens in the shared board; tickets need an owner and an estimate before the sprint starts.{tail()}",
    ])


# ------------------------------------------------------------------
# CONFIDENTIAL
# ------------------------------------------------------------------

def confidential():
    kind = rng.randint(0, 8)
    person = pick(PEOPLE)
    customer = pick(CUSTOMERS)
    if kind == 0:
        return pick([
            f"Bonus pool split for the {pick(TEAMS)} team: {person} gets {money()}, the rest is shared equally.{tail()}",
            f"{person}'s work visa expires on {date()} and the renewal was refused once.{tail()}",
            f"Home addresses and personal phone numbers of the {pick(TEAMS)} team, for the emergency contact sheet.{tail()}",
        ])
    if kind == 1:
        return pick([
            f"Whistleblower report about {person}'s handling of vendor payments, received {date()}.{tail()}",
            f"{person} disclosed a disability and asked for adjusted working hours; HR approved.{tail()}",
        ])
    if kind == 2:
        return pick([
            f"Negotiation plan for {customer}: open at list price, concede up to {percent()} if they sign before quarter end.{tail()}",
            f"{customer}'s usage dropped {percent()} this quarter; their account manager thinks they are evaluating a competitor.{tail()}",
            f"Signed order form from {customer}: {money()} over {rng.randint(2, 5)} years, payment net 60.{tail()}",
        ])
    if kind == 3:
        return pick([
            f"Draft term sheet from the lead investor: {money()} at a pre-money valuation we have not disclosed.{tail()}",
            f"Unreleased product codename {pick(WORDS).capitalize()}: launch planned for {pick(MONTHS)} with {customer} as the design partner.{tail()}",
        ])
    if kind == 4:
        return pick([
            f"Due diligence notes on a possible partner: weak cash position, two pending lawsuits.{tail()}",
            f"We intend to raise prices by {percent()} for all {pick(CITIES)} customers next year; sales will be briefed first.{tail()}",
        ])
    if kind == 5:
        return pick([
            f"Support ticket from {customer}: their admin {person} shared a spreadsheet of patient appointments by mistake.{tail()}",
            f"Exported list of {rng.randint(50, 900)} trial users with emails and company sizes for the sales team.{tail()}",
        ])
    if kind == 6:
        return pick([
            f"External auditor's draft letter: control weaknesses in revenue recognition at {customer}'s subsidiary.{tail()}",
            f"Regulator inquiry about our handling of {customer} data, with our draft response.{tail()}",
        ])
    if kind == 7:
        return pick([
            f"Exit interview notes: {person} cited their manager as the main reason for leaving.{tail()}",
            f"Promotion committee minutes: {person} not promoted this cycle, feedback to be delivered privately.{tail()}",
        ])
    return pick([
        f"Margin analysis by customer: {customer} is loss-making at the current discount.{tail()}",
        f"Insurance claim after the {pick(CITIES)} office flood, with the payout we expect.{tail()}",
    ])


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
            text = text[0].upper() + text[1:]
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
