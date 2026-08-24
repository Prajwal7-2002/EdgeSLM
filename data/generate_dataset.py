import json
import random
from pathlib import Path


SEED = 42
TOTAL_PER_CLASS = 250

OUTPUT_PATH = Path("data/raw/generated_documents.jsonl")

random.seed(SEED)


DEPARTMENTS = [
    "engineering",
    "finance",
    "human resources",
    "sales",
    "customer support",
    "security",
    "operations",
    "product",
]


# ============================================================
# PUBLIC
# ============================================================

PUBLIC_TEMPLATES = [
    "The company announced {topic} as part of its public product update.",
    "The company published documentation explaining {topic}.",
    "A public announcement describes the company's work on {topic}.",
    "The company's website provides customers with information about {topic}.",
    "The product documentation explains how customers can use {topic}.",
]

PUBLIC_TOPICS = [
    "a new cloud storage feature",
    "the latest product release",
    "customer account configuration",
    "two-factor authentication setup",
    "supported file formats",
    "product pricing",
    "available API features",
    "the company's sustainability initiative",
    "new customer-facing integrations",
    "the public security documentation",
    "mobile application settings",
    "account recovery options",
    "notification preferences",
    "data export capabilities",
    "application accessibility features",
    "customer onboarding resources",
    "API documentation",
    "service availability information",
    "integration documentation",
    "developer documentation",
    "billing configuration",
    "subscription management",
    "product compatibility information",
    "system requirements",
    "release notes",
    "feature documentation",
    "public support resources",
    "customer help documentation",
    "security best practices",
    "password management guidance",
    "authentication documentation",
    "application usage instructions",
    "supported integrations",
    "cloud deployment documentation",
    "data import functionality",
    "account management features",
    "user profile settings",
    "public product roadmap",
    "product comparison information",
    "technical documentation",
    "customer service information",
    "platform documentation",
    "software installation instructions",
    "browser compatibility information",
    "public compliance documentation",
    "privacy documentation",
    "terms of service",
    "public status information",
    "customer notification settings",
    "product configuration options",
]


def generate_public_examples():
    examples = []

    for template in PUBLIC_TEMPLATES:
        for topic in PUBLIC_TOPICS:
            examples.append(
                template.format(topic=topic)
            )

    return examples


# ============================================================
# INTERNAL
# ============================================================

INTERNAL_TEMPLATES = [
    "The {department} team documented its internal procedure for {topic}.",
    "Employees in the {department} department should follow the internal workflow for {topic}.",
    "The engineering organization uses an internal process for {topic}.",
    "This internal document describes how the {department} team handles {topic}.",
    "The team prepared internal guidance covering {topic}.",
]

INTERNAL_TOPICS = [
    "deploying applications to the staging environment",
    "reviewing pull requests",
    "handling internal support tickets",
    "planning the next sprint",
    "onboarding new employees",
    "managing development environments",
    "conducting internal testing",
    "requesting access to development systems",
    "coordinating release schedules",
    "reviewing internal operational procedures",
    "managing internal documentation",
    "approving software changes",
    "handling development incidents",
    "coordinating engineering tasks",
    "maintaining internal dashboards",
    "reviewing deployment requests",
    "managing internal repositories",
    "conducting team planning",
    "tracking engineering work",
    "processing internal requests",
]


def generate_internal_examples():
    candidates = []

    for template in INTERNAL_TEMPLATES:
        for department in DEPARTMENTS:
            for topic in INTERNAL_TOPICS:
                candidates.append(
                    template.format(
                        department=department,
                        topic=topic,
                    )
                )

    return candidates


# ============================================================
# CONFIDENTIAL
# ============================================================

CONFIDENTIAL_TEMPLATES = [
    "The {department} report contains customer information including account identifiers and contact details related to {topic}.",
    "The {department} quarterly report contains non-public financial information about {topic}.",
    "The {department} document contains confidential business information concerning {topic}.",
    "The {department} customer analysis includes account history, contact information, and business metrics related to {topic}.",
    "The {department} internal business report contains confidential information about {topic}.",
]

CONFIDENTIAL_TOPICS = [
    "customer retention",
    "enterprise accounts",
    "regional revenue",
    "customer support activity",
    "sales performance",
    "employee compensation",
    "business forecasts",
    "customer usage patterns",
    "contract negotiations",
    "product adoption",
    "customer churn",
    "regional sales",
    "account activity",
    "business performance",
    "customer acquisition",
    "subscription revenue",
    "sales pipeline",
    "account growth",
    "customer demographics",
    "financial projections",
]


def generate_confidential_examples():
    candidates = []

    for template in CONFIDENTIAL_TEMPLATES:
        for department in DEPARTMENTS:
            for topic in CONFIDENTIAL_TOPICS:
                candidates.append(
                    template.format(
                        department=department,
                        topic=topic,
                    )
                )

    return candidates


# ============================================================
# RESTRICTED
# ============================================================

RESTRICTED_TEMPLATES = [
    "The security document contains privileged credentials used to access {system}.",
    "The incident report contains private authentication material associated with {system}.",
    "The document contains production administrator credentials for {system}.",
    "The security record contains private encryption keys used by {system}.",
    "The restricted security report contains highly sensitive access information for {system}.",
]

SYSTEMS = [
    "the production database",
    "the identity service",
    "the cloud infrastructure",
    "the production API",
    "the internal authentication service",
    "the deployment platform",
    "the backup system",
    "the monitoring infrastructure",
    "the payment processing service",
    "the customer database",
    "the employee directory",
    "the authentication gateway",
    "the production cluster",
    "the secrets management service",
    "the database administration system",
    "the internal API gateway",
    "the logging infrastructure",
    "the container registry",
    "the CI/CD infrastructure",
    "the production Kubernetes cluster",
    "the service account infrastructure",
    "the internal messaging system",
    "the identity management platform",
    "the privileged access system",
    "the production storage system",
    "the internal network gateway",
    "the security monitoring platform",
    "the infrastructure management system",
    "the production cache",
    "the internal service registry",
    "the database replication system",
    "the production load balancer",
    "the authentication database",
    "the internal secrets vault",
    "the production job scheduler",
    "the deployment controller",
    "the infrastructure provisioning system",
    "the internal certificate service",
    "the production configuration service",
    "the privileged administrator portal",
    "the production message queue",
    "the internal audit system",
    "the security incident platform",
    "the production backup service",
    "the internal access gateway",
    "the service credential store",
    "the production monitoring service",
    "the internal authorization service",
    "the infrastructure credential system",
    "the production administration portal",
]


def generate_restricted_examples():
    examples = []

    for template in RESTRICTED_TEMPLATES:
        for system in SYSTEMS:
            examples.append(
                template.format(system=system)
            )

    return examples


# ============================================================
# DATASET GENERATION
# ============================================================

GENERATORS = {
    "PUBLIC": generate_public_examples,
    "INTERNAL": generate_internal_examples,
    "CONFIDENTIAL": generate_confidential_examples,
    "RESTRICTED": generate_restricted_examples,
}


def generate_dataset():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    examples = []

    for label, generator in GENERATORS.items():

        candidates = generator()

        # Remove duplicates inside the class
        candidates = list(dict.fromkeys(candidates))

        if len(candidates) < TOTAL_PER_CLASS:
            raise ValueError(
                f"{label} only has {len(candidates)} unique "
                f"examples, but {TOTAL_PER_CLASS} are required."
            )

        # Randomly select exactly 250 unique examples
        selected = random.sample(
            candidates,
            TOTAL_PER_CLASS,
        )

        for text in selected:
            examples.append(
                {
                    "text": text,
                    "label": label,
                }
            )

    # Global uniqueness check
    texts = [x["text"] for x in examples]

    if len(texts) != len(set(texts)):
        raise ValueError(
            "Duplicate texts detected in generated dataset."
        )

    random.shuffle(examples)

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        for example in examples:

            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("Dataset generated successfully.")
    print("Total examples:", len(examples))
    print("Examples per class:", TOTAL_PER_CLASS)
    print("Unique examples:", len(set(texts)))
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    generate_dataset()