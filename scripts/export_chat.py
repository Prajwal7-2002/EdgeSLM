import json
from pathlib import Path


SOURCE = Path("chat.json")
OUTPUT = Path("chat_transcript.md")


def response_text(response):
    parts = []
    for item in response or []:
        if item.get("kind") in {"thinking", "toolInvocationSerialized", "mcpServersStarting"}:
            continue
        value = item.get("value")
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n\n".join(parts)


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    lines = [
        "# EdgeSLM Coding Agent Session",
        "",
        "> Cleaned transcript for technical screening review.",
        "> Internal reasoning, tool payloads, IDs, and UI metadata have been removed.",
        "",
    ]

    for index, request in enumerate(data.get("requests", []), start=1):
        user_text = request.get("message", {}).get("text", "").strip()
        assistant_text = response_text(request.get("response", []))
        if not user_text and not assistant_text:
            continue

        lines.extend([f"## Turn {index}", ""])
        if user_text:
            lines.extend(["### User", "", user_text, ""])
        if assistant_text:
            lines.extend(["### GitHub Copilot", "", assistant_text, ""])

        OUTPUT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8-sig")
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    main()
