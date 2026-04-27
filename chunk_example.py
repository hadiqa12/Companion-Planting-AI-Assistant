"""Example: load plants.json and build one text chunk for embedding (RAG)."""
import json
from pathlib import Path

# Same folder as this script
JSON_PATH = Path(__file__).resolve().parent / "plants.json"


def plant_to_chunk(p: dict) -> str:
    lines = [
        f"Plant: {p['plant']} ({p['id']}). Family: {p['family']}.",
        f"Companions: {', '.join(p['companions'])}.",
        f"Antagonists: {', '.join(p['antagonists'])}.",
        f"Why companions help: {p['companion_benefits']}",
        f"Why avoid antagonists: {p['antagonist_reasons']}",
        (
            f"Growing: {p['sun']} sun, water {p['water']}, "
            f"spacing {p['spacing_inches']} inches, {p['nutrient_role']}."
        ),
    ]
    if p.get("pests_deterred"):
        lines.append(f"Pests deterred: {', '.join(p['pests_deterred'])}.")
    if p.get("attracts"):
        lines.append(f"Attracts: {', '.join(p['attracts'])}.")
    return "\n".join(lines)


def main() -> None:
    with JSON_PATH.open(encoding="utf-8") as f:
        plants = json.load(f)

    tomato = next(p for p in plants if p["plant"] == "Tomato")
    chunk = plant_to_chunk(tomato)
    print(chunk)
    print()
    print(f"--- Chunk length: {len(chunk)} characters ---")


if __name__ == "__main__":
    main()
