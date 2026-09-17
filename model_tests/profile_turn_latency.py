# model_tests/profile_turn_latency.py
"""Runs interactive ORCA chat handle_turn on sample queries to profile per-stage latency breakdown."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = REPO_ROOT / "Proto"
if str(PROTO_DIR) not in sys.path:
    sys.path.insert(0, str(PROTO_DIR))

# pyrefly: ignore [missing-import]
from chat import build_engine, handle_turn
# pyrefly: ignore [missing-import]
from agents.ocean.model import load_environmental_datasets
# pyrefly: ignore [missing-import]
from agents.pfz.model import get_pfz_model
# pyrefly: ignore [missing-import]
from agents.weather.model import get_weather_model
# pyrefly: ignore [missing-import]
from agents.productivity.model import get_productivity_model
# pyrefly: ignore [missing-import]
from conversation.model import get_conversation_model
# pyrefly: ignore [missing-import]
from conversation.state import ConversationState
# pyrefly: ignore [missing-import]
from main import extract_location

def main():
    print("Pre-warming models and datasets...")
    load_environmental_datasets()
    get_pfz_model()
    get_weather_model()
    get_productivity_model()
    conv_model = get_conversation_model()
    engine = build_engine()
    print("✓ Models pre-warmed.\n")

    queries = [
        "Kal ke ki kelarar asepase kono jaygay mach dhorte jawa hobe???",
        "kal ke ki kelara r asepase kono jaygay mach dhorte jawa ajbe?",
        "Delhi ke as pas main koi fishing zone hai kya?"
    ]

    for q in queries:
        state = ConversationState()
        print("=" * 65)
        print(f"QUERY: \"{q}\"")
        handle_turn(q, conv_model, engine, state, debug=True)
        print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
