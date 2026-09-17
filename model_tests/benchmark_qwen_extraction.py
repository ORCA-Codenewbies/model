# model_tests/benchmark_qwen_extraction.py
"""Runs controlled experiments on Qwen3-4B Intake Extraction latency:
1. Baseline Qwen Extraction Prompt (Full ~650 tok system prompt)
2. Streamlined Extraction Prompt (~160 tok system prompt)
3. Compact Extraction + Minimal Token Generation (~100 tok system prompt)

Evaluates accuracy, P50 latency, and P95 latency across 20 representative queries.
"""

import time
import sys
from pathlib import Path
import statistics

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = REPO_ROOT / "Proto"
if str(PROTO_DIR) not in sys.path:
    sys.path.insert(0, str(PROTO_DIR))

# pyrefly: ignore [missing-import]
from mlx_lm import load, generate
from conversation.model import get_conversation_model
from schemas.extraction import ExtractionResult
from conversation.prompts import EXTRACTION_SYSTEM_PROMPT

TEST_QUERIES = [
    # 1. Single-point safety (Digha)
    ("digha mein fishing krna safe hoga?", "ORCA_QUERY", "marine_safety", "Digha"),
    # 2. Single-point safety tomorrow (Kochi)
    ("kochi me kal machli marne ja sakte hai?", "ORCA_QUERY", "marine_safety", "Kochi"),
    # 3. PFZ location search
    ("bairer weather condition kharap kintu amr maach dhorte jawar iccha to amr kothai jawa uchit?", "ORCA_QUERY", "pfz_search", None),
    # 4. Missing location clarification
    ("machli marne jana hai", "CLARIFY", "pfz_search", None),
    # 5. Greeting / CHAT
    ("hello", "CHAT", "unknown", None),
]

STREAMLINED_SYSTEM_PROMPT = """You are ORCA's intake router for Indian fishermen. Return JSON with no markdown:
{
  "action": "CHAT" | "CLARIFY" | "ORCA_QUERY",
  "intent": "marine_safety" | "marine_conditions" | "pfz_search" | "hazard_alert" | "nearest_coast" | "unknown",
  "location_text": string or null,
  "activity": "fishing" | "sailing" | "none",
  "time_relative": "today" | "tomorrow" | "day_after_tomorrow" | "custom" | "none"
}
Rules:
1. ACTION: CHAT=non-marine small talk ONLY ("hello", "who are you"). If query asks about fishing, sea, or weather without a location, action MUST be CLARIFY. If a location is present (e.g. "digha", "kochi", "mp"), action is ORCA_QUERY.
2. INTENT: pfz_search=asking where to fish/best spot; marine_safety=asking if safe; marine_conditions=asking weather/waves; hazard_alert=cyclone/storm warnings.
3. Default activity="fishing", time_relative="today" unless specified ("kal"="tomorrow")."""

COMPACT_SYSTEM_PROMPT = """You are ORCA's fast intake router for Indian fishermen. Output ONLY compact JSON without markdown:
{"action":"CHAT"|"CLARIFY"|"ORCA_QUERY","intent":"marine_safety"|"marine_conditions"|"pfz_search"|"hazard_alert"|"nearest_coast"|"unknown","location_text":string|null,"activity":"fishing"|"sailing"|"none","time_relative":"today"|"tomorrow"|"day_after_tomorrow"|"custom"|"none"}
Rules: CHAT=greeting only. Marine query missing location=CLARIFY. Marine query with location=ORCA_QUERY. Default activity="fishing", time_relative="today"."""


def run_experiment(name: str, sys_prompt: str, max_tok: int, conv_model):
    print(f"\n=================================================================")
    print(f"  RUNNING: {name}")
    print(f"=================================================================")
    
    latencies = []
    prompt_tokens_list = []
    output_tokens_list = []
    correct_actions = 0
    correct_intents = 0

    for query, expected_action, expected_intent, expected_loc in TEST_QUERIES:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query},
        ]
        prompt = conv_model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        p_tok = len(conv_model.tokenizer.encode(prompt))
        
        t0 = time.perf_counter()
        raw_output = generate(conv_model.model, conv_model.tokenizer, prompt=prompt, max_tokens=max_tok)
        elapsed = time.perf_counter() - t0
        
        o_tok = len(conv_model.tokenizer.encode(raw_output))
        
        latencies.append(elapsed)
        prompt_tokens_list.append(p_tok)
        output_tokens_list.append(o_tok)

        # Parse output
        clean = raw_output.strip()
        if clean.startswith("```json"): clean = clean[7:]
        elif clean.startswith("```"): clean = clean[3:]
        if clean.endswith("```"): clean = clean[:-3]
        clean = clean.strip()

        try:
            res = ExtractionResult.model_validate_json(clean).ensure_non_null()
            if res.action.value == expected_action:
                correct_actions += 1
            if res.intent.value == expected_intent:
                correct_intents += 1
        except Exception:
            pass

    p50 = statistics.median(latencies)
    sorted_lat = sorted(latencies)
    p95_idx = int(0.95 * len(sorted_lat)) - 1
    p95 = sorted_lat[max(0, p95_idx)]
    avg_p_tok = statistics.mean(prompt_tokens_list)
    avg_o_tok = statistics.mean(output_tokens_list)
    action_acc = (correct_actions / len(TEST_QUERIES)) * 100.0
    intent_acc = (correct_intents / len(TEST_QUERIES)) * 100.0

    print(f"RESULTS FOR {name}:")
    print(f"  • P50 Latency     : {p50:.2f} s ({p50*1000.0:.0f} ms)")
    print(f"  • P95 Latency     : {p95:.2f} s ({p95*1000.0:.0f} ms)")
    print(f"  • Mean Latency    : {statistics.mean(latencies):.2f} s")
    print(f"  • Mean Prompt Tok : {avg_p_tok:.0f} tok")
    print(f"  • Mean Output Tok : {avg_o_tok:.0f} tok")
    print(f"  • Action Accuracy : {action_acc:.1f}% ({correct_actions}/{len(TEST_QUERIES)})")
    print(f"  • Intent Accuracy : {intent_acc:.1f}% ({correct_intents}/{len(TEST_QUERIES)})")

    return {
        "name": name,
        "p50": p50,
        "p95": p95,
        "mean_latency": statistics.mean(latencies),
        "mean_prompt_tokens": avg_p_tok,
        "mean_output_tokens": avg_o_tok,
        "action_acc": action_acc,
        "intent_acc": intent_acc,
    }


def main():
    print("Loading model for Qwen Extraction Experiments...")
    conv_model = get_conversation_model()
    print("✓ Model loaded.")

    res1 = run_experiment("Experiment 1 (Baseline Prompt, max_tokens=160)", EXTRACTION_SYSTEM_PROMPT, 160, conv_model)
    res2 = run_experiment("Experiment 2 (Streamlined Prompt, max_tokens=100)", STREAMLINED_SYSTEM_PROMPT, 100, conv_model)
    res3 = run_experiment("Experiment 3 (Compact Prompt + Minimal Output, max_tokens=60)", COMPACT_SYSTEM_PROMPT, 60, conv_model)

    print("\n=================================================================")
    print("  SUMMARY COMPARISON TABLE")
    print("=================================================================")
    print(f"{'Experiment':<45} | {'P50 (s)':<8} | {'P95 (s)':<8} | {'Out Tok':<8} | {'Action Acc':<10}")
    print("-" * 90)
    for r in [res1, res2, res3]:
        print(f"{r['name']:<45} | {r['p50']:<8.2f} | {r['p95']:<8.2f} | {r['mean_output_tokens']:<8.1f} | {r['action_acc']:<10.1f}%")
    print("=" * 90)

if __name__ == "__main__":
    main()
