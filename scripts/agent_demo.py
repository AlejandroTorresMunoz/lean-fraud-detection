"""Demo: run the fraud-triage agent against a local Ollama model on one held-out transaction.

Overrides the config to provider=ollama (configs/base.yaml stays on `mock` for CI), draws a RANDOM
transaction from the time-based TEST split (so every run shows a different, genuinely held-out case),
and triages it — printing the ReAct trace (the reason/act loop) and the final Decision.

Run from the repo root, with Ollama running and the chosen model pulled:

    uv run python scripts/agent_demo.py
    uv run python scripts/agent_demo.py --model qwen2.5:7b
    uv run python scripts/agent_demo.py --fraud-only --seed 0

If the model fails tool-calling, the structured decision comes back empty and triage() falls back to
a threshold decision (rationale='fallback') — the pipeline never hangs on the model.
"""

from __future__ import annotations

import argparse
import os

# Turn LangSmith tracing off (must be set BEFORE importing langchain) so the demo never POSTs traces.
# Hard-assign (not setdefault): override any LANGSMITH_TRACING/API key already set in the shell.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import numpy as np  # noqa: E402

from lean_fraud.agent.graph import _extract_decision, _format_alert, build_agent  # noqa: E402
from lean_fraud.agent.llm import build_chat_model  # noqa: E402
from lean_fraud.agent.schema import AlertContext  # noqa: E402
from lean_fraud.agent.store import TransactionStore  # noqa: E402
from lean_fraud.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fraud-triage agent against Ollama.")
    parser.add_argument("--model", default=None, help="override the Ollama model, e.g. qwen2.5:7b")
    parser.add_argument("--config", default="configs/base.yaml", help="path to the YAML config")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (omit for a random draw)")
    parser.add_argument(
        "--fraud-only", action="store_true", help="only sample transactions labelled fraud"
    )
    parser.add_argument(
        "--score", type=float, default=0.97, help="synthetic TCN score for the alert"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["agent"]["provider"] = "ollama"  # override; base.yaml stays on mock
    if args.model:
        cfg["agent"]["model"] = args.model
    print(f"provider={cfg['agent']['provider']}  model={cfg['agent'].get('model')}")

    store = TransactionStore.from_config(cfg)  # loads the real raw Sparkov CSVs
    tx = store.sample_test_transaction(
        test_size=cfg["dataset"]["test_size"],
        val_size=cfg["dataset"]["val_size"],
        rng=np.random.default_rng(args.seed),
        fraud_only=args.fraud_only,
    )
    card_id = str(tx["cc_num"])
    print(f"card={card_id}  is_fraud={tx.get('is_fraud')}")
    print(f"profile={store.card_profile(card_id)}\n")

    # A real held-out transaction, presented as an alert the upstream TCN flagged.
    ctx = AlertContext.from_alert({"tx": tx, "prob": args.score})

    model = build_chat_model(cfg)  # built explicitly so we can reuse it for the extraction step
    agent = build_agent(cfg, store=store, model=model)
    result = agent.invoke(
        {"messages": [("user", _format_alert(ctx))]},
        config={"recursion_limit": cfg["agent"]["recursion_limit"]},
    )

    print("=== reason/act trace (the ReAct loop) ===")
    for message in result["messages"]:
        message.pretty_print()

    # Same two-step triage() uses: tier-1 structured_response, else extract the decision from prose.
    decision = result.get("structured_response") or _extract_decision(model, result["messages"])
    print("\n=== structured decision (what triage() returns) ===")
    print(decision)


if __name__ == "__main__":
    main()
