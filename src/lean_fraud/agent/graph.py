"""The fraud-triage agent: a `create_agent` ReAct graph wrapped in a deterministic safety net.

`triage()` runs a `create_agent` graph whose tools inspect the card behind a flagged transaction
(profile, recent activity, segment fraud rate), then coerces the outcome into the `Decision` schema
in up to three tiers:

  1. `structured_response` — if the model called `response_format`'s structured-output tool, use it.
  2. structured extraction — small local models reliably USE tools and REASON but often decide in
     prose instead of emitting the schema, so we re-ask via `with_structured_output(Decision)` (a
     separate, tool-free call they format far better).
  3. deterministic fallback — if both yield nothing (or anything raises / `recursion_limit` trips),
     decide by threshold on the TCN score. The pipeline ALWAYS returns a valid Decision, never hangs.

The store, model and compiled agent are all injectable so tests drive the graph with the mock
backend (which cannot bind tools) and exercise every tier without a network or Ollama.
"""

from __future__ import annotations
import logging
from langchain.agents.factory import create_agent
from lean_fraud.agent.llm import build_chat_model
from lean_fraud.agent.schema import AlertContext, Decision
from lean_fraud.agent.store import TransactionStore
from lean_fraud.agent.tools import build_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a fraud analyst reviewing a card payment that an upstream model flagged as suspicious. "
    "Use the tools to compare the flagged transaction against the card's baseline behaviour, its "
    "recent activity, and the fraud base rate for its (category, state) segment. Then decide: "
    "'block' for clear fraud, 'review' when it is suspicious and needs a human, or 'allow' when it "
    "is likely legitimate. Give a short rationale grounded in what the tools returned."
)


def _format_alert(ctx: AlertContext) -> str:
    t = ctx.transaction
    return (
        f"Flagged transaction on card {ctx.card_id} (model fraud score {ctx.score:.3f}).\n"
        f"Amount: {t.amt}, category: {t.category}, state: {t.state}, unix_time: {t.unix_time}."
    )


def build_agent(cfg: dict, store: TransactionStore | None = None, model=None):
    """Compile the ReAct agent: tools bound to the store, final answer forced into `Decision`."""
    store = store or TransactionStore.from_config(cfg)
    model = model or build_chat_model(cfg)
    return create_agent(
        model,
        build_tools(store),
        system_prompt=SYSTEM_PROMPT,
        response_format=Decision,
    )


def _extract_decision(model, messages) -> Decision | None:
    """Re-ask the model to restate its analysis as a structured Decision (tool-free, native schema).

    The reasoning + tool observations already live in `messages`; small models format a Decision far
    more reliably as this separate call than while also driving the tool loop. Best-effort: on any
    error the caller falls through to the deterministic fallback.
    """
    try:
        prompt = list(messages) + [("user", "Return your final triage decision.")]
        decision = model.with_structured_output(Decision).invoke(prompt)
        return decision if isinstance(decision, Decision) else None
    except Exception as exc:
        logger.warning("structured extraction failed (%s)", exc)
        return None


def _fallback(ctx: AlertContext, cfg: dict) -> Decision:
    """Decide by threshold on the TCN score when the LLM yields no valid Decision."""
    agent_cfg = cfg["agent"]
    if ctx.score >= agent_cfg.get("block_threshold", 0.9):
        action = "block"
    elif ctx.score >= agent_cfg.get("review_threshold", 0.5):
        action = "review"
    else:
        action = "allow"
    return Decision(action=action, rationale="fallback", confidence=ctx.score)


def triage(
    context: AlertContext, cfg: dict, agent=None, store: TransactionStore | None = None, model=None
) -> Decision:
    """Triage a flagged transaction. Always returns a Decision — never raises, never hangs."""
    recursion_limit = cfg["agent"].get("recursion_limit", 8)
    try:
        if agent is None:
            model = model or build_chat_model(cfg)  # keep a handle for the extraction tier
            agent = build_agent(cfg, store=store, model=model)
        result = agent.invoke(
            {"messages": [("user", _format_alert(context))]},
            config={"recursion_limit": recursion_limit},
        )
        # Tier 1: the model called the structured-output tool.
        decision = result.get("structured_response")
        if isinstance(decision, Decision):
            return decision
        # Tier 2: it reasoned in prose — extract the decision in a separate structured call.
        if model is not None:
            decision = _extract_decision(model, result["messages"])
            if isinstance(decision, Decision):
                return decision
        logger.warning("agent returned no structured Decision; using threshold fallback.")
    except Exception as e:
        logger.warning("agent failed (%s); using threshold fallback.", e, exc_info=True)
    return _fallback(context, cfg)
