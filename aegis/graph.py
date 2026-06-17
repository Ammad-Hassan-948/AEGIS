"""AEGIS LangGraph workflow.

Node topology:
  intake
    └─► parallel_agents (appropriateness + contraindication + dosing run concurrently)
          └─► supervisor_compiler
                └─► safety_gate ──┬─► counsellor ─► finalizer ─► END
                                  │
                                  └─► supervisor_compiler  (retry loop, max 3 times)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any, Dict

from langgraph.graph import StateGraph, END

from state import AegisState
from agents.intake_agent import run_intake_agent
from agents.appropriateness_agent import run_appropriateness_agent
from agents.contraindication_agent import run_contraindication_agent
from agents.dosing_agent import run_dosing_agent
from agents.supervisor_agent import run_supervisor_agent
from agents.counsellor_agent import run_counsellor_agent
from gates.safety_gate import safety_gate_node

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Async helper
# ─────────────────────────────────────────────────────────────────────────────

def _run_async(coro) -> Any:
    """Run an async coroutine safely from a synchronous LangGraph node.

    On Windows, ProactorEventLoop is thread-unsafe so we always use
    SelectorEventLoop when creating a loop in a background thread.
    """
    import sys
    try:
        asyncio.get_running_loop()
        # Already inside a running loop — offload to a fresh thread with its own loop
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_run_in_new_loop, coro).result()
    except RuntimeError:
        # No running loop in this thread — create one directly
        return _run_in_new_loop(coro)


def _run_in_new_loop(coro) -> Any:
    """Create a SelectorEventLoop (thread-safe on Windows) and run coro to completion."""
    import sys
    if sys.platform == "win32":
        loop = asyncio.SelectorEventLoop()
    else:
        loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()
        asyncio.set_event_loop(None)


# ─────────────────────────────────────────────────────────────────────────────
# Node functions (synchronous wrappers around async agents)
# ─────────────────────────────────────────────────────────────────────────────

def node_intake(state: AegisState) -> Dict[str, Any]:
    logger.info("[graph] ► intake")
    return _run_async(run_intake_agent(state))


def node_parallel_agents(state: AegisState) -> Dict[str, Any]:
    """Run all three clinical agents concurrently and merge their results."""
    logger.info("[graph] ► parallel_agents")

    async def _gather():
        results = await asyncio.gather(
            run_appropriateness_agent(state),
            run_contraindication_agent(state),
            run_dosing_agent(state),
        )
        merged: Dict[str, Any] = {}
        all_chunks = list(state.retrieved_chunks)
        for result in results:
            chunks = result.pop("retrieved_chunks", [])
            all_chunks.extend(chunks)
            merged.update(result)
        merged["retrieved_chunks"] = all_chunks
        return merged

    return _run_async(_gather())


def node_supervisor(state: AegisState) -> Dict[str, Any]:
    logger.info("[graph] ► supervisor_compiler (iteration=%d)", state.iteration_count)
    return _run_async(run_supervisor_agent(state))


def node_safety_gate(state: AegisState) -> Dict[str, Any]:
    logger.info("[graph] ► safety_gate")
    result = safety_gate_node(state)
    # Persist next_step into state so the routing function can read it
    return result


def node_counsellor(state: AegisState) -> Dict[str, Any]:
    logger.info("[graph] ► counsellor")
    return _run_async(run_counsellor_agent(state))


def node_finalizer(state: AegisState) -> Dict[str, Any]:
    logger.info("[graph] ► finalizer — pipeline complete")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Conditional routing
# ─────────────────────────────────────────────────────────────────────────────

def route_after_safety_gate(state: AegisState) -> str:
    """Return the name of the next node based on safety_gate's next_step signal."""
    next_step = state.next_step
    if next_step == "retry_compilation":
        logger.info("[graph] safety_gate → supervisor_compiler (retry)")
        return "supervisor_compiler"
    logger.info("[graph] safety_gate → counsellor")
    return "counsellor"


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    builder = StateGraph(AegisState)

    # Nodes
    builder.add_node("intake", node_intake)
    builder.add_node("parallel_agents", node_parallel_agents)
    builder.add_node("supervisor_compiler", node_supervisor)
    builder.add_node("safety_gate", node_safety_gate)
    builder.add_node("counsellor", node_counsellor)
    builder.add_node("finalizer", node_finalizer)

    # Entry point
    builder.set_entry_point("intake")

    # Edges
    builder.add_edge("intake", "parallel_agents")
    builder.add_edge("parallel_agents", "supervisor_compiler")
    builder.add_edge("supervisor_compiler", "safety_gate")

    # Conditional: safety_gate → counsellor OR → supervisor_compiler (retry)
    builder.add_conditional_edges(
        "safety_gate",
        route_after_safety_gate,
        {
            "counsellor": "counsellor",
            "supervisor_compiler": "supervisor_compiler",
        },
    )

    builder.add_edge("counsellor", "finalizer")
    builder.add_edge("finalizer", END)

    return builder


# Build and export
aegis_graph = build_graph().compile()
logger.info("[graph] AEGIS graph compiled.")
