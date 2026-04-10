#!/usr/bin/env python3
"""
Multi-Agent Orchestration: Coordinator → Analysts → Synthesis

THE PATTERN EXPLAINED
─────────────────────
Most "AI agents" are just one Claude call in a loop. This file shows a different
structure: multiple Claude instances with different roles, each unaware of the others.

  COORDINATOR   Reads the user's question. Decides WHICH analysts are needed
                and WHAT to ask each one. Returns a JSON plan. Has no tools.
                Think of it as a project manager who assigns work.

  ANALYSTS      Each is a separate Claude call with its own system prompt and
                its own Salesforce tool loop. Three available:
                  • pipeline       — opportunities, amounts, stages, close rates
                  • customer_health — support cases, at-risk accounts
                  • activity        — tasks, engagement, rep activity
                Each analyst only knows about its own domain. Zero coordination
                with the other analysts — that's by design.

  SYNTHESIS     Receives ALL analyst outputs. Has NO tools — it only reasons.
                Combines the findings into a single executive report.
                Think of it as a senior consultant who reads all the memos
                and writes the board presentation.

WHY THIS WORKS BETTER THAN ONE BIG AGENT
─────────────────────────────────────────
• Each analyst has a tight, focused system prompt — less likely to go off-track.
• Analysts can run in parallel (see asyncio.gather below).
• You can swap out one analyst without touching the others.
• The synthesis agent never queries Salesforce — it can't corrupt data.
• The coordinator's plan is inspectable — you can log it, test it, override it.

FLOW
────
User question
    │
    ▼
[Coordinator Claude]  — no tools, returns JSON plan
    │
    ├── plan.analysts = [pipeline, customer_health, activity]
    │
    ▼  (parallel)
[Pipeline Analyst]    [Customer Health Analyst]    [Activity Analyst]
  SOQL tool loop          SOQL tool loop               SOQL tool loop
    │                         │                             │
    └──────────┬──────────────┘                             │
               └─────────────────────────────┬─────────────┘
                                             ▼
                                   [Synthesis Claude]  — no tools, streams report
                                             │
                                             ▼
                                    Executive Report
"""

import asyncio
import json
import os
import re
from typing import AsyncGenerator

import anthropic
from mcp import ClientSession
from mcp.client.sse import sse_client

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")


# ── Shared helpers ────────────────────────────────────────────────────────────

def _to_dict(obj):
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return json.loads(json.dumps(obj, default=str))


async def _load_tools(sess: ClientSession):
    resp  = await sess.list_tools()
    tools = [{"name": t.name, "description": t.description or "", "input_schema": _to_dict(t.inputSchema)} for t in resp.tools]
    names = {t.name for t in resp.tools}
    return tools, names


async def _call_tool(sess: ClientSession, name: str, inputs: dict, names: set) -> str:
    if name not in names:
        return f"Unknown tool: {name}"
    r = await sess.call_tool(name, inputs)
    return r.content[0].text if r.content else "No data."


# ═══════════════════════════════════════════════════════════════════════════════
# COORDINATOR
# Reads the user question, returns a JSON plan with:
#   analysts      — which analysts to run, what to ask each
#   synthesis_focus — what the synthesis agent should emphasise
# ═══════════════════════════════════════════════════════════════════════════════

_COORDINATOR_SYSTEM = """You are a Salesforce business intelligence coordinator.

Your job is to read the user's question and decide which specialist analysts to invoke.

Available analysts and what they know:
  pipeline         Opportunity records: stages, amounts, close dates, win rates, pipeline trends
  customer_health  Case records: open cases, priorities, at-risk accounts, escalations
  activity         Task records: rep activity, calls logged, overdue tasks, dormant accounts

Return ONLY valid JSON, no markdown fences, no explanation outside the JSON:
{
  "reasoning": "one sentence explaining your analyst selection",
  "analysts": [
    {"name": "pipeline",         "question": "precise question tailored to the user's query"},
    {"name": "customer_health",  "question": "..."},
    {"name": "activity",         "question": "..."}
  ],
  "synthesis_focus": "what the final report should emphasise"
}

Rules:
- Only include analysts whose domain is relevant to the question
- If the question is broad or says "full report", include all three
- Make each question specific — the analyst only answers the question you give it
- synthesis_focus tells the synthesis agent what angle to take"""


async def _run_coordinator(client: anthropic.Anthropic, user_question: str) -> dict:
    """
    No tools. Just Claude reading the request and returning a plan.
    This is fast — single API call, no tool loop.
    """
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=_COORDINATOR_SYSTEM,
        messages=[{"role": "user", "content": user_question}],
    )
    raw = next((b.text for b in response.content if b.type == "text"), "{}")
    m   = re.search(r"\{.*\}", raw, re.DOTALL)
    return json.loads(m.group(0) if m else raw)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSTS
# Each has its own focused system prompt.
# They share a Salesforce MCP session but know nothing about each other.
# ═══════════════════════════════════════════════════════════════════════════════

_ANALYST_SYSTEMS = {
    "pipeline": """You are a Salesforce pipeline analyst. You ONLY look at Opportunity data.
Use run_soql to answer the question. Useful queries:
  SELECT StageName, COUNT(Id) cnt, SUM(Amount) total FROM Opportunity WHERE IsClosed=false GROUP BY StageName
  SELECT Name, Amount, CloseDate, StageName, Account.Name FROM Opportunity WHERE IsClosed=false ORDER BY Amount DESC LIMIT 10
  SELECT COUNT(Id) won, SUM(Amount) rev FROM Opportunity WHERE IsWon=true AND CloseDate = THIS_QUARTER
Return a focused 3-4 paragraph analysis with specific numbers, deal names, and dates.""",

    "customer_health": """You are a Salesforce customer health analyst. You ONLY look at Case data.
Use run_soql to answer the question. Useful queries:
  SELECT Priority, COUNT(Id) cnt FROM Case WHERE Status != 'Closed' GROUP BY Priority
  SELECT Account.Name, COUNT(Id) cnt FROM Case WHERE Status != 'Closed' GROUP BY Account.Name HAVING COUNT(Id) > 1
  SELECT CaseNumber, Subject, Priority, Account.Name, CreatedDate FROM Case WHERE Priority='High' AND Status!='Closed' ORDER BY CreatedDate DESC LIMIT 10
Return a focused 3-4 paragraph analysis highlighting risks and at-risk accounts.""",

    "activity": """You are a Salesforce activity analyst. You ONLY look at Task data.
Use run_soql to answer the question. Useful queries:
  SELECT Status, COUNT(Id) cnt FROM Task WHERE CreatedDate = THIS_MONTH GROUP BY Status
  SELECT Account.Name FROM Task WHERE ActivityDate < TODAY AND Status = 'Not Started' GROUP BY Account.Name LIMIT 10
  SELECT WhatId, COUNT(Id) cnt FROM Task WHERE CreatedDate = THIS_MONTH AND Type = 'Call' GROUP BY WhatId ORDER BY COUNT(Id) DESC LIMIT 5
Return a focused 3-4 paragraph analysis on rep engagement and dormant accounts.""",
}


async def _run_analyst(
    client: anthropic.Anthropic,
    name: str,
    question: str,
    sess: ClientSession,
    tools: list,
    tool_names: set,
) -> tuple[str, str]:
    """
    Runs one analyst. Returns (name, analysis_text).
    Each analyst is a completely independent Claude call — it doesn't know about
    the coordinator's plan or what the other analysts are doing.
    """
    system   = _ANALYST_SYSTEMS.get(name, "You are a Salesforce analyst. Use run_soql to answer the question.")
    messages = [{"role": "user", "content": question}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            tools=tools,
            messages=messages,
        )
        if response.stop_reason != "tool_use":
            break
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result_text = await _call_tool(sess, block.name, dict(block.input), tool_names)
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result_text})
        messages.append({"role": "assistant", "content": [_to_dict(b) for b in response.content]})
        messages.append({"role": "user", "content": tool_results})

    text = next((b.text for b in response.content if b.type == "text"), "No data available.")
    return name, text


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHESIS AGENT
# Receives all analyst outputs. Has NO tools — it only reasons.
# Streams its output so the UI can display it token by token.
# ═══════════════════════════════════════════════════════════════════════════════

_SYNTHESIS_SYSTEM = """You are a business intelligence synthesis agent.
You receive reports from specialist analysts and write a final executive report.

You have NO access to Salesforce. You only reason over what the analysts found.
Never make up numbers — only reference figures from the analyst reports.

Write a structured executive report:
## Executive Summary
(2-3 sentences answering the user's original question directly)

## Key Findings
(one bullet per analyst, with the most important data point)

## Risk Flags
(specific risks identified in the data, e.g. "3 high-priority cases open on top account")

## Recommended Actions
(3-5 prioritised, concrete actions based on the data)"""


async def _run_synthesis(
    client: anthropic.Anthropic,
    user_question: str,
    analyst_outputs: dict[str, str],
    synthesis_focus: str,
) -> AsyncGenerator[str, None]:
    """Streams the synthesis. No tools, just reasoning over analyst outputs."""
    sections = "\n\n".join(
        f"=== {name.upper().replace('_', ' ')} ANALYST ===\n{text}"
        for name, text in analyst_outputs.items()
    )
    user_msg = (
        f"User question: {user_question}\n\n"
        f"Focus: {synthesis_focus}\n\n"
        f"ANALYST REPORTS:\n{sections}\n\n"
        f"Write the executive report now."
    )
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=_SYNTHESIS_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        for chunk in stream.text_stream:
            yield chunk


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# Yields events the FastAPI endpoint can stream to the UI.
# ═══════════════════════════════════════════════════════════════════════════════

async def run_report(api_key: str, sf_token: str, sf_instance: str, user_question: str):
    """
    Yields tuples:
      ("coordinator", {"reasoning": str, "analysts": [...]} )
      ("analyst_start", analyst_name)
      ("analyst_done",  analyst_name)
      ("synthesis_start", "")
      ("text", chunk)
    """
    client  = anthropic.Anthropic(api_key=api_key)
    mcp_url = f"{MCP_SERVER_URL}?sf_token={sf_token}&sf_instance={sf_instance}"

    # ── Step 1: Coordinator ───────────────────────────────────────────────────
    plan = await _run_coordinator(client, user_question)
    yield ("coordinator", plan)

    async with sse_client(url=mcp_url) as (read, write):
        async with ClientSession(read, write) as sess:
            await sess.initialize()
            tools, tool_names = await _load_tools(sess)

            # ── Step 2: Run analysts in PARALLEL ─────────────────────────────
            # asyncio.gather runs all analysts concurrently — each is independent,
            # so there's no reason to wait for one before starting the next.
            analyst_tasks = [
                _run_analyst(client, item["name"], item["question"], sess, tools, tool_names)
                for item in plan["analysts"]
            ]

            # Emit "starting" events before gather
            for item in plan["analysts"]:
                yield ("analyst_start", item["name"])

            results = await asyncio.gather(*analyst_tasks, return_exceptions=True)

            analyst_outputs: dict[str, str] = {}
            for result in results:
                if isinstance(result, Exception):
                    # One analyst failed — continue without it
                    continue
                name, text = result
                analyst_outputs[name] = text
                yield ("analyst_done", name)

    # ── Step 3: Synthesis ─────────────────────────────────────────────────────
    # Note: synthesis happens OUTSIDE the MCP connection context because it
    # doesn't need Salesforce. This is intentional — the synthesis agent is
    # a pure reasoning step with no data access.
    yield ("synthesis_start", "")
    async for chunk in _run_synthesis(client, user_question, analyst_outputs, plan.get("synthesis_focus", "")):
        yield ("text", chunk)
