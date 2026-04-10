#!/usr/bin/env python3
"""
Three agent patterns for the Salesforce AI Assistant.

  AGENT 1 — DealInvestigator
    Multi-hop: finds a deal → digs into account, contacts, open cases → deal briefing.
    WHY AGENT: the path is dynamic — what to query next depends on what was found.

  AGENT 2 — BackgroundMonitor
    Autonomous asyncio task, no user session. Polls Salesforce every N minutes,
    stores a JSON health report in memory.
    WHY AGENT: proactive — nobody has to ask. Runs silently in the background.

  AGENT 3 — BusinessReview (subagents)
    Orchestrator Claude has two tools: analyze_pipeline + analyze_customer_health.
    Each tool spawns a SEPARATE Claude call (a subagent) with its own tool loop.
    Orchestrator synthesizes both reports into one executive review.
    WHY SUBAGENTS: parallel specialisation — each sub-Claude focuses on one domain
    and the orchestrator never has to know the SOQL details.
"""

import asyncio
import json
import os
import re
from datetime import datetime

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


async def _load_mcp_tools(mcp_session: ClientSession):
    resp = await mcp_session.list_tools()
    tools = [
        {"name": t.name, "description": t.description or "", "input_schema": _to_dict(t.inputSchema)}
        for t in resp.tools
    ]
    names = {t.name for t in resp.tools}
    return tools, names


async def _call_mcp(mcp_session: ClientSession, tool_name: str, tool_input: dict, tool_names: set) -> str:
    if tool_name not in tool_names:
        return f"Unknown tool: {tool_name}"
    result = await mcp_session.call_tool(tool_name, tool_input)
    return result.content[0].text if result.content else "No data."


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — Deal Investigator
# Yields: ("tool", tool_name)  or  ("text", chunk)
# ═══════════════════════════════════════════════════════════════════════════════

_INVESTIGATOR_SYSTEM = """You are a Salesforce deal investigator.
Given a company or deal name, work through these steps IN ORDER using run_soql:

1. Find the opportunity:
   SELECT Id, Name, StageName, Amount, CloseDate, LastActivityDate, AccountId
   FROM Opportunity WHERE Name LIKE '%<company>%' LIMIT 3

2. Get the account profile (use the AccountId from step 1):
   SELECT Id, Name, Industry, AnnualRevenue, Type, BillingCity, BillingCountry
   FROM Account WHERE Id = '<AccountId>'

3. Get key contacts at that account:
   SELECT FirstName, LastName, Title, Email, Phone
   FROM Contact WHERE AccountId = '<AccountId>' LIMIT 5

4. Check for open support cases:
   SELECT CaseNumber, Subject, Status, Priority
   FROM Case WHERE AccountId = '<AccountId>' AND Status != 'Closed' LIMIT 5

Write a structured DEAL BRIEFING:
## Deal Status
## Account Profile
## Key Contacts
## Risks (open cases, stale activity, overdue close date)
## Recommended Next Steps (3 specific actions)"""


async def deal_investigator(api_key: str, sf_token: str, sf_instance: str, company_name: str):
    """Async generator: yields ("tool", name) and ("text", chunk)."""
    client = anthropic.Anthropic(api_key=api_key)
    mcp_url = f"{MCP_SERVER_URL}?sf_token={sf_token}&sf_instance={sf_instance}"

    async with sse_client(url=mcp_url) as (read, write):
        async with ClientSession(read, write) as sess:
            await sess.initialize()
            tools, tool_names = await _load_mcp_tools(sess)
            messages = [{"role": "user", "content": f"Investigate this company / deal: {company_name}"}]

            # ── Tool-call loop ────────────────────────────────────────
            while True:
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=2048,
                    system=_INVESTIGATOR_SYSTEM,
                    tools=tools,
                    messages=messages,
                )
                if response.stop_reason != "tool_use":
                    break

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        yield ("tool", block.name)
                        result_text = await _call_mcp(sess, block.name, dict(block.input), tool_names)
                        tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result_text})

                messages.append({"role": "assistant", "content": [_to_dict(b) for b in response.content]})
                messages.append({"role": "user", "content": tool_results})

            # ── Stream final briefing ─────────────────────────────────
            with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=_INVESTIGATOR_SYSTEM,
                tools=tools,
                messages=messages,
            ) as stream:
                for chunk in stream.text_stream:
                    yield ("text", chunk)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — Background Monitor
# Runs on a schedule, stores a JSON health report in memory.
# Uses env-var SF credentials (no user session required).
# ═══════════════════════════════════════════════════════════════════════════════

monitor_state: dict = {
    "running":     False,
    "last_run":    None,
    "run_count":   0,
    "last_report": None,
    "error":       None,
}

# Credentials set when start_monitor() is called
_monitor_creds: dict = {}  # {api_key, sf_token, sf_instance}

_MONITOR_SYSTEM = """You are an automated Salesforce pipeline monitor.
Run exactly these three SOQL queries using run_soql, then return ONLY valid JSON.

Queries:
1. SELECT StageName, COUNT(Id) cnt FROM Opportunity WHERE IsClosed=false GROUP BY StageName
2. SELECT Name, Amount, StageName, LastActivityDate FROM Opportunity WHERE IsClosed=false AND LastActivityDate < LAST_N_DAYS:30 LIMIT 10
3. SELECT Name, Amount, StageName, CloseDate FROM Opportunity WHERE CloseDate = THIS_WEEK AND IsClosed=false LIMIT 10

Return ONLY this JSON (no markdown, no explanation):
{
  "pipeline_by_stage": {"<StageName>": <count>, ...},
  "stale_deals": [{"name": "...", "amount": ..., "stage": "..."}, ...],
  "closing_this_week": [{"name": "...", "amount": ..., "close_date": "..."}, ...],
  "alerts": ["...", "..."]
}
Populate alerts with plain-English flags: stale deals, risky close dates, low-pipeline stages."""


async def _monitor_once() -> dict:
    api_key     = _monitor_creds.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
    sf_token    = _monitor_creds.get("sf_token")
    sf_instance = _monitor_creds.get("sf_instance")

    client = anthropic.Anthropic(api_key=api_key)

    # Build MCP URL: with user token if available, else bare (env-var client creds)
    if sf_token and sf_instance:
        mcp_url = f"{MCP_SERVER_URL}?sf_token={sf_token}&sf_instance={sf_instance}"
    else:
        mcp_url = MCP_SERVER_URL

    async with sse_client(url=mcp_url) as (read, write):
        async with ClientSession(read, write) as sess:
            await sess.initialize()
            tools, tool_names = await _load_mcp_tools(sess)
            messages = [{"role": "user", "content": "Run the pipeline health check and return the JSON report."}]

            while True:
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1024,
                    system=_MONITOR_SYSTEM,
                    tools=tools,
                    messages=messages,
                )
                if response.stop_reason != "tool_use":
                    break
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result_text = await _call_mcp(sess, block.name, dict(block.input), tool_names)
                        tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result_text})
                messages.append({"role": "assistant", "content": [_to_dict(b) for b in response.content]})
                messages.append({"role": "user", "content": tool_results})

    raw = next((b.text for b in response.content if b.type == "text"), "{}")
    # Strip markdown fences if Claude wrapped it
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    return json.loads(m.group(1) if m else raw)


async def _monitor_loop(interval_seconds: int):
    while monitor_state["running"]:
        try:
            report = await _monitor_once()
            report["timestamp"] = datetime.utcnow().isoformat() + "Z"
            monitor_state.update(last_report=report, last_run=report["timestamp"],
                                 error=None, run_count=monitor_state["run_count"] + 1)
        except Exception as e:
            monitor_state.update(error=str(e), last_run=datetime.utcnow().isoformat() + "Z")
        await asyncio.sleep(interval_seconds)
    monitor_state["running"] = False


def start_monitor(interval_seconds: int = 300, api_key: str = None,
                  sf_token: str = None, sf_instance: str = None):
    """Start the background monitor. Safe to call multiple times — ignores if already running."""
    if monitor_state["running"]:
        return
    # Store credentials so _monitor_once can use them
    _monitor_creds["api_key"]     = api_key
    _monitor_creds["sf_token"]    = sf_token
    _monitor_creds["sf_instance"] = sf_instance
    monitor_state["running"] = True
    asyncio.create_task(_monitor_loop(interval_seconds))


def stop_monitor():
    monitor_state["running"] = False


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Business Review with Subagents
#
# How subagents work here:
#   Orchestrator Claude sees two tools: analyze_pipeline + analyze_customer_health.
#   When it calls one, your code spawns a SEPARATE Claude call (_run_subagent)
#   with its own system prompt and its own MCP tool loop.
#   The subagent's final text is returned to the orchestrator as a tool result.
#   The orchestrator then synthesizes both into an executive review.
#
# Yields: ("tool", label)  or  ("text", chunk)
# ═══════════════════════════════════════════════════════════════════════════════

_ORCHESTRATOR_SYSTEM = """You are a Salesforce executive business review orchestrator.
You have two specialist subagents:
  - analyze_pipeline        → deep dive into open opportunities and revenue
  - analyze_customer_health → review support cases and customer risk

Call BOTH tools first, then write a structured executive review:
## Revenue Outlook
## Customer Health
## Top 3 Risks
## Recommended Actions"""

_PIPELINE_SUBAGENT_SYSTEM = """You are a pipeline analyst with access to Salesforce.
Use run_soql to answer:
1. Total pipeline value + deal count by stage
2. Top 5 open deals by Amount (name, amount, stage, close date)
3. How many deals close this month?
Be concise. Include specific names and dollar amounts."""

_CUSTOMER_SUBAGENT_SYSTEM = """You are a customer health analyst with access to Salesforce.
Use run_soql to answer:
1. Open case count by Priority
2. Which accounts have more than 1 open case?
3. Show the 5 most recent high-priority cases
Be concise. Highlight the biggest customer risks."""


async def _run_subagent(client, system_prompt: str, user_message: str,
                        mcp_session: ClientSession, mcp_tools: list, tool_names: set) -> str:
    """A standalone Claude loop. Returns the final text response."""
    messages = [{"role": "user", "content": user_message}]
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system_prompt,
            tools=mcp_tools,
            messages=messages,
        )
        if response.stop_reason != "tool_use":
            break
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result_text = await _call_mcp(mcp_session, block.name, dict(block.input), tool_names)
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result_text})
        messages.append({"role": "assistant", "content": [_to_dict(b) for b in response.content]})
        messages.append({"role": "user", "content": tool_results})
    return next((b.text for b in response.content if b.type == "text"), "")


async def business_review(api_key: str, sf_token: str, sf_instance: str):
    """Async generator: yields ("tool", label) and ("text", chunk)."""
    client = anthropic.Anthropic(api_key=api_key)
    mcp_url = f"{MCP_SERVER_URL}?sf_token={sf_token}&sf_instance={sf_instance}"

    # These are the "subagent tools" the orchestrator can call.
    # They look like normal tools but their implementations are other Claude calls.
    subagent_tools = [
        {
            "name": "analyze_pipeline",
            "description": "Spawn the pipeline analyst subagent to analyze open opportunities and revenue.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "analyze_customer_health",
            "description": "Spawn the customer health subagent to analyze support cases and at-risk accounts.",
            "input_schema": {"type": "object", "properties": {}},
        },
    ]

    async with sse_client(url=mcp_url) as (read, write):
        async with ClientSession(read, write) as sess:
            await sess.initialize()
            mcp_tools, tool_names = await _load_mcp_tools(sess)

            messages = [{"role": "user", "content": "Run a full executive business review of the Salesforce org."}]

            # ── Orchestrator loop ─────────────────────────────────────
            while True:
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=2048,
                    system=_ORCHESTRATOR_SYSTEM,
                    tools=subagent_tools,
                    messages=messages,
                )
                if response.stop_reason != "tool_use":
                    break

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        yield ("tool", f"Subagent → {block.name}")

                        # Dispatch to the right subagent
                        if block.name == "analyze_pipeline":
                            result = await _run_subagent(
                                client, _PIPELINE_SUBAGENT_SYSTEM,
                                "Analyse the current sales pipeline.",
                                sess, mcp_tools, tool_names,
                            )
                        elif block.name == "analyze_customer_health":
                            result = await _run_subagent(
                                client, _CUSTOMER_SUBAGENT_SYSTEM,
                                "Analyse customer health from support cases.",
                                sess, mcp_tools, tool_names,
                            )
                        else:
                            result = f"Unknown subagent: {block.name}"

                        tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

                messages.append({"role": "assistant", "content": [_to_dict(b) for b in response.content]})
                messages.append({"role": "user", "content": tool_results})

            # ── Stream final executive review ─────────────────────────
            with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=_ORCHESTRATOR_SYSTEM,
                tools=subagent_tools,
                messages=messages,
            ) as stream:
                for chunk in stream.text_stream:
                    yield ("text", chunk)
