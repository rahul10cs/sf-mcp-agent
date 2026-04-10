#!/usr/bin/env python3
"""
Chat Backend — MCP Client
Handles Salesforce OAuth login, Anthropic API key entry, and the MCP tool-call loop.
"""

import json
import os
import secrets
import traceback
from urllib.parse import urlencode

import anthropic
import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic import BaseModel

from agents import (
    deal_investigator,
    business_review,
    start_monitor,
    stop_monitor,
    monitor_state,
)
from multi_agent import run_report

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup():
    # Monitor starts later — triggered by /set-api-key once the user has logged in.
    # Nothing to do here.
    pass

MCP_SERVER_URL        = os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")
MCP_WEBSEARCH_URL     = os.getenv("MCP_WEBSEARCH_URL", "http://localhost:8001/sse")
SF_CLIENT_ID     = os.getenv("SF_CLIENT_ID")
SF_CLIENT_SECRET = os.getenv("SF_CLIENT_SECRET")
SF_LOGIN_URL     = os.getenv("SF_LOGIN_URL", "https://login.salesforce.com")
SF_CALLBACK_URL  = os.getenv("SF_CALLBACK_URL", "http://localhost:3000/auth/salesforce/callback")

SESSION_COOKIE = "mcp_session"

# In-memory session store: { session_id: { sf_token, sf_instance_url, anthropic_key, oauth_state } }
sessions: dict = {}


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def get_session(request: Request) -> dict | None:
    sid = request.cookies.get(SESSION_COOKIE)
    return sessions.get(sid) if sid else None


def require_session(request: Request, response: Response) -> tuple[str, dict]:
    """Return existing session or create a new one."""
    sid = request.cookies.get(SESSION_COOKIE)
    if sid and sid in sessions:
        return sid, sessions[sid]
    sid = secrets.token_urlsafe(32)
    sessions[sid] = {}
    response.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
    return sid, sessions[sid]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_dict(obj):
    """Convert Pydantic model or dict to plain dict (for JSON serialization)."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return json.loads(json.dumps(obj, default=str))


def friendly_api_error(e: anthropic.APIStatusError) -> str:
    try:
        msg = e.response.json().get("error", {}).get("message", str(e))
    except Exception:
        msg = str(e)
    if "credit balance is too low" in msg:
        return "Your Anthropic API credit balance is too low. Please add credits at console.anthropic.com → Plans & Billing."
    if e.status_code == 401:
        return "Invalid Anthropic API key. Please re-enter your key."
    if e.status_code == 429:
        return "Anthropic API rate limit hit. Please wait a moment and try again."
    return f"Anthropic API error ({e.status_code}): {msg}"


# ---------------------------------------------------------------------------
# Auth status — used by UI to decide which screen to show
# ---------------------------------------------------------------------------

@app.get("/auth/status")
async def auth_status(request: Request):
    session = get_session(request)
    return {
        "sf_authenticated":     bool(session and session.get("sf_token")),
        "anthropic_configured": bool(session and session.get("anthropic_key")),
    }


# ---------------------------------------------------------------------------
# Salesforce OAuth — Authorization Code Flow
# Step 1: redirect browser to Salesforce login
# ---------------------------------------------------------------------------

@app.get("/auth/salesforce")
async def salesforce_login(request: Request, response: Response):
    sid, session = require_session(request, response)

    state = secrets.token_urlsafe(16)
    session["oauth_state"] = state

    params = {
        "response_type": "code",
        "client_id":     SF_CLIENT_ID,
        "redirect_uri":  SF_CALLBACK_URL,
        "state":         state,
        "scope":         "api refresh_token",
    }
    auth_url = f"{SF_LOGIN_URL}/services/oauth2/authorize?{urlencode(params)}"

    redirect = RedirectResponse(url=auth_url)
    redirect.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
    return redirect


# ---------------------------------------------------------------------------
# Salesforce OAuth — Authorization Code Flow
# Step 2: Salesforce redirects back here with ?code=...
# ---------------------------------------------------------------------------

@app.get("/auth/salesforce/callback")
async def salesforce_callback(request: Request):
    code  = request.query_params.get("code")
    state = request.query_params.get("state")
    error = request.query_params.get("error")

    if error:
        return RedirectResponse(url=f"/?error={error}")

    sid = request.cookies.get(SESSION_COOKIE)
    session = sessions.get(sid)

    if not session:
        return RedirectResponse(url="/?error=session_missing")

    # Validate state to prevent CSRF
    if state != session.get("oauth_state"):
        return RedirectResponse(url="/?error=invalid_state")

    # Exchange authorization code for access token
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{SF_LOGIN_URL}/services/oauth2/token",
            data={
                "grant_type":    "authorization_code",
                "code":          code,
                "client_id":     SF_CLIENT_ID,
                "client_secret": SF_CLIENT_SECRET,
                "redirect_uri":  SF_CALLBACK_URL,
            },
        )

    if resp.status_code != 200:
        err = resp.json().get("error_description", "token_exchange_failed")
        return RedirectResponse(url=f"/?error={err}")

    token_data = resp.json()
    session["sf_token"]        = token_data["access_token"]
    session["sf_instance_url"] = token_data["instance_url"]
    session.pop("oauth_state", None)

    redirect = RedirectResponse(url="/")
    redirect.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
    return redirect


# ---------------------------------------------------------------------------
# Anthropic API key — user submits via UI form
# ---------------------------------------------------------------------------

class ApiKeyRequest(BaseModel):
    api_key: str


@app.post("/set-api-key")
async def set_api_key(req: ApiKeyRequest, request: Request):
    session = get_session(request)
    if not session:
        return JSONResponse({"error": "No session found. Please refresh and try again."}, status_code=401)
    if not session.get("sf_token"):
        return JSONResponse({"error": "Please connect Salesforce first."}, status_code=401)

    session["anthropic_key"] = req.api_key.strip()

    # Start the background monitor now that we have both Salesforce token and
    # Anthropic key. start_monitor() is a no-op if already running.
    start_monitor(
        interval_seconds=300,
        api_key=session["anthropic_key"],
        sf_token=session["sf_token"],
        sf_instance=session["sf_instance_url"],
    )

    return {"ok": True}


# ---------------------------------------------------------------------------
# Logout — clears session and cookie
# ---------------------------------------------------------------------------

@app.post("/auth/logout")
async def logout(request: Request, response: Response):
    sid = request.cookies.get(SESSION_COOKIE)
    if sid:
        sessions.pop(sid, None)
    stop_monitor()  # stops the background task
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}


# ---------------------------------------------------------------------------
# /chat endpoint — the core MCP client loop
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# System prompt as a cached content block.
#
# CONCEPT: Prompt Caching
# By wrapping the system prompt in a list with cache_control, Claude caches
# the processed prompt on Anthropic's servers for 5 minutes. Subsequent
# requests that share the same prefix skip reprocessing it — you only pay
# ~10% of the input token cost instead of 100%.
#
# The response.usage object then tells you:
#   cache_creation_input_tokens → tokens written to cache (first call)
#   cache_read_input_tokens     → tokens served from cache (subsequent calls)
#   input_tokens                → uncached tokens (your new message)
# ---------------------------------------------------------------------------
_CHAT_SYSTEM = [
    {
        "type": "text",
        "cache_control": {"type": "ephemeral"},   # ← prompt caching marker
        "text": """You are a Salesforce AI assistant with web search capability.

For Salesforce data questions, always use the run_soql tool to build a dynamic SOQL query.
For current events, news, or any up-to-date information, use the web_search tool.
Do NOT use the predefined tools (get_accounts, get_contacts, etc.) unless the user asks for a simple default list.

Rules for building SOQL:
- Only use SELECT statements
- Select only the fields relevant to the user's question
- Add WHERE clauses, ORDER BY, and LIMIT based on what the user asks
- If the user asks for "all" records, use LIMIT 200 at most
- Use relationships where needed e.g. Account.Name on Contact, Opportunity

Common Salesforce objects and their key fields:
- Account: Id, Name, Industry, Phone, Website, AnnualRevenue, BillingCity, BillingCountry, Type, OwnerId
- Contact: Id, FirstName, LastName, Email, Phone, Title, Department, AccountId, Account.Name
- Opportunity: Id, Name, StageName, Amount, CloseDate, Probability, AccountId, Account.Name, OwnerId
- Case: Id, CaseNumber, Subject, Status, Priority, Origin, AccountId, Account.Name, ContactId
- Lead: Id, FirstName, LastName, Email, Phone, Company, Status, Industry, LeadSource
- Task: Id, Subject, Status, Priority, ActivityDate, WhoId, WhatId
- User: Id, Name, Email, Title, Department, IsActive
- Organization: Id, Name, OrganizationType, IsSandbox"""
    }
]


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    session = get_session(request)

    if not session or not session.get("sf_token"):
        return JSONResponse({"reply": "You are not connected to Salesforce. Please log in first."})
    if not session.get("anthropic_key"):
        return JSONResponse({"reply": "No Anthropic API key set. Please enter your API key first."})

    sf_token        = session["sf_token"]
    sf_instance_url = session["sf_instance_url"]
    claude          = anthropic.Anthropic(api_key=session["anthropic_key"])
    mcp_url         = f"{MCP_SERVER_URL}?sf_token={sf_token}&sf_instance={sf_instance_url}"

    # CONCEPT: Multi-turn conversation history
    # We keep the last 40 messages (20 exchanges) in the session so Claude
    # remembers context across requests. Each entry is {"role": ..., "content": ...}.
    history = session.get("conversation_history", [])

    async def stream_response():
        try:
            async with sse_client(url=mcp_url) as (sf_read, sf_write):
                async with ClientSession(sf_read, sf_write) as sf_session:
                    await sf_session.initialize()

                    async with sse_client(url=MCP_WEBSEARCH_URL) as (ws_read, ws_write):
                        async with ClientSession(ws_read, ws_write) as ws_session:
                            await ws_session.initialize()

                            sf_tools_resp = await sf_session.list_tools()
                            ws_tools_resp = await ws_session.list_tools()

                            def to_tool_dict(t):
                                return {"name": t.name, "description": t.description or "", "input_schema": to_dict(t.inputSchema)}

                            tools         = [to_tool_dict(t) for t in sf_tools_resp.tools] + [to_tool_dict(t) for t in ws_tools_resp.tools]
                            sf_tool_names = {t.name for t in sf_tools_resp.tools}
                            ws_tool_names = {t.name for t in ws_tools_resp.tools}

                            # History + new message
                            messages = history + [{"role": "user", "content": req.message}]

                            # Track token usage across every API call in this request
                            usage = {"input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_creation": 0}

                            def _add_usage(u):
                                usage["input_tokens"]   += u.input_tokens
                                usage["output_tokens"]  += u.output_tokens
                                usage["cache_read"]     += getattr(u, "cache_read_input_tokens", 0)
                                usage["cache_creation"] += getattr(u, "cache_creation_input_tokens", 0)

                            try:
                                # ── Tool call loop ───────────────────────────
                                while True:
                                    response = claude.messages.create(
                                        model="claude-sonnet-4-6",
                                        max_tokens=1024,
                                        system=_CHAT_SYSTEM,   # ← cached system prompt
                                        tools=tools,
                                        messages=messages,
                                    )
                                    _add_usage(response.usage)

                                    if response.stop_reason != "tool_use":
                                        break

                                    tool_blocks  = [b for b in response.content if b.type == "tool_use"]
                                    tool_results = []

                                    for tool_block in tool_blocks:
                                        yield f"data: {json.dumps({'type': 'tool', 'name': tool_block.name})}\n\n"

                                        mcp_sess = sf_session if tool_block.name in sf_tool_names else (ws_session if tool_block.name in ws_tool_names else None)
                                        if mcp_sess is None:
                                            tool_results.append({"type": "tool_result", "tool_use_id": tool_block.id, "content": f"Unknown tool: {tool_block.name}"})
                                            continue

                                        tool_result = await mcp_sess.call_tool(tool_block.name, dict(tool_block.input))
                                        result_text = tool_result.content[0].text if tool_result.content else "No data returned."
                                        tool_results.append({"type": "tool_result", "tool_use_id": tool_block.id, "content": result_text})

                                    messages.append({"role": "assistant", "content": [to_dict(b) for b in response.content]})
                                    messages.append({"role": "user", "content": tool_results})

                                # ── Stream final response ────────────────────
                                full_text = ""
                                with claude.messages.stream(
                                    model="claude-sonnet-4-6",
                                    max_tokens=1024,
                                    system=_CHAT_SYSTEM,
                                    tools=tools,
                                    messages=messages,
                                ) as stream:
                                    for chunk in stream.text_stream:
                                        full_text += chunk
                                        yield f"data: {json.dumps({'type': 'text', 'chunk': chunk})}\n\n"
                                    _add_usage(stream.get_final_message().usage)

                                # ── Save conversation history ────────────────
                                updated = history + [
                                    {"role": "user",      "content": req.message},
                                    {"role": "assistant", "content": full_text},
                                ]
                                session["conversation_history"] = updated[-40:]  # keep last 20 exchanges

                                # ── Emit token usage ─────────────────────────
                                # CONCEPT: This is the usage object from the API.
                                # cache_read > 0 means the system prompt was served from cache.
                                # cache_creation > 0 means this call wrote it to cache.
                                yield f"data: {json.dumps({'type': 'usage', **usage, 'model': 'claude-sonnet-4-6'})}\n\n"
                                yield "data: [DONE]\n\n"

                            except anthropic.APIStatusError as e:
                                yield f"data: {json.dumps({'type': 'error', 'message': friendly_api_error(e)})}\n\n"

        except Exception as e:
            print(traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Clear conversation history
# ---------------------------------------------------------------------------

@app.post("/chat/clear")
async def clear_chat(request: Request):
    session = get_session(request)
    if session:
        session["conversation_history"] = []
    return {"ok": True}


# ---------------------------------------------------------------------------
# AGENT 1 — /agent/investigate  (deal investigator, multi-hop)
# ---------------------------------------------------------------------------

class InvestigateRequest(BaseModel):
    company: str


@app.post("/agent/investigate")
async def agent_investigate(req: InvestigateRequest, request: Request):
    session = get_session(request)
    if not session or not session.get("sf_token"):
        return JSONResponse({"error": "Not connected to Salesforce."}, status_code=401)
    if not session.get("anthropic_key"):
        return JSONResponse({"error": "No Anthropic API key set."}, status_code=401)

    async def stream():
        try:
            async for event_type, data in deal_investigator(
                api_key=session["anthropic_key"],
                sf_token=session["sf_token"],
                sf_instance=session["sf_instance_url"],
                company_name=req.company,
            ):
                yield f"data: {json.dumps({'type': event_type, 'name' if event_type == 'tool' else 'chunk': data})}\n\n"
            yield "data: [DONE]\n\n"
        except anthropic.APIStatusError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': friendly_api_error(e)})}\n\n"
        except Exception as e:
            print(traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# AGENT 2 — /agent/monitor/status  (background monitor status)
# ---------------------------------------------------------------------------

@app.get("/agent/monitor/status")
async def agent_monitor_status():
    return JSONResponse(monitor_state)


# ---------------------------------------------------------------------------
# AGENT 3 — /agent/review  (business review with subagents)
# ---------------------------------------------------------------------------

@app.post("/agent/review")
async def agent_review(request: Request):
    session = get_session(request)
    if not session or not session.get("sf_token"):
        return JSONResponse({"error": "Not connected to Salesforce."}, status_code=401)
    if not session.get("anthropic_key"):
        return JSONResponse({"error": "No Anthropic API key set."}, status_code=401)

    async def stream():
        try:
            async for event_type, data in business_review(
                api_key=session["anthropic_key"],
                sf_token=session["sf_token"],
                sf_instance=session["sf_instance_url"],
            ):
                yield f"data: {json.dumps({'type': event_type, 'name' if event_type == 'tool' else 'chunk': data})}\n\n"
            yield "data: [DONE]\n\n"
        except anthropic.APIStatusError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': friendly_api_error(e)})}\n\n"
        except Exception as e:
            print(traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# MULTI-AGENT — /agent/report  (coordinator → analysts → synthesis)
# ---------------------------------------------------------------------------

class ReportRequest(BaseModel):
    question: str


@app.post("/agent/report")
async def agent_report(req: ReportRequest, request: Request):
    session = get_session(request)
    if not session or not session.get("sf_token"):
        return JSONResponse({"error": "Not connected to Salesforce."}, status_code=401)
    if not session.get("anthropic_key"):
        return JSONResponse({"error": "No Anthropic API key set."}, status_code=401)

    async def stream():
        try:
            async for event_type, data in run_report(
                api_key=session["anthropic_key"],
                sf_token=session["sf_token"],
                sf_instance=session["sf_instance_url"],
                user_question=req.question,
            ):
                if event_type == "coordinator":
                    analysts = ", ".join(a["name"] for a in data.get("analysts", []))
                    payload  = {"type": "stage", "label": f"Coordinator → running: {analysts}", "detail": data.get("reasoning", "")}
                elif event_type == "analyst_start":
                    payload = {"type": "stage", "label": f"Analyst running: {data}"}
                elif event_type == "analyst_done":
                    payload = {"type": "stage", "label": f"Analyst done: {data}"}
                elif event_type == "synthesis_start":
                    payload = {"type": "stage", "label": "Synthesis agent writing report…"}
                elif event_type == "text":
                    payload = {"type": "text", "chunk": data}
                else:
                    continue
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
        except anthropic.APIStatusError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': friendly_api_error(e)})}\n\n"
        except Exception as e:
            print(traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# /agent/pipeline — Sales pipeline health-check agent
# ---------------------------------------------------------------------------

@app.post("/agent/pipeline")
async def pipeline_agent(request: Request):
    session = get_session(request)

    if not session or not session.get("sf_token"):
        return JSONResponse({"reply": "You are not connected to Salesforce. Please log in first."})
    if not session.get("anthropic_key"):
        return JSONResponse({"reply": "No Anthropic API key set. Please enter your API key first."})

    sf_token        = session["sf_token"]
    sf_instance_url = session["sf_instance_url"]
    claude          = anthropic.Anthropic(api_key=session["anthropic_key"])

    mcp_url = f"{MCP_SERVER_URL}?sf_token={sf_token}&sf_instance={sf_instance_url}"

    agent_system_prompt = """You are a Salesforce sales analyst agent.
Use the available tools to perform a complete pipeline health check.

Your analysis must cover:
1. Pipeline summary — total open pipeline value + deal count grouped by stage (use run_soql with GROUP BY StageName)
2. At-risk deals — open opportunities with no activity in the last 30 days (LastActivityDate < LAST_N_DAYS:30)
3. Closing soon — deals with CloseDate in the next 14 days that are still open
4. Top 5 open deals by amount

Always use run_soql for aggregations. Reference specific deal names and dollar amounts.
End with a "Focus This Week" list of 3-5 concrete, prioritised actions."""

    async def stream_response():
        try:
            async with sse_client(url=mcp_url) as (read, write):
                async with ClientSession(read, write) as mcp_session:
                    await mcp_session.initialize()

                    tools_resp = await mcp_session.list_tools()
                    tools = [
                        {
                            "name":         t.name,
                            "description":  t.description or "",
                            "input_schema": to_dict(t.inputSchema),
                        }
                        for t in tools_resp.tools
                    ]
                    tool_names = {t.name for t in tools_resp.tools}

                    messages = [{"role": "user", "content": (
                        "Give me a full pipeline health check. "
                        "Highlight at-risk deals with no recent activity, "
                        "deals closing in the next 14 days, and where to focus this week."
                    )}]

                    try:
                        # ── Agent tool-call loop ─────────────────────────────
                        while True:
                            response = claude.messages.create(
                                model="claude-sonnet-4-6",
                                max_tokens=2048,
                                system=agent_system_prompt,
                                tools=tools,
                                messages=messages,
                            )

                            if response.stop_reason != "tool_use":
                                break

                            tool_blocks  = [b for b in response.content if b.type == "tool_use"]
                            tool_results = []

                            for tool_block in tool_blocks:
                                yield f"data: {json.dumps({'type': 'tool', 'name': tool_block.name})}\n\n"

                                if tool_block.name in tool_names:
                                    result      = await mcp_session.call_tool(tool_block.name, dict(tool_block.input))
                                    result_text = result.content[0].text if result.content else "No data."
                                else:
                                    result_text = f"Unknown tool: {tool_block.name}"

                                tool_results.append({
                                    "type":        "tool_result",
                                    "tool_use_id": tool_block.id,
                                    "content":     result_text,
                                })

                            assistant_content = [
                                to_dict(b) if not isinstance(b, dict) else b
                                for b in response.content
                            ]
                            messages.append({"role": "assistant", "content": assistant_content})
                            messages.append({"role": "user",      "content": tool_results})

                        # ── Stream final text response ───────────────────────
                        with claude.messages.stream(
                            model="claude-sonnet-4-6",
                            max_tokens=2048,
                            system=agent_system_prompt,
                            tools=tools,
                            messages=messages,
                        ) as stream:
                            for text_chunk in stream.text_stream:
                                yield f"data: {json.dumps({'type': 'text', 'chunk': text_chunk})}\n\n"

                        yield "data: [DONE]\n\n"

                    except anthropic.APIStatusError as e:
                        yield f"data: {json.dumps({'type': 'error', 'message': friendly_api_error(e)})}\n\n"

        except Exception as e:
            print(traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# /agent/analyze — Deep analysis using claude-opus-4-6 with adaptive thinking
#
# CONCEPT: Extended Thinking / Adaptive Thinking
# claude-opus-4-6 supports thinking: {type: "adaptive"} which lets the model
# decide *when* and *how much* to think before responding. The model emits
# thinking blocks (type=="thinking") BEFORE the final answer blocks.
#
# This is different from the /chat endpoint which uses claude-sonnet-4-6.
# Use this when you want the model to reason deeply over complex Salesforce data
# before producing an answer — e.g., "why are we losing deals in the enterprise tier?"
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    question: str


@app.post("/agent/analyze")
async def agent_analyze(req: AnalyzeRequest, request: Request):
    session = get_session(request)
    if not session or not session.get("sf_token"):
        return JSONResponse({"error": "Not connected to Salesforce."}, status_code=401)
    if not session.get("anthropic_key"):
        return JSONResponse({"error": "No Anthropic API key set."}, status_code=401)

    sf_token        = session["sf_token"]
    sf_instance_url = session["sf_instance_url"]
    claude          = anthropic.Anthropic(api_key=session["anthropic_key"])
    mcp_url         = f"{MCP_SERVER_URL}?sf_token={sf_token}&sf_instance={sf_instance_url}"

    analyze_system = """You are a senior Salesforce business analyst with deep expertise in revenue operations.

You have access to live Salesforce data via SOQL tools. Use them freely to gather evidence before forming conclusions.

Your job:
1. Run the SOQL queries needed to fully understand the question
2. Think carefully (you have extended thinking enabled — use it)
3. Produce a structured, insight-driven analysis with evidence from the data

Always reference specific numbers, deal names, and trends from the data. Avoid generic advice."""

    async def stream():
        try:
            async with sse_client(url=mcp_url) as (read, write):
                async with ClientSession(read, write) as mcp_session:
                    await mcp_session.initialize()

                    tools_resp = await mcp_session.list_tools()
                    tools = [
                        {
                            "name":         t.name,
                            "description":  t.description or "",
                            "input_schema": to_dict(t.inputSchema),
                        }
                        for t in tools_resp.tools
                    ]
                    tool_names = {t.name for t in tools_resp.tools}

                    messages = [{"role": "user", "content": req.question}]

                    # Track usage
                    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_creation": 0}

                    def _add_usage(u):
                        usage["input_tokens"]   += u.input_tokens
                        usage["output_tokens"]  += u.output_tokens
                        usage["cache_read"]     += getattr(u, "cache_read_input_tokens", 0)
                        usage["cache_creation"] += getattr(u, "cache_creation_input_tokens", 0)

                    try:
                        # ── Tool-call loop with adaptive thinking ────────────
                        # CONCEPT: betas=["interleaved-thinking-2025-05-14"] enables
                        # thinking blocks between tool calls, not just at the end.
                        while True:
                            response = claude.messages.create(
                                model="claude-opus-4-6",
                                max_tokens=4096,
                                thinking={"type": "adaptive"},
                                system=analyze_system,
                                tools=tools,
                                messages=messages,
                                betas=["interleaved-thinking-2025-05-14"],
                            )
                            _add_usage(response.usage)

                            if response.stop_reason != "tool_use":
                                # Emit any thinking blocks from final response
                                for block in response.content:
                                    if block.type == "thinking":
                                        yield f"data: {json.dumps({'type': 'thinking', 'text': block.thinking})}\n\n"
                                break

                            tool_blocks  = [b for b in response.content if b.type == "tool_use"]
                            tool_results = []

                            # Emit thinking blocks that appeared before tool calls
                            for block in response.content:
                                if block.type == "thinking":
                                    yield f"data: {json.dumps({'type': 'thinking', 'text': block.thinking})}\n\n"

                            for tool_block in tool_blocks:
                                yield f"data: {json.dumps({'type': 'tool', 'name': tool_block.name})}\n\n"

                                if tool_block.name in tool_names:
                                    result      = await mcp_session.call_tool(tool_block.name, dict(tool_block.input))
                                    result_text = result.content[0].text if result.content else "No data."
                                else:
                                    result_text = f"Unknown tool: {tool_block.name}"

                                tool_results.append({
                                    "type":        "tool_result",
                                    "tool_use_id": tool_block.id,
                                    "content":     result_text,
                                })

                            messages.append({"role": "assistant", "content": [to_dict(b) for b in response.content]})
                            messages.append({"role": "user", "content": tool_results})

                        # ── Stream final text response ────────────────────────
                        # After tool calls are done, stream the final answer.
                        # We use messages.create (not stream) since thinking+streaming
                        # requires the beta SDK. Instead we emit the final text in chunks
                        # from the last response we already have.
                        final_text = ""
                        for block in response.content:
                            if hasattr(block, "text"):
                                final_text += block.text

                        # Stream the final text word-by-word for UI effect
                        words = final_text.split(" ")
                        for i, word in enumerate(words):
                            chunk = word + (" " if i < len(words) - 1 else "")
                            yield f"data: {json.dumps({'type': 'text', 'chunk': chunk})}\n\n"

                        # Emit usage
                        yield f"data: {json.dumps({'type': 'usage', **usage, 'model': 'claude-opus-4-6'})}\n\n"
                        yield "data: [DONE]\n\n"

                    except anthropic.APIStatusError as e:
                        yield f"data: {json.dumps({'type': 'error', 'message': friendly_api_error(e)})}\n\n"

        except Exception as e:
            print(traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Serve the chat UI
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse("static/index.html")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)
