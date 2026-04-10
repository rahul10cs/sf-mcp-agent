# Salesforce AI Assistant — MCP + Claude Agents

A full-stack AI chat assistant that connects to a live Salesforce org using the **Model Context Protocol (MCP)** and **Anthropic Claude**. Includes five agent patterns built from scratch: single-agent tool loops, background monitors, subagents, and a full coordinator → analyst → synthesis multi-agent pipeline.

---

## What's in this repo

```
sf-mcp-agent/
├── mcp_chat/               ← Chat UI + FastAPI backend + all agents
│   ├── app.py              ← FastAPI server, OAuth, all API endpoints
│   ├── agents.py           ← Agent 1 (Deal Investigator), Agent 2 (Background Monitor), Agent 3 (Business Review)
│   ├── multi_agent.py      ← Multi-agent: Coordinator → Analysts → Synthesis
│   ├── requirements.txt
│   ├── .env.example
│   └── static/
│       └── index.html      ← Chat UI (plain HTML/JS, no framework)
│
├── mcp_salesforce/         ← Salesforce MCP server
│   ├── server.py           ← MCP server exposing Salesforce as tools
│   └── requirements.txt
│
└── scripts/
    └── sales_agent.py      ← Standalone CLI agent (connect + run from terminal)
```

---

## Architecture overview

```
Browser
  │  HTTP / SSE
  ▼
mcp_chat/app.py  (FastAPI, port 3000)
  │  MCP over SSE
  ├──► mcp_salesforce/server.py  (port 8000)  — Salesforce SOQL tools
  │
  └──► Anthropic Claude API  (claude-sonnet-4-6)
            │
            └── Tool-use loop: Claude calls tools → gets results → responds
```

The chat server is the **MCP client**. The Salesforce server is the **MCP server**. Claude sits in the middle, deciding which tools to call.

---

## Quick start

### 1. Start the Salesforce MCP server

```bash
cd mcp_salesforce
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Set env vars (create a `.env` or export directly):

```bash
# For per-user OAuth flow (used by the chat UI)
SF_CLIENT_ID=your_connected_app_client_id
SF_CLIENT_SECRET=your_connected_app_client_secret
SF_LOGIN_URL=https://login.salesforce.com
SF_CALLBACK_URL=http://localhost:3000/auth/salesforce/callback

# For background monitor / CLI agent (client credentials flow)
SF_INSTANCE_URL=https://yourorg.my.salesforce.com
```

```bash
TRANSPORT=sse python server.py   # starts on port 8000
```

### 2. Start the chat server

```bash
cd mcp_chat
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # edit MCP_SERVER_URL if needed
python app.py                   # starts on port 3000
```

### 3. Open the UI

Go to `http://localhost:3000` → Login with Salesforce → Enter Anthropic API key → Start chatting.

---

## Authentication flow (step by step)

```
1. User opens http://localhost:3000
   app.py: GET /auth/status → returns {sf_authenticated: false}
   UI shows: "Login with Salesforce" screen

2. User clicks Login
   app.py: GET /auth/salesforce
     - Creates session cookie (secrets.token_urlsafe)
     - Stores random `state` in session (CSRF protection)
     - Redirects browser to Salesforce OAuth authorize URL

3. Salesforce redirects back to /auth/salesforce/callback?code=...&state=...
   app.py: validates state, POSTs to Salesforce token endpoint
     - Receives access_token + instance_url
     - Stores both in in-memory session dict
     - Redirects to /

4. UI shows: "Enter Anthropic API Key" screen
   User submits key → POST /set-api-key → stored in session

5. UI shows: chat screen
   Every request from here uses session cookie to look up:
     session["sf_token"], session["sf_instance_url"], session["anthropic_key"]
```

---

## Chat — how a message flows end to end

This is the core `/chat` endpoint in `app.py`. Everything else is built on this same loop.

```
User types: "Show me high-value opportunities"
              │
              ▼
POST /chat  {message: "Show me high-value opportunities"}
              │
              ▼
app.py opens TWO MCP connections:
  sse_client(mcp_salesforce)  →  ClientSession.initialize()  →  list_tools()
  sse_client(mcp_websearch)   →  ClientSession.initialize()  →  list_tools()

tools = sf_tools + websearch_tools   (merged into one list for Claude)

              │
              ▼
TOOL-CALL LOOP:
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  claude.messages.create(model, tools, messages)     │
  │         │                                           │
  │         ▼                                           │
  │  response.stop_reason == "tool_use"?                │
  │    YES → for each tool_use block:                   │
  │            - yield SSE: {"type":"tool","name":"..."}│
  │            - call mcp_session.call_tool(name, input)│
  │            - append tool_result to messages         │
  │          loop again ──────────────────────────────► │
  │    NO  → break                                      │
  └─────────────────────────────────────────────────────┘
              │
              ▼
STREAMING FINAL RESPONSE:
  claude.messages.stream(...)
    for chunk in stream.text_stream:
      yield SSE: {"type":"text","chunk":"..."}

  yield SSE: "[DONE]"
              │
              ▼
Browser JS reads the SSE stream:
  type=="tool"  → updates typing indicator: "Using tool: run_soql..."
  type=="text"  → appends chunk to chat bubble (streaming effect)
  "[DONE]"      → re-enables input
```

---

## The Salesforce MCP Server (`mcp_salesforce/server.py`)

Exposes 6 tools over SSE that any MCP client can call:

| Tool | What it does |
|---|---|
| `get_accounts` | SELECT from Account with optional name filter |
| `get_contacts` | SELECT from Contact with optional email/name filter |
| `get_opportunities` | SELECT from Opportunity with optional stage filter |
| `get_cases` | SELECT from Case with optional status filter |
| `run_soql` | Run any SELECT query (blocks INSERT/UPDATE/DELETE) |
| `get_org_info` | Returns org name, ID, type, sandbox flag |

### Two auth modes

```python
# Mode 1: Per-user OAuth token (used by chat UI)
# mcp_chat passes ?sf_token=...&sf_instance=... on the SSE URL
# server.py captures it from query params and stores against session_id

# Mode 2: Client credentials (used by background monitor / CLI scripts)
# Falls back to SF_CLIENT_ID + SF_CLIENT_SECRET + SF_INSTANCE_URL env vars
# No user session needed
```

### SOQL injection protection

```python
def safe_str(value: str) -> str:
    return value.replace("'", "\\'")   # escapes single quotes in all filters
```

---

## Agent patterns (`mcp_chat/agents.py`)

### Agent 1 — Deal Investigator (`/agent/investigate`)

**Trigger:** Click "🔍 Investigate Deal", enter a company name.

**Why it's an agent and not a function:** Each query depends on the result of the previous one. The AccountId from step 1 is needed to run steps 2–4. You can't write all queries upfront.

```
User: "Investigate Edge Communications"
         │
         ▼
deal_investigator() in agents.py
         │
         ▼
TOOL LOOP:
  1. run_soql → find Opportunity WHERE Name LIKE '%Edge%'
                  → gets: Id, Name, StageName, Amount, AccountId
  2. run_soql → find Account WHERE Id = '<AccountId from step 1>'
                  → gets: Industry, AnnualRevenue, BillingCity
  3. run_soql → find Contacts WHERE AccountId = '<AccountId>'
                  → gets: Name, Title, Email
  4. run_soql → find Cases WHERE AccountId = '<AccountId>' AND Status != 'Closed'
                  → gets: open support issues (risk flag)
         │
         ▼
Streams deal briefing:
  ## Deal Status
  ## Account Profile
  ## Key Contacts
  ## Risks
  ## Recommended Next Steps
```

**Code path:**
```
POST /agent/investigate (app.py:392)
  └── deal_investigator(api_key, sf_token, sf_instance, company_name)  (agents.py:70)
        └── _run_tool_loop() — yields ("tool", name) and ("text", chunk)
              └── mcp_session.call_tool(name, input)  (Salesforce MCP server)
```

---

### Agent 2 — Background Monitor (`/agent/monitor/status`)

**Trigger:** Starts automatically at server startup if `SF_CLIENT_ID` + `SF_CLIENT_SECRET` + `SF_INSTANCE_URL` are set.

**Why it's needed:** Proactive. Nobody asks. The server already knows the pipeline state.

```
app.py startup event
  └── start_monitor(interval_seconds=300)  (agents.py:152)
        └── asyncio.create_task(_monitor_loop())

_monitor_loop():
  while running:
    _monitor_once()
      → open MCP connection (env-var credentials, no user session)
      → Claude runs 3 SOQL queries:
          1. Pipeline by stage (GROUP BY StageName)
          2. Stale deals (LastActivityDate < LAST_N_DAYS:30)
          3. Deals closing this week
      → Claude returns JSON report
      → stored in monitor_state dict
    sleep(300)

GET /agent/monitor/status
  → returns monitor_state (last_report, last_run, run_count, error)

UI "📡 Monitor Status" button
  → fetches /agent/monitor/status
  → formats and displays as chat message
```

---

### Agent 3 — Business Review with Subagents (`/agent/review`)

**Trigger:** Click "📊 Business Review".

**What subagents mean here:** The orchestrator Claude sees two tools (`analyze_pipeline`, `analyze_customer_health`). When it calls one, your code runs a **completely separate Claude call** with its own system prompt and its own Salesforce tool loop. That inner Claude's output is returned as the tool result to the outer Claude.

```
business_review() in agents.py
         │
         ▼
Orchestrator Claude  (system: "You are a business review orchestrator")
  tools: [analyze_pipeline, analyze_customer_health]
         │
         ├── calls analyze_pipeline
         │       └── _run_subagent() → NEW Claude call
         │             system: "You are a pipeline analyst"
         │             tools: all Salesforce MCP tools
         │             → runs SOQL queries
         │             → returns pipeline summary text
         │                         ↓ (returned as tool_result to orchestrator)
         │
         └── calls analyze_customer_health
                 └── _run_subagent() → NEW Claude call
                       system: "You are a customer health analyst"
                       tools: all Salesforce MCP tools
                       → runs SOQL queries
                       → returns case/customer summary text
                                   ↓ (returned as tool_result to orchestrator)
         │
         ▼
Orchestrator synthesizes both → streams executive review
```

---

## Multi-agent pipeline (`mcp_chat/multi_agent.py`)

**Trigger:** Click "🤖 AI Report", enter any business question.

This is the full coordinator → analyst → synthesis pattern. The most sophisticated agent in this repo.

```
User: "Are we on track to hit quota this quarter?"
         │
         ▼
run_report() in multi_agent.py
         │
         ▼
STEP 1 — COORDINATOR (no tools, no Salesforce access)
  Claude reads the question and returns a JSON plan:
  {
    "reasoning": "Need pipeline data and activity data to assess quota attainment",
    "analysts": [
      {"name": "pipeline", "question": "What is total pipeline this quarter vs closed won?"},
      {"name": "activity", "question": "How active have reps been this month?"}
    ],
    "synthesis_focus": "Whether current trajectory will hit quota"
  }
  yield ("coordinator", plan)

         │
         ▼
STEP 2 — ANALYSTS run in PARALLEL (asyncio.gather)
  Each analyst is a separate Claude call, knows nothing about the others.

  Pipeline Analyst                  Activity Analyst
  system: "You are a pipeline       system: "You are an activity
           analyst..."                       analyst..."
  tools: Salesforce MCP tools       tools: Salesforce MCP tools
  runs SOQL on Opportunity          runs SOQL on Task
  returns pipeline summary          returns activity summary
         │                                  │
         └──────────────┬───────────────────┘
                        ▼
                analyst_outputs = {
                  "pipeline": "Total Q2 pipeline $2.4M...",
                  "activity": "87 calls logged this month..."
                }

         │
         ▼
STEP 3 — SYNTHESIS (no tools, no Salesforce access)
  Claude reads all analyst reports.
  Streams executive report:
    ## Executive Summary
    ## Key Findings
    ## Risk Flags
    ## Recommended Actions
```

### Why each role has no access to the others' concerns

| Agent | Has Salesforce tools? | Knows about other agents? | Job |
|---|---|---|---|
| Coordinator | No | No | Reads question → writes plan |
| Pipeline Analyst | Yes | No | Answers ONE pipeline question |
| Customer Health Analyst | Yes | No | Answers ONE cases question |
| Activity Analyst | Yes | No | Answers ONE activity question |
| Synthesis | No | No | Reads all outputs → writes report |

This separation means you can improve, replace, or parallelize any agent without touching the others.

---

## SSE event format (all endpoints use the same format)

Every agent endpoint returns a `StreamingResponse` with `text/event-stream`. The browser JS reads it with `ReadableStream`.

```
data: {"type": "tool",  "name": "run_soql"}          ← Claude called a tool
data: {"type": "stage", "label": "Analyst running: pipeline"}  ← multi-agent progress
data: {"type": "text",  "chunk": "Here are the top"}  ← streaming response text
data: {"type": "error", "message": "..."}             ← something went wrong
data: [DONE]                                           ← stream complete
```

Browser handler in `index.html`:
```javascript
// type=="tool"  → updates typing indicator
// type=="stage" → updates typing indicator (multi-agent steps)
// type=="text"  → appends to chat bubble (streaming effect)
// type=="error" → shows error message
// "[DONE]"      → re-enables input, cleans up
```

---

## All API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Serves the chat UI |
| GET | `/auth/status` | Returns `{sf_authenticated, anthropic_configured}` |
| GET | `/auth/salesforce` | Starts OAuth flow, redirects to Salesforce |
| GET | `/auth/salesforce/callback` | OAuth callback, exchanges code for token |
| POST | `/set-api-key` | Saves Anthropic API key to session |
| POST | `/auth/logout` | Clears session and cookie |
| POST | `/chat` | Main chat — MCP tool loop + streaming response |
| POST | `/agent/investigate` | Deal Investigator agent (SSE stream) |
| GET | `/agent/monitor/status` | Returns last background monitor report |
| POST | `/agent/review` | Business Review with subagents (SSE stream) |
| POST | `/agent/report` | Multi-agent report: coordinator → analysts → synthesis (SSE stream) |
| POST | `/agent/pipeline` | Pipeline health check agent (SSE stream) |

---

## Standalone CLI agent (`scripts/sales_agent.py`)

Connects directly to the MCP server from your terminal. Uses client-credentials auth (env vars). Good for testing without the UI.

```bash
# MCP server must be running first
cd mcp_salesforce && TRANSPORT=sse python server.py

# In another terminal
cd scripts
ANTHROPIC_API_KEY=sk-ant-... python sales_agent.py
```

---

## Environment variables reference

### `mcp_chat/.env`
```bash
ANTHROPIC_API_KEY=sk-ant-...           # for background monitor (agents.py)
MCP_SERVER_URL=http://localhost:8000/sse
MCP_WEBSEARCH_URL=http://localhost:8001/sse
SF_CLIENT_ID=...                       # Salesforce connected app
SF_CLIENT_SECRET=...
SF_LOGIN_URL=https://login.salesforce.com
SF_CALLBACK_URL=http://localhost:3000/auth/salesforce/callback
PORT=3000
```

### `mcp_salesforce/.env`
```bash
SF_CLIENT_ID=...
SF_CLIENT_SECRET=...
SF_INSTANCE_URL=https://yourorg.my.salesforce.com  # for client credentials fallback
PORT=8000
TRANSPORT=sse
```

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude (claude-sonnet-4-6) |
| Agent protocol | Model Context Protocol (MCP) over SSE |
| Backend | FastAPI + uvicorn |
| Salesforce client | simple-salesforce |
| Frontend | Vanilla HTML/JS (no framework) |
| Auth | Salesforce OAuth 2.0 Authorization Code Flow |
| Async | Python asyncio (background tasks, parallel analysts) |
