#!/usr/bin/env python3
"""
MCP Server — Salesforce Org Tools
Gives Claude (or any MCP client) the ability to query your Salesforce org:
  • get_accounts       → list accounts with optional name filter
  • get_contacts       → list contacts with optional email filter
  • get_opportunities  → list opportunities with optional stage filter
  • get_cases          → list support cases with optional status filter
  • run_soql           → run any read-only SOQL query
  • get_org_info       → basic info about the connected Salesforce org

Auth modes:
  1. Per-user OAuth token  — passed via ?sf_token=&sf_instance= on the SSE URL (used by mcp-chat)
  2. Client Credentials    — fallback using SF_CLIENT_ID / SF_CLIENT_SECRET env vars
"""

import asyncio
import os
import re
from contextvars import ContextVar
from urllib.parse import parse_qs

import requests
import uvicorn
from dotenv import load_dotenv
from simple_salesforce import Salesforce, SalesforceAuthenticationFailed
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.stdio import stdio_server
from mcp import types

load_dotenv()

server = Server("salesforce-mcp")

# ---------------------------------------------------------------------------
# Per-session credential store
#
# When mcp-chat connects via SSE it passes the user's SF access token as
# query params: GET /sse?sf_token=<token>&sf_instance=<url>
#
# We capture the session_id assigned by the SSE transport and store the
# token against it. When a tool call POST arrives (/messages/?session_id=<id>)
# we look up the token and inject it into the async context so get_sf() can use it.
# ---------------------------------------------------------------------------

_session_creds: dict = {}                                       # { session_id: { token, instance_url } }
_current_sf_creds: ContextVar = ContextVar("current_sf_creds", default=None)


# ---------------------------------------------------------------------------
# Salesforce connection
# ---------------------------------------------------------------------------

def get_sf() -> Salesforce:
    # Priority 1: per-user token injected via context (OAuth Authorization Code flow)
    creds = _current_sf_creds.get()
    if creds and creds.get("token"):
        return Salesforce(
            instance_url=creds["instance_url"],
            session_id=creds["token"],
        )

    # Priority 2: Client Credentials flow (env vars — for CLI / direct use)
    client_id     = os.getenv("SF_CLIENT_ID")
    client_secret = os.getenv("SF_CLIENT_SECRET")
    instance_url  = os.getenv("SF_INSTANCE_URL")

    if not all([client_id, client_secret, instance_url]):
        raise ValueError(
            "No Salesforce credentials available. "
            "Connect via the chat UI or set SF_CLIENT_ID, SF_CLIENT_SECRET, SF_INSTANCE_URL in .env."
        )

    resp = requests.post(
        f"{instance_url}/services/oauth2/token",
        data={
            "grant_type":    "client_credentials",
            "client_id":     client_id,
            "client_secret": client_secret,
        },
    )
    resp.raise_for_status()
    token = resp.json()

    return Salesforce(
        instance_url=token["instance_url"],
        session_id=token["access_token"],
    )


def safe_str(value: str) -> str:
    """Escape single quotes to prevent SOQL injection."""
    return value.replace("'", "\\'")


def records_to_text(records: list, empty_msg: str = "No records found.") -> str:
    if not records:
        return empty_msg
    lines = []
    for i, rec in enumerate(records, 1):
        clean = {k: v for k, v in rec.items() if k != "attributes"}
        lines.append(f"[{i}] " + " | ".join(f"{k}: {v}" for k, v in clean.items()))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_accounts",
            description=(
                "Fetch accounts from Salesforce. "
                "Optionally filter by account name (partial match). "
                "Returns Id, Name, Industry, Phone, Website, AnnualRevenue."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name_filter": {
                        "type": "string",
                        "description": "Optional partial name to search for (e.g. 'Acme')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of records to return (default 10, max 50)",
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_contacts",
            description=(
                "Fetch contacts from Salesforce. "
                "Optionally filter by email or last name. "
                "Returns Id, FirstName, LastName, Email, Phone, Account Name."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "Optional email to filter by (exact match)",
                    },
                    "last_name": {
                        "type": "string",
                        "description": "Optional last name to filter by (partial match)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of records to return (default 10, max 50)",
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_opportunities",
            description=(
                "Fetch opportunities from Salesforce. "
                "Optionally filter by stage. "
                "Returns Id, Name, StageName, Amount, CloseDate, Account Name."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "stage": {
                        "type": "string",
                        "description": "Optional stage filter e.g. 'Prospecting', 'Closed Won', 'Closed Lost'",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of records to return (default 10, max 50)",
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_cases",
            description=(
                "Fetch support cases from Salesforce. "
                "Optionally filter by status. "
                "Returns Id, CaseNumber, Subject, Status, Priority, Account Name."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Optional status filter e.g. 'New', 'Working', 'Closed'",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of records to return (default 10, max 50)",
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="run_soql",
            description=(
                "Run a dynamic read-only SOQL query against the Salesforce org. "
                "Use this tool to answer any question about Salesforce data by building the right query. "
                "Only SELECT statements are allowed — no INSERT, UPDATE, or DELETE. "
                "\n\nCommon objects and fields:"
                "\n- Account: Id, Name, Industry, Phone, Website, AnnualRevenue, BillingCity, BillingCountry, Type"
                "\n- Contact: Id, FirstName, LastName, Email, Phone, Title, Department, Account.Name"
                "\n- Opportunity: Id, Name, StageName, Amount, CloseDate, Probability, Account.Name"
                "\n- Case: Id, CaseNumber, Subject, Status, Priority, Origin, Account.Name"
                "\n- Lead: Id, FirstName, LastName, Email, Company, Status, Industry, LeadSource"
                "\n- Task: Id, Subject, Status, Priority, ActivityDate"
                "\n- User: Id, Name, Email, Title, IsActive"
                "\n\nDate literals: TODAY, YESTERDAY, THIS_WEEK, LAST_WEEK, THIS_MONTH, LAST_MONTH, THIS_YEAR, LAST_N_DAYS:n"
                "\n\nExamples:"
                "\n- SELECT Id, Name, AnnualRevenue FROM Account ORDER BY AnnualRevenue DESC LIMIT 5"
                "\n- SELECT Id, CaseNumber, Subject, Status FROM Case WHERE Priority = 'High' AND Status != 'Closed'"
                "\n- SELECT Id, Name, Amount, CloseDate FROM Opportunity WHERE CloseDate = THIS_MONTH ORDER BY Amount DESC"
                "\n- SELECT Id, FirstName, LastName, Email FROM Contact WHERE Account.Name LIKE '%Acme%'"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A valid SOQL SELECT query built dynamically based on the user's request",
                    }
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get_org_info",
            description="Return basic information about the connected Salesforce org (name, id, org type).",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
    ]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    # ── get_accounts ─────────────────────────────────────────────────────────
    if name == "get_accounts":
        try:
            sf    = get_sf()
            limit = min(int(arguments.get("limit", 10)), 50)
            query = "SELECT Id, Name, Industry, Phone, Website, AnnualRevenue FROM Account"
            if arguments.get("name_filter"):
                query += f" WHERE Name LIKE '%{safe_str(arguments['name_filter'])}%'"
            query += f" LIMIT {limit}"
            result = sf.query(query)
            return [types.TextContent(type="text", text=records_to_text(result["records"]))]
        except SalesforceAuthenticationFailed:
            return [types.TextContent(type="text", text="Salesforce authentication failed. Please reconnect via the chat UI.")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {e}")]

    # ── get_contacts ─────────────────────────────────────────────────────────
    if name == "get_contacts":
        try:
            sf      = get_sf()
            limit   = min(int(arguments.get("limit", 10)), 50)
            query   = "SELECT Id, FirstName, LastName, Email, Phone, Account.Name FROM Contact"
            filters = []
            if arguments.get("email"):
                filters.append(f"Email = '{safe_str(arguments['email'])}'")
            if arguments.get("last_name"):
                filters.append(f"LastName LIKE '%{safe_str(arguments['last_name'])}%'")
            if filters:
                query += " WHERE " + " AND ".join(filters)
            query += f" LIMIT {limit}"
            result = sf.query(query)
            return [types.TextContent(type="text", text=records_to_text(result["records"]))]
        except SalesforceAuthenticationFailed:
            return [types.TextContent(type="text", text="Salesforce authentication failed. Please reconnect via the chat UI.")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {e}")]

    # ── get_opportunities ────────────────────────────────────────────────────
    if name == "get_opportunities":
        try:
            sf    = get_sf()
            limit = min(int(arguments.get("limit", 10)), 50)
            query = "SELECT Id, Name, StageName, Amount, CloseDate, Account.Name FROM Opportunity"
            if arguments.get("stage"):
                query += f" WHERE StageName = '{safe_str(arguments['stage'])}'"
            query += f" ORDER BY CloseDate DESC LIMIT {limit}"
            result = sf.query(query)
            return [types.TextContent(type="text", text=records_to_text(result["records"]))]
        except SalesforceAuthenticationFailed:
            return [types.TextContent(type="text", text="Salesforce authentication failed. Please reconnect via the chat UI.")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {e}")]

    # ── get_cases ────────────────────────────────────────────────────────────
    if name == "get_cases":
        try:
            sf    = get_sf()
            limit = min(int(arguments.get("limit", 10)), 50)
            query = "SELECT Id, CaseNumber, Subject, Status, Priority, Account.Name FROM Case"
            if arguments.get("status"):
                query += f" WHERE Status = '{safe_str(arguments['status'])}'"
            query += f" ORDER BY CreatedDate DESC LIMIT {limit}"
            result = sf.query(query)
            return [types.TextContent(type="text", text=records_to_text(result["records"]))]
        except SalesforceAuthenticationFailed:
            return [types.TextContent(type="text", text="Salesforce authentication failed. Please reconnect via the chat UI.")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {e}")]

    # ── run_soql ─────────────────────────────────────────────────────────────
    if name == "run_soql":
        try:
            query = arguments.get("query", "").strip()
            if not re.match(r"^\s*SELECT\b", query, re.IGNORECASE):
                return [types.TextContent(type="text", text="Only SELECT queries are allowed.")]
            sf     = get_sf()
            result = sf.query(query)
            return [types.TextContent(type="text", text=records_to_text(result["records"]))]
        except SalesforceAuthenticationFailed:
            return [types.TextContent(type="text", text="Salesforce authentication failed. Please reconnect via the chat UI.")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {e}")]

    # ── get_org_info ─────────────────────────────────────────────────────────
    if name == "get_org_info":
        try:
            sf     = get_sf()
            result = sf.query("SELECT Id, Name, OrganizationType, IsSandbox FROM Organization LIMIT 1")
            if result["records"]:
                rec  = result["records"][0]
                info = (
                    f"Org Name    : {rec.get('Name')}\n"
                    f"Org ID      : {rec.get('Id')}\n"
                    f"Org Type    : {rec.get('OrganizationType')}\n"
                    f"Is Sandbox  : {rec.get('IsSandbox')}"
                )
            else:
                info = "Could not retrieve org info."
            return [types.TextContent(type="text", text=info)]
        except SalesforceAuthenticationFailed:
            return [types.TextContent(type="text", text="Salesforce authentication failed. Please reconnect via the chat UI.")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {e}")]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


# ---------------------------------------------------------------------------
# Entry point — auto-selects transport based on TRANSPORT env var
# ---------------------------------------------------------------------------

async def run_stdio() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def run_sse() -> None:
    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        # Extract per-user SF token passed by mcp-chat as query params
        sf_token    = request.query_params.get("sf_token")
        sf_instance = request.query_params.get("sf_instance")

        session_id_holder: dict = {}

        # Intercept the SSE endpoint event to capture the session_id the transport assigns,
        # so we can store the token against it and look it up on incoming POSTs.
        original_send = request._send

        async def capturing_send(message):
            if message["type"] == "http.response.body" and "id" not in session_id_holder:
                body = message.get("body", b"").decode("utf-8", errors="ignore")
                m = re.search(r"session_id=([^\s\"&\n]+)", body)
                if m and sf_token:
                    sid = m.group(1)
                    session_id_holder["id"] = sid
                    _session_creds[sid] = {"token": sf_token, "instance_url": sf_instance}
            await original_send(message)

        try:
            async with sse_transport.connect_sse(
                request.scope, request.receive, capturing_send
            ) as streams:
                await server.run(
                    streams[0], streams[1], server.create_initialization_options()
                )
        finally:
            # Clean up session creds when connection closes
            sid = session_id_holder.get("id")
            if sid:
                _session_creds.pop(sid, None)

    async def handle_post_with_creds(scope, receive, send):
        """Wrap handle_post_message to inject the right SF creds into async context."""
        qs         = parse_qs(scope.get("query_string", b"").decode())
        session_id = (qs.get("session_id") or [None])[0]
        creds      = _session_creds.get(session_id) if session_id else None

        token = _current_sf_creds.set(creds)
        try:
            await sse_transport.handle_post_message(scope, receive, send)
        finally:
            _current_sf_creds.reset(token)

    starlette_app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=handle_post_with_creds),
        ]
    )

    port = int(os.getenv("PORT", 8000))
    print(f"Salesforce MCP server running on http://0.0.0.0:{port}/sse")
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    transport = os.getenv("TRANSPORT", "stdio")
    if transport == "sse":
        run_sse()
    else:
        asyncio.run(run_stdio())
