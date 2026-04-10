#!/usr/bin/env python3
"""
Sales Analysis Agent — uses the real Salesforce MCP server.
Requires the MCP server to be running (python mcp_salesforce/server.py)
and SF_CLIENT_ID / SF_CLIENT_SECRET / SF_INSTANCE_URL set in .env
"""

import asyncio
import json
import os

import anthropic
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client

load_dotenv()

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")

SYSTEM_PROMPT = """You are a Salesforce sales analyst agent.
Use the available tools to perform a complete pipeline health check.

Your analysis must cover:
1. Pipeline summary — total open pipeline value + deal count grouped by stage (use run_soql with GROUP BY StageName)
2. At-risk deals — open opportunities with no activity in the last 30 days (LastActivityDate < LAST_N_DAYS:30)
3. Closing soon — deals with CloseDate in the next 14 days that are still open
4. Top 5 open deals by amount

Always use run_soql for aggregations. Reference specific deal names and dollar amounts.
End with a "Focus This Week" list of 3-5 concrete, prioritised actions."""


def to_dict(obj):
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return json.loads(json.dumps(obj, default=str))


async def run_agent(user_request: str):
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    async with sse_client(url=MCP_SERVER_URL) as (read, write):
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

            messages = [{"role": "user", "content": user_request}]

            print(f"\nUser: {user_request}\n{'─' * 60}\n")

            while True:
                response = client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=4096,
                    thinking={"type": "adaptive"},
                    system=SYSTEM_PROMPT,
                    tools=tools,
                    messages=messages,
                )

                for block in response.content:
                    if block.type == "text":
                        print(block.text)

                if response.stop_reason == "end_turn":
                    break

                messages.append({"role": "assistant", "content": [to_dict(b) for b in response.content]})

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"\n  [Tool] {block.name}({json.dumps(block.input)})")
                        if block.name in tool_names:
                            result = await mcp_session.call_tool(block.name, dict(block.input))
                            result_text = result.content[0].text if result.content else "No data."
                        else:
                            result_text = f"Unknown tool: {block.name}"
                        print(f"  [Result] {result_text[:200]}{'...' if len(result_text) > 200 else ''}")
                        tool_results.append({
                            "type":        "tool_result",
                            "tool_use_id": block.id,
                            "content":     result_text,
                        })

                messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    asyncio.run(run_agent(
        "Give me a full pipeline health check. "
        "Highlight at-risk deals with no recent activity, deals closing in the next 14 days, "
        "and tell me where to focus this week."
    ))
