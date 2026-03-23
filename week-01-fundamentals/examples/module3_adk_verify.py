"""
Module 3: Google ADK Setup Verification
=========================================
Verifies that Google ADK is correctly installed and working.
Requires GOOGLE_API_KEY in your environment.

Run: python week-01-fundamentals/examples/module3_adk_verify.py
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Create a simple ADK agent
agent = LlmAgent(
    name="test_agent",
    model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
    instruction="You are a helpful test agent. Reply with: ADK is working!",
)

# Create a session and runner
session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name="test_app", session_service=session_service)


async def test_adk():
    session = await session_service.create_session(
        app_name="test_app", user_id="test_user"
    )
    async for event in runner.run_async(
        user_id="test_user",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="Hello")]),
    ):
        if event.is_final_response():
            print(f"Agent: {event.content.parts[0].text}")


asyncio.run(test_adk())
print("\n[OK] ADK setup verified!")
