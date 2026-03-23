"""
Exercise 3: Name Memory Agent
================================
Difficulty: Intermediate | Time: 2.5 hours

Task:
Build a LangGraph agent that:
1. Greets the user and asks for their name
2. Remembers the name across conversation turns
3. Tracks how many messages have been exchanged
4. Uses the name naturally in all responses

Instructions:
1. Define a ChatState TypedDict with: messages, user_name, message_count
2. Create a chat_node that uses the LLM with state context
3. Build a LangGraph StateGraph with conditional edges
4. Add a simple loop: continue chatting until user says "bye"
5. Bonus: Add Phoenix tracing and observe the conversation flow

Run: python exercise_03_name_memory_agent.py
"""

import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END, add_messages
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor

class ChatState(TypedDict):
    """Define your agent state here."""
    messages: Annotated[list, add_messages]
    user_name: Optional[str]
    message_count: int

def chat_node(state: ChatState):
    """The main chat node."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    
    user_name = state.get("user_name")
    message_count = state.get("message_count", 0) + 1
    messages = state["messages"]
    
    if not user_name and messages:
        last_message = messages[-1].content
        class NameExtract(BaseModel):
            name: Optional[str] = Field(description="The user's first name if explicitly mentioned, else None")
            
        try:
            extractor = ChatGroq(model="llama-3.3-70b-versatile", temperature=0).with_structured_output(NameExtract)
            extracted = extractor.invoke(f"Extract the user's name from this message if present: '{last_message}'")
            if extracted and extracted.name:
                user_name = extracted.name
        except Exception:
            pass # Failsafe incase extraction fails

    sys_prompt = f"You are a friendly AI agent. This is turn number {message_count} in the conversation.\n"
    if user_name:
        sys_prompt += f"You are talking to {user_name}. Use their name naturally in your response."
    else:
        sys_prompt += "You don't know the user's name yet. If this is the first message, politely ask for it."

    full_messages = [{"role": "system", "content": sys_prompt}] + messages
    response = llm.invoke(full_messages)
    
    return {
        "messages": [response],
        "message_count": message_count,
        "user_name": user_name
    }

def create_memory_agent():
    """Build the name-memory agent."""
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)
    return graph.compile()

if __name__ == "__main__":
    load_dotenv("config/.env")
    
    px.launch_app()
    LangChainInstrumentor().instrument()
    
    print("Name Memory Agent (provider: Groq)")
    print("=" * 40)
    print("Type 'quit' or 'bye' to exit\n")

    agent = create_memory_agent()
    state = {"messages": [], "user_name": None, "message_count": 0}

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Agent: Goodbye! See you next time.")
                break
                
            if "messages" not in state or not state["messages"]:
                state["messages"] = [HumanMessage(content=user_input)]
            else:
                state["messages"].append(HumanMessage(content=user_input))
            
            state = agent.invoke(state)
            
            print(f"Agent: {state['messages'][-1].content}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Agent Error: {e}")
