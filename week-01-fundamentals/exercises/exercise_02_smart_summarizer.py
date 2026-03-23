"""
Exercise 2: Smart Summarizer
==============================
Difficulty: Beginner | Time: 2 hours

Task:
Build an LLM-powered text analyzer that returns structured output
with: summary, key_terms, and sentiment.

Instructions:
1. Set up your OpenAI API key
2. Create a Pydantic model for the output schema
3. Use .with_structured_output() for guaranteed schema compliance
4. Test with at least 2 different paragraphs
5. Bonus: Add Phoenix tracing to view the LLM call

Run: python exercise_02_smart_summarizer.py
"""
import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing import List, Literal

from langchain_groq import ChatGroq
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor

class TextAnalysis(BaseModel):
    """Define the output schema here."""
    summary: str = Field(description="A brief summary of the text.")
    key_terms: List[str] = Field(description="List of key concepts or entities found.")
    sentiment: Literal["positive", "neutral", "negative"] = Field(description="The general sentiment of the text.")

def analyze_text(text: str) -> TextAnalysis:
    """Analyze text and return structured insights.

    Args:
        text: The text to analyze

    Returns:
        TextAnalysis with summary, key_terms, and sentiment
    """
    # 1. Initialize ChatGroq
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    # 2. Use .with_structured_output(TextAnalysis)
    structured_llm = llm.with_structured_output(TextAnalysis)
    
    # 3. Create a clear prompt asking for summary, key terms, and sentiment
    prompt = f"Please analyze the text below and extract its summary, key terms, and sentiment.\n\nText:\n{text}"
    
    # 4. Return the structured result
    return structured_llm.invoke(prompt)

if __name__ == "__main__":
    load_dotenv("config/.env")
    
    px.launch_app()
    LangChainInstrumentor().instrument()

    sample1 = """
    AI agents represent a paradigm shift in software development.
    Unlike traditional programs, agents can reason about their
    environment, use tools autonomously, and improve through
    self-reflection. Companies like Klarna have deployed agents
    that handle millions of customer interactions.
    """

    sample2 = """
    The latest quarterly earnings report showed a significant decline
    in hardware sales, leading to a 15% drop in stock price. However,
    the software services division saw record-breaking growth, which
    management believes will offset the hardware losses next year.
    """

    print("--- Paragraph 1 ---")
    result1 = analyze_text(sample1)
    print(f"Summary: {result1.summary}")
    print(f"Key Terms: {result1.key_terms}")
    print(f"Sentiment: {result1.sentiment}\n")

    print("--- Paragraph 2 ---")
    result2 = analyze_text(sample2)
    print(f"Summary: {result2.summary}")
    print(f"Key Terms: {result2.key_terms}")
    print(f"Sentiment: {result2.sentiment}\n")

    print("Phoenix dashboard is running locally! You can view it at http://localhost:6006")
