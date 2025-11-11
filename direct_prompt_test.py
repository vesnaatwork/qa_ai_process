# Test script for DirectPromptAgent class

from base_agents import DirectPromptAgent        
import os


prompt = "What is the capital of France?"
agent = DirectPromptAgent(model="llama3.2")
response = agent.respond(prompt)

print("\n=== DirectPromptAgent Test ===")
print(f"Prompt: {prompt}")
print(f"Response: {response}")
print("\nKnowledge Source: The DirectPromptAgent used its inherent knowledge from the model's training data. No external knowledge sources were accessed.")


