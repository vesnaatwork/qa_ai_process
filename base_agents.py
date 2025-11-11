import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime
import gc
import psutil
import requests

OLLAMA_URL = "http://localhost:11434"

def debug_log(method_name, additional_info=""):
    print(f"[DEBUG] Calling method: {method_name} {additional_info}")

def ollama_chat(model, messages):
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["message"]["content"]

def ollama_generate(model, prompt):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["response"]

class DirectPromptAgent:
    def __init__(self, model="llama3.2"):
        self.model = model

    def respond(self, prompt):
        return ollama_generate(self.model, prompt)

class AugmentedPromptAgent:
    def __init__(self, persona, model="llama3.2"):
        self.persona = persona
        self.model = model

    def respond(self, input_text):
        prompt = f"{self.persona} Always begin your answer with: 'Dear students,'. Forget all previous context.\n{input_text}"
        return ollama_generate(self.model, prompt)

class KnowledgeAugmentedPromptAgent:
    def __init__(self, persona, knowledge, model="llama3.2"):
        self.persona = persona
        self.knowledge = knowledge
        self.model = model

    def respond(self, input_text):
        prompt = (
            f"You are {self.persona} knowledge-based assistant. Forget all previous context. "
            f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge} "
            "Answer the prompt based on this knowledge, not your own.\n"
            f"{input_text}"
        )
        return ollama_generate(self.model, prompt)

class RAGKnowledgePromptAgent:
    """
    Retrieval-Augmented Generation agent for knowledge-based responses.
    """
    def __init__(self, persona, knowledge_corpus, model="llama3.2"):
        self.persona = persona
        self.knowledge_corpus = knowledge_corpus
        self.model = model

    def respond(self, prompt):
        # For demonstration, just use the knowledge_corpus directly
        system_prompt = (
            f"You are {self.persona}. Use only the following knowledge to answer: {self.knowledge_corpus}. "
            "Forget all previous context."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        return ollama_chat(self.model, messages)

class EvaluationAgent:
    def __init__(self, worker_agent, evaluation_criteria, persona="Evaluator", model="llama3.2", max_interactions=3):
        self.worker_agent = worker_agent
        self.evaluation_criteria = evaluation_criteria
        self.persona = persona
        self.model = model
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt):
        prompt_to_evaluate = initial_prompt
        final_response = None
        evaluation = None
        iterations = 0

        for i in range(self.max_interactions):
            iterations += 1
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            messages = [
                {"role": "system", "content": f"You are {self.persona}. Evaluate the following answer."},
                {"role": "user", "content": eval_prompt}
            ]
            evaluation = ollama_chat(self.model, messages)

            if evaluation.lower().startswith("yes"):
                final_response = response_from_worker
                break
            else:
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                )
                messages = [
                    {"role": "system", "content": f"You are {self.persona}. Provide correction instructions."},
                    {"role": "user", "content": instruction_prompt}
                ]
                instructions = ollama_chat(self.model, messages)
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )
                final_response = response_from_worker

        return {
            "final_response": final_response,
            "evaluation": evaluation,
            "iterations": iterations
        }

class RoutingAgent:
    def __init__(self, agents, model="llama3.2"):
        """
        Initialize the RoutingAgent with a list of agents and a model name.
        Each agent should be a dictionary with 'description', 'name', and 'func' (callable).
        """
        self.agents = agents  # List of agent dicts: {'description': str, 'name': str, 'func': callable}
        self.model = model

    def route(self, user_input):
        """
        Route the user prompt to the most appropriate agent based on description similarity.
        Returns the response from the selected agent.
        """
        best_agent = None
        best_score = -1

        for agent in self.agents:
            # Simple similarity: count overlapping words between input and description
            score = len(set(user_input.lower().split()) & set(agent['description'].lower().split()))
            if score > best_score:
                best_score = score
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score})")
        return best_agent["func"](user_input)