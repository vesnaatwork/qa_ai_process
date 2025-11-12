#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import json
import requests
from dotenv import load_dotenv
import re
from pprint import pprint

from base_agents import RAGKnowledgePromptAgent, EvaluationAgent

def call_model(agent, prompt, system_prompt=None, temperature=0.2, max_tokens=2000):
    """
    Calls the agent's generate method with the given prompt and system prompt.
    """
    return agent.respond(
        prompt=prompt,
    )

def extract_analytics_data(excel_path):
    """
    Extracts and summarizes analytics data from the provided Excel file.
    Returns a formatted string to append to the system prompt.
    """
    xl = pd.ExcelFile(excel_path)
    summary = []

    # Device sheets
    for sheet in ["Device - standtogether.org", "Device - afp"]:
        df = xl.parse(sheet)
        total_sessions = df["Sessions"].sum()
        summary.append(f"\n--- {sheet} ---")
        summary.append(f"Total sessions: {total_sessions}")
        # Group by Device category and brand/model
        cat_summary = df.groupby("Device category")["Sessions"].sum().reset_index()
        summary.append("Device category breakdown:")
        for _, row in cat_summary.iterrows():
            percent = (row["Sessions"] / total_sessions) * 100 if total_sessions else 0
            summary.append(f"  {row['Device category']}: {row['Sessions']} sessions ({percent:.1f}%)")
        # Top brands/models
        top_devices = df.groupby(["Device brand", "Device model"])["Sessions"].sum().reset_index()
        top_devices = top_devices.sort_values("Sessions", ascending=False).head(5)
        summary.append("Top devices:")
        for _, row in top_devices.iterrows():
            summary.append(f"  {row['Device brand']} {row['Device model']}: {row['Sessions']} sessions")

    # Browser sheets
    for sheet in ["Browser - standtogether.org", "Browser - afp"]:
        df = xl.parse(sheet)
        total_sessions = df["Sessions"].sum()
        summary.append(f"\n--- {sheet} ---")
        summary.append(f"Total sessions: {total_sessions}")
        # Group by Browser/version
        browser_summary = df.groupby(["Browser", "Browser version"])["Sessions"].sum().reset_index()
        browser_summary = browser_summary.sort_values("Sessions", ascending=False)
        summary.append("Top browsers and versions:")
        for _, row in browser_summary.head(5).iterrows():
            percent = (row["Sessions"] / total_sessions) * 100 if total_sessions else 0
            summary.append(f"  {row['Browser']} {row['Browser version']}: {row['Sessions']} sessions ({percent:.1f}%)")
        # Screen resolution breakdown
        res_summary = df.groupby("Screen resolution")["Sessions"].sum().reset_index()
        res_summary = res_summary.sort_values("Sessions", ascending=False).head(3)
        summary.append("Top screen resolutions:")
        for _, row in res_summary.iterrows():
            percent = (row["Sessions"] / total_sessions) * 100 if total_sessions else 0
            summary.append(f"  {row['Screen resolution']}: {row['Sessions']} sessions ({percent:.1f}%)")

    return "\n".join(summary)

def main(knowledge, analysis, latest_os_versions):
    # Create system prompt with knowledge and specific instructions
    system_prompt = f"""You are an experienced QA Lead specializing in cross-device testing strategies. You MUST use ONLY the following Google Analytics data to inform your recommendations:

{knowledge}

CRITICAL INSTRUCTIONS:
1. Base ALL recommendations STRICTLY on the provided analytics data - do not fabricate statistics or percentages
2. Use ONLY the ACTUAL percentages and counts shown in the data summary
3. CRUCIAL: When determining Tier 1 and Tier 2 classifications, use percentages WITHIN EACH CATEGORY, not overall percentages
   - For example, Chrome might be 32% of all traffic but 80% of desktop traffic - classify based on the 80% figure
   - For example, Safari might be 10% of all traffic but 60% of mobile traffic - classify based on the 60% figure
4. Your device/OS recommendations MUST match the actual distribution in the data
5. Use the ACTUAL browser version ranges from the data (like Chrome 134-139, Safari 18)
6. DO NOT make up specific device models unless they appear explicitly in the data
7. If specific models aren't available, use "Latest iPhone model with iOS {latest_os_versions.get('iOS', '17-18')}" instead of inventing models
8. DO NOT create numbers or statistics that aren't directly derived from the provided data
9. ENSURE CONSISTENCY: If a browser is listed in Tier 1 or Tier 2, it CANNOT also appear in "Not Tested"
10. CALCULATE IN-CATEGORY PERCENTAGES: Show what percentage each browser represents WITHIN its category (desktop/mobile/tablet), not as percentage of all traffic
11. EXPLICITLY address data limitations and their impact on your recommendations
12. For device counts, explain that Apple's privacy restrictions make device model details unavailable
13. Acknowledge when data is limited or unclear rather than making assumptions
14. DO NOT CONFUSE percentages across different dimensions - keep desktop, mobile, and tablet calculations separate
"""

    prompt = f"""
Based ONLY on the provided Google Analytics data, create a data-driven cross-device testing strategy that accurately reflects actual user patterns.

YOUR TASK:
1. Analyze ONLY the provided data to determine the most common device categories, browsers, and operating systems
2. Recommend testing configurations based EXCLUSIVELY on the actual usage percentages shown
3. Create a two-tier testing approach prioritizing the platforms with highest actual traffic

CRITICAL: OS PERCENTAGES MUST MATCH THE DATA EXACTLY
- Windows is approximately {analysis.get('desktop_os_breakdown', {}).get('Windows/Other', {}).get('percentage', 0)}% of desktop traffic (NOT 100%)
- macOS is approximately {analysis.get('desktop_os_breakdown', {}).get('macOS', {}).get('percentage', 0)}% of desktop traffic
- iOS is approximately {analysis.get('mobile_os_breakdown', {}).get('iOS', {}).get('percentage', 0)}% of mobile traffic (NOT 100%)
- Android is approximately {analysis.get('mobile_os_breakdown', {}).get('Android', {}).get('percentage', 0)}% of mobile traffic

STRICT TIER CLASSIFICATION RULES:
- Tier 1: Include ONLY browsers, devices or OS with >20% share IN THEIR SPECIFIC CATEGORY
  - For example, Chrome with >20% of DESKTOP traffic (not >20% of all traffic)
  - For example, Safari with >20% of MOBILE traffic (not >20% of all traffic)
- Tier 2: Include ONLY browsers, devices or OS with 5-20% share IN THEIR SPECIFIC CATEGORY
- Configurations with <5% share in their specific category should be explicitly listed as "Not Tested"
- A browser or device cannot appear in both Tier 1 and "Not Tested" sections

FORMAT YOUR RESPONSE AS:

## Cross-Device Testing Strategy

### Data Quality Assessment
- [Assess the quality and limitations of the analytics data]
- [Explain how these limitations affect your recommendations]
- [Note any potential biases in the data that impact testing priorities]

### Device Category Distribution
- [Present the ACTUAL percentage breakdown between Desktop, Mobile, and Tablet]
- [Explain how this distribution informs your testing prioritization]

### Browser Distribution BY CATEGORY
**Desktop browsers:** [List top browser percentages WITHIN DESKTOP category]
**Mobile browsers:** [List top browser percentages WITHIN MOBILE category]
**Tablet browsers:** [List top browser percentages WITHIN TABLET category]

### Operating System Distribution
**Desktop:**
- Windows: {analysis.get('desktop_os_breakdown', {}).get('Windows/Other', {}).get('percentage', 0)}% of desktop traffic
- macOS: {analysis.get('desktop_os_breakdown', {}).get('macOS', {}).get('percentage', 0)}% of desktop traffic

**Mobile:**
- iOS: {analysis.get('mobile_os_breakdown', {}).get('iOS', {}).get('percentage', 0)}% of mobile traffic
- Android: {analysis.get('mobile_os_breakdown', {}).get('Android', {}).get('percentage', 0)}% of mobile traffic

**Tablet:** [List iPadOS and Android percentages WITHIN tablet category]

### Tier 1 Browsers (Primary - Full Regression Testing)
**Desktop:**
- [List ONLY browsers with >20% share WITHIN desktop traffic with EXACT percentages]

**Mobile:**
- [List ONLY browsers with >20% share WITHIN mobile traffic with EXACT percentages]

**Tablet:** (if applicable)
- [List ONLY browsers with >20% share WITHIN tablet traffic with EXACT percentages]

### Tier 2 Browsers (Secondary - Smoke Testing Only)
**Desktop:**
- [List ONLY browsers with 5-20% share WITHIN desktop traffic with EXACT percentages]

**Mobile:**
- [List ONLY browsers with 5-20% share WITHIN mobile traffic with EXACT percentages]

**Tablet:**
- [List ONLY browsers with 5-20% share WITHIN tablet traffic with EXACT percentages]

### Tier 1 Devices and Operating Systems (Primary - Full Regression Testing)
**Desktop:**
- Windows: {analysis.get('desktop_os_breakdown', {}).get('Windows/Other', {}).get('percentage', 0)}% of desktop traffic
  - [List relevant browser(s) based on the data]
  - [Recommend screen resolution: 1920x1080 and 1440x900]

- macOS: {analysis.get('desktop_os_breakdown', {}).get('macOS', {}).get('percentage', 0)}% of desktop traffic (Include in Tier 1 ONLY if >20%)
  - [List relevant browser(s) based on the data]
  - [Recommend screen resolution: 1440x900]

**Mobile:**
- iOS: {analysis.get('mobile_os_breakdown', {}).get('iOS', {}).get('percentage', 0)}% of mobile traffic
  - Latest iPhone model with iOS {latest_os_versions.get('iOS', '17-18')}
  - [Recommend screen resolution: 390x844]

- Android: {analysis.get('mobile_os_breakdown', {}).get('Android', {}).get('percentage', 0)}% of mobile traffic (Include in Tier 1 ONLY if >20%)
  - Latest Samsung Galaxy model with Android {latest_os_versions.get('Android', '13-15')}
  - [Recommend screen resolution: 393x852]

**Tablet:** (if applicable)
- [List ONLY operating systems with >20% share in tablet category]
  - [For each, list relevant device types based on the data]
  - [Include screen resolution recommendations for tablet testing]

### Tier 2 Devices and Operating Systems (Secondary - Smoke Testing Only)
- [List ONLY device types with 5-20% share in their category]
- [For each, list relevant OS version(s) based on the data]
- [Include screen resolution recommendations for testing]

### Testing Approach & Coverage Strategy
- **Tier 1:** Complete end-to-end regression testing across all features and user flows
- **Tier 2:** Smoke testing covering major functionality and critical user paths only
- **Coverage Calculation:** 
  - For each category, calculate SEPARATE coverage percentages:
    - Desktop browsers: [X% in Tier 1] + [Y% in Tier 2] = [Z% total covered within desktop]
    - Desktop operating systems: [X% in Tier 1] + [Y% in Tier 2] = [Z% total covered within desktop]
    - Mobile browsers: [X% in Tier 1] + [Y% in Tier 2] = [Z% total covered within mobile]
    - Mobile operating systems: [X% in Tier 1] + [Y% in Tier 2] = [Z% total covered within mobile]
    - Tablet browsers: [X% in Tier 1] + [Y% in Tier 2] = [Z% total covered within tablet]

### Not Tested
**Desktop:** [List browsers and operating systems with <5% share WITHIN desktop category]
**Mobile:** [List browsers and operating systems with <5% share WITHIN mobile category]
**Tablet:** [List browsers and operating systems with <5% share WITHIN tablet category]

### Data-Driven Recommendations
- [Suggest improvements to data collection if needed]
- [Recommend how frequently to update the test matrix based on data trends]
"""

    evaluation_criteria = (
        "The QA matrix should be accurate, clear, and adhere strictly to the provided instructions. "
        "It must reflect the actual usage patterns from the analytics data without fabricating any statistics. "
        "The tier classifications must be correct based on the specified percentage thresholds. "
        "The response should be well-structured and easy to understand."
    )

    # Instantiate the RAG agent (worker)
    rag_agent = RAGKnowledgePromptAgent(persona="Software Quality Assurance Expert", knowledge_corpus=knowledge, model="llama3.2")

    # Generate QA matrix using RAG agent
    qa_matrix = call_model(rag_agent, prompt, system_prompt)
    # Instantiate the evaluator agent
    evaluator_agent = EvaluationAgent(rag_agent, evaluation_criteria, persona="Software Quality Assurance Expert Evaluator", model="llama3.2")

    # Evaluate the QA matrix
    evaluation_prompt = f"Evaluate the following QA matrix for accuracy, clarity, and adherence to the instructions:\n\n{qa_matrix}"
    evaluation = evaluator_agent.evaluate(evaluation_prompt)

    print("=== QA Matrix ===")
    print(qa_matrix)
    print("\n=== Evaluation ===")
    print(evaluation)

# Example usage:
if __name__ == "__main__":
    # Example usage of extract_analytics_data
    analytics_file = "analytics_data.xlsx"  # Update path if needed
    extracted_data = extract_analytics_data(analytics_file)
    knowledge = extracted_data  # Use extracted data as knowledge
    analysis = {}      # Replace with your actual analysis dict
    latest_os_versions = {}  # Replace with your actual OS versions dict
    main(knowledge, analysis, latest_os_versions)
