#!/usr/bin/env python3
"""
Isolates ResumeParserAgent from the rest of the pipeline to pinpoint
why resume_06_ayesha_naeem.pdf produces an empty candidate profile,
given document parsing itself is already confirmed working (1418+
alpha chars extracted). No workflow, no other agents, no DB writes --
just: parse document -> build prompt -> call LLM -> inspect raw response.
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.document_parser import parse_document
from src.agents.resume_parser import ResumeParserAgent

RESUME_PATH = "sample_data/resumes/resume_06_ayesha_naeem.pdf"


async def main():
    parsed = parse_document(RESUME_PATH)
    print(f"Parse result: success={parsed.success}  chars={len(parsed.text)}  "
          f"confidence={parsed.confidence:.2f}  error={parsed.error_message or 'none'}")

    if not parsed.success:
        print("Document parsing failed -- this IS the root cause. Stop here.")
        return

    print("\nSanitized text sample (repr, first 400 chars):")
    print(repr(parsed.text[:400]))
    print(f"\nContains raw NUL (0x00): {'\\x00' in parsed.text}")

    agent = ResumeParserAgent()
    prompt = agent._build_parsing_prompt(parsed.text)
    print(f"\nCalling LLM with prompt length={len(prompt)} chars...")

    raw_response = await agent._call_llm_async(prompt)
    print("\n--- RAW LLM RESPONSE (first 1000 chars) ---")
    print(raw_response[:1000])
    print("--- END ---\n")

    if raw_response.startswith("Error calling LLM:"):
        print("ROOT CAUSE: the LLM API call itself threw an exception (see "
              "above). BaseAgent._call_llm_async() silently turns this into "
              "a fake response string, which fails JSON parsing downstream "
              "and produces the empty/low-confidence result you saw.")
    else:
        data = agent._extract_json_from_response(raw_response)
        if data is None:
            print("LLM responded, but the response is not valid JSON -- a "
                  "JSON-parsing issue, not an API-call or document-parsing issue.")
        else:
            print("LLM response WAS valid JSON.")
            print("contact:", data.get("contact"))
            print("skills_section:", data.get("skills_section"))
            print("parsing_confidence:", data.get("parsing_confidence"))
            print("\nIf contact.name is still empty here, the model itself "
                  "failed to extract the name despite valid input -- a "
                  "model/prompt quality issue, not a code defect.")


if __name__ == "__main__":
    asyncio.run(main())