"""Extract political position holders from a page via the OpenAI API.

Tier 1 feeds the rendered page text. Tier 2 adds the full-page screenshot for
pages tier 1 missed. Both use strict structured outputs, a cached static prompt
prefix (the instructions), and low reasoning effort.

Bump PROMPT_VERSION whenever the instructions, schema, or model change — it is
part of the cache key, so stale extractions get re-run.
"""

import base64

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

client = AsyncOpenAI()

MODEL = "gpt-5.4-mini"
PROMPT_VERSION = "v2"
REASONING_EFFORT = "low"

INSTRUCTIONS = """\
You extract political position holders from the text of a single web page.

A "holder" is a specific, named human who holds a named position (office, seat,
title, or role) at the organisation the page is about — e.g. a council member,
board director, judge, minister, or chair.

Rules:
- Return one entry per (human, position) pair. If one person holds two
  positions, return two entries.
- Use the person's full name exactly as written on the page.
- Use the most specific position title shown. If the page is about an
  organisation and the title omits it, you may name the body (e.g. "Council
  Member").
- Do not invent people. Only extract humans actually named on the page.
- Ignore names that are not position holders (authors, contacts, mentions).
- If the page names no position holders, return an empty list.
"""


class Holder(BaseModel):
    human: str = Field(
        description="Full name of the person, exactly as written on the page."
    )
    position: str = Field(
        description="Specific position title the person holds, e.g. 'Council Member'."
    )


class Extraction(BaseModel):
    holders: list[Holder] = Field(
        description="Position holders found on the page. Empty if none."
    )


async def _parse(page_content: list[dict]) -> tuple[Extraction, dict]:
    response = await client.responses.parse(
        model=MODEL,
        instructions=INSTRUCTIONS,
        input=[{"role": "user", "content": page_content}],
        reasoning={"effort": REASONING_EFFORT},
        text_format=Extraction,
    )
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    # Guard against refusals — the model may decline for safety reasons.
    if response.output_parsed is None:
        raise ValueError(
            f"Model returned no parsed output (possible refusal): {response.output}"
        )
    return response.output_parsed, usage


async def extract_from_text(text: str) -> tuple[Extraction, dict]:
    """Tier 1: extract holders from the page text alone."""
    return await _parse([{"type": "input_text", "text": text}])


async def extract_from_text_and_image(
    text: str, screenshot: bytes
) -> tuple[Extraction, dict]:
    """Tier 2: extract holders from the page text plus its screenshot."""
    data_url = "data:image/png;base64," + base64.b64encode(screenshot).decode()
    return await _parse(
        [
            {"type": "input_text", "text": text},
            {"type": "input_image", "image_url": data_url},
        ]
    )
