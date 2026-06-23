"""Extract political position holders from a page via the OpenAI API.

Stage 1 feeds the rendered page text. Stage 2 retries stage-1 misses with the
full-page screenshot on its own (stage 1 already failed on the text, so the
text isn't re-sent), tiled into overlapping squares so the model never has to
rescale a page larger than its image cap. Both use strict structured outputs,
a cached static prompt prefix (the instructions), and low reasoning effort.
"""

import base64
import logging
import os

from kolkhoz import pravda
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

client = AsyncOpenAI()

REASONING_EFFORT = "low"

INSTRUCTIONS = """\
You extract political position holders from the content of a single web page.

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
        model=os.environ["OPENAI_MODEL"],
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
    """Stage 1: extract holders from the page text alone."""
    return await _parse([{"type": "input_text", "text": text}])


async def extract_from_image(screenshot: bytes) -> tuple[Extraction, dict]:
    """Stage 2: extract holders from the full-page screenshot alone.

    The screenshot is tiled into overlapping squares under the model's image
    cap; all tiles are sent in a single request.
    """
    tile = int(os.environ["IMAGE_TILE_SIZE"])
    overlap = float(os.environ["IMAGE_TILE_OVERLAP"])
    tiles = pravda.split_image(screenshot, tile, overlap)
    log.info("tiled screenshot into %d piece(s)", len(tiles))

    # Tell the model the images are tiles of one screenshot, not separate
    # pages: tiles overlap, so the same person/position may recur.
    content = [
        {
            "type": "input_text",
            "text": (
                f"The following {len(tiles)} image(s) are overlapping tiles of a "
                "single full-page screenshot of one web page, split to fit the "
                "size limit. Read them together as one page. Tiles overlap, so "
                "the same person or position may appear in more than one tile — "
                "extract each holder once."
            ),
        },
        *[
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,"
                + base64.b64encode(piece).decode(),
            }
            for piece in tiles
        ],
    ]
    return await _parse(content)
