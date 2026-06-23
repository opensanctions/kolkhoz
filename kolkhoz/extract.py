"""Extract political position holders from a page via the OpenAI API.

Single agentic step: always sends the rendered page *text* to the model, and
exposes a ``get_screenshot`` function tool so the model can pull the full-page
screenshot (tiled into overlapping squares) only when the text is insufficient.
This avoids wasting expensive tiled-image vision on pages that genuinely have
no holders, while still rescuing names that only live in the screenshot.

Uses the OpenAI Responses API with structured outputs and low reasoning effort.
"""

import base64
import logging
import os
from enum import Enum

from kolkhoz import pravda
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

client = AsyncOpenAI()

REASONING_EFFORT = "low"
MAX_ROUNDS = 2

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
- First, classify the page as `roster`, `profile`, or `other` (see field
  descriptions).
- The page TEXT is given to you. The full-page SCREENSHOT is NOT included by
  default. If the text is insufficient to read the holders — names appear to
  be in images, the page is JS-rendered, or the text is too thin even to tell
  what kind of page it is — call `get_screenshot` and then extract. Otherwise
  do not call it; answer from the text.
- `page_type=other` should be used for generic pages (about, contact, article,
  landing) that are not in the business of listing position holders.
"""

TOOLS = [
    {
        "type": "function",
        "name": "get_screenshot",
        "description": (
            "Return the full-page screenshot of this page, tiled into overlapping squares. "
            "Call this ONLY when the text alone is not enough to read the position holders — "
            "for example the names seem to be in images, a JS-rendered org chart, or the text "
            "is too thin to tell what the page is. Do NOT call it if the text already names "
            "the holders, or if the page clearly has no roster."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
]


class PageType(str, Enum):
    roster = "roster"  # page that lists named position holders (board, council, staff, directory)
    profile = "profile"  # a single person's bio / CV / appointment page
    other = "other"  # about/contact/landing/article — not expected to list holders


class Holder(BaseModel):
    human: str = Field(
        description="Full name of the person, exactly as written on the page."
    )
    position: str = Field(
        description="Specific position title the person holds, e.g. 'Council Member'."
    )


class Extraction(BaseModel):
    page_type: PageType = Field(description="The kind of page this is.")
    holders: list[Holder] = Field(
        description="Position holders found on the page. Empty if none."
    )


async def extract(
    text: str, screenshot_blob: bytes | None
) -> tuple[Extraction, list[dict]]:
    """Extract holders from one page.

    Always sends *text*. The model may pull the *screenshot* via the
    ``get_screenshot`` tool.

    Returns (extraction, provenance) where provenance is the full dump of
    every model response, one per round (id, model, token usage, reasoning,
    function calls, and the final parsed answer all live in there). Whether
    the model pulled the screenshot is recoverable from the function_call
    items in those dumps.
    """
    tile = int(os.environ["IMAGE_TILE_SIZE"])
    overlap = float(os.environ["IMAGE_TILE_OVERLAP"])

    input_items = [{"role": "user", "content": [{"type": "input_text", "text": text}]}]
    provenance: list[dict] = []

    for _ in range(MAX_ROUNDS):
        response = await client.responses.parse(
            model=os.environ["OPENAI_MODEL"],
            instructions=INSTRUCTIONS,
            input=input_items,
            tools=TOOLS,
            text_format=Extraction,
            reasoning={"effort": REASONING_EFFORT},
        )
        provenance.append(response.model_dump(mode="json"))

        # Does the model want the screenshot?
        tool_calls = [
            o
            for o in response.output
            if o.type == "function_call" and o.name == "get_screenshot"
        ]
        if not tool_calls:
            # No tool call -> final structured answer.
            if response.output_parsed is None:
                raise ValueError(
                    f"Model returned no parsed output (possible refusal): {response.output}"
                )
            return response.output_parsed, provenance

        # It asked for the screenshot. Append the assistant's call(s) + our
        # tool result, loop again.
        if screenshot_blob is None:
            # Can't fulfil the request. Tell the model so it can answer from
            # text alone.
            out_content = "No screenshot is available for this page."
        else:
            tiles = pravda.split_image(screenshot_blob, tile, overlap)
            out_content = [
                {
                    "type": "input_text",
                    "text": (
                        f"The {len(tiles)} image(s) below are overlapping tiles "
                        "of a single full-page screenshot of the same page. Read "
                        "them together as one page. Tiles overlap, so the same "
                        "person/position may recur — extract each holder once."
                    ),
                },
                *[
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,"
                        + base64.b64encode(t).decode(),
                    }
                    for t in tiles
                ],
            ]
        input_items += (
            response.output
        )  # echoes the assistant's function_call item(s) back
        for tc in tool_calls:
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": out_content,
                }
            )

    raise RuntimeError("extraction loop exhausted without a final answer")
