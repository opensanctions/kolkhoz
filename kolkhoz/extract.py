"""LLM extraction of position holders from a single page.

Owns the structured-output schema (``Person`` / ``Position`` /
``Extraction``), the model instructions, the text/screenshot decision
(``screenshot_reason``), and the ``extract()`` call. ``screenshot_parts``
tiles a screenshot blob into model input parts using ``split_image`` from
:mod:`kolkhoz.capture`.

The OpenAI client is an explicit argument: it is constructed at the CLI
boundary, not held as module-global state.
"""

import base64
import logging

from openai import OpenAI
from pydantic import BaseModel, Field

from kolkhoz.capture import split_image
from kolkhoz.config import ImageConfig, ModelConfig

log = logging.getLogger("kolkhoz")

REASONING_EFFORT = "low"

# Content signals that force the full-page screenshot into the model
# input. Below MIN_TEXT_WORDS the plaintext is too thin to read holders
# from; with at least MIN_IMAGES and an imgs-per-word ratio over
# MAX_IMG_DENSITY the page is image-dominated (org charts, headshot
# rosters) and the names/titles are likely baked into the images rather
# than the text. The decision is made in code, not by the model.
MIN_TEXT_WORDS = 200
MIN_IMAGES = 10
MAX_IMG_DENSITY = 0.1

_INSTRUCTIONS_HEAD = """\
# Role

You extract stated person-position relationships from the content of a single
web page and return structured data only. Extract what the source says; do not
decide whether a position is politically relevant or useful downstream.

# Definitions

- A holder is a named human whom the page explicitly ties to a named position:
  an office, seat, title, role, or membership.
- Include every explicit person-position relationship, whether current,
  former, honorary, incidental, or stated in contact information. Relevance
  and selection are not part of extraction.
- A person can hold several positions; list each as a separate entry under
  that person. The person's own facts (date of birth, biography, country)
  are stated once on the person, not repeated per position.
- A name, action, or personal relationship on its own is not a position. Do
  not turn source wording such as "founded by", "married to", or "spoke at"
  into invented titles such as "Founder", "Spouse", or "Speaker".

# Goal and success

Return one person record per holder, with every explicitly stated position and
all facts the page states. The result is correct when every stated
person-position relationship is captured once and nothing is invented. If the
page states none, return an empty persons list.

# How to work

For each position, first find the verbatim phrase on the page that ties the
person to the role. Put that phrase in evidence_quotes, then fill the other
fields only from what the page states. Every field must trace to text that is
actually on the page. Repeated mentions are supporting evidence, not separate
positions: emit the same person-position relationship only once.

# Constraints

- Only extract humans the page names. Never invent a person or a position.
- Copy names, position titles, countries, and dates exactly as written (e.g.
  "3 May 2022"). Never normalize, singularize, expand, translate, or reformat.
- Leave a field null, and evidence_quotes empty, when the page does not state
  it. Never infer a value from context or fill it from world knowledge.
- person.country is the person's country; position.jurisdiction is an
  explicitly stated geographic area the position covers. An organisation or
  employer is not a jurisdiction; null unless the source states the area.
- position.description is a stated mandate, remit, or set of responsibilities.
  A title, employer, date, or circumstance of departure is not a description.
"""

_INSTRUCTIONS_WITH_SCREENSHOT = """\

# Screenshot

The full-page screenshot is attached as overlapping image tiles. Read the
tiles together with the text as one page. Tiles overlap, so the same person or
position may recur across tiles — extract each holder once.
"""


# Structured-output schema for extraction. Nested: a Person holds many
# Positions; person-level facts live on the person. Storage and export stay
# flat (one row per person-position), so ``flatten_persons`` is the single
# boundary that owns the nested→flat mapping. Because Structured Outputs emit
# fields in schema order, ``evidence_quotes`` is first on Position so the model
# commits to a verbatim anchor before filling the other fields.
class Position(BaseModel):
    evidence_quotes: list[str] = Field(
        default_factory=list,
        description=(
            "Short verbatim phrases from the page that tie this person to this "
            "position. Lift them first; they anchor the other fields. Empty if "
            "stated only in a table or list with no prose."
        ),
    )
    name: str | None = Field(
        default=None,
        description=(
            "Name of the position the person holds, e.g. 'Council Member'. Copy "
            "the most specific title exactly as shown; do not normalize or "
            "singularize it. Null only if the page names the person but states "
            "no title."
        ),
    )
    description: str | None = Field(
        default=None,
        description=(
            "Responsibilities, mandate, or remit of the position as stated on the "
            "page; not its title, employer, dates, or circumstances of departure."
        ),
    )
    jurisdiction: str | None = Field(
        default=None,
        description=(
            "Explicitly stated geographic area the position covers, as written on "
            "the page. An organisation or employer is not a jurisdiction."
        ),
    )
    start_date: str | None = Field(
        default=None, description="When the person started, as written on the page."
    )
    end_date: str | None = Field(
        default=None,
        description="When the person left or will leave, as written on the page.",
    )


class Person(BaseModel):
    name: str = Field(description="Full name of the person, exactly as written.")
    dob: str | None = Field(
        default=None, description="Date of birth as written on the page."
    )
    bio: str | None = Field(
        default=None, description="Short biographical note from the page."
    )
    country: str | None = Field(
        default=None,
        description="Country associated with the person, as written on the page.",
    )
    # A holder holds at least one position; we never emit a person with none.
    positions: list[Position] = Field(
        min_length=1, description="Every position this person holds on the page."
    )


class Extraction(BaseModel):
    persons: list[Person] = Field(
        default_factory=list,
        description="Position holders stated on the page. Empty list if none.",
    )


def flatten_persons(extraction: Extraction) -> list[dict]:
    """Flatten the nested extraction into one dict per (person, position).

    Person-level fields repeat across a person's positions so every flat row
    is self-contained — matching the Holder table and the JSONL export. The
    dict keys are the flat storage/export names; this is the only place that
    knows how nested maps to flat.
    """
    rows: list[dict] = []
    for person in extraction.persons:
        for position in person.positions:
            rows.append(
                {
                    "person_name": person.name,
                    "position_name": position.name,
                    "person_dob": person.dob,
                    "person_bio": person.bio,
                    "person_country": person.country,
                    "position_description": position.description,
                    "position_jurisdiction": position.jurisdiction,
                    "position_start_date": position.start_date,
                    "position_end_date": position.end_date,
                    "evidence_quotes": position.evidence_quotes,
                }
            )
    return rows


def extract(
    client: OpenAI,
    model: ModelConfig,
    image: ImageConfig,
    text: str,
    screenshot_blob: bytes | None,
) -> Extraction:
    """Extract holders from a single page.

    Always sends the page *text*. When the content signals in the caller
    fire, *screenshot_blob* is tiled and attached as overlapping image
    parts alongside the text. There is no model-driven tool call: the
    decision to include the screenshot is made in code from text length
    and image density, not delegated to the model.
    """
    parts: list[dict] = [{"type": "input_text", "text": text}]
    if screenshot_blob is not None:
        parts.extend(screenshot_parts(image, screenshot_blob))

    response = client.responses.parse(
        model=model.name,
        instructions=_INSTRUCTIONS_HEAD
        + (_INSTRUCTIONS_WITH_SCREENSHOT if screenshot_blob is not None else ""),
        input=[{"role": "user", "content": parts}],
        text_format=Extraction,
        reasoning={"effort": REASONING_EFFORT},
    )
    extraction = response.output_parsed
    positions = sum(len(p.positions) for p in extraction.persons)
    log.info(
        "  → final: %d person(s), %d position(s)",
        len(extraction.persons),
        positions,
    )
    return extraction


def screenshot_parts(image: ImageConfig, screenshot_blob: bytes) -> list[dict]:
    """Tile a full-page screenshot into overlapping input parts for the model."""
    tiles = split_image(screenshot_blob, image.tile_size, image.tile_overlap)
    return [
        {
            "type": "input_text",
            "text": "Overlapping tiles of the full-page screenshot follow:",
        },
        *[
            {
                "type": "input_image",
                "image_url": "data:image/png;base64," + base64.b64encode(t).decode(),
            }
            for t in tiles
        ],
    ]


def screenshot_reason(text: str, html: str) -> str | None:
    """Return why the screenshot should be attached, or None to skip it.

    Any one signal forces the screenshot into the model input:
    - *thin text*: fewer than ``MIN_TEXT_WORDS`` words of plaintext, i.e.
      the names/titles are unlikely to be in the text at all.
    - *image-dense*: at least ``MIN_IMAGES`` images and an imgs-per-word
      ratio above ``MAX_IMG_DENSITY``, i.e. the page is dominated by
      images that likely carry the names/titles (org charts, headshot
      rosters).
    """
    words = len(text.split())
    imgs = html.count("<img")
    if words < MIN_TEXT_WORDS:
        return f"thin text ({words} words)"
    if imgs >= MIN_IMAGES and imgs / words > MAX_IMG_DENSITY:
        return f"image-dense ({imgs} imgs / {words} words)"
    return None
