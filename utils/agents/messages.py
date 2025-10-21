import base64
from io import BytesIO
from typing import Any

import tiktoken
from PIL import Image
from tiktoken.core import Encoding

# Rough estimate: 512x512 ~ 85 tokens based on OpenAI
TILE_SIZE = 512
TOKENS_PER_TILE = 85
CONTEXT_BUDGET = 0.90

Message = dict[str, Any]
# Tiktoken Text Encoder
encoder: Encoding = tiktoken.get_encoding('cl100k_base')


def _rough_text_tokens(s: str) -> int:
    """Rough token estimation using tiktoken encoder"""
    return len(encoder.encode(s))


def _rough_image_tokens(b64: str) -> int:
    """Rough token estimation using 512x512 = 85 tokens"""
    if not b64.startswith('data:image'):
        return 0
    try:
        b64 = b64.split(',', 1)[1]
        raw: bytes = base64.b64decode(b64)
        with Image.open(BytesIO(raw)) as im:
            w, h = im.width, im.height
        tiles_w = (w + TILE_SIZE - 1) // TILE_SIZE
        tiles_h = (h + TILE_SIZE - 1) // TILE_SIZE
        return tiles_w * tiles_h * TOKENS_PER_TILE
    except Exception:
        return 0


def _rough_message_tokens(msg: dict[str, Any]) -> int:
    """
    Rough token count for one message. Supports both string content and
    OpenAI-style content arrays with text and image_url parts (data: only).
    """
    content = msg.get('content')
    if isinstance(content, str):
        return _rough_text_tokens(content)

    total = 0
    for part in content or []:
        ptype = part.get('type')
        if ptype == 'text':
            total += _rough_text_tokens(part.get('text', '') or '')
        elif ptype == 'image_url':
            url = (part.get('image_url') or {}).get('url') or ''
            total += _rough_image_tokens(url)
    return total


def fit_context(
    history: list[dict[str, Any]], max_context_tokens: int
) -> list[dict[str, Any]]:
    """
    Keep System + newest turns, drop oldest when we exceed the context.
    Works on OpenAI-style dict messages.
    """

    if not history:
        return []

    # Extract last system message if present; otherwise, use the first message defensively.
    systems: list[dict[str, Any]] = [m for m in history if m.get('role') == 'system']
    sys_msg: dict[str, Any] = systems[-1] if systems else history[0]
    sys_tokens: int = _rough_message_tokens(sys_msg)

    # The rest of the messages (non-system), newest to oldest.
    rest: list[dict[str, Any]] = [m for m in history if m is not sys_msg]
    rest_rev = list(reversed(rest))

    kept: list[dict[str, Any]] = []
    total = sys_tokens
    limit = int(max_context_tokens * CONTEXT_BUDGET)

    # Iterate messages from newest to oldest.
    for m in rest_rev:
        t = _rough_message_tokens(m)
        # If we exceed 90% of the total context, we stop.
        if total + t > limit:
            break
        kept.append(m)
        total += t

    kept.reverse()
    return [sys_msg] + kept


def _merge_parts(a: Any, b: Any) -> Any:
    """Merge OpenAI-style content: str or list[{'type',...}] (text/image_url)."""
    if a is None:
        return b
    if b is None:
        return a
    if isinstance(a, str) and isinstance(b, str):
        return (a + '\n\n' + b).strip()
    if isinstance(a, list) and isinstance(b, list):
        return a + b
    if isinstance(a, str) and isinstance(b, list):
        return [{'type': 'text', 'text': a}] + b
    if isinstance(a, list) and isinstance(b, str):
        return a + [{'type': 'text', 'text': b}]
    return (str(a) + '\n\n' + str(b)).strip()


def normalize_messages(
    messages: list[Message],
    *,
    merge_system_into_user: bool = False,
    keep_system: bool = True,
) -> list[Message]:
    """
    Minimal normalizer:
      - (optional) merge all system content into the first user turn
      - drop system turns if keep_system=False
      - drop leading assistant(s)
      - merge adjacent same-role turns
      - enforce strict alternation starting with 'user'
    """
    if not messages:
        return []

    msgs = messages

    # 1) Optionally fold ALL system content into the first 'user'
    sys_content = None
    if merge_system_into_user:
        for m in msgs:
            if m.get('role') == 'system':
                sys_content = _merge_parts(sys_content, m.get('content'))

        new_msgs: list[Message] = []
        inserted = False
        for m in msgs:
            r = m.get('role')
            if r == 'system':
                continue
            if not inserted and r == 'user':
                new_msgs.append(
                    {
                        'role': 'user',
                        'content': _merge_parts(sys_content, m.get('content')),
                    }
                )
                inserted = True
            else:
                new_msgs.append({'role': r, 'content': m.get('content')})
        if sys_content is not None and not inserted:
            # No user at all → synthesize a first user with system content
            new_msgs.insert(0, {'role': 'user', 'content': sys_content})
        msgs = new_msgs

    # 2) Drop system if requested
    if not keep_system:
        msgs = [m for m in msgs if m.get('role') != 'system']

    # 3) Drop leading assistants until first user
    while msgs and msgs[0].get('role') == 'assistant':
        msgs.pop(0)
    if not msgs:
        return []

    # 4) Merge adjacent same-role turns
    merged: list[Message] = []
    for m in msgs:
        if merged and merged[-1].get('role') == m.get('role'):
            merged[-1]['content'] = _merge_parts(
                merged[-1].get('content'), m.get('content')
            )
        else:
            merged.append({'role': m.get('role'), 'content': m.get('content')})

    # 5) Enforce alternation: user -> assistant -> user -> ...
    alt: list[Message] = []
    expect = 'user'
    for m in merged:
        r = m.get('role')
        if r not in ('user', 'assistant'):
            continue
        if r == expect:
            alt.append(m)
            expect = 'assistant' if expect == 'user' else 'user'

    # If we somehow still start with assistant, drop it
    if alt and alt[0]['role'] != 'user':
        alt = alt[1:]

    return alt


def _get_media_type(raw: bytes) -> str | None:
    # Quick magic-byte checks first (fast path)
    if raw.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if raw.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if raw.startswith((b'GIF87a', b'GIF89a')):
        return 'image/gif'
    if raw[:4] == b'RIFF' and raw[8:12] == b'WEBP':
        return 'image/webp'
    # Fallback to PIL (slower but tolerant)
    try:
        with Image.open(BytesIO(raw)) as im:
            fmt = (im.format or '').upper()
        return {
            'PNG': 'image/png',
            'JPEG': 'image/jpeg',
            'JPG': 'image/jpeg',
            'GIF': 'image/gif',
            'WEBP': 'image/webp',
        }.get(fmt)
    except Exception:
        return None


def to_antrophic(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get('content')
    if isinstance(content, str):
        return [{'type': 'text', 'text': content}]

    blocks: list[dict[str, Any]] = []
    for part in content or []:
        if part.get('type') == 'text':
            blocks.append({'type': 'text', 'text': part.get('text')})
        elif part.get('type') == 'image_url':
            url = (part.get('image_url') or {}).get('url') or ''
            if not url:
                continue
            if url.startswith('data:image'):
                header, b64 = url.split(',', 1)
                raw = base64.b64decode(b64, validate=True)
                sniffed = _get_media_type(raw)
                header_mime = header.split(':', 1)[1].split(';', 1)[0]
                mime = sniffed or header_mime

                blocks.append(
                    {
                        'type': 'image',
                        'source': {'type': 'base64', 'media_type': mime, 'data': b64},
                    }
                )
            else:
                blocks.append({'type': 'image', 'source': {'type': 'url', 'url': url}})
    return blocks
