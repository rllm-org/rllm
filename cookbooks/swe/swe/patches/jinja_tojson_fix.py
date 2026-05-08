"""Patch Jinja2's tojson filter to not HTML-escape output.

Jinja2's htmlsafe_json_dumps escapes &, ', <, > for HTML safety.
This is counterproductive when output is consumed by LLMs — models
see \\u0026 instead of & and construct sed patterns that never match.
"""

import json

import jinja2.filters
import jinja2.utils
import markupsafe

_patched = False


def apply_jinja_tojson_patch() -> bool:
    global _patched
    if _patched:
        return False

    def _plain_json_dumps(obj, dumps=None, **kwargs):
        if dumps is None:
            dumps = json.dumps
        return markupsafe.Markup(dumps(obj, **kwargs))

    # Patch both the canonical location and the reference captured by do_tojson
    jinja2.utils.htmlsafe_json_dumps = _plain_json_dumps
    jinja2.filters.htmlsafe_json_dumps = _plain_json_dumps
    _patched = True
    return True
