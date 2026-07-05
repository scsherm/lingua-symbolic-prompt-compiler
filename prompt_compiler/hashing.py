from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    payload = value if isinstance(value, str) else stable_json(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

