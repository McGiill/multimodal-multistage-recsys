"""Read the closet's CURRENT user snapshot; no event writes or stale cache."""
from __future__ import annotations

import json
import logging
from typing import Any, Mapping
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.error import URLError

from src.common.config import env, load_config
from src.recommendation.preferences.signal import parse_weights

logger = logging.getLogger(__name__)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        return None


def fetch_outfit_preferences(user_id: str, config: Mapping[str, Any] | None = None
                            ) -> tuple[dict[str, float], str]:
    """Fail open to baseline on outage/invalid snapshots; never reuse old signal."""
    cfg = config if config is not None else load_config('outfit_preferences')
    if not cfg['enabled']:
        return {}, 'disabled'
    endpoint = env('CLOSET_PREFERENCE_URL', cfg['source_url'])
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError('Invalid configured closet preference URL')
    timeout, max_bytes = float(cfg['timeout_seconds']), int(cfg['max_response_bytes'])
    if not 0 < timeout <= 5 or not 1 <= max_bytes <= 4_000_000:
        raise ValueError('Invalid preference transport bounds')
    request = Request(endpoint + '?' + urlencode({'user_id': user_id}),
                      headers={'Accept': 'application/json'})
    try:
        with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
            raw = response.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError('Preference response too large')
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError('Invalid preference snapshot')
        weights = parse_weights(payload, user_id)
        return weights, 'loaded' if weights else 'empty'
    except (URLError, OSError, ValueError):
        logger.warning('Closet preference snapshot unavailable or invalid; using baseline')
        return {}, 'unavailable'
