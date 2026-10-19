"""Shared HTTP session with retries and a timeout."""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TIMEOUT = 30

_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        retry = Retry(total=3, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.headers["User-Agent"] = "lms-model (+https://github.com/Kbrown176/LMS-Model)"
        _session = s
    return _session


def get(url: str, **kwargs) -> requests.Response:
    resp = session().get(url, timeout=TIMEOUT, **kwargs)
    resp.raise_for_status()
    return resp
