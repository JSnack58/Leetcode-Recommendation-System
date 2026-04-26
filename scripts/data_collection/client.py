# HOW TO GET YOUR LEETCODE COOKIES
# 1. Log into leetcode.com in Chrome
# 2. Press F12 -> Application -> Cookies -> https://leetcode.com
# 3. Copy "LEETCODE_SESSION", "csrftoken", and "cf_clearance" into .env.scraper

import os
from pathlib import Path
from curl_cffi import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env.scraper")

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)


class LeetCodeClient:
    """Authenticated curl_cffi session that mimics Chrome's TLS fingerprint to pass Cloudflare."""

    BASE_URL = "https://leetcode.com"

    def __init__(self, session_cookie: str, csrf_token: str, cf_clearance: str) -> None:
        self._session = requests.Session(impersonate="chrome")
        self._session.headers.update(
            {
                "User-Agent": _USER_AGENT,
                "Cookie": (
                    f"LEETCODE_SESSION={session_cookie}; "
                    f"csrftoken={csrf_token}; "
                    f"cf_clearance={cf_clearance}"
                ),
                "Referer": self.BASE_URL + "/",
            }
        )

    def get(self, url: str, **kwargs):
        kwargs.setdefault("timeout", 10)
        return self._session.get(url, **kwargs)

    @classmethod
    def from_env(cls) -> "LeetCodeClient":
        session = os.environ.get("LEETCODE_SESSION")
        csrf = os.environ.get("LEETCODE_CSRF")
        cf = os.environ.get("LEETCODE_CF_CLEARANCE")

        missing = [
            name
            for name, val in [
                ("LEETCODE_SESSION", session),
                ("LEETCODE_CSRF", csrf),
                ("LEETCODE_CF_CLEARANCE", cf),
            ]
            if not val
        ]
        if missing:
            raise ValueError(
                f"Missing credentials: {', '.join(missing)}\n"
                "Fill in .env.scraper at the project root."
            )
        return cls(session, csrf, cf)
