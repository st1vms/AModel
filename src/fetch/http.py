"""Shared HTTP helper: polite delays, retries with backoff."""
import random
import time

import requests

import config


def polite_sleep():
    lo, hi = config.REQUEST_DELAY
    time.sleep(random.uniform(lo, hi))


def get_text(url: str, retries: int = config.MAX_RETRIES, extra_headers: dict | None = None) -> str:
    """GET a URL and return the response text. Raises on final failure."""
    headers = {"User-Agent": config.USER_AGENT}
    if extra_headers:
        headers.update(extra_headers)
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.text
            last_err = f"HTTP {resp.status_code}"
            if resp.status_code == 404:
                raise FileNotFoundError(f"404 Not Found: {url}")
        except requests.RequestException as e:
            last_err = str(e)
        if attempt < retries:
            wait = config.RETRY_BACKOFF * attempt
            print(f"  retry {attempt}/{retries - 1} after {wait:.0f}s ({last_err})")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")
