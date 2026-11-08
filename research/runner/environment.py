"""The environment a runner builds walls in: no paid API, and from .env.local only what the case needs."""

import os

from research.paths import REPO

DROPPED_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "STONEPACKING_ALLOW_PAID_API", "OPENAI_BASE_URL")
KEPT_KEYS = ("MOSEKLM_LICENSE_FILE", "PYTHONNOUSERSITE")


def kept_settings(text):
    """The values of KEPT_KEYS in the text of .env.local, a relative license path made absolute; the rest is ignored."""
    pairs = (raw.strip().partition("=") for raw in text.splitlines())
    settings = ((key, value.strip().strip("\"'")) for key, _, value in pairs)
    kept = {key: value for key, value in settings if key in KEPT_KEYS and value}
    license_file = kept.get("MOSEKLM_LICENSE_FILE")
    if license_file and not os.path.isabs(license_file):
        kept["MOSEKLM_LICENSE_FILE"] = str(REPO / license_file)
    return kept


def prepare_environment(env_file, tmp):
    """Remove the API keys, take only KEPT_KEYS from `env_file` and send temporary files to `tmp`."""
    for key in DROPPED_KEYS:
        os.environ.pop(key, None)
    os.environ.update(kept_settings(env_file.read_text(encoding="utf-8")))
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", TEMP=str(tmp), TMP=str(tmp))
