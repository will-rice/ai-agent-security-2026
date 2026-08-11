"""Test-wide setup for the jed_attack suite.

Sets ``JED_MIN_SHIP_MESSAGES=1`` BEFORE any ``jed_attack`` import so
``Submission.messages`` binds a floor of 1 (its ``Field(min_length)`` reads
``config.MIN_SHIP_MESSAGES`` at import time). Fixtures throughout the suite build
small single-shape submissions to exercise scoring/fill/validation in isolation;
production keeps the real ``MIN_SHIP_MESSAGES`` floor (100) unchanged.
"""

import os

# setdefault: never clobber an explicit floor a specific test sets in its own env.
os.environ.setdefault("JED_MIN_SHIP_MESSAGES", "1")
