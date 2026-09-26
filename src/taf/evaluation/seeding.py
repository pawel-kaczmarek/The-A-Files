"""Per-trial seed derivation.

An experiment has one base seed. Every random quantity in it - each message,
each attack realisation - gets its own seed derived from the base seed and
from what identifies that quantity, never from its position in a list. So

* adding a payload length or a file does not change the messages or the
  noise of any other trial;
* messages of different lengths are independent draws, not prefixes of one
  another;
* the attack seed of a trial depends on the file, the repetition and the
  attack, but not on the method. Every method therefore meets the same noise
  realisation on the same file (common random numbers), which is what makes
  a paired comparison between methods fair, while repetitions and files
  sample the channel independently.
"""

from __future__ import annotations

import hashlib

import numpy as np


def stable_key(value: int | str) -> int:
    """A 64-bit integer for ``value`` that is the same on every run and machine.

    Python's ``hash()`` of a string is salted per process, so it cannot be
    used to key reproducible seeds.
    """
    if isinstance(value, int) and value >= 0:
        return value
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def derive_seed(base_seed: int, *keys: int | str) -> int:
    """A 32-bit seed determined by the base seed and the identifying keys."""
    entropy = [stable_key(base_seed), *(stable_key(key) for key in keys)]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def message_seed(base_seed: int, length: int, occurrence: int = 0) -> int:
    """Seed for the random messages of one length.

    ``occurrence`` separates two message specifications that happen to share
    a length within one configuration.
    """
    return derive_seed(base_seed, "message", length, occurrence)


def attack_seed(base_seed: int, file_key: str, repetition: int, attack: str) -> int:
    """Seed for one attack realisation, shared by all methods and payloads."""
    return derive_seed(base_seed, "attack", file_key, repetition, attack)


__all__ = ["attack_seed", "derive_seed", "message_seed", "stable_key"]
