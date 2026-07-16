"""Encoders produce guardrail-safe payloads the exfil predicate still detects."""

from aicomp_sdk.core.predicates import payload_exfiltrates_secret

from jed_attack.submission.encoders import SCHEMES, encode_payload, is_guardrail_safe

SECRET_VALUE = "MySuperSecretPassword123!"  # value-after-'=' form of a secret line


def test_every_scheme_is_detected_as_exfil() -> None:
    """Each encoding round-trips: the predicate still flags the secret."""
    for scheme in SCHEMES:
        payload = encode_payload(SECRET_VALUE, scheme)
        assert payload_exfiltrates_secret(
            payload, read_secrets=[SECRET_VALUE]
        ), f"{scheme} not detected"


def test_encoded_schemes_are_guardrail_safe() -> None:
    """base64/hex/reverse payloads carry no ultra-dangerous substring."""
    for scheme in ("base64", "hex", "reverse"):
        assert is_guardrail_safe(encode_payload(SECRET_VALUE, scheme))
