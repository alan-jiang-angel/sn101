"""Wallet signing helpers for task-server requests."""

from __future__ import annotations

import json
from typing import Any


def wallet_hotkey(wallet: Any) -> str:
    hotkey = getattr(wallet, "hotkey", None)
    ss58 = getattr(hotkey, "ss58_address", None)
    if ss58:
        return str(ss58)
    return str(getattr(wallet, "ss58_address", ""))


def sign_model(wallet: Any, model: Any) -> str:
    hotkey = getattr(wallet, "hotkey", None)
    signer = getattr(hotkey, "sign", None)
    if signer is None:
        return ""
    payload = canonical_model_payload(model)
    signed = signer(payload)
    if hasattr(signed, "hex"):
        return signed.hex()
    return str(signed)


def verify_model_signature(hotkey: str, signature: str, model: Any) -> bool:
    if not signature:
        return False
    import bittensor as bt

    message = canonical_model_payload(model)
    try:
        keypair = bt.Keypair(ss58_address=hotkey)
    except Exception:
        return False
    for candidate in signature_candidates(signature):
        try:
            if bool(keypair.verify(message, candidate)):
                return True
        except Exception:
            continue
    return False


def canonical_model_payload(model: Any) -> str:
    if hasattr(model, "model_dump_json"):
        value = json.loads(model.model_dump_json())
    else:
        value = model
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except TypeError:
        return str(model)


def signature_candidates(signature: str) -> list[str | bytes]:
    value = signature.strip()
    candidates: list[str | bytes] = [value]
    hex_value = value[2:] if value.startswith("0x") else value
    if not value.startswith("0x"):
        candidates.append(f"0x{value}")
    try:
        candidates.append(bytes.fromhex(hex_value))
    except ValueError:
        pass
    return candidates
