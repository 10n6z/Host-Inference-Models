#!/usr/bin/env python3
"""Mint a short-lived model-gateway JWT.

Usage:
  MODEL_GATEWAY_JWT_SECRET=... ./mint_gateway_token.py <tasks> [models] [ttl_seconds]

  tasks  -- comma-separated, e.g. text-to-speech,automatic-speech-recognition
  models -- comma-separated model ids; required for POST /generate, may be
            empty for GET /models (discovery is scoped by task only).
"""
import base64
import hashlib
import hmac
import json
import os
import sys
import time


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def main() -> int:
    secret = os.environ.get("MODEL_GATEWAY_JWT_SECRET", "").strip()
    if not secret:
        print("MODEL_GATEWAY_JWT_SECRET is not set", file=sys.stderr)
        return 1
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 1

    tasks = [t for t in sys.argv[1].split(",") if t]
    models = [m for m in (sys.argv[2] if len(sys.argv) > 2 else "").split(",") if m]
    ttl = int(sys.argv[3]) if len(sys.argv) > 3 else 900

    header = b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = b64url(
        json.dumps(
            {
                "iss": "sw4e-control-plane",
                "aud": "model-gateway",
                "userId": os.environ.get("USER", "cli"),
                "tenantId": os.environ.get("GATEWAY_TENANT_ID", "cli"),
                "permittedTasks": tasks,
                "permittedModelIds": models,
                "exp": int(time.time()) + ttl,
            }
        ).encode()
    )
    signature = b64url(
        hmac.new(secret.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
    )
    print(f"{header}.{payload}.{signature}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
