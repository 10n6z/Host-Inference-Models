from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

from fastapi import UploadFile

from tenant_store import TenantAsset, record_asset

# 50 MB matches the plan's bounded-document upload ceiling (one PDF/TIFF).
MAX_ASSET_BYTES = int(os.getenv("GATEWAY_MAX_ASSET_BYTES", str(50 * 1024 * 1024)))
ASSETS_ROOT = Path(
    os.getenv("GATEWAY_ASSETS_ROOT", "/gpt-lab/long/computer-vision/assets")
)
CHUNK_BYTES = 1024 * 1024


class AssetTooLargeError(Exception):
    pass


async def save_tenant_asset(tenant_id: str, upload: UploadFile) -> TenantAsset:
    asset_id = uuid.uuid4().hex
    destination_dir = ASSETS_ROOT / tenant_id
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / asset_id
    digest = hashlib.sha256()
    size = 0
    try:
        with destination.open("wb") as handle:
            while True:
                chunk = await upload.read(CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_ASSET_BYTES:
                    raise AssetTooLargeError()
                digest.update(chunk)
                handle.write(chunk)
    except AssetTooLargeError:
        destination.unlink(missing_ok=True)
        raise

    asset = TenantAsset(
        id=asset_id,
        tenant_id=tenant_id,
        path=destination,
        mime_type=upload.content_type or "application/octet-stream",
        byte_length=size,
        sha256=digest.hexdigest(),
    )
    record_asset(asset)
    return asset
