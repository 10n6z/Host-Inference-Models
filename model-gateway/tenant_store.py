from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class TenantAsset:
    id: str
    tenant_id: str
    path: Path
    mime_type: str
    byte_length: int
    sha256: str


@dataclass
class TenantGatewayJob:
    id: str
    tenant_id: str
    model: str
    status: str
    model_job_id: Optional[str]
    response: dict[str, Any]
    updated_at: str


# In-memory, keyed by (tenant_id, id): matches the process-local `gateway_jobs`
# dict already used for job bookkeeping. Durable storage is out of scope until
# the Durable Multi-Host Production Platform milestone.
_assets: dict[tuple[str, str], TenantAsset] = {}
_jobs: dict[tuple[str, str], TenantGatewayJob] = {}


def record_asset(asset: TenantAsset) -> None:
    _assets[(asset.tenant_id, asset.id)] = asset


def get_asset(tenant_id: str, asset_id: str) -> Optional[TenantAsset]:
    return _assets.get((tenant_id, asset_id))


def record_job(job: TenantGatewayJob) -> None:
    _jobs[(job.tenant_id, job.id)] = job


def get_job(tenant_id: str, job_id: str) -> Optional[TenantGatewayJob]:
    return _jobs.get((tenant_id, job_id))
