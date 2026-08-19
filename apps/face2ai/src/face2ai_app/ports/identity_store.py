from __future__ import annotations

from typing import Protocol

from face2ai_app.domain.models import IdentityRecord


class IdentityStore(Protocol):
    """Persistence for enrolled identities.

    **A closed error set**, enforced by ``apps/face2ai/tests/test_port_conformance.py``: every
    method raises only

    * ``IdentityStoreCorrupted`` — the stored data cannot be decoded or validated. The store must
      never overwrite data it could not read.
    * ``IdentityStoreUnavailable`` — the store could not be reached (an ``OSError``: a missing
      mount, a permission change, a full disk). The data may be perfectly fine.

    A raw ``OSError`` reaching the API layer becomes an HTTP 500 that blames the request for an
    infrastructure fault; both members above map to 503.

    Reads are consistent within one call: the service reads the store once per operation.
    """

    def list(self) -> list[IdentityRecord]: ...
    def add(self, identity: IdentityRecord) -> IdentityRecord: ...
    def delete(self, identity_id: str) -> bool: ...
    def clear(self) -> int: ...
