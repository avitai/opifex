"""Cross-process persistence tests for the Neural Functional Registry.

These tests exercise the real async-SQLAlchemy persistence backend wired
into :class:`~opifex.platform.registry.core.RegistryService` (feature F23).
They register, retrieve, search and delete functionals against a temporary
on-disk SQLite database and verify that a *fresh* ``RegistryService``
instance pointed at the same database file (simulating a separate process)
observes the persisted state.
"""

import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from opifex.platform.registry.core import RegistryService
from opifex.platform.registry.exceptions import (
    AccessDenied,
    FunctionalNotFound,
    VersionNotFound,
)


AUTHOR = "author-001"
OTHER = "intruder-002"


def _metadata(name: str = "Persisted Optimizer", *, is_public: bool = True) -> dict:
    """Build a valid metadata payload for tests."""
    return {
        "name": name,
        "type": "l2o",
        "description": "Persisted across processes",
        "tags": ["Optimization", "Persistence"],
        "domain": "fluid_dynamics",
        "is_public": is_public,
        "performance": {"accuracy": 0.95},
    }


def _functional(seed: float = 1.0) -> dict:
    """Build a serialized-functional payload for tests."""
    return {"module_type": "PersistOpt", "weights": [seed, seed + 1.0]}


@pytest_asyncio.fixture
async def db_path() -> AsyncIterator[Path]:
    """Temporary directory that holds both the SQLite DB and artifact files."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def _service(root: Path) -> RegistryService:
    """Build a service whose DB + storage both live under ``root``."""
    return RegistryService(
        storage_path=str(root / "storage"),
        database_url=f"sqlite+aiosqlite:///{root / 'registry.db'}",
        max_file_size=1024 * 1024,
    )


@pytest.mark.asyncio
async def test_register_then_retrieve_round_trip(db_path: Path) -> None:
    """A registered functional is retrievable with all fields intact."""
    service = _service(db_path)
    try:
        functional_id = await service.register_functional(
            functional=_functional(), metadata=_metadata(), user_id=AUTHOR, version_tag="v1.0.0"
        )
        retrieved = await service.retrieve_functional(functional_id)
    finally:
        await service.close()

    assert retrieved["id"] == functional_id
    assert retrieved["name"] == "Persisted Optimizer"
    assert retrieved["version"] == "v1.0.0"
    assert retrieved["author_id"] == AUTHOR
    assert retrieved["tags"] == ["optimization", "persistence"]
    assert retrieved["data"] == _functional()
    assert retrieved["metadata"]["performance"] == {"accuracy": 0.95}


@pytest.mark.asyncio
async def test_persistence_survives_fresh_instance(db_path: Path) -> None:
    """A new RegistryService on the same DB file retrieves prior writes.

    This is the cross-process guarantee: the first instance is closed
    entirely before a second instance reopens the same database file.
    """
    writer = _service(db_path)
    try:
        functional_id = await writer.register_functional(
            functional=_functional(2.0), metadata=_metadata("Survivor"), user_id=AUTHOR
        )
    finally:
        await writer.close()

    reader = _service(db_path)
    try:
        retrieved = await reader.retrieve_functional(functional_id)
    finally:
        await reader.close()

    assert retrieved["id"] == functional_id
    assert retrieved["name"] == "Survivor"
    assert retrieved["data"] == _functional(2.0)


@pytest.mark.asyncio
async def test_version_tag_resolution(db_path: Path) -> None:
    """Retrieval resolves the latest version when no tag is given.

    Each ``register_functional`` mints a distinct functional id with a single
    initial version flagged ``is_latest``. Retrieval with ``version_tag=None``
    must return that latest version; an explicit tag returns that exact one;
    an unknown tag raises :class:`VersionNotFound`.
    """
    service = _service(db_path)
    try:
        functional_id = await service.register_functional(
            functional=_functional(1.0), metadata=_metadata(), user_id=AUTHOR, version_tag="v1.0.0"
        )
        latest = await service.retrieve_functional(functional_id)
        explicit = await service.retrieve_functional(functional_id, version_tag="v1.0.0")
        with pytest.raises(VersionNotFound):
            await service.retrieve_functional(functional_id, version_tag="v9.9.9")
    finally:
        await service.close()

    assert latest["version"] == "v1.0.0"
    assert explicit["version"] == "v1.0.0"


@pytest.mark.asyncio
async def test_search_by_type_and_tags(db_path: Path) -> None:
    """Search filters by functional type, author and tags."""
    service = _service(db_path)
    try:
        await service.register_functional(
            functional=_functional(), metadata=_metadata("Alpha"), user_id=AUTHOR
        )
        pinn_meta = _metadata("Beta")
        pinn_meta["type"] = "pinn"
        pinn_meta["tags"] = ["physics"]
        await service.register_functional(
            functional=_functional(), metadata=pinn_meta, user_id=OTHER
        )

        all_results = await service.search_functionals()
        l2o_results = await service.search_functionals(functional_type="l2o")
        author_results = await service.search_functionals(author_id=AUTHOR)
        tag_results = await service.search_functionals(tags=["physics"])
        name_results = await service.search_functionals(query="Alph")
    finally:
        await service.close()

    assert len(all_results) == 2
    assert {r["name"] for r in l2o_results} == {"Alpha"}
    assert {r["name"] for r in author_results} == {"Alpha"}
    assert {r["name"] for r in tag_results} == {"Beta"}
    assert {r["name"] for r in name_results} == {"Alpha"}


@pytest.mark.asyncio
async def test_access_control_private_functional(db_path: Path) -> None:
    """A private functional is denied to non-owners and granted to the owner."""
    service = _service(db_path)
    try:
        functional_id = await service.register_functional(
            functional=_functional(),
            metadata=_metadata("Secret", is_public=False),
            user_id=AUTHOR,
        )

        owner_view = await service.retrieve_functional(functional_id, user_id=AUTHOR)
        with pytest.raises(AccessDenied):
            await service.retrieve_functional(functional_id, user_id=OTHER)
    finally:
        await service.close()

    assert owner_view["name"] == "Secret"


@pytest.mark.asyncio
async def test_delete_complete_removes_everything(db_path: Path) -> None:
    """Deleting without a version tag removes the functional entirely."""
    service = _service(db_path)
    try:
        functional_id = await service.register_functional(
            functional=_functional(), metadata=_metadata(), user_id=AUTHOR, version_tag="v1.0.0"
        )
        deleted = await service.delete_functional(functional_id, user_id=AUTHOR)
        with pytest.raises(FunctionalNotFound):
            await service.retrieve_functional(functional_id)
    finally:
        await service.close()

    assert deleted is True


@pytest.mark.asyncio
async def test_delete_specific_version(db_path: Path) -> None:
    """Deleting a specific version removes only that version's record/file."""
    service = _service(db_path)
    try:
        functional_id = await service.register_functional(
            functional=_functional(), metadata=_metadata(), user_id=AUTHOR, version_tag="v1.0.0"
        )
        await service.delete_functional(functional_id, user_id=AUTHOR, version_tag="v1.0.0")

        # The functional record still exists, but the version is gone.
        with pytest.raises(VersionNotFound):
            await service.retrieve_functional(functional_id, version_tag="v1.0.0")
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_delete_requires_ownership(db_path: Path) -> None:
    """Only the owning author may delete a functional."""
    service = _service(db_path)
    try:
        functional_id = await service.register_functional(
            functional=_functional(), metadata=_metadata(), user_id=AUTHOR
        )
        with pytest.raises(AccessDenied):
            await service.delete_functional(functional_id, user_id=OTHER)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_retrieve_unknown_functional_raises(db_path: Path) -> None:
    """Retrieving an unknown id raises FunctionalNotFound."""
    import uuid

    service = _service(db_path)
    try:
        with pytest.raises(FunctionalNotFound):
            await service.retrieve_functional(str(uuid.uuid4()))
    finally:
        await service.close()
