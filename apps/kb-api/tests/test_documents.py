"""Tests for /v1/kb/documents CRUD endpoints (TDD: red → green)."""

from __future__ import annotations

import httpx
import pytest

SAMPLE_MD = """\
# Test Policy

## Section 1

This is a test markdown document with enough content to produce at least one chunk.
It covers basic policy information for testing purposes.

## Section 2

More content here to ensure the chunker produces multiple segments when possible.
Policy terms and conditions apply. Please read carefully.
"""


# ---------------------------------------------------------------------------
# Happy-path CRUD
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_list_get_delete(api_client: httpx.AsyncClient):
    """Full lifecycle: upload → list → get → delete → 404 → idempotent delete."""
    # 1. Upload
    response = await api_client.post(
        "/v1/kb/documents",
        files={"file": ("policy.md", SAMPLE_MD.encode(), "text/plain")},
    )
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["title"] == "policy"
    assert data["source_name"] == "policy.md"
    assert data["chunk_count"] > 0
    doc_id = data["id"]
    chunk_count = data["chunk_count"]

    # 2. List
    response = await api_client.get("/v1/kb/documents")
    assert response.status_code == 200
    docs = response.json()
    assert len(docs) == 1
    assert docs[0]["id"] == doc_id
    assert docs[0]["chunk_count"] == chunk_count
    assert "created_at" in docs[0]

    # 3. Get
    response = await api_client.get(f"/v1/kb/documents/{doc_id}")
    assert response.status_code == 200
    detail = response.json()
    assert detail["id"] == doc_id
    assert detail["content"] == SAMPLE_MD
    assert "created_at" in detail

    # 4. Delete
    response = await api_client.delete(f"/v1/kb/documents/{doc_id}")
    assert response.status_code == 204

    # 5. Get → 404
    response = await api_client.get(f"/v1/kb/documents/{doc_id}")
    assert response.status_code == 404

    # 6. Idempotent delete
    response = await api_client.delete(f"/v1/kb/documents/{doc_id}")
    assert response.status_code == 204


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_wrong_extension_returns_400(api_client: httpx.AsyncClient):
    """Non-.md/.txt file should return 400."""
    response = await api_client.post(
        "/v1/kb/documents",
        files={"file": ("doc.pdf", b"%PDF content", "application/pdf")},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_upload_empty_content_returns_400(api_client: httpx.AsyncClient):
    """Empty .md file should return 400 with 'no chunks' in detail."""
    response = await api_client.post(
        "/v1/kb/documents",
        files={"file": ("empty.md", b"", "text/plain")},
    )
    assert response.status_code == 400
    assert "no chunks" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_whitespace_only_returns_400(api_client: httpx.AsyncClient):
    """Whitespace-only .md file should return 400."""
    response = await api_client.post(
        "/v1/kb/documents",
        files={"file": ("blank.md", b"   \n\n   ", "text/plain")},
    )
    assert response.status_code == 400
    assert "no chunks" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_non_utf8_returns_400(api_client: httpx.AsyncClient):
    """Non-UTF-8 bytes in a .md file should return 400 with 'not valid UTF-8' detail."""
    response = await api_client.post(
        "/v1/kb/documents",
        files={"file": ("bad_encoding.md", b"\xff\xfe\x00 not utf8", "text/plain")},
    )
    assert response.status_code == 400
    assert "not valid utf-8" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Bearer auth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auth_no_header_returns_401(auth_client_no_header: httpx.AsyncClient):
    """With ANVIL_KB_API_KEY set, no Authorization header → 401."""
    response = await auth_client_no_header.get("/v1/kb/documents")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_auth_wrong_key_returns_401(auth_client_no_header: httpx.AsyncClient):
    """With ANVIL_KB_API_KEY set, wrong Bearer key → 401."""
    response = await auth_client_no_header.get(
        "/v1/kb/documents", headers={"Authorization": "Bearer wrong-key"}
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_auth_correct_key_returns_200(auth_client_correct_key: httpx.AsyncClient):
    """With ANVIL_KB_API_KEY set, correct Bearer key → 200."""
    response = await auth_client_correct_key.get(
        "/v1/kb/documents", headers={"Authorization": "Bearer secret-key"}
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Idempotent re-upload (same source_name → replace)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reupload_same_filename_replaces(api_client: httpx.AsyncClient):
    """Uploading same filename twice → list still has 1 doc, id changes."""
    # First upload
    r1 = await api_client.post(
        "/v1/kb/documents",
        files={"file": ("policy.md", SAMPLE_MD.encode(), "text/plain")},
    )
    assert r1.status_code == 201
    id1 = r1.json()["id"]

    # Second upload with same filename
    updated_text = SAMPLE_MD + "\n## Extra\nExtra content.\n"
    r2 = await api_client.post(
        "/v1/kb/documents",
        files={"file": ("policy.md", updated_text.encode(), "text/plain")},
    )
    assert r2.status_code == 201
    id2 = r2.json()["id"]

    # id should differ (new doc created)
    assert id1 != id2

    # List should contain exactly 1 document
    r_list = await api_client.get("/v1/kb/documents")
    assert r_list.status_code == 200
    docs = r_list.json()
    assert len(docs) == 1
    assert docs[0]["id"] == id2
