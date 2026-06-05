import uuid

from anvil_kb.store.base import Chunk, Document


def _vec(i: int) -> list[float]:
    v = [0.0] * 512
    v[i] = 1.0
    return v


def _doc(name="d1"):
    return Document(id=uuid.uuid4(), title=name, source_name=name, content="全文")


def _chunk(doc, seq, vec):
    return Chunk(id=uuid.uuid4(), document_id=doc.id, seq=seq, content=f"c{seq}",
                 header_path="", start_offset=0, end_offset=2, embedding=vec)


async def test_search_orders_by_cosine_similarity(kb_store):
    doc = _doc()
    await kb_store.upsert_chunks(doc, [_chunk(doc, 0, _vec(0)), _chunk(doc, 1, _vec(1))])
    hits = await kb_store.search(_vec(0), k=2)
    assert [h.chunk.seq for h in hits] == [0, 1]
    assert hits[0].score > 0.99 and hits[1].score < 0.01  # 同向≈1,正交≈0


async def test_upsert_same_source_name_replaces(kb_store):
    d1 = _doc("same")
    await kb_store.upsert_chunks(d1, [_chunk(d1, 0, _vec(0))])
    d2 = _doc("same")
    await kb_store.upsert_chunks(d2, [_chunk(d2, 0, _vec(1)), _chunk(d2, 1, _vec(2))])
    hits = await kb_store.search(_vec(1), k=10)
    assert all(h.chunk.document_id == d2.id for h in hits) and len(hits) == 2


async def test_delete_document_removes_chunks(kb_store):
    doc = _doc()
    await kb_store.upsert_chunks(doc, [_chunk(doc, 0, _vec(0))])
    await kb_store.delete_document(doc.id)
    assert await kb_store.search(_vec(0), k=5) == []


async def test_delete_document_nonexistent_is_silent(kb_store):
    await kb_store.delete_document(uuid.uuid4())  # 不应抛出
