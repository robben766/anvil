import uuid

from sqlalchemy import select

from anvil_kb.db import ChunkRow, DocumentRow


async def test_roundtrip_document_and_chunk(kb_session):
    doc = DocumentRow(id=uuid.uuid4(), title="t", source_name="s", content="全文")
    kb_session.add(doc)
    await kb_session.flush()
    chunk = ChunkRow(
        id=uuid.uuid4(),
        document_id=doc.id,
        seq=0,
        content="片段",
        header_path="t",
        start_offset=0,
        end_offset=2,
        embedding=[0.1] * 512,
    )
    kb_session.add(chunk)
    await kb_session.commit()
    row = (await kb_session.execute(select(ChunkRow))).scalar_one()
    assert row.document_id == doc.id
    assert len(list(row.embedding)) == 512


async def test_delete_document_cascades(kb_session):
    doc = DocumentRow(id=uuid.uuid4(), title="t", source_name="s", content="x")
    kb_session.add(doc)
    await kb_session.flush()
    kb_session.add(
        ChunkRow(
            id=uuid.uuid4(),
            document_id=doc.id,
            seq=0,
            content="c",
            header_path="",
            start_offset=0,
            end_offset=1,
            embedding=[0.0] * 512,
        )
    )
    await kb_session.commit()
    await kb_session.delete(doc)
    await kb_session.commit()
    assert (await kb_session.execute(select(ChunkRow))).first() is None
