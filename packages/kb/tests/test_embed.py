import math

from anvil_kb.embed import FastEmbedEmbedder


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


def test_dim_and_shapes():
    e = FastEmbedEmbedder()
    vecs = e.embed_texts(["等待期为90天", "保险金额"])
    assert len(vecs) == 2 and all(len(v) == e.dim == 512 for v in vecs)


def test_semantic_sanity():
    e = FastEmbedEmbedder()
    q = e.embed_query("等待期是多少天?")
    related = e.embed_texts(["本合同等待期为90天。"])[0]
    unrelated = e.embed_texts(["今天天气晴朗适合郊游。"])[0]
    assert _cos(q, related) > _cos(q, unrelated)
