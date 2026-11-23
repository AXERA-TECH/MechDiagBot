import asyncio
from dataclasses import replace

from cat_assistant.adapters.knowledge import InMemoryKnowledge, demo_documents
from cat_assistant.adapters.telemetry import _FLEET_FAULT_POOL, demo_snapshots


def test_demo_knowledge_covers_every_simulated_fault_for_its_machine():
    # The fleet simulator can surface any code from a machine's pool. Every one
    # must have a matching manual, otherwise a demo diagnosis says "手册未覆盖"
    # for a code the console itself is showing.
    index = InMemoryKnowledge(demo_documents())
    fleet = demo_snapshots()
    for machine_id, pool in _FLEET_FAULT_POOL.items():
        base = fleet[machine_id]
        for code in pool:
            machine = replace(base, fault_codes=(code,))
            hits = asyncio.run(
                index.search("当前异常故障码处理建议", machine=machine, limit=1)
            )
            hit = f"{hits[0].content} {hits[0].source}"
            assert code in hit, f"{base.model} missing manual for {code}: {hit}"
            assert base.model in hit, f"wrong-model manual for {code}: {hit}"


def test_query_tokens_affect_demo_ranking():
    index = InMemoryKnowledge(demo_documents())
    machine = demo_snapshots()["cat-306-demo"]
    hits = asyncio.run(index.search("E456 冷却液温度", machine=machine, limit=1))
    assert "E456" in hits[0].content


def test_search_scopes_out_other_device_manuals():
    index = InMemoryKnowledge(demo_documents())
    machine = demo_snapshots()["cat-950gc-demo"]  # Cat 950 GC, faults E45/F210
    hits = asyncio.run(index.search("故障码处理建议", machine=machine, limit=5))
    sources = [hit.source for hit in hits]
    # A Cat 950 GC turn must not cite another device's manual (e.g. Cat 306 CR) …
    assert all("306" not in source for source in sources), sources
    # … while this model's manuals and model-agnostic safety docs stay eligible.
    assert any("Cat 950 GC" in source for source in sources), sources
    assert any("Safety" in source for source in sources), sources
