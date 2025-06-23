import asyncio
import os
import shutil
import tempfile
from pathlib import Path

from ncOS import (
    extract_and_validate_uploaded_archive,
    summarize_workspace_memory,
    intelligently_route_user_request_to_best_agent,
    automatically_optimize_memory_and_consolidate_session_data,
    detect_and_recover_from_system_errors_automatically,
)


async def _run(coro):
    return await coro


def test_extract_and_validate_uploaded_archive():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "payload"
        data_dir.mkdir()
        (data_dir / "config.yaml").write_text("name: test")
        (data_dir / "script.py").write_text("print('hi')")
        archive = shutil.make_archive(str(Path(tmpdir) / "bundle"), "zip", data_dir)

        result = asyncio.run(
            extract_and_validate_uploaded_archive(archive)
        )

        assert result["status"] == "success"
        assert result["file_count"] == 2
        assert result["archive"] == archive
        assert os.path.exists(result["extracted_to"])


def test_summarize_workspace_memory():
    class Snapshot:
        def __init__(self):
            self.memory_usage_mb = 42.5
            self.processed_files = ["a", "b"]
            self.active_agents = ["agent1"]
            self.trading_signals = ["sig1", "sig2", "sig3"]

    state = Snapshot()
    result = asyncio.run(summarize_workspace_memory(state))

    assert result == {
        "memory_usage_mb": 42.5,
        "processed_files": 2,
        "active_agents": 1,
        "trading_signals": 3,
    }


def test_intelligently_route_user_request_to_best_agent():
    registry = {"alpha": {"id": 1}, "default": {"id": 0}}
    request = {"type": "alpha"}

    result = asyncio.run(
        intelligently_route_user_request_to_best_agent(request, registry)
    )

    assert result["status"] == "routed"
    assert result["selected_agent"] == "alpha"
    assert result["reason"] == "matched_by_type"


def test_automatically_optimize_memory_and_consolidate_session_data():
    class Session:
        def __init__(self):
            self.processed_files = [1, 2, 3, 4, 5]
            self.memory_usage_mb = 200.0

    session = Session()
    result = asyncio.run(
        automatically_optimize_memory_and_consolidate_session_data(
            session, max_files=3, max_memory_mb=150.0
        )
    )

    assert result["status"] == "optimized"
    assert result["files_consolidated"] == 2
    assert result["memory_usage_mb"] == 150.0
    assert result["memory_freed_mb"] == 50.0
    assert len(session.processed_files) == 3
    assert session.memory_usage_mb == 150.0


def test_detect_and_recover_from_system_errors_automatically():
    err = asyncio.TimeoutError("timeout")
    context = {"type": "agent_timeout"}

    result = asyncio.run(
        detect_and_recover_from_system_errors_automatically(err, context)
    )

    assert result["status"] == "recovered"
    assert result["recovery_action"] == "retry"
    assert "timeout" in result["error"]
