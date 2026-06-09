from anvil_ai_employee.db import JobRow, MemoryRow, ScheduleRow


def test_rows_have_expected_columns():
    # schedule
    cols = ScheduleRow.__table__.columns.keys()
    assert {
        "id", "name", "cron_expr", "skill", "payload", "next_run_at", "enabled", "created_at"
    } <= set(cols)
    # job
    jcols = JobRow.__table__.columns.keys()
    assert {"id", "schedule_id", "skill", "payload", "status", "result", "error", "locked_by",
            "created_at", "started_at", "finished_at"} <= set(jcols)
    # memory
    mcols = MemoryRow.__table__.columns.keys()
    assert {"id", "employee", "kind", "content", "created_at"} <= set(mcols)


def test_tablenames():
    assert ScheduleRow.__tablename__ == "ae_schedules"
    assert JobRow.__tablename__ == "ae_jobs"
    assert MemoryRow.__tablename__ == "ae_memories"


def test_memory_has_embedding_and_session_row():
    from anvil_ai_employee.db import MemoryRow, SessionRow
    assert "embedding" in MemoryRow.__table__.columns.keys()
    scols = SessionRow.__table__.columns.keys()
    assert {"id", "employee", "messages", "status", "created_at", "updated_at"} <= set(scols)
    assert SessionRow.__tablename__ == "ae_sessions"
