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


async def test_goal_row_and_job_fleet_columns(session_factory):
    import uuid

    from anvil_ai_employee.db import GoalRow, JobRow
    from sqlalchemy import select

    gid = uuid.uuid4()
    async with session_factory() as s:
        async with s.begin():
            s.add(GoalRow(id=gid, objective="调研并产周报", status="running"))
            s.add(
                JobRow(
                    skill="kb_digest",
                    payload={"task": "查 X"},
                    goal_id=gid,
                    employee="researcher",
                )
            )
    async with session_factory() as s:
        goal = (await s.execute(select(GoalRow).where(GoalRow.id == gid))).scalar_one()
        assert goal.objective == "调研并产周报"
        assert goal.status == "running"
        assert goal.result is None
        job = (
            await s.execute(select(JobRow).where(JobRow.goal_id == gid))
        ).scalar_one()
        assert job.employee == "researcher"
        assert job.goal_id == gid


async def test_job_columns_default_null(session_factory):
    """M1 jobs created without goal_id/employee must still work (nullable)."""
    import uuid

    from anvil_ai_employee.db import JobRow
    from sqlalchemy import select

    jid = uuid.uuid4()
    async with session_factory() as s:
        async with s.begin():
            s.add(JobRow(id=jid, skill="kb_digest", payload={}))
    async with session_factory() as s:
        job = (await s.execute(select(JobRow).where(JobRow.id == jid))).scalar_one()
        assert job.goal_id is None
        assert job.employee is None
