from app.main import build_debate_jobs, build_debate_pairs, resolve_agents
from app.models import RunRequest, SourceAgentSpec, SourceResult


class TestBuildDebatePairs:
    def test_single_agent_pairs_with_fusion_model(self):
        pairs = build_debate_pairs(["a1"], "full", "fusion-model")
        assert pairs == [("fusion-model", "a1")]

    def test_partial_rotates_reviewers(self):
        pairs = build_debate_pairs(["a", "b", "c"], "partial", "fusion")
        assert pairs == [("b", "a"), ("c", "b"), ("a", "c")]

    def test_full_reviews_every_peer(self):
        pairs = build_debate_pairs(["a", "b", "c"], "full", "fusion")
        assert len(pairs) == 6
        assert all(reviewer != target for reviewer, target in pairs)

    def test_off_mode_uses_rotation(self):
        pairs = build_debate_pairs(["a", "b"], "off", "fusion")
        assert pairs == [("b", "a"), ("a", "b")]

    def test_empty_agents(self):
        assert build_debate_pairs([], "full", "fusion") == []


class TestResolveAgents:
    def _request(self, **overrides) -> RunRequest:
        payload = {
            "prompt": "q",
            "source_models": ["m1", "m2"],
            "fusion_model": "f",
        }
        payload.update(overrides)
        return RunRequest(**payload)

    def test_explicit_agents_win(self):
        agents = [
            SourceAgentSpec(id="x", model="m1"),
            SourceAgentSpec(id="y", model="m1"),
        ]
        assert resolve_agents(self._request(source_agents=agents)) == agents

    def test_falls_back_to_source_models(self):
        agents = resolve_agents(self._request())
        assert [a.id for a in agents] == ["m1", "m2"]
        assert all(a.id == a.model for a in agents)


class TestBuildDebateJobs:
    def _source(self, agent_id: str, model: str) -> SourceResult:
        return SourceResult(
            model=model, agent_id=agent_id, content="ok", status="ok"
        )

    def test_jobs_materialize_in_pair_order(self):
        jobs = build_debate_jobs(
            debate_pairs=[("b", "a"), ("a", "b")],
            source_by_agent={"a": self._source("a", "m1"), "b": self._source("b", "m2")},
            agent_by_id={
                "a": SourceAgentSpec(id="a", model="m1"),
                "b": SourceAgentSpec(id="b", model="m2"),
            },
            user_prompt="q",
        )
        assert [job.pair_index for job in jobs] == [0, 1]
        assert jobs[0].reviewer_model == "m2"
        assert jobs[0].target_model == "m1"
        assert "## Original Question" in jobs[0].prompt

    def test_skips_missing_target_result(self):
        jobs = build_debate_jobs(
            debate_pairs=[("b", "gone")],
            source_by_agent={"a": self._source("a", "m1")},
            agent_by_id={},
            user_prompt="q",
        )
        assert jobs == []

    def test_reviewer_model_falls_back_to_id(self):
        jobs = build_debate_jobs(
            debate_pairs=[("unknown-reviewer", "a")],
            source_by_agent={"a": self._source("a", "m1")},
            agent_by_id={},
            user_prompt="q",
        )
        assert jobs[0].reviewer_model == "unknown-reviewer"
