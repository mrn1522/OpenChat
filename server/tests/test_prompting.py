from app.models import PersonaAssignment, SourceAgentSpec, SourceResult
from app.prompting import (
    build_debate_markdown,
    build_debate_prompt,
    build_persona_generation_prompt,
    build_source_system_prompt,
    build_synth_prompt,
    build_system_prompt_with_current_aest,
)


def _result(model: str, content: str, status: str = "ok", agent_id: str = "") -> SourceResult:
    return SourceResult(
        model=model, content=content, status=status, agent_id=agent_id
    )


class TestSystemPrompt:
    def test_includes_aest_datetime(self):
        prompt = build_system_prompt_with_current_aest()
        assert "Current datetime (AEST, Australia/Hobart):" in prompt

    def test_appends_datetime_to_base(self):
        prompt = build_system_prompt_with_current_aest("You are helpful.")
        assert prompt.startswith("You are helpful.")
        assert "Current datetime (AEST" in prompt

    def test_blank_base_treated_as_none(self):
        prompt = build_system_prompt_with_current_aest("   ")
        assert "You are helpful" not in prompt
        assert "Current datetime (AEST" in prompt


class TestPersonaGenerationPrompt:
    def test_includes_prompt_and_agent_ids(self):
        agents = [
            SourceAgentSpec(id="agent-1", model="openai/a"),
            SourceAgentSpec(id="agent-2", model="openai/b"),
        ]
        prompt = build_persona_generation_prompt("  My question  ", agents)
        assert "My question" in prompt
        assert "agent_id: agent-1 | model: openai/a" in prompt
        assert "agent_id: agent-2 | model: openai/b" in prompt


class TestSourceSystemPrompt:
    def test_no_persona(self):
        prompt = build_source_system_prompt(None)
        assert "Persona assignment" not in prompt
        assert "helpful assistant" in prompt

    def test_includes_persona_details(self):
        persona = PersonaAssignment(
            model="m", title="Skeptic", description="Question assumptions"
        )
        prompt = build_source_system_prompt(persona)
        assert "Persona: Skeptic" in prompt
        assert "Question assumptions" in prompt


class TestDebatePrompt:
    def test_wraps_target_and_peers_as_untrusted(self):
        prompt = build_debate_prompt(
            user_prompt="Q?",
            target_result=_result("openai/a", "answer a", agent_id="a1"),
            peer_results=[_result("openai/b", "answer b")],
        )
        assert "## Original Question" in prompt
        assert '<target_response model="openai/a (agent: a1)">' in prompt
        assert "answer a" in prompt
        assert '<peer_response model="openai/b (agent: openai/b)">' in prompt
        assert "untrusted DATA" in prompt

    def test_error_result_shows_error_body(self):
        prompt = build_debate_prompt(
            user_prompt="Q?",
            target_result=_result(
                "openai/a", "unused", status="error", agent_id="a1"
            ),
            peer_results=[],
        )
        prompt_target = prompt.split("<target_response", 1)[1]
        assert "ERROR:" in prompt_target
        assert "unused" not in prompt_target.split("</target_response>")[0]


class TestDebateMarkdown:
    def test_labels_and_fallback_critique(self):
        markdown = build_debate_markdown(
            [
                ("openai/a", "a1", "openai/b", "b1", "  looks wrong  "),
                ("openai/c", "", "openai/d", "", "   "),
            ]
        )
        assert "# Debate Report" in markdown
        assert "## Target: openai/a (agent: a1)" in markdown
        assert "Reviewer: openai/b (agent: b1)" in markdown
        assert "looks wrong" in markdown
        assert "## Target: openai/c" in markdown
        assert "(No critique returned)" in markdown


class TestSynthPrompt:
    def test_wraps_sources_and_analysis(self):
        prompt = build_synth_prompt(
            user_prompt="Q?",
            source_results=[_result("openai/a", "ans")],
            debate_markdown="critique here",
        )
        assert '<response model="openai/a (agent: openai/a)">' in prompt
        assert "ans" in prompt
        assert "<analysis>\ncritique here\n</analysis>" in prompt
        assert "untrusted DATA" in prompt
        assert "Return ONLY the final answer in Markdown." in prompt
