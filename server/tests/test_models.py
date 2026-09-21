import pytest
from pydantic import ValidationError

from app.models import (
    DirectChatMessage,
    DirectChatRequest,
    PersonaAssignment,
    ReasoningConfig,
    RunRequest,
    WorkflowConfig,
    WorkflowCreateRequest,
)


def _run_payload(**overrides) -> dict:
    payload = {
        "prompt": "What is 2+2?",
        "source_models": ["openai/model-a", "openai/model-b"],
        "fusion_model": "openai/fusion",
    }
    payload.update(overrides)
    return payload


class TestRunRequest:
    def test_minimal_valid_defaults(self):
        request = RunRequest(**_run_payload())
        assert request.debate_mode == "partial"
        assert request.temperature == 0.2
        assert request.max_output_tokens == 1000
        assert request.reasoning.effort == "medium"

    def test_rejects_empty_prompt(self):
        with pytest.raises(ValidationError):
            RunRequest(**_run_payload(prompt=""))

    def test_rejects_empty_source_models(self):
        with pytest.raises(ValidationError):
            RunRequest(**_run_payload(source_models=[]))

    @pytest.mark.parametrize("temperature", [0, 0.7, 2])
    def test_temperature_bounds(self, temperature):
        assert RunRequest(**_run_payload(temperature=temperature)).temperature == temperature

    @pytest.mark.parametrize("temperature", [-0.01, 2.01])
    def test_temperature_out_of_bounds(self, temperature):
        with pytest.raises(ValidationError):
            RunRequest(**_run_payload(temperature=temperature))

    def test_max_output_tokens_bounds(self):
        with pytest.raises(ValidationError):
            RunRequest(**_run_payload(max_output_tokens=127))
        with pytest.raises(ValidationError):
            RunRequest(**_run_payload(max_output_tokens=10001))
        assert RunRequest(**_run_payload(max_output_tokens=128)).max_output_tokens == 128

    def test_rejects_unknown_debate_mode(self):
        with pytest.raises(ValidationError):
            RunRequest(**_run_payload(debate_mode="everything"))

    def test_rejects_over_five_attachments(self):
        attachments = [
            {"name": f"file-{i}", "size": 1, "content": "x"} for i in range(6)
        ]
        with pytest.raises(ValidationError):
            RunRequest(**_run_payload(attachments=attachments))

    def test_rejects_over_48_persona_overrides(self):
        overrides = [
            {"model": "m", "title": "t", "description": "d"} for _ in range(49)
        ]
        with pytest.raises(ValidationError):
            RunRequest(**_run_payload(persona_assignments_override=overrides))


class TestReasoningConfig:
    @pytest.mark.parametrize(
        "effort", ["max", "xhigh", "high", "medium", "low", "minimal", "none"]
    )
    def test_valid_efforts(self, effort):
        assert ReasoningConfig(effort=effort).effort == effort

    def test_rejects_unknown_effort(self):
        with pytest.raises(ValidationError):
            ReasoningConfig(effort="ultra")


class TestDirectChatRequest:
    def test_valid(self):
        request = DirectChatRequest(
            model="openai/model",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert request.messages[0].role == "user"

    def test_rejects_system_role(self):
        with pytest.raises(ValidationError):
            DirectChatMessage(role="system", content="behave")

    def test_rejects_empty_messages(self):
        with pytest.raises(ValidationError):
            DirectChatRequest(model="m", messages=[])

    def test_rejects_over_200_messages(self):
        messages = [{"role": "user", "content": "x"}] * 201
        with pytest.raises(ValidationError):
            DirectChatRequest(model="m", messages=messages)


class TestPersonaAssignment:
    @pytest.mark.parametrize("temperature", [0.5, 0.7, 1.2])
    def test_temperature_bounds(self, temperature):
        persona = PersonaAssignment(
            model="m", title="t", description="d", temperature=temperature
        )
        assert persona.temperature == temperature

    @pytest.mark.parametrize("temperature", [0.49, 1.21])
    def test_temperature_out_of_bounds(self, temperature):
        with pytest.raises(ValidationError):
            PersonaAssignment(
                model="m", title="t", description="d", temperature=temperature
            )


class TestWorkflowModels:
    def test_valid_workflow(self):
        request = WorkflowCreateRequest(
            name="my flow",
            config={
                "source_models": ["a", "b"],
                "fusion_model": "f",
            },
        )
        assert request.config.debate_mode == "partial"

    def test_rejects_missing_fusion_model(self):
        with pytest.raises(ValidationError):
            WorkflowConfig(source_models=["a"])

    def test_rejects_name_over_96_chars(self):
        with pytest.raises(ValidationError):
            WorkflowCreateRequest(
                name="x" * 97,
                config={"source_models": ["a"], "fusion_model": "f"},
            )
