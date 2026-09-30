import pytest
from pydantic import ValidationError

from app.models import (
    MAX_IMAGE_ATTACHMENT_BYTES,
    MAX_IMAGE_BASE64_CHARS,
    MAX_PDF_ATTACHMENT_BYTES,
    MAX_PDF_BASE64_CHARS,
    MAX_TEXT_ATTACHMENT_BYTES,
    MAX_TEXT_ATTACHMENT_CHARS,
    AttachmentInput,
    DirectChatFileMeta,
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

    def test_rejects_image_attachments(self):
        with pytest.raises(ValidationError):
            RunRequest(
                **_run_payload(
                    attachments=[
                        {
                            "name": "shot.png",
                            "size": 10,
                            "content_type": "image/png",
                            "content": "QUJD",
                        }
                    ]
                )
            )

    def test_rejects_pdf_attachments(self):
        with pytest.raises(ValidationError):
            RunRequest(
                **_run_payload(
                    attachments=[
                        {
                            "name": "doc.pdf",
                            "size": 10,
                            "content_type": "application/pdf",
                            "content": "JVBERi0xLjc=",
                        }
                    ]
                )
            )

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


class TestAttachmentInput:
    def test_text_attachment_valid(self):
        attachment = AttachmentInput(
            name="notes.txt",
            size=100,
            content_type="text/plain",
            content="hello",
        )
        assert not attachment.is_image

    def test_image_attachment_valid(self):
        attachment = AttachmentInput(
            name="shot.png",
            size=10,
            content_type="image/png",
            content="QUJD",  # b"ABC"
        )
        assert attachment.is_image

    def test_image_attachment_rejects_unsupported_type(self):
        with pytest.raises(ValidationError):
            AttachmentInput(
                name="icon.svg",
                size=10,
                content_type="image/svg+xml",
                content="QUJD",
            )

    def test_image_attachment_rejects_invalid_base64(self):
        with pytest.raises(ValidationError):
            AttachmentInput(
                name="shot.png",
                size=10,
                content_type="image/png",
                content="not base64!!!",
            )

    def test_image_attachment_rejects_oversize_file(self):
        with pytest.raises(ValidationError):
            AttachmentInput(
                name="big.png",
                size=MAX_IMAGE_ATTACHMENT_BYTES + 1,
                content_type="image/png",
                content="QUJD",
            )

    def test_image_attachment_rejects_oversize_base64(self):
        with pytest.raises(ValidationError):
            AttachmentInput(
                name="big.png",
                size=MAX_IMAGE_ATTACHMENT_BYTES,
                content_type="image/png",
                content="a" * (MAX_IMAGE_BASE64_CHARS + 1),
            )

    def test_image_attachment_rejects_oversize_decoded_bytes(self):
        # At the encoded limit the payload still decodes to >5MiB.
        with pytest.raises(ValidationError):
            AttachmentInput(
                name="big.png",
                size=10,
                content_type="image/png",
                content="a" * MAX_IMAGE_BASE64_CHARS,
            )

    def test_pdf_attachment_valid(self):
        attachment = AttachmentInput(
            name="doc.pdf",
            size=10,
            content_type="application/pdf",
            content="JVBERi0xLjc=",  # b"%PDF-1.7"
        )
        assert attachment.is_pdf
        assert not attachment.is_text

    def test_pdf_attachment_rejects_invalid_base64(self):
        with pytest.raises(ValidationError):
            AttachmentInput(
                name="doc.pdf",
                size=10,
                content_type="application/pdf",
                content="not base64!!!",
            )

    def test_pdf_attachment_rejects_oversize_file(self):
        with pytest.raises(ValidationError):
            AttachmentInput(
                name="big.pdf",
                size=MAX_PDF_ATTACHMENT_BYTES + 1,
                content_type="application/pdf",
                content="JVBERi0xLjc=",
            )

    def test_pdf_attachment_rejects_non_pdf_bytes(self):
        with pytest.raises(ValidationError):
            AttachmentInput(
                name="doc.pdf",
                size=10,
                content_type="application/pdf",
                content="QUJD",  # b"ABC" — valid base64, wrong magic
            )

    def test_text_attachment_rejects_oversize_file(self):
        with pytest.raises(ValidationError):
            AttachmentInput(
                name="big.txt",
                size=MAX_TEXT_ATTACHMENT_BYTES + 1,
                content_type="text/plain",
                content="a",
            )

    def test_text_attachment_rejects_oversize_content(self):
        with pytest.raises(ValidationError):
            AttachmentInput(
                name="big.txt",
                size=100,
                content_type="text/plain",
                content="a" * (MAX_TEXT_ATTACHMENT_CHARS + 1),
            )


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

    def test_total_binary_cap_boundary_counts_decoded_bytes(self):
        # 4 x 5 MiB images decode to exactly MAX_TOTAL_BINARY_BYTES: the cap
        # must account for base64 padding per string, not len() * 3 // 4.
        import base64

        from app.models import MAX_TOTAL_BINARY_BYTES, base64_decoded_len

        each = MAX_TOTAL_BINARY_BYTES // 4
        chunk = base64.b64encode(b"\0" * each).decode()
        assert base64_decoded_len(chunk) == each

        def image() -> AttachmentInput:
            return AttachmentInput(
                name="i.png",
                size=each,
                content_type="image/png",
                content=chunk,
            )

        ok = DirectChatRequest(
            model="m",
            messages=[{"role": "user", "content": "look"}],
            attachments=[image(), image(), image(), image()],
        )
        assert len(ok.attachments) == 4
        with pytest.raises(ValidationError):
            DirectChatRequest(
                model="m",
                messages=[{"role": "user", "content": "look"}],
                attachments=[image(), image(), image(), image(), image()],
            )

    def test_total_binary_cap_counts_pdf_bytes(self):
        import base64

        chunk = base64.b64encode(
            b"%PDF-1.7" + b"\0" * (MAX_PDF_ATTACHMENT_BYTES - 8)
        ).decode()

        def pdf() -> AttachmentInput:
            return AttachmentInput(
                name="d.pdf",
                size=MAX_PDF_ATTACHMENT_BYTES,
                content_type="application/pdf",
                content=chunk,
            )

        ok = DirectChatRequest(
            model="m",
            messages=[{"role": "user", "content": "read"}],
            attachments=[pdf(), pdf()],
        )
        assert len(ok.attachments) == 2
        with pytest.raises(ValidationError):
            DirectChatRequest(
                model="m",
                messages=[{"role": "user", "content": "read"}],
                attachments=[pdf(), pdf(), pdf()],
            )

    def test_raw_binary_bound_rejects_before_nested_decode(self):
        # Encoded input that can never satisfy the decoded cap is rejected
        # up front instead of after base64-decoding every payload.
        oversized = "a" * MAX_PDF_BASE64_CHARS
        with pytest.raises(ValidationError, match="binary payload"):
            DirectChatRequest(
                model="m",
                messages=[{"role": "user", "content": "x"}],
                attachments=[
                    {
                        "name": f"d{i}.pdf",
                        "size": 1,
                        "content_type": "application/pdf",
                        "content": oversized,
                    }
                    for i in range(3)
                ],
            )

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


class TestDirectChatFileMeta:
    def test_metadata_only_valid(self):
        meta = DirectChatFileMeta(name="doc.pdf", content_type="application/pdf")
        assert meta.content is None

    def test_rejects_unsupported_type(self):
        with pytest.raises(ValidationError):
            DirectChatFileMeta(name="doc.docx", content_type="application/msword")

    def test_rejects_invalid_base64(self):
        with pytest.raises(ValidationError):
            DirectChatFileMeta(
                name="doc.pdf",
                content_type="application/pdf",
                content="not base64!!!",
            )

    def test_rejects_non_pdf_bytes(self):
        with pytest.raises(ValidationError):
            DirectChatFileMeta(
                name="doc.pdf",
                content_type="application/pdf",
                content="QUJD",
            )

    def test_message_accepts_files(self):
        message = DirectChatMessage.model_validate(
            {
                "role": "user",
                "content": "hi",
                "files": [{"name": "doc.pdf", "content_type": "application/pdf"}],
            }
        )
        assert message.files[0].name == "doc.pdf"


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
