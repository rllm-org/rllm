"""Tests for rllm session/harness metadata extraction."""

from rllm_model_gateway.metadata import (
    HEADER_HARNESS,
    HEADER_PARENT_SPAN_ID,
    HEADER_PROJECT,
    HEADER_RUN_ID,
    HEADER_SESSION_ID,
    HEADER_STEP_ID,
    extract_metadata,
    headers_from_scope,
)


class TestExtractMetadataFromHeaders:
    def test_all_headers_present(self):
        headers = {
            HEADER_SESSION_ID: "sess-1",
            HEADER_RUN_ID: "run-7",
            HEADER_HARNESS: "claude-code",
            HEADER_STEP_ID: "3",
            HEADER_PARENT_SPAN_ID: "span-parent",
            HEADER_PROJECT: "demo",
        }
        md = extract_metadata(headers=headers)
        assert md.session_id == "sess-1"
        assert md.run_id == "run-7"
        assert md.harness == "claude-code"
        assert md.step_id == 3
        assert md.parent_span_id == "span-parent"
        assert md.project == "demo"
        assert md.experiment is None

    def test_empty_header_value_treated_as_missing(self):
        md = extract_metadata(headers={HEADER_SESSION_ID: "  "})
        assert md.session_id is None

    def test_invalid_step_id_falls_back_to_none(self):
        md = extract_metadata(headers={HEADER_STEP_ID: "not-a-number"})
        assert md.step_id is None


class TestExtractMetadataBodyFallback:
    def test_metadata_rllm_nested(self):
        body = {"metadata": {"rllm": {"session_id": "sess-2", "harness": "react"}}}
        md = extract_metadata(body=body)
        assert md.session_id == "sess-2"
        assert md.harness == "react"

    def test_top_level_rllm_key(self):
        body = {"rllm": {"session_id": "sess-3", "run_id": "run-1", "step_id": 5}}
        md = extract_metadata(body=body)
        assert md.session_id == "sess-3"
        assert md.run_id == "run-1"
        assert md.step_id == 5

    def test_body_ignored_when_not_dict(self):
        md = extract_metadata(body={"metadata": "not-a-dict"})
        assert md.session_id is None


class TestExtractMetadataPathFallback:
    def test_session_path_legacy(self):
        md = extract_metadata(path="/sessions/legacy-sid/v1/chat/completions")
        assert md.session_id == "legacy-sid"

    def test_session_path_short(self):
        md = extract_metadata(path="/sessions/foo")
        assert md.session_id == "foo"

    def test_no_session_in_path(self):
        md = extract_metadata(path="/v1/chat/completions")
        assert md.session_id is None


class TestExtractMetadataPrecedence:
    def test_header_beats_body(self):
        headers = {HEADER_SESSION_ID: "from-header"}
        body = {"metadata": {"rllm": {"session_id": "from-body"}}}
        md = extract_metadata(headers=headers, body=body)
        assert md.session_id == "from-header"

    def test_header_beats_path(self):
        headers = {HEADER_SESSION_ID: "from-header"}
        md = extract_metadata(headers=headers, path="/sessions/from-path/v1/foo")
        assert md.session_id == "from-header"

    def test_body_beats_path(self):
        body = {"rllm": {"session_id": "from-body"}}
        md = extract_metadata(body=body, path="/sessions/from-path/v1/foo")
        assert md.session_id == "from-body"

    def test_field_level_merge(self):
        # Header has session_id; body has run_id. Both contribute.
        headers = {HEADER_SESSION_ID: "h-sess"}
        body = {"rllm": {"run_id": "b-run"}}
        md = extract_metadata(headers=headers, body=body)
        assert md.session_id == "h-sess"
        assert md.run_id == "b-run"


class TestHeadersFromScope:
    def test_lowercases_and_decodes(self):
        scope = {"headers": [(b"X-RLLM-Session-Id", b"abc"), (b"Content-Type", b"application/json")]}
        out = headers_from_scope(scope)
        assert out["x-rllm-session-id"] == "abc"
        assert out["content-type"] == "application/json"

    def test_empty(self):
        assert headers_from_scope({}) == {}
        assert headers_from_scope({"headers": []}) == {}
