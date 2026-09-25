import base64
import json
import uuid
from pathlib import Path
from typing import Final

import anthropic
import pytest
from integration._support.client import Gateway, Scenario
from integration._support.process import owned_proxy_process
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

MODEL_ID: Final = "us.anthropic.claude-sonnet-5"
TOKEN: Final = "synthetic-bedrock-bearer"
INVOKE: Final = f"/model/{MODEL_ID}/invoke"
INVOKE_STREAM: Final = f"/model/{MODEL_ID}/invoke-with-response-stream"
BODY: Final = TypeAdapter(dict[str, JsonValue])
MESSAGES: Final = TypeAdapter(list[dict[str, JsonValue]])
TOOL_ADDITION: Final = {"type": "tool_addition", "tool_reference": {"type": "tool_reference", "tool_name": "Read"}}
TOOL_USE: Final = {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "/tmp/a.txt"}}
TOOL_RESULT: Final = {"type": "tool_result", "tool_use_id": "toolu_1", "content": "hello world"}
READING: Final = {"type": "text", "text": "Reading it now."}
REJECTED: Final = "Bedrock Invoke rejects the extension: "


def _output_config_turns(tag: str) -> list[dict[str, JsonValue]]:
    return [
        {"role": "user", "content": f"read the file /tmp/a.txt {tag}"},
        {"role": "assistant", "output_config": {"effort": "high"}, "content": [READING, TOOL_USE]},
        {"role": "user", "content": [TOOL_RESULT]},
    ]


def _tool_addition_turns(tag: str) -> list[dict[str, JsonValue]]:
    return [
        {"role": "user", "content": f"read /tmp/a.txt {tag}"},
        {"role": "assistant", "content": [TOOL_ADDITION, {"type": "text", "text": "ok"}]},
        {"role": "user", "content": "continue"},
    ]


def _without_extensions(messages: list[dict[str, JsonValue]]) -> list[dict[str, JsonValue]]:
    cleaned: Final = tuple(
        {
            key: (
                [block for block in value if not (isinstance(block, dict) and block.get("type") == "tool_addition")]
                if key == "content" and isinstance(value, list)
                else value
            )
            for key, value in message.items()
            if key != "output_config"
        }
        for message in messages
    )
    return [
        {**message, "content": [{"type": "text", "text": "Please continue."}]}
        if message.get("content") == []
        else message
        for message in cleaned
    ]


def _first_tag(body: dict[str, JsonValue]) -> str:
    first: Final = MESSAGES.validate_python(body["messages"])[0]["content"]
    text: Final = first if isinstance(first, str) else MESSAGES.validate_python(first)[0]["text"]
    assert isinstance(text, str), first
    return text.rsplit(" ", 1)[-1]


def _rejection(body: dict[str, JsonValue]) -> str | None:
    thinking: Final = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("display") not in (None, "summarized", "omitted"):
        return f"thinking.display={thinking['display']!r}"
    messages: Final = MESSAGES.validate_python(body["messages"])
    if any("output_config" in message for message in messages):
        return "message.output_config"
    if any(
        isinstance(block, dict) and block.get("type") == "tool_addition"
        for message in messages
        if isinstance(message["content"], list)
        for block in message["content"]
    ):
        return "tool_addition block"
    return None


def _message(tag: str) -> bytes:
    return json.dumps(
        {
            "id": f"msg_{tag}",
            "type": "message",
            "role": "assistant",
            "model": MODEL_ID,
            "content": [{"type": "text", "text": f"answer {tag}"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 11, "output_tokens": 4},
        }
    ).encode()


def _chunk(payload: dict[str, JsonValue]) -> bytes:
    encoded: Final = base64.b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()
    return _aws_event_frame("chunk", {"bytes": encoded}, "", "")


def _stream(tag: str) -> bytes:
    return (
        _chunk(
            {
                "type": "message_start",
                "message": {
                    "id": f"msg_{tag}",
                    "type": "message",
                    "role": "assistant",
                    "model": MODEL_ID,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 11, "output_tokens": 0},
                },
            }
        )
        + _chunk({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
        + _chunk({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": f"answer {tag}"}})
        + _chunk({"type": "content_block_stop", "index": 0})
        + _chunk({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}})
        + _chunk({"type": "message_stop"})
    )


def lenient_bedrock_peer(request: Request) -> Reply:
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    tag: Final = _first_tag(BODY.validate_python(json.loads(request.body)))
    if request.target == INVOKE_STREAM:
        return Reply(content_type="application/vnd.amazon.eventstream", chunks=(_stream(tag),))
    assert request.target == INVOKE, request.target
    return Reply(body=_message(tag))


def bedrock_peer(request: Request) -> Reply:
    rejected: Final = _rejection(BODY.validate_python(json.loads(request.body)))
    if rejected is not None:
        return Reply(status=400, body=json.dumps({"message": REJECTED + rejected}).encode())
    return lenient_bedrock_peer(request)


def _register(scenario: Scenario, api_base: str, *, drop_params: bool = False) -> str:
    return scenario.model(
        model=f"bedrock/invoke/{MODEL_ID}",
        api_key=TOKEN,
        aws_region_name="us-east-1",
        api_base=api_base,
        drop_params=drop_params,
    )


def _sent(wire_requests: tuple[Request, ...], target: str) -> dict[str, JsonValue]:
    assert len(wire_requests) == 1, [request.target for request in wire_requests]
    assert wire_requests[0].target == target, wire_requests[0].target
    return BODY.validate_python(json.loads(wire_requests[0].body))


def test_per_message_output_config_is_rejected_with_actionable_error_without_drop_params(
    gateway: Gateway,
) -> None:
    tag: Final = uuid.uuid4().hex
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire.url)
        response: Final = gateway.request(
            "POST", "/v1/messages", {"model": model, "max_tokens": 300, "messages": _output_config_turns(tag)}
        )
        assert response.status_code == 400, response.text
        assert "messages[1].output_config" in response.text and "drop_params" in response.text, response.text
        assert wire.drain() == (), "rejected request must not reach Bedrock"


def test_per_message_output_config_is_dropped_before_bedrock_invoke_with_drop_params(gateway: Gateway) -> None:
    tag: Final = uuid.uuid4().hex
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire.url, drop_params=True)
        response: Final = gateway.request(
            "POST", "/v1/messages", {"model": model, "max_tokens": 300, "messages": _output_config_turns(tag)}
        )
        assert response.status_code == 200, response.text
        assert response.json()["content"] == [{"type": "text", "text": f"answer {tag}"}], response.text
        sent: Final = _sent(wire.drain(), INVOKE)
        assert sent["messages"] == _without_extensions(_output_config_turns(tag)), sent
        assert "output_config" not in json.dumps(sent), sent
        assert sent["max_tokens"] == 300 and "model" not in sent, sent


def test_thinking_display_updates_is_rejected_without_drop_params(gateway: Gateway) -> None:
    tag: Final = uuid.uuid4().hex
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire.url)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 300,
                "thinking": {"type": "adaptive", "display": "updates"},
                "messages": [{"role": "user", "content": f"what is 2+2? think briefly {tag}"}],
            },
        )
        assert response.status_code == 400, response.text
        assert "thinking.display" in response.text and "drop_params" in response.text, response.text
        assert wire.drain() == (), "rejected request must not reach Bedrock"


def test_thinking_display_updates_is_dropped_with_drop_params(gateway: Gateway) -> None:
    tag: Final = uuid.uuid4().hex
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire.url, drop_params=True)
        client: Final = anthropic.Anthropic(
            api_key=gateway.key, base_url=str(gateway.client.base_url).rstrip("/"), max_retries=0
        )
        message: Final = client.messages.create(
            model=model,
            max_tokens=300,
            thinking={"type": "adaptive", "display": "updates"},  # pyright: ignore[reportArgumentType]  # Claude Code sends this shape; the SDK types lag
            messages=[{"role": "user", "content": f"what is 2+2? think briefly {tag}"}],
        )
        assert message.role == "assistant" and message.id == f"msg_{tag}", message
        sent: Final = _sent(wire.drain(), INVOKE)
        assert sent["thinking"] == {"type": "adaptive"}, sent
        assert sent["messages"] == [{"role": "user", "content": f"what is 2+2? think briefly {tag}"}], sent


def test_tool_addition_block_is_rejected_with_actionable_error_without_modify_params(gateway: Gateway) -> None:
    tag: Final = uuid.uuid4().hex
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire.url, drop_params=True)
        response: Final = gateway.request(
            "POST", "/v1/messages", {"model": model, "max_tokens": 300, "messages": _tool_addition_turns(tag)}
        )
        assert response.status_code == 400, response.text
        assert "messages[1].content[0]" in response.text and "modify_params" in response.text, response.text
        assert wire.drain() == (), "rejected request must not reach Bedrock"


def test_tool_addition_block_is_removed_before_bedrock_invoke_with_modify_params(
    gateway: Gateway, tmp_path: Path
) -> None:
    tag: Final = uuid.uuid4().hex
    with (
        wire_server(bedrock_peer) as wire,
        owned_proxy_process(gateway, tmp_path, {"LITELLM_MODIFY_PARAMS": "True"}) as owned,
        owned.gateway.scenario() as scenario,
    ):
        model: Final = _register(scenario, wire.url)
        response: Final = owned.gateway.request(
            "POST", "/v1/messages", {"model": model, "max_tokens": 300, "messages": _tool_addition_turns(tag)}
        )
        assert response.status_code == 200, response.text
        sent: Final = _sent(wire.drain(), INVOKE)
        assert sent["messages"] == [
            {"role": "user", "content": f"read /tmp/a.txt {tag}"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": "continue"},
        ], sent


def test_all_three_extensions_are_sanitized_on_the_streaming_invoke_path(gateway: Gateway, tmp_path: Path) -> None:
    tag: Final = uuid.uuid4().hex
    turns: Final = _output_config_turns(tag) + _tool_addition_turns(tag)[1:]
    with (
        wire_server(bedrock_peer) as wire,
        owned_proxy_process(gateway, tmp_path, {"LITELLM_MODIFY_PARAMS": "True"}) as owned,
        owned.gateway.scenario() as scenario,
    ):
        model: Final = _register(scenario, wire.url, drop_params=True)
        response: Final = owned.gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 300,
                "stream": True,
                "thinking": {"type": "adaptive", "display": "updates"},
                "messages": turns,
            },
        )
        assert response.status_code == 200, response.text
        events: Final = tuple(
            json.loads(line.removeprefix("data:")) for line in response.text.splitlines() if line.startswith("data:")
        )
        assert events[-1]["type"] == "message_stop", response.text
        assert (
            "".join(event["delta"]["text"] for event in events if event["type"] == "content_block_delta")
            == f"answer {tag}"
        ), response.text
        sent: Final = _sent(wire.drain(), INVOKE_STREAM)
        assert sent["messages"] == _without_extensions(turns), sent
        assert sent["thinking"] == {"type": "adaptive"}, sent
        assert "tool_addition" not in json.dumps(sent) and "output_config" not in json.dumps(sent), sent


def test_message_holding_only_tool_addition_blocks_gets_the_placeholder_text(gateway: Gateway, tmp_path: Path) -> None:
    tag: Final = uuid.uuid4().hex
    turns: Final = [
        {"role": "user", "content": f"read /tmp/a.txt {tag}"},
        {"role": "assistant", "content": [TOOL_ADDITION, TOOL_ADDITION]},
        {"role": "user", "content": "continue"},
    ]
    with (
        wire_server(bedrock_peer) as wire,
        owned_proxy_process(gateway, tmp_path, {"LITELLM_MODIFY_PARAMS": "True"}) as owned,
        owned.gateway.scenario() as scenario,
    ):
        model: Final = _register(scenario, wire.url)
        response: Final = owned.gateway.request(
            "POST", "/v1/messages", {"model": model, "max_tokens": 300, "messages": turns}
        )
        assert response.status_code == 200, response.text
        sent: Final = _sent(wire.drain(), INVOKE)
        assert sent["messages"][1] == {
            "role": "assistant",
            "content": [{"type": "text", "text": "Please continue."}],
        }, sent


@pytest.mark.parametrize("display", ["summarized", "omitted", None])
def test_supported_or_absent_thinking_display_reaches_bedrock_unchanged(gateway: Gateway, display: str | None) -> None:
    tag: Final = uuid.uuid4().hex
    thinking: Final = {"type": "adaptive", **({"display": display} if display is not None else {})}
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire.url)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {"model": model, "max_tokens": 300, "thinking": thinking, "messages": [{"role": "user", "content": tag}]},
        )
        assert response.status_code == 200, response.text
        sent: Final = _sent(wire.drain(), INVOKE)
        assert sent["thinking"] == thinking, sent


@pytest.mark.parametrize(
    "output_config", [7, ["high"], "", "x" * 5000], ids=["int", "list", "empty_string", "five_kb_string"]
)
def test_malformed_per_message_output_config_is_dropped_on_every_message(
    gateway: Gateway, output_config: JsonValue
) -> None:
    tag: Final = uuid.uuid4().hex
    turns: Final = [
        {"role": "user", "content": f"read {tag}", "output_config": output_config},
        {"role": "assistant", "content": [READING], "output_config": output_config},
        {"role": "user", "content": "continue", "output_config": output_config},
    ]
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire.url, drop_params=True)
        response: Final = gateway.request(
            "POST", "/v1/messages", {"model": model, "max_tokens": 300, "messages": turns}
        )
        assert response.status_code == 200, response.text
        sent: Final = _sent(wire.drain(), INVOKE)
        assert sent["messages"] == _without_extensions(turns), sent


def test_non_string_thinking_display_and_bare_string_blocks_are_forwarded_as_sent(gateway: Gateway) -> None:
    tag: Final = uuid.uuid4().hex
    turns: Final = [
        {"role": "user", "content": f"read {tag}"},
        {"role": "assistant", "content": ["ok"]},
        {"role": "user", "content": "continue"},
    ]
    with wire_server(lenient_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire.url)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {"model": model, "max_tokens": 300, "thinking": {"type": "adaptive", "display": 7}, "messages": turns},
        )
        assert response.status_code == 200, response.text
        sent: Final = _sent(wire.drain(), INVOKE)
        assert sent["thinking"] == {"type": "adaptive", "display": 7}, sent
        assert sent["messages"] == turns, sent


def test_anthropic_direct_deployment_forwards_the_extensions_verbatim(gateway: Gateway) -> None:
    tag: Final = uuid.uuid4().hex
    turns: Final = _output_config_turns(tag) + _tool_addition_turns(tag)[1:]

    def anthropic_peer(request: Request) -> Reply:
        assert request.target == "/v1/messages", request.target
        return Reply(body=_message(tag))

    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-5", api_base=wire.url, api_key="synthetic-anthropic-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 300,
                "thinking": {"type": "adaptive", "display": "updates"},
                "messages": turns,
            },
        )
        assert response.status_code == 200, response.text
        sent: Final = _sent(wire.drain(), "/v1/messages")
        assert sent["messages"] == turns, sent
        assert sent["thinking"] == {"type": "adaptive", "display": "updates"}, sent


def test_bedrock_invoke_chat_completions_are_untouched_by_the_messages_sanitizer(gateway: Gateway) -> None:
    tag: Final = uuid.uuid4().hex
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire.url)
        body: Final = gateway.chat(model, text=f"chat control {tag}")
        assert body["choices"][0]["message"]["content"] == f"answer {tag}", body
        sent: Final = _sent(wire.drain(), INVOKE)
        assert sent["messages"] == [{"role": "user", "content": [{"type": "text", "text": f"chat control {tag}"}]}], (
            sent
        )
