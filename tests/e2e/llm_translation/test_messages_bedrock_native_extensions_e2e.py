from __future__ import annotations

from typing import cast

import anthropic
import pytest
from anthropic.types import Message, MessageParam, TextBlock, ThinkingConfigParam
from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import NO_PROXY_CACHE, SdkClients

pytestmark = pytest.mark.e2e

BEDROCK_BACKEND = "bedrock/us.anthropic.claude-sonnet-5"


def _register(proxy: ProxyClient, resources: ResourceManager, *, drop_params: bool) -> tuple[str, str]:
    model = f"e2e-bedrock-msgs-ext-{unique_marker()}"
    model_id = proxy.create_model(
        model,
        LiteLLMParamsBody(
            model=BEDROCK_BACKEND,
            aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
            aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
            aws_region_name="us-east-1",
            drop_params=drop_params,
        ),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, resources.key()


def _output_config_messages() -> list[MessageParam]:
    return [
        {"role": "user", "content": "read the file /tmp/a.txt"},
        cast(
            MessageParam,
            {
                "role": "assistant",
                "output_config": {"effort": "high"},
                "content": [
                    {"type": "text", "text": "Reading it now."},
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "/tmp/a.txt"}},
                ],
            },
        ),
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "hello world"}],
        },
    ]


def _tool_addition_messages() -> list[MessageParam]:
    return [
        {"role": "user", "content": "read /tmp/a.txt"},
        cast(
            MessageParam,
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_addition", "tool_reference": {"type": "tool_reference", "tool_name": "Read"}},
                    {"type": "text", "text": "ok"},
                ],
            },
        ),
        {"role": "user", "content": "continue"},
    ]


def _text(message: Message) -> str:
    return "".join(block.text for block in message.content if isinstance(block, TextBlock))


def _assert_actionable_error(error: anthropic.BadRequestError, *, knob: str) -> None:
    assert knob in str(error), str(error)
    assert "messages[" in str(error), str(error)


class TestBedrockMessagesNativeExtensions:
    @pytest.mark.covers("llm.messages.bedrock_invoke.native_extensions.nonstream.works")
    def test_drop_params_strips_nested_output_config(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources, drop_params=True)
        client = sdk.anthropic(key)

        message = client.messages.create(
            model=model,
            max_tokens=300,
            messages=_output_config_messages(),
            extra_body=NO_PROXY_CACHE,
        )
        assert message.role == "assistant", f"unexpected role: {message.role!r}"
        assert _text(message).strip(), f"/v1/messages returned no text: {message.content!r}"

    @pytest.mark.covers("llm.messages.bedrock_invoke.native_extensions.nonstream.works")
    def test_drop_params_alone_does_not_remove_tool_addition_block(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources, drop_params=True)
        client = sdk.anthropic(key)

        with pytest.raises(anthropic.BadRequestError) as exc_info:
            client.messages.create(
                model=model,
                max_tokens=300,
                messages=_tool_addition_messages(),
                extra_body=NO_PROXY_CACHE,
            )
        _assert_actionable_error(exc_info.value, knob="modify_params")

    @pytest.mark.covers("llm.messages.bedrock_invoke.native_extensions.nonstream.works")
    def test_drop_params_drops_thinking_display_updates(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources, drop_params=True)
        client = sdk.anthropic(key)

        message = client.messages.create(
            model=model,
            max_tokens=300,
            thinking=cast(ThinkingConfigParam, {"type": "adaptive", "display": "updates"}),
            messages=[{"role": "user", "content": "what is 2+2? think briefly"}],
            extra_body=NO_PROXY_CACHE,
        )
        assert message.role == "assistant", f"unexpected role: {message.role!r}"
        assert _text(message).strip(), f"/v1/messages returned no text: {message.content!r}"

    @pytest.mark.covers("llm.messages.bedrock_invoke.native_extensions.nonstream.works")
    def test_nested_output_config_rejected_with_actionable_error(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources, drop_params=False)
        client = sdk.anthropic(key)

        with pytest.raises(anthropic.BadRequestError) as exc_info:
            client.messages.create(
                model=model,
                max_tokens=300,
                messages=_output_config_messages(),
                extra_body=NO_PROXY_CACHE,
            )
        _assert_actionable_error(exc_info.value, knob="drop_params")

    @pytest.mark.covers("llm.messages.bedrock_invoke.native_extensions.nonstream.works")
    def test_tool_addition_block_rejected_with_actionable_error(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources, drop_params=False)
        client = sdk.anthropic(key)

        with pytest.raises(anthropic.BadRequestError) as exc_info:
            client.messages.create(
                model=model,
                max_tokens=300,
                messages=_tool_addition_messages(),
                extra_body=NO_PROXY_CACHE,
            )
        _assert_actionable_error(exc_info.value, knob="modify_params")

    @pytest.mark.covers("llm.messages.bedrock_invoke.native_extensions.nonstream.works")
    def test_thinking_display_updates_rejected_with_actionable_error(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources, drop_params=False)
        client = sdk.anthropic(key)

        with pytest.raises(anthropic.BadRequestError) as exc_info:
            client.messages.create(
                model=model,
                max_tokens=300,
                thinking=cast(ThinkingConfigParam, {"type": "adaptive", "display": "updates"}),
                messages=[{"role": "user", "content": "what is 2+2? think briefly"}],
                extra_body=NO_PROXY_CACHE,
            )
        assert "thinking.display" in str(exc_info.value), str(exc_info.value)
        assert "drop_params" in str(exc_info.value), str(exc_info.value)
