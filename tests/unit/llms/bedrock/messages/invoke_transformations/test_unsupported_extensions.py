import copy

import pytest

from litellm.llms.bedrock.messages.invoke_transformations.unsupported_extensions import (
    Offender,
    OptInKnob,
    OptIns,
    Refused,
    Sanitized,
    Unchanged,
    sanitize_for_bedrock_invoke,
)

ALL_OPT_INS = OptIns(drop_params=True, modify_params=True)
TOOL_ADDITION = {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "mcp__files__read"}}
TOOL_USE = {
    "type": "tool_use",
    "id": "toolu_1",
    "name": "Read",
    "input": {"path": "/tmp/a.txt", "flags": [1, 2.5, None]},
}


def _claude_code_request():
    return {
        "max_tokens": 100,
        "metadata": {"user_id": "u"},
        "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
        "thinking": {"type": "adaptive", "display": "updates"},
        "messages": [
            {"role": "user", "content": "read /tmp/a.txt"},
            {
                "role": "assistant",
                "output_config": {"effort": "high"},
                "content": [
                    {"type": "text", "text": "ok", "cache_control": {"type": "ephemeral"}},
                    TOOL_ADDITION,
                    TOOL_USE,
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "hi"}]},
            {"role": "assistant", "content": [TOOL_ADDITION]},
            {"role": "user", "content": []},
        ],
        "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
    }


@pytest.mark.parametrize(
    "request_body",
    [
        {"messages": "not a list", "thinking": "not a dict"},
        {"messages": ["bare string", 7, None], "thinking": {"type": "adaptive"}},
        {"messages": [{"role": "assistant", "content": ["bare string", 7, {"type": "text", "text": "ok"}]}]},
        {"messages": [{"role": "assistant", "content": [{"type": ["unhashable"]}]}], "thinking": {"display": ["x"]}},
        {"messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "adaptive", "display": "summarized"}},
        {"messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "adaptive", "display": "omitted"}},
    ],
)
def test_sanitize_leaves_requests_without_offenders_alone(request_body):
    snapshot = copy.deepcopy(request_body)

    outcome = sanitize_for_bedrock_invoke(request_body, OptIns(drop_params=False, modify_params=False))

    assert outcome == Unchanged()
    assert request_body == snapshot


def test_sanitize_removes_only_the_offenders_and_keeps_every_message():
    request_body = _claude_code_request()
    snapshot = copy.deepcopy(request_body)
    expected = copy.deepcopy(request_body)
    del expected["thinking"]["display"]
    del expected["messages"][1]["output_config"]
    expected["messages"][1]["content"].remove(TOOL_ADDITION)
    expected["messages"][3]["content"] = [{"type": "text", "text": "Please continue."}]

    outcome = sanitize_for_bedrock_invoke(request_body, ALL_OPT_INS)

    assert isinstance(outcome, Sanitized)
    assert {**request_body, **outcome.replacements} == expected
    assert request_body == snapshot
    assert outcome.removed == (
        Offender(path="thinking.display (value 'updates')", knob=OptInKnob.DROP_PARAMS),
        Offender(path="messages[1].output_config", knob=OptInKnob.DROP_PARAMS),
        Offender(path="messages[1].content[1] (type 'tool_addition')", knob=OptInKnob.MODIFY_PARAMS),
        Offender(path="messages[3].content[0] (type 'tool_addition')", knob=OptInKnob.MODIFY_PARAMS),
    )


def test_sanitize_replaces_only_the_fields_the_request_sets():
    outcome = sanitize_for_bedrock_invoke(
        {"max_tokens": 10, "messages": [{"role": "assistant", "output_config": {"effort": "high"}, "content": "hi"}]},
        ALL_OPT_INS,
    )

    assert outcome == Sanitized(
        replacements={"messages": [{"role": "assistant", "content": "hi"}]},
        removed=(Offender(path="messages[0].output_config", knob=OptInKnob.DROP_PARAMS),),
    )


@pytest.mark.parametrize(
    ("opt_ins", "blocked"),
    [
        (
            OptIns(drop_params=True, modify_params=False),
            (
                Offender(path="messages[1].content[1] (type 'tool_addition')", knob=OptInKnob.MODIFY_PARAMS),
                Offender(path="messages[3].content[0] (type 'tool_addition')", knob=OptInKnob.MODIFY_PARAMS),
            ),
        ),
        (
            OptIns(drop_params=False, modify_params=True),
            (
                Offender(path="thinking.display (value 'updates')", knob=OptInKnob.DROP_PARAMS),
                Offender(path="messages[1].output_config", knob=OptInKnob.DROP_PARAMS),
            ),
        ),
    ],
)
def test_sanitize_refuses_with_only_the_offenders_whose_opt_in_is_off(opt_ins, blocked):
    assert sanitize_for_bedrock_invoke(_claude_code_request(), opt_ins) == Refused(offenders=blocked)
