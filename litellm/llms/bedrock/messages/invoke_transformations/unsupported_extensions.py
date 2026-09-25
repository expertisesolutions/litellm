from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import chain
from types import MappingProxyType
from typing import Annotated, Final, NoReturn, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

import litellm
from litellm.constants import (
    BEDROCK_INVOKE_SUPPORTED_THINKING_DISPLAY_VALUES,
    BEDROCK_INVOKE_UNSUPPORTED_MESSAGE_CONTENT_BLOCK_TYPES,
)
from litellm.litellm_core_utils.prompt_templates.factory import DEFAULT_USER_CONTINUE_MESSAGE


class OptInKnob(Enum):
    DROP_PARAMS = "drop_params"
    MODIFY_PARAMS = "modify_params"


@dataclass(frozen=True, slots=True)
class OptIns:
    drop_params: bool
    modify_params: bool

    def allows(self, knob: OptInKnob) -> bool:
        return self.drop_params if knob is OptInKnob.DROP_PARAMS else self.modify_params


@dataclass(frozen=True, slots=True)
class Offender:
    path: str
    knob: OptInKnob


@dataclass(frozen=True, slots=True)
class Unchanged:
    pass


@dataclass(frozen=True, slots=True)
class Sanitized:
    replacements: Mapping[str, JsonValue]
    removed: tuple[Offender, ...]


@dataclass(frozen=True, slots=True)
class Refused:
    offenders: tuple[Offender, ...]


SanitizeOutcome: TypeAlias = Unchanged | Sanitized | Refused


class _Block(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    type: str

    @property
    def is_unsupported(self) -> bool:
        return self.type in BEDROCK_INVOKE_UNSUPPORTED_MESSAGE_CONTENT_BLOCK_TYPES


class _Message(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    content: str | tuple[Annotated[_Block | JsonValue, Field(union_mode="left_to_right")], ...] | None = None
    output_config: JsonValue = None


class _Thinking(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    display: JsonValue = None


class _Request(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    messages: tuple[Annotated[_Message | JsonValue, Field(union_mode="left_to_right")], ...] | None = None
    thinking: Annotated[_Thinking | JsonValue, Field(union_mode="left_to_right")] = None


_PLACEHOLDER_BLOCK: Final = _Block.model_validate(
    MappingProxyType({"type": "text", "text": DEFAULT_USER_CONTINUE_MESSAGE["content"]})
)
_WITHOUT_OUTPUT_CONFIG: Final = MappingProxyType({"output_config": True})
_WITHOUT_DISPLAY: Final = MappingProxyType({"display": True})
_SANITIZED_FIELDS: Final = MappingProxyType({"messages": True, "thinking": True})


def _unsupported_display(thinking: _Thinking | JsonValue) -> str | None:
    if not isinstance(thinking, _Thinking) or not isinstance(thinking.display, str):
        return None
    return None if thinking.display in BEDROCK_INVOKE_SUPPORTED_THINKING_DISPLAY_VALUES else thinking.display


def _blocks(message: _Message) -> tuple[_Block | JsonValue, ...]:
    return message.content if isinstance(message.content, tuple) else ()


def _thinking_offenders(thinking: _Thinking | JsonValue) -> tuple[Offender, ...]:
    display: Final = _unsupported_display(thinking)
    if display is None:
        return ()
    return (Offender(path=f"thinking.display (value '{display}')", knob=OptInKnob.DROP_PARAMS),)


def _message_offenders(index: int, message: _Message | JsonValue) -> tuple[Offender, ...]:
    if not isinstance(message, _Message):
        return ()
    output_config: Final = (
        (Offender(path=f"messages[{index}].output_config", knob=OptInKnob.DROP_PARAMS),)
        if "output_config" in message.model_fields_set
        else ()
    )
    blocks: Final = tuple(
        Offender(path=f"messages[{index}].content[{block_index}] (type '{block.type}')", knob=OptInKnob.MODIFY_PARAMS)
        for block_index, block in enumerate(_blocks(message))
        if isinstance(block, _Block) and block.is_unsupported
    )
    return output_config + blocks


def _offenders(request: _Request) -> tuple[Offender, ...]:
    per_message: Final = chain.from_iterable(
        _message_offenders(index, message) for index, message in enumerate(request.messages or ())
    )
    return _thinking_offenders(request.thinking) + tuple(per_message)


def _kept_blocks(blocks: tuple[_Block | JsonValue, ...]) -> tuple[_Block | JsonValue, ...]:
    kept: Final = tuple(block for block in blocks if not (isinstance(block, _Block) and block.is_unsupported))
    return kept if kept or not blocks else (_PLACEHOLDER_BLOCK,)


def _sanitized_message(message: _Message | JsonValue) -> JsonValue:
    if not isinstance(message, _Message):
        return message
    without_blocks: Final = (
        message.model_copy(update=MappingProxyType({"content": _kept_blocks(message.content)}))
        if isinstance(message.content, tuple)
        else message
    )
    return without_blocks.model_dump(mode="json", exclude_unset=True, exclude=_WITHOUT_OUTPUT_CONFIG)


def _sanitized_thinking(thinking: _Thinking | JsonValue) -> _Thinking | JsonValue:
    if not isinstance(thinking, _Thinking) or _unsupported_display(thinking) is None:
        return thinking
    return thinking.model_dump(mode="json", exclude_unset=True, exclude=_WITHOUT_DISPLAY)


def _replacements(request: _Request) -> Mapping[str, JsonValue]:
    sanitized: Final = (
        ("messages", tuple(_sanitized_message(message) for message in request.messages or ())),
        ("thinking", _sanitized_thinking(request.thinking)),
    )
    update: Final = MappingProxyType({field: value for field, value in sanitized if field in request.model_fields_set})
    return MappingProxyType(
        request.model_copy(update=update).model_dump(mode="json", exclude_unset=True, include=_SANITIZED_FIELDS)
    )


def sanitize_for_bedrock_invoke(request: Mapping[str, object], opt_ins: OptIns) -> SanitizeOutcome:
    try:
        view: Final = _Request.model_validate(request)
    except ValidationError:
        return Unchanged()
    offenders: Final = _offenders(view)
    blocked: Final = tuple(offender for offender in offenders if not opt_ins.allows(offender.knob))
    if blocked:
        return Refused(offenders=blocked)
    if not offenders:
        return Unchanged()
    return Sanitized(replacements=_replacements(view), removed=offenders)


def _paths(refused: Refused, knob: OptInKnob) -> str:
    return ", ".join(offender.path for offender in refused.offenders if offender.knob is knob)


def _drop_params_hint(paths: str) -> str:
    if not paths:
        return ""
    return (
        f"Bedrock Invoke does not accept {paths}. Set `drop_params: true` in this deployment's `litellm_params` "
        "(or `litellm_settings.drop_params: true` on the proxy, `litellm.drop_params = True` in the SDK) to have "
        "LiteLLM drop them (an unsupported thinking.display falls back to the model default), or remove them from "
        "the request."
    )


def _modify_params_hint(paths: str) -> str:
    if not paths:
        return ""
    return (
        f"Bedrock Invoke does not accept {paths}. Set `litellm_settings.modify_params: true` on the proxy or "
        "`litellm.modify_params = True` in the SDK to have LiteLLM remove those blocks (a message left empty gets "
        f"the placeholder text '{DEFAULT_USER_CONTINUE_MESSAGE['content']}'), or remove them from the request."
    )


def raise_refusal(refused: Refused, model: str) -> NoReturn:
    param_paths: Final = _paths(refused, OptInKnob.DROP_PARAMS)
    block_paths: Final = _paths(refused, OptInKnob.MODIFY_PARAMS)
    hints: Final = tuple(hint for hint in (_drop_params_hint(param_paths), _modify_params_hint(block_paths)) if hint)
    error_class: Final = litellm.UnsupportedParamsError if param_paths else litellm.BadRequestError
    raise error_class(message=" ".join(hints), model=model, llm_provider="bedrock")
