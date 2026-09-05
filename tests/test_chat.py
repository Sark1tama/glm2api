import pytest

from glm2api.config import (
    BUILTIN_EXPOSED_MODELS,
    BUILTIN_IMAGE_MODELS,
    BUILTIN_MODEL_CATALOG,
    BUILTIN_TEXT_MODELS,
    BUILTIN_VIDEO_MODELS,
    DEFAULT_CHAT_MODEL_NAME,
)
from glm2api.glm.chat import (
    get_model_multimodal_capability,
    resolve_chat_mode,
    resolve_networking,
    validate_text_model,
)


def test_model_capability_uses_exact_model_name():
    assert get_model_multimodal_capability("glm-5.3") is False
    assert get_model_multimodal_capability("glm-5.3-flash") is True
    assert get_model_multimodal_capability("glm-5.3-flash-search") is None
    assert get_model_multimodal_capability("custom-model") is None


def test_text_model_validation_accepts_public_names_and_rejects_unknown_names():
    assert validate_text_model("glm-5.3") == "glm-5.3"

    with pytest.raises(ValueError, match="不支持模型 glm-5.3-search") as raised:
        validate_text_model("glm-5.3-search")

    assert "glm-5.3, glm-5.3-flash" in str(raised.value)


def test_networking_is_controlled_only_by_explicit_request_capability():
    assert resolve_networking(None) is False
    assert resolve_networking(False) is False
    assert resolve_networking(True) is True


def test_chat_mode_is_controlled_only_by_explicit_request_capabilities():
    assert resolve_chat_mode(None) == ""
    assert resolve_chat_mode("none") == ""
    assert resolve_chat_mode("low") == ""
    assert resolve_chat_mode("medium") == "thinking"
    assert resolve_chat_mode("high") == "deep_thinking"
    assert resolve_chat_mode("xhigh") == "deep_thinking"
    assert resolve_chat_mode("max") == "deep_thinking"


def test_current_models_are_exposed_without_virtual_variants():
    assert BUILTIN_TEXT_MODELS == ("glm-5.3", "glm-5.3-flash")
    assert BUILTIN_IMAGE_MODELS == ("glm-image-1",)
    assert BUILTIN_VIDEO_MODELS == ("glm-video-1",)
    assert BUILTIN_EXPOSED_MODELS == (
        "glm-5.3",
        "glm-5.3-flash",
        "glm-image-1",
        "glm-video-1",
    )
    assert DEFAULT_CHAT_MODEL_NAME == "glm-5.3-flash"
    assert BUILTIN_MODEL_CATALOG["glm-5.3"].multimodal is False
    assert BUILTIN_MODEL_CATALOG["glm-5.3-flash"].multimodal is True
    assert BUILTIN_MODEL_CATALOG["glm-image-1"].kind == "image"
    assert BUILTIN_MODEL_CATALOG["glm-video-1"].kind == "video"
