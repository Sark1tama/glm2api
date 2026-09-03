from glm2api.config import (
    BUILTIN_EXPOSED_MODELS,
    BUILTIN_IMAGE_MODELS,
    BUILTIN_MODEL_ALIASES,
    BUILTIN_TEXT_MODELS,
    BUILTIN_VIDEO_MODELS,
    DEFAULT_CHAT_MODEL_NAME,
)
from glm2api.glm.chat import (
    expand_model_variants,
    get_model_multimodal_capability,
    split_model_features,
)
from glm2api.glm.chat import resolve_chat_mode, resolve_networking, resolve_upstream_model


class _Config:
    glm_assistant_id = "65940acff94777010aa6b796"
    model_aliases = {"glm-4": "glm-4", "custom": "65940acff94777010aa6b797"}


def test_expand_model_variants_adds_think_search_and_combined_suffixes():
    models = expand_model_variants(("glm-4", "glm-image-1"), excluded_models={"glm-image-1"})

    assert models == [
        "glm-4",
        "glm-4-think",
        "glm-4-search",
        "glm-4-think-search",
        "glm-image-1",
    ]


def test_split_model_features_accepts_think_search_in_either_order():
    assert split_model_features("glm-4-think-search") == ("glm-4", {"think", "search"})
    assert split_model_features("glm-4-search-think") == ("glm-4", {"think", "search"})
    assert split_model_features("glm-deep-research") == ("glm-deep-research", set())


def test_model_multimodal_capability_is_model_aware_and_suffix_safe():
    assert get_model_multimodal_capability("glm-5.3") is False
    assert get_model_multimodal_capability("glm-5.3-think-search") is False
    assert get_model_multimodal_capability("glm-5.3-flash") is True
    assert get_model_multimodal_capability("glm-5.3-flash-search") is True
    assert get_model_multimodal_capability("custom-model") is None


def test_variant_model_resolves_to_base_upstream_model():
    upstream_model, assistant_id = resolve_upstream_model("glm-4-think-search", _Config())
    custom_upstream, custom_assistant_id = resolve_upstream_model("custom-search", _Config())

    assert upstream_model == "glm-4"
    assert assistant_id == _Config.glm_assistant_id
    assert custom_upstream == "65940acff94777010aa6b797"
    assert custom_assistant_id == "65940acff94777010aa6b797"


def test_model_suffixes_resolve_chat_mode_and_networking_matrix():
    assert resolve_chat_mode("glm-4-think", None, None) == "thinking"
    assert resolve_networking("glm-4-think", None) is False

    assert resolve_chat_mode("glm-4-search", None, None) == ""
    assert resolve_networking("glm-4-search", None) is True

    assert resolve_chat_mode("glm-4-think-search", None, None) == "thinking"
    assert resolve_networking("glm-4-think-search", None) is True

    assert resolve_chat_mode("glm-4", None, None) == ""
    assert resolve_networking("glm-4", None) is False


def test_existing_thinking_model_name_still_enables_chat_mode():
    assert resolve_chat_mode("glm-4.1v-thinking-flashx", None, None) == "thinking"


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
    assert BUILTIN_MODEL_ALIASES == {
        "glm-5.3": "glm-5.3",
        "glm-5.3-flash": "glm-5.3-flash",
    }


def test_current_reasoning_effort_mapping_matches_chatglm_web_modes():
    assert resolve_chat_mode("glm-5.3", "none", None) == ""
    assert resolve_chat_mode("glm-5.3", "low", None) == ""
    assert resolve_chat_mode("glm-5.3", "medium", None) == "thinking"
    assert resolve_chat_mode("glm-5.3", "high", None) == "deep_thinking"
    assert resolve_chat_mode("glm-5.3", "xhigh", None) == "deep_thinking"


def test_deep_research_uses_supported_deep_thinking_mode():
    assert resolve_chat_mode("glm-5.3", None, True) == "deep_thinking"
