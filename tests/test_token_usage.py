from glm2api.core.usage import TokenUsage
from glm2api.api.adapters.anthropic.messages import serialize_anthropic_usage
from glm2api.api.adapters.openai.chat_completions import serialize_openai_usage
from glm2api.api.adapters.openai.responses import serialize_openai_responses_usage


def test_token_usage_maps_to_each_protocol_without_exposing_source():
    usage = TokenUsage.estimated(input_tokens=12, output_tokens=8)

    assert usage.source == "estimated"
    assert usage.total_tokens == 20
    assert serialize_openai_usage(usage) == {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "total_tokens": 20,
    }
    assert serialize_anthropic_usage(usage) == {"input_tokens": 12, "output_tokens": 8}
    assert serialize_openai_responses_usage(usage, include_output_details=True) == {
        "input_tokens": 12,
        "output_tokens": 8,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 20,
    }


def test_upstream_usage_replaces_estimates_and_estimates_cannot_overwrite_it():
    estimated = TokenUsage.estimated(input_tokens=120, output_tokens=80)
    upstream = estimated.with_upstream({"prompt_tokens": 7, "completion_tokens": 5})

    assert upstream.source == "upstream"
    assert serialize_openai_usage(upstream) == {
        "prompt_tokens": 7,
        "completion_tokens": 5,
        "total_tokens": 12,
    }
    assert upstream.with_estimated(input_tokens=999, output_tokens=999) == upstream


def test_partial_upstream_usage_is_marked_mixed_and_keeps_the_other_estimate():
    usage = TokenUsage.estimated(input_tokens=120, output_tokens=80).with_upstream(
        {"prompt_tokens": 7}
    )

    assert usage.source == "mixed"
    assert usage.input_tokens == 7
    assert usage.output_tokens == 80


def test_usage_plus_sums_retry_attempts_and_keeps_conservative_provenance():
    first = TokenUsage.estimated(input_tokens=10, output_tokens=3)
    second = TokenUsage.from_upstream({"prompt_tokens": 7, "completion_tokens": 2})

    combined = first.plus(second)

    assert combined.input_tokens == 17
    assert combined.output_tokens == 5
    assert combined.source == "estimated"
