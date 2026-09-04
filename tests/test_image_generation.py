from __future__ import annotations

from types import SimpleNamespace

import pytest

from glm2api.media import images as images_module
from glm2api.glm.client import GLMWebClient, QueueLease, UpstreamAPIError
from glm2api.media.images import (
    GLMImageEventAccumulator,
    build_glm_image_payload,
    GeneratedImage,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageResponseConversionError,
    ImageService,
    internal_to_openai_images_response,
    openai_images_to_internal,
)


def test_glm_image_accumulator_preserves_upstream_order_and_deduplicates_snapshots():
    accumulator = GLMImageEventAccumulator()
    accumulator.consume_event(
        {
            "status": "init",
            "parts": [{"logic_id": "10", "status": "init", "content": []}],
        }
    )
    accumulator.consume_event(
        {
            "conversation_id": "conversation-1",
            "status": "finish",
            "parts": [
                {
                    "logic_id": "10",
                    "status": "finish",
                    "content": [
                        {
                            "type": "image",
                            "code": "猫咪插画",
                            "image": [{"image_url": "https://cdn.example/first.png"}],
                        }
                    ],
                },
                {
                    "logic_id": "2",
                    "status": "finish",
                    "content": [
                        {
                            "type": "image",
                            "image": [
                                {"image_url": "https://cdn.example/second.png"},
                                {"image_url": "https://cdn.example/second.png"},
                            ],
                        }
                    ],
                },
            ],
        }
    )

    result = accumulator.build_result(created=123)

    assert accumulator.conversation_id == "conversation-1"
    assert result.created == 123
    assert [image.url for image in result.images] == [
        "https://cdn.example/first.png",
        "https://cdn.example/second.png",
    ]
    assert result.images[0].revised_prompt == "猫咪插画"


def test_glm_image_accumulator_accepts_json_image_envelope():
    accumulator = GLMImageEventAccumulator()

    accumulator.consume_event(
        {
            "status": 0,
            "result": {"data": [{"url": "https://cdn.example/json.png", "revised_prompt": "改写"}]},
        }
    )

    result = accumulator.build_result(created=456)

    assert result.images == (
        GeneratedImage(url="https://cdn.example/json.png", revised_prompt="改写"),
    )


def test_openai_images_adapter_validates_request_boundary():
    valid = openai_images_to_internal(
        {"model": "glm-image-1", "prompt": "画一只猫", "n": 2, "response_format": "B64_JSON"},
        default_model="glm-image-1",
    )
    assert valid.response_format == "b64_json"
    assert valid.n == 2

    with pytest.raises(ValueError, match="prompt"):
        openai_images_to_internal({}, default_model="glm-image-1")
    with pytest.raises(ValueError, match="n"):
        openai_images_to_internal({"prompt": "猫", "n": 0}, default_model="glm-image-1")
    with pytest.raises(ValueError, match="response_format"):
        openai_images_to_internal({"prompt": "猫", "response_format": "binary"}, default_model="glm-image-1")
    with pytest.raises(ValueError, match="不支持模型 glm-5.3"):
        openai_images_to_internal({"model": "glm-5.3", "prompt": "猫"}, default_model="glm-image-1")


def test_openai_images_adapter_serializes_url_and_base64_results():
    request = ImageGenerationRequest(model="glm-image-1", prompt="猫", n=2, response_format="b64_json")
    result = ImageGenerationResult(
        created=789,
        images=(
            GeneratedImage(url="https://cdn.example/a.png", revised_prompt="a"),
            GeneratedImage(base64_data="already-encoded"),
        ),
    )

    response = internal_to_openai_images_response(
        request,
        result,
        download_image=lambda url: f"downloaded:{url}",
    )

    assert response == {
        "created": 789,
        "data": [
            {"b64_json": "downloaded:https://cdn.example/a.png", "revised_prompt": "a"},
            {"b64_json": "already-encoded"},
        ],
    }


def test_openai_images_adapter_marks_unrepresentable_upstream_result():
    request = ImageGenerationRequest(model="glm-image-1", prompt="猫", response_format="url")
    result = ImageGenerationResult(created=789, images=(GeneratedImage(base64_data="already-encoded"),))

    with pytest.raises(ImageResponseConversionError, match="没有可转换"):
        internal_to_openai_images_response(
            request,
            result,
            download_image=lambda url: "unused",
        )


def test_generate_images_propagates_upstream_error_event():
    client = object.__new__(GLMWebClient)
    client.config = SimpleNamespace(debug_dump_all=False)
    client.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    client.images = ImageService(client)
    released: list[int] = []
    client.request_queue = SimpleNamespace(
        acquire=lambda name: QueueLease(ticket=0, release_callback=lambda ticket: released.append(ticket))
    )
    client.get_preferred_account_index = lambda ticket: None
    client.images._open_stream = lambda request, preferred_account_index=None: (object(), "assistant")
    client.iter_sse_events = lambda response: iter(
        [{"status": "error", "last_error": {"error_code": "quota", "err_msg": "额度不足"}}]
    )
    client.delete_conversation = lambda *args, **kwargs: None

    with pytest.raises(UpstreamAPIError, match="额度不足"):
        client.generate_images(ImageGenerationRequest(model="glm-image-1", prompt="猫"))

    assert released == [0]


def test_generate_images_accepts_json_upstream_result():
    client = object.__new__(GLMWebClient)
    client.config = SimpleNamespace(debug_dump_all=False)
    client.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    client.images = ImageService(client)
    client.request_queue = SimpleNamespace(
        acquire=lambda name: QueueLease(ticket=0, release_callback=lambda ticket: None)
    )
    client.get_preferred_account_index = lambda ticket: None
    client.images._open_stream = lambda request, preferred_account_index=None: (
        {"status": 0, "result": {"data": [{"url": "https://cdn.example/json.png"}]}},
        "assistant",
    )
    client.delete_conversation = lambda *args, **kwargs: None

    result = client.generate_images(ImageGenerationRequest(model="glm-image-1", prompt="猫"))

    assert result.images == (GeneratedImage(url="https://cdn.example/json.png"),)


@pytest.mark.parametrize(
    "event",
    [
        {"status": None, "code": 13},
        {"status": "13"},
    ],
)
def test_event_error_detection_uses_code_and_numeric_string_status(event):
    client = object.__new__(GLMWebClient)

    with pytest.raises(UpstreamAPIError):
        client.raise_for_event_error(event, stream=False)


def test_image_accumulator_treats_intervene_as_terminal_for_image_parts():
    accumulator = GLMImageEventAccumulator()

    accumulator.consume_event(
        {
            "status": "intervene",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [{"type": "image", "image": [{"image_url": "https://cdn.example/intervene.png"}]}],
                }
            ],
        }
    )

    assert accumulator.build_result().images == (
        GeneratedImage(url="https://cdn.example/intervene.png"),
    )


def test_prepare_image_response_keeps_json_payload_and_closes_source():
    client = object.__new__(GLMWebClient)
    client.auth = SimpleNamespace(read_json_response=lambda response: {"status": 0, "data": []})
    client.config = SimpleNamespace(debug_dump_all=False)
    image_service = ImageService(client)
    closed: list[bool] = []

    response = SimpleNamespace(
        headers={"Content-Type": "application/json"},
        close=lambda: closed.append(True),
    )

    assert image_service._prepare_response(response) == {"status": 0, "data": []}
    assert closed == [True]


def test_prepare_image_response_rejects_numeric_error_and_closes_source():
    client = object.__new__(GLMWebClient)
    client.auth = SimpleNamespace(read_json_response=lambda response: {"status": 13, "message": "失败"})
    client.config = SimpleNamespace(debug_dump_all=False)
    image_service = ImageService(client)
    closed: list[bool] = []
    response = SimpleNamespace(
        headers={"Content-Type": "application/json"},
        close=lambda: closed.append(True),
    )

    with pytest.raises(UpstreamAPIError):
        image_service._prepare_response(response)
    assert closed == [True]


def test_download_image_as_base64_rejects_unbounded_response(monkeypatch):
    client = object.__new__(GLMWebClient)
    client.config = SimpleNamespace(request_timeout=1)

    class FakeResponse:
        headers = {"Content-Type": "image/png"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, size=-1):
            return b"123456"

    class FakeOpener:
        def open(self, url, timeout):
            return FakeResponse()

    image_service = ImageService(client)
    monkeypatch.setattr(images_module, "IMAGE_DOWNLOAD_MAX_BYTES", 5)
    monkeypatch.setattr(
        images_module.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    monkeypatch.setattr(images_module.urllib.request, "build_opener", lambda *args: FakeOpener())

    with pytest.raises(UpstreamAPIError, match="超过允许的大小"):
        image_service.download_as_base64("https://cdn.example/large.png")


def test_download_image_as_base64_rejects_non_http_url():
    client = object.__new__(GLMWebClient)
    client.config = SimpleNamespace(request_timeout=1)
    image_service = ImageService(client)

    with pytest.raises(UpstreamAPIError, match="协议不受支持"):
        image_service.download_as_base64("file:///etc/passwd")


def test_download_image_as_base64_rejects_url_credentials():
    client = object.__new__(GLMWebClient)
    client.config = SimpleNamespace(request_timeout=1)
    image_service = ImageService(client)

    with pytest.raises(UpstreamAPIError, match="协议不受支持"):
        image_service.download_as_base64("https://user:password@cdn.example/image.png")


def test_download_image_as_base64_rejects_private_dns_host(monkeypatch):
    client = object.__new__(GLMWebClient)
    client.config = SimpleNamespace(request_timeout=1)
    image_service = ImageService(client)

    monkeypatch.setattr(
        images_module.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("10.0.0.8", 0))],
    )

    with pytest.raises(UpstreamAPIError, match="非公网"):
        image_service.download_as_base64("https://cdn.example/private.png")


def test_download_image_as_base64_allows_official_chatglm_image_host(monkeypatch):
    client = object.__new__(GLMWebClient)
    client.config = SimpleNamespace(request_timeout=1)
    image_service = ImageService(client)

    class FakeResponse:
        headers = {"Content-Type": "image/jpeg"}
        chunks = [b"jpeg-data", b""]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, size=-1):
            return self.chunks.pop(0)

        def geturl(self):
            return "https://sfile.chatglm.cn/testpath/image.jpg"

    class FakeOpener:
        def open(self, url, timeout):
            assert url == "https://sfile.chatglm.cn/testpath/image.jpg"
            return FakeResponse()

    monkeypatch.setattr(
        images_module.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("198.18.0.200", 0))],
    )
    monkeypatch.setattr(images_module.urllib.request, "build_opener", lambda *args: FakeOpener())

    assert image_service.download_as_base64("https://sfile.chatglm.cn/testpath/image.jpg")


def test_image_redirect_handler_rejects_private_destination(monkeypatch):
    monkeypatch.setattr(
        images_module.socket,
        "getaddrinfo",
        lambda host, *args, **kwargs: (
            [(2, 1, 6, "", ("93.184.216.34", 0))]
            if host == "cdn.example"
            else [(2, 1, 6, "", ("10.0.0.8", 0))]
        ),
    )
    handler = images_module._PublicImageRedirectHandler()
    request = images_module.urllib.request.Request("https://cdn.example/image.png")

    with pytest.raises(UpstreamAPIError, match="非公网"):
        handler.redirect_request(
            request,
            SimpleNamespace(),
            302,
            "Found",
            {},
            "http://internal.example/image.png",
        )
