from types import SimpleNamespace

import pytest

from glm2api.api.adapters.anthropic.messages import anthropic_messages_to_internal
from glm2api.glm.client import GLMWebClient, UpstreamAPIError
from glm2api.glm.files import FileService, _build_uploaded_reference
from glm2api.glm.translator import messages_to_glm_payload


def test_uploaded_image_reference_uses_file_id_and_returned_url():
    reference = _build_uploaded_reference(
        {"file_id": "file_123", "file_url": "https://cdn.example/image.png"},
        "https://source.example/image.png",
        is_image=True,
    )

    assert reference == {
        "type": "image_url",
        "image_url": {"url": "https://cdn.example/image.png"},
    }


def test_uploaded_file_reference_maps_file_id_to_source_id():
    reference = _build_uploaded_reference(
        {"file_id": "file_123", "file_url": "https://cdn.example/document.pdf"},
        "https://source.example/document.pdf",
        is_image=False,
    )

    assert reference == {
        "type": "file",
        "file": [{"source_id": "file_123", "file_url": "https://cdn.example/document.pdf"}],
    }


def test_upload_failure_is_not_silently_replaced_with_text_placeholder(monkeypatch):
    client = GLMWebClient.__new__(GLMWebClient)
    file_service = FileService(client)

    def fail_fetch(_service, _file_url):
        raise ValueError("invalid data URL")

    monkeypatch.setattr(FileService, "fetch_file_payload", fail_fetch)

    with pytest.raises(ValueError, match="invalid data URL"):
        file_service._upload_file_reference("data:application/pdf;base64,invalid", is_image=False)


def test_fetch_file_payload_accepts_valid_bounded_data_url():
    client = GLMWebClient.__new__(GLMWebClient)
    client.config = SimpleNamespace(request_timeout=1)
    filename, mime_type, payload = FileService(client).fetch_file_payload("data:text/plain;base64,SGk=")

    assert filename.endswith(".txt")
    assert mime_type == "text/plain"
    assert payload == b"Hi"


@pytest.mark.parametrize(
    "file_url",
    [
        "file:///etc/passwd",
        "http://127.0.0.1/secret",
        "data:text/plain;base64,not-valid-base64",
        "data:text/plain,plain-text",
    ],
)
def test_fetch_file_payload_rejects_unsafe_or_invalid_urls(file_url):
    client = GLMWebClient.__new__(GLMWebClient)
    client.config = SimpleNamespace(request_timeout=1)

    with pytest.raises(ValueError):
        FileService(client).fetch_file_payload(file_url)


def test_fetch_file_payload_enforces_data_url_decoded_size(monkeypatch):
    client = GLMWebClient.__new__(GLMWebClient)
    client.config = SimpleNamespace(request_timeout=1)
    monkeypatch.setattr("glm2api.glm.files.FILE_SIZE_LIMIT", 2)

    with pytest.raises(ValueError, match="超过 100MB"):
        FileService(client).fetch_file_payload("data:application/octet-stream;base64,YWJj")


def test_upstream_upload_failure_is_exposed_as_bad_gateway(monkeypatch):
    client = GLMWebClient.__new__(GLMWebClient)
    file_service = FileService(client)
    monkeypatch.setattr(
        FileService,
        "fetch_file_payload",
        lambda _service, _file_url: ("document.pdf", "application/pdf", b"pdf"),
    )
    monkeypatch.setattr(FileService, "_build_multipart", lambda *_args: b"multipart")

    def fail_upload(*_args, **_kwargs):
        raise RuntimeError("upstream unavailable")

    client.call_with_account_failover = fail_upload
    client.logger = type("Logger", (), {"warning": lambda *_args, **_kwargs: None})()

    with pytest.raises(UpstreamAPIError, match="GLM 附件上传失败"):
        file_service._upload_file_reference("https://source.example/document.pdf", is_image=False)


def test_uploaded_reference_keeps_legacy_source_id_response():
    reference = _build_uploaded_reference(
        {"source_id": "source_123"},
        "https://source.example/document.pdf",
        is_image=False,
    )

    assert reference == {
        "type": "file",
        "file": [{"source_id": "source_123", "file_url": "https://source.example/document.pdf"}],
    }


@pytest.mark.parametrize("result", [None, {}, {"file_url": "https://cdn.example/file"}, "invalid"])
def test_uploaded_file_reference_requires_an_uploaded_id(result):
    assert _build_uploaded_reference(result, "https://source.example/file", is_image=False) is None


def test_anthropic_document_reuses_chatglm_file_upload_path(monkeypatch):
    request = anthropic_messages_to_internal(
        {
            "model": "glm-5.3-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "AQI=",
                            },
                        }
                    ],
                }
            ],
        }
    )
    client = GLMWebClient.__new__(GLMWebClient)
    client.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    file_service = FileService(client)
    uploaded = []
    monkeypatch.setattr(
        FileService,
        "_upload_file_reference",
        lambda _service, url, is_image: uploaded.append((url, is_image))
        or {
            "type": "file",
            "file": [{"source_id": "file_123", "file_url": "https://cdn.example/report.pdf"}],
        },
    )

    refs = file_service.upload_referenced_files(messages_to_glm_payload(request.messages))

    assert uploaded == [("data:application/pdf;base64,AQI=", False)]
    assert refs == [
        {
            "type": "file",
            "file": [{"source_id": "file_123", "file_url": "https://cdn.example/report.pdf"}],
        }
    ]
