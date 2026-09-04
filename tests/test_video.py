from __future__ import annotations

import json
import threading
import time
import urllib.request
from types import SimpleNamespace

import pytest

from glm2api.api import server as server_module
from glm2api.glm.client import GLMWebClient, QueueLease
from glm2api.glm.errors import UpstreamAPIError
from glm2api.media.videos import (
    NormalizedVideoRequest,
    VideoService,
    VideoJobStore,
    VideoJob,
    build_upstream_video_payload,
    new_video_job,
    normalize_video_request,
)
from glm2api.media import videos as videos_module
from glm2api.media.images import ImageGenerationRequest, ImageService


def test_text_video_payload_matches_chatglm_web_shape():
    request = normalize_video_request(
        {
            "model": "glm-video-1",
            "prompt": "一只猫在窗边看雨",
            "seconds": "5",
            "size": "1280x720",
        }
    )

    payload = build_upstream_video_payload(request)

    assert payload["prompt"] == "一只猫在窗边看雨"
    assert payload["conversation_id"] == ""
    assert payload["advanced_parameter_extra"] == {}
    assert "source_list" not in payload
    assert payload["base_parameter_extra"] == {
        "generation_pattern": 1,
        "resolution": 2,
        "fps": 0,
        "duration": 1,
        "generation_ai_audio": 1,
        "generation_ratio_width": 16,
        "generation_ratio_height": 9,
        "activity_type": 0,
        "label_watermark": 1,
        "generation_type": "",
        "prompt": "一只猫在窗边看雨",
    }


def test_video_request_rejects_unknown_model():
    with pytest.raises(ValueError, match="当前视频接口不支持模型 glm-video-1-search"):
        normalize_video_request({"model": "glm-video-1-search", "prompt": "测试"})


def test_image_video_payload_uses_source_list_and_forces_five_seconds_for_two_sources():
    request = normalize_video_request({"prompt": "让画面动起来", "seconds": "10", "size": "720x1280"})

    payload = build_upstream_video_payload(request, source_ids=["source_a", "source_b"])

    assert payload["source_list"] == ["source_a", "source_b"]
    assert "advanced_parameter_extra" not in payload
    assert payload["base_parameter_extra"]["duration"] == 1
    assert payload["base_parameter_extra"]["generation_type"] == "reference_img"
    assert payload["base_parameter_extra"]["generation_ratio_width"] == 9
    assert payload["base_parameter_extra"]["generation_ratio_height"] == 16


def test_video_request_accepts_multipart_style_reference_file():
    request = normalize_video_request(
        {
            "prompt": "平滑移动",
            "_input_reference_file": {
                "filename": "input.png",
                "mime_type": "image/png",
                "content": b"png-bytes",
            },
        }
    )

    assert request.input_reference is not None
    assert request.input_reference.filename == "input.png"
    assert request.input_reference.content == b"png-bytes"


def test_video_reference_upload_enforces_media_size_limit(monkeypatch):
    client = object.__new__(GLMWebClient)
    service = VideoService(client)
    monkeypatch.setattr(videos_module, "VIDEO_MAX_UPLOAD_BYTES", 1)

    reference = videos_module.VideoInputReference(
        filename="input.png",
        mime_type="image/png",
        content=b"too-large",
    )

    with pytest.raises(ValueError, match="超过 100MB"):
        service._upload_video_reference(reference, fallback_ratio=(16, 9), preferred_account_index=None)


def test_video_state_extracts_completed_output_url():
    client = object.__new__(GLMWebClient)
    video_service = VideoService(client)

    state = video_service._extract_video_state(
        {
            "result": {
                "status": "success",
                "progress": 100,
                "output": {"file_list": [{"url": "https://example.test/video.mp4"}]},
            }
        }
    )

    assert state == {
        "completed": True,
        "failed": False,
        "progress": 100,
        "error": None,
        "video_url": "https://example.test/video.mp4",
    }


def test_video_result_url_is_limited_to_chatglm_hosts():
    client = object.__new__(GLMWebClient)
    client.config = SimpleNamespace(glm_base_url="https://chatglm.cn/chatglm")
    service = VideoService(client)

    assert service._normalize_video_url("/chat/video/result.mp4") == "https://chatglm.cn/chat/video/result.mp4"
    assert service._normalize_video_url("https://sfile.chatglm.cn/video/result.mp4") == "https://sfile.chatglm.cn/video/result.mp4"
    with pytest.raises(UpstreamAPIError, match="不受信任"):
        service._normalize_video_url("https://127.0.0.1/secret.mp4")


def test_video_content_redirect_handler_rejects_untrusted_target():
    client = object.__new__(GLMWebClient)
    client.config = SimpleNamespace(glm_base_url="https://chatglm.cn/chatglm")
    service = VideoService(client)
    handler = videos_module._TrustedVideoRedirectHandler(service._normalize_video_url)
    request = urllib.request.Request("https://chatglm.cn/video/result.mp4")

    with pytest.raises(UpstreamAPIError, match="不受信任"):
        handler.redirect_request(
            request,
            SimpleNamespace(),
            302,
            "Found",
            {},
            "https://evil.example/video.mp4",
        )


def test_video_job_store_prunes_old_terminal_jobs():
    store = VideoJobStore(max_jobs=1, retention_seconds=10)
    old_job = VideoJob(
        id="video_old",
        model="glm-video-1",
        prompt="old",
        seconds="5",
        size="1280x720",
        created_at=1,
        status="failed",
    )
    current_job = VideoJob(
        id="video_current",
        model="glm-video-1",
        prompt="current",
        seconds="5",
        size="1280x720",
        created_at=int(time.time()),
    )
    store.add(old_job)
    store.add(current_job)

    assert [job.id for job in store.list()] == ["video_current"]


def test_image_request_includes_current_chatglm_selected_model(monkeypatch):
    client = object.__new__(GLMWebClient)
    client.config = SimpleNamespace(
        glm_image_assistant_id="image-assistant",
        chat_stream_url="https://chatglm.test/stream",
        request_timeout=1,
        debug_dump_all=False,
    )
    client.logger = SimpleNamespace(info=lambda *args, **kwargs: None, debug=lambda *args, **kwargs: None)
    client.auth = SimpleNamespace(
        get_browser_headers=lambda: {"Content-Type": "application/json"},
    )
    image_service = ImageService(client)
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        return object()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    image_service._prepare_response = lambda response: response
    client.call_with_account_failover = lambda name, operation, preferred_account_index=None: operation(0, "access")

    image_service._open_stream(ImageGenerationRequest(model="glm-image-1", prompt="画一只猫"))

    body = json.loads(captured["request"].data)
    assert body["meta_data"]["selected_model"] == "glm-5.3-flash"


def test_video_worker_polls_and_persists_content_url_without_network():
    client = object.__new__(GLMWebClient)
    client.config = SimpleNamespace(
        glm_base_url="https://chatglm.cn/chatglm",
        video_chat_url="https://chatglm.cn/chatglm/video-api/v1/chat",
        video_api_base_url="https://chatglm.cn/chatglm/video-api/v1",
    )
    client.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    from glm2api.media.videos import VideoService

    client.videos = VideoService(client)
    client.videos.jobs = VideoJobStore()
    client.get_preferred_account_index = lambda ticket: 0
    calls = []

    def fake_video_json_request(method, url, payload, preferred_account_index, request_name):
        calls.append(request_name)
        if request_name == "video_create":
            return {"result": {"chat_id": "upstream-test"}}, 0
        if request_name == "video_status":
            return {"result": {"status": "success", "progress": 100}}, 0
        return {"result": {"output": {"file_list": [{"url": "/chat/video/test.mp4"}]} }}, 0

    client.videos._video_json_request = fake_video_json_request
    request = normalize_video_request({"prompt": "hello"})
    job = new_video_job(request)
    released = []
    lease = QueueLease(ticket=0, release_callback=lambda ticket: released.append(ticket))

    client.videos._run_video_job(job, request, lease)

    snapshot = job.snapshot()
    assert calls == ["video_create", "video_status", "video_detail"]
    assert snapshot["status"] == "completed"
    assert snapshot["progress"] == 100
    assert released == [0]
    assert job.content_url == "https://chatglm.cn/chat/video/test.mp4"


def test_video_http_endpoint_accepts_json_and_returns_task():
    class FakeGLM:
        def __init__(self):
            self.payload = None
            self.payloads = []

        def create_video(self, payload):
            self.payload = payload
            self.payloads.append(payload)
            return {"id": "video_test", "object": "video", "status": "queued", "progress": 0}

        def list_videos(self):
            return {"object": "list", "data": []}

        def retrieve_video(self, video_id):
            return {"id": video_id, "object": "video", "status": "queued", "progress": 0}

        def open_video_content(self, video_id):
            raise AssertionError(video_id)

    class FakeLogger:
        def debug(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    config = SimpleNamespace(
        host="127.0.0.1",
        port=0,
        api_prefix="/v1",
        cors_allow_origin="*",
        server_api_keys=[],
        debug_dump_all=False,
    )
    fake_glm = FakeGLM()
    http_server = server_module.GLM2APIServer(config, fake_glm, FakeLogger())
    thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    thread.start()
    port = http_server._server.server_address[1]
    try:
        body = json.dumps({"prompt": "hello", "seconds": "5"}).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/videos",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            created = json.loads(response.read())

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/videos/video_test", timeout=5) as response:
            retrieved = json.loads(response.read())

        boundary = "----glm2api-test"
        multipart_body = (
            f"--{boundary}\r\n"
            "Content-Disposition: form-data; name=\"prompt\"\r\n\r\n"
            "animate\r\n"
            f"--{boundary}\r\n"
            "Content-Disposition: form-data; name=\"input_reference\"; filename=\"input.png\"\r\n"
            "Content-Type: image/png\r\n\r\n"
        ).encode() + b"png-bytes\r\n" + f"--{boundary}--\r\n".encode()
        multipart_request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/videos",
            data=multipart_body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(multipart_request, timeout=5) as response:
            multipart_created = json.loads(response.read())
    finally:
        http_server.shutdown()
        thread.join(timeout=1)

    assert isinstance(fake_glm.payloads[0], NormalizedVideoRequest)
    assert fake_glm.payloads[0].prompt == "hello"
    assert fake_glm.payloads[0].seconds == "5"
    assert created["id"] == "video_test"
    assert retrieved["status"] == "queued"
    assert multipart_created["id"] == "video_test"
    assert isinstance(fake_glm.payloads[1], NormalizedVideoRequest)
    assert fake_glm.payloads[1].prompt == "animate"
    assert fake_glm.payloads[1].input_reference is not None
    assert fake_glm.payloads[1].input_reference.content == b"png-bytes"
