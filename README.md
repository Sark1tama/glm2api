# glm2api

`glm2api` 是一个本地协议代理：它把 ChatGLM 网页端的私有接口转换成可供常用客户端使用的兼容 API。项目面向本地部署，主线支持 OpenAI Chat Completions、OpenAI Responses 和 Anthropic Messages；图片、视频属于次级兼容能力。

## 功能概览

- 文本：`/v1/chat/completions`、`/v1/responses`、`/v1/messages`
- Anthropic 估算：`/v1/messages/count_tokens`
- 图片（次级）：`/v1/images/generations`
- 视频（次级）：`/v1/videos`、视频查询和 `/content` 下载
- 模型列表：`/v1/models`
- 多账号轮换、游客模式、请求队列和上游失败重试
- 工具调用、图片/文件输入和保守的 token usage 估算

上游是 ChatGLM 网页端的私有协议；接口字段或行为变化时，需要同步调整兼容层并重新验证。

## 快速开始

要求 Python 3.14+ 和 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/Sark1tama/glm2api.git
cd glm2api
cp .env.example .env
uv sync
uv run python main.py
```

将登录 ChatGLM 后取得的 `refresh_token` 写入 `.env`：

```env
GLM_REFRESH_TOKEN=你的_refresh_token
```

也可以不登录而使用游客模式：

```env
GLM_USE_GUEST_REFRESH_TOKEN=true
```

启动后检查：

```bash
curl http://127.0.0.1:8000/health
```

## Docker

Compose 默认把宿主机 `18080` 映射到容器 `8000`，并要求外部 Docker 网络 `shared-net`：

```bash
docker network create shared-net  # 仅首次需要
cp .env.example .env
docker compose up --build -d
curl http://127.0.0.1:18080/health
```

Docker 单账号直接在 `.env` 配置 `GLM_REFRESH_TOKEN`。需要多账号时，先在项目根目录创建 `token.txt`，再使用可选覆盖文件挂载：

```bash
touch token.txt
docker compose -f docker-compose.yml -f docker-compose.tokens.yml up --build -d
```

可用 `GLM2API_HOST_PORT` 修改宿主机端口，例如：

```bash
GLM2API_HOST_PORT=18081 docker compose up -d
```

容器网络内的服务通过 `http://glm2api:8000` 访问。

## 常用配置

配置文件为 `.env`；如果不存在，程序会从 `.env.example` 自动创建。

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `HOST` | `127.0.0.1` | 本地监听地址 |
| `PORT` | `8000` | 非 Docker 启动端口 |
| `API_PREFIX` | `/v1` | API 路径前缀 |
| `GLM_TOKEN_FILE` | `token.txt` | 多账号 token 文件，每行一个 token |
| `GLM_MAX_CONCURRENCY` | `3` | 上游并发槽位数量 |
| `MAX_REQUEST_BODY_BYTES` | `105906176` | 所有 HTTP 请求体上限（101 MiB） |
| `GLM_DELETE_CONVERSATION` | `true` | 请求结束后删除网页会话 |
| `SERVER_API_KEYS` | 空 | 本地 Bearer/x-api-key 认证，逗号分隔 |
| `LOG_LEVEL` | `INFO` | `DEBUG` 会写入 `log/glm2api_debug.log` |
| `DEBUG_DUMP_ALL` | `false` | 打印完整入站、上游和出站 payload；调试后应关闭 |

游客模式会按 `GLM_MAX_CONCURRENCY` 创建游客账号槽位。配置 `token.txt` 时，程序可在上游返回新 token 后自动写回对应行。

## 接口示例

### Chat Completions

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"你好"}]}'
```

流式请求只需增加 `"stream":true`。OpenAI Responses 和 Anthropic Messages 使用各自原生 JSON/SSE 格式：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
response = client.responses.create(
    model="glm-5.3-flash",
    input="介绍一下你自己",
)
print(response.output_text)
```

### 图片生成

```bash
curl http://127.0.0.1:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-image-1","prompt":"一只橘猫","size":"1024x1024"}'
```

支持 `url` 和 `b64_json` 两种 `response_format`，`n` 为 1 到 10。

### 视频生成

视频创建是异步的：

```bash
curl http://127.0.0.1:8000/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-video-1","prompt":"一只猫在窗边看雨","seconds":"5","size":"1280x720"}'

curl http://127.0.0.1:8000/v1/videos/video_xxx
curl http://127.0.0.1:8000/v1/videos/video_xxx/content -o result.mp4
```

也可以用 JSON 的 `input_reference.image_url` 或 multipart 文件进行图生视频。任务状态只保存在当前进程内，服务重启后不会恢复。

## 模型和输入边界

- `glm-5.3`：只接受文本输入。
- `glm-5.3-flash`：文本聊天支持图片和文件引用。
- `glm-image-1`：只用于 `/v1/images/generations`。
- `glm-video-1`：只用于 `/v1/videos`。

聊天图片和文件会先上传到 ChatGLM 网页端。Anthropic `document`、Responses `input_file.file_data/file_url` 可转换；外部 `file_id` 没有本地文件资源映射，会返回 400。

ChatGLM 网页协议没有通用的 `temperature`、`top_p`、`stop` 或 `response_format` 字段；这些参数会明确返回 400。Anthropic 的 `max_tokens`、Chat Completions 的 `max_tokens`/`max_completion_tokens` 以及 Responses 的 `max_output_tokens` 会映射到统一的本地输出 token 预算，并使用保守估算限制返回内容；该限制不会减少 GLM 上游已经开始的生成计算。

网页 SSE 当前不提供 token 统计，因此响应中的 `usage` 是基于原始请求和转换后 prompt 的保守估算，不代表计费精度；若上游将来返回统计值，则优先使用上游字段。

## 工具调用

公共协议中的工具定义先进入内部工具对象，再由 GLM 文本桥接层构造成网页端可识别的 DSML。上游返回的 DSML、旧 XML 变体和流式片段会被解析回标准工具调用。工具名称、参数名和 `tool_choice` 会在转换前校验；网页端内置的浏览器工具不会自动暴露给客户端。

## 项目结构

```text
src/glm2api/
├── api/server.py                     HTTP 路由和响应写回
├── api/{errors,sse}.py               错误映射和 SSE 写出
├── api/adapters/                     文本公共协议边界转换
├── core/models.py                    内部请求、结果和流事件
├── core/usage.py                     usage 来源追踪和保守估算
├── glm/tools/{dsml,parser}.py        GLM 工具协议序列化和解析
├── glm/{chat,translator,events}.py   GLM 模型映射、prompt 和 SSE 事件
├── glm/{client,auth,files}.py        上游 HTTP、鉴权、队列和附件上传
├── media/{images,videos}.py          图片和视频垂直切片
└── infrastructure/logging.py         日志与调试输出
```

开发命令：

```bash
uv run pytest                 # 完整测试
uv run pytest tests/test_tool_parser.py -q
uv build                      # 构建发行包
```

测试不需要真实 token 或网络，应使用 mock 上游响应。

## 安全提示

不要提交 `.env`、`token.txt`、refresh token、API key 或调试日志。`DEBUG_DUMP_ALL=true` 会记录完整请求和响应，可能包含敏感内容；仅在本地排查问题时短暂开启。

远程图片/文档引用只接受公网 HTTP(S)，data URL 必须是 base64 且不超过 100 MiB；HTTP 请求体默认上限为 101 MiB（可用 `MAX_REQUEST_BODY_BYTES` 调整）。调试日志会脱敏常见认证字段，但仍不应在生产环境长期开启完整 payload 记录。

项目许可证见 [`LICENSE`](LICENSE)。
