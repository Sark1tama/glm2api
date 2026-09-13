# glm2api

`glm2api` 是一个本地协议代理：它把 ChatGLM 网页端接口转换成可供常用客户端使用的兼容 API。项目面向本地部署，主线支持 OpenAI Chat Completions、OpenAI Responses 和 Anthropic Messages；图片、视频属于次级兼容能力。

> [!IMPORTANT]
> 本项目仅供学习交流与个人研究使用，与上游服务提供方无关联。请自行确保使用行为符合相关服务的条款与当地法律法规，使用风险由使用者自行承担。

## 功能概览

- 文本：`/v1/chat/completions`、`/v1/responses`、`/v1/messages`
- Anthropic 估算：`/v1/messages/count_tokens`
- 图片（次级）：`/v1/images/generations`
- 视频（次级）：`/v1/videos`、视频查询和 `/content` 下载
- 模型列表：`/v1/models`
- 请求队列、上游失败重试和保守的 token usage 估算
- 工具调用、图片/文件输入

上游是 ChatGLM 网页端协议，非官方开放 API；接口字段或行为变化时，需要同步调整兼容层并重新验证。

## 快速开始

要求 Python 3.14+ 和 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/Sark1tama/glm2api.git
cd glm2api
cp .env.example .env
uv sync
uv run python main.py
```

在 `.env` 中配置你本人账号的 `refresh_token`：

```env
GLM_REFRESH_TOKEN=你的_refresh_token
```

也可以使用游客模式（无需配置账号）：

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

流式请求只需增加 `"stream":true`。声明客户端工具时，推理和经工具解析器过滤后的正文也会增量发送；工具调用前的正文可以与工具调用共存，并共同计入输出预算；完整 DSML 工具块作为本轮交接点，其后正文不再返回，代理停止读取后续上游事件。沙箱替代重试仅在尚未发送内容时执行，已发送内容后遇到错误会直接失败收尾。OpenAI Responses 和 Anthropic Messages 使用各自原生 JSON/SSE 格式；Responses 会把 GLM 推理作为独立的 `reasoning` output item 返回。上游流式错误会按协议终止：Chat Completions 返回错误事件和 `[DONE]`，Responses 追加 `response.failed`，Anthropic 返回 `error` 事件后关闭流：

Chat Completions 如需使用 GLM 远端联网，可显式传入代理扩展字段 `"web_search":true`，或在 `tools` 中声明 `type` 以 `web_search` 开头的工具；普通 `function` 工具（即使名称为 `web_search`）仍按客户端工具处理。

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

聊天图片和文件会先上传到 ChatGLM 网页端。Anthropic 普通消息及 `tool_result.content` 中的 `document`、Responses `input_file.file_data/file_url` 可转换；外部 `file_id` 没有本地文件资源映射，会返回 400。

Anthropic Messages 的 `messages[].role` 支持 `user`、`assistant` 和 `system`。中途 `system` 消息会按原始位置转换，当前支持字符串或 `text` blocks；Anthropic 工具结果应放在 `user` 消息的 `tool_result` block 中，而不是使用 `role: "tool"`。消息级 `clear_at`、`output_config` 以及 `tool_addition`/`tool_removal` 尚无等价的 GLM 映射，会明确返回 400。

ChatGLM 网页协议没有通用的 `temperature`、`top_p`、停止序列或结构化输出字段；合法的 `temperature` 和 `top_p` 会被接受但忽略，并记录警告。OpenAI Chat Completions 的 `stop` 和 Anthropic Messages 的 `stop_sequences` 由代理在本地流式匹配，命中内容及其后续输出不会返回。Chat Completions 的 `response_format`、Responses 的 `text.format` 和 Anthropic 的 `output_config.format` 会转换为统一的内部结构化输出约束，并通过后置提示尽力要求 GLM 返回 JSON；由于上游没有原生 JSON Schema 解码器，`strict: true` 不代表可达到原厂级严格保证。Anthropic 的 `thinking.type="adaptive"` 与顶层 `output_config.effort` 会映射为 GLM 思考模式；未指定 effort 的 adaptive thinking 按 Anthropic 默认值 `high` 处理，GLM 推理会以 Anthropic `thinking` block 返回，其 `signature` 是仅用于本代理多轮往返的兼容标识，不是 Anthropic 原厂签名。Anthropic 的 `max_tokens`、Chat Completions 的 `max_tokens`/`max_completion_tokens` 以及 Responses 的 `max_output_tokens` 会映射到统一的本地输出 token 预算，并使用保守估算限制返回内容；这些本地限制不会减少 GLM 在代理检测生效前已经生成的少量内容。

网页 SSE 当前不提供 token 统计，因此响应中的 `usage` 是基于原始请求和转换后 prompt 的保守估算，不代表计费精度；若上游将来返回统计值，则优先使用上游字段。

Anthropic 的 `thinking.display="omitted"` 可与 `enabled` 或 `adaptive` 一起使用：保留推理过程、usage 与输出预算，只隐藏返回的推理文字。流式不发送 `thinking_delta`，有推理时仍返回空 thinking 块和兼容 signature。该签名无法恢复隐藏推理；多轮依靠调用方传回的正文和工具历史，不等价于 Anthropic 原厂的加密推理恢复，也不保证降低 GLM 生成延迟。

## 工具调用

各文本接口均接受 `tools: []`，与不提供工具定义等价；显式要求调用工具时仍需提供可用工具。

OpenAI 的 `parallel_tool_calls=false` 与 Anthropic 的 `tool_choice.disable_parallel_tool_use=true` 会限制每轮最多交付一个客户端工具调用：提示词要求串行调用，若模型仍生成多个调用，代理只返回第一个完整调用。停止序列仅匹配 DSML 解析后的正文，不匹配工具名称或参数。Responses 返回的工具定义、选择策略和并行设置来自本次内部请求。

Responses 暂不保存可供 `previous_response_id` 引用的历史；传入非 null 的该字段会返回 400，请每轮在 `input` 中提供完整对话历史。

公共协议中的工具定义先进入内部工具对象，再由 GLM 文本桥接层构造成网页端可识别的 DSML。客户端工具优先通过正文 DSML 解析；如果上游平台错误返回了本次请求声明的 `client__` 别名结构化调用，代理也会恢复为客户端工具调用。隐藏推理中的示例或草稿不会执行。工具名称、参数名和 `tool_choice` 会在转换前校验；网页端内置的浏览器工具不会自动暴露给客户端。

## 项目结构

```text
src/glm2api/
├── {app,__main__}.py                 应用装配和启动入口
├── api/server.py                     HTTP 路由和响应写回
├── api/{errors,sse}.py               错误映射和 SSE 写出
├── api/adapters/                     文本公共协议边界转换
├── core/models.py                    内部请求、结果和流事件
├── core/output_budget.py             本地输出 token 预算
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

项目许可证见 [`LICENSE`](LICENSE)。
