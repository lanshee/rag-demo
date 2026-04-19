# 机试

这是一个基于 **FastAPI + Milvus + Ollama** 构建的简易 RAG（针对pdf）系统。

---

## 1. PDF 解析方案说明
为了应对工业场景中常见的扫描件和高质量原生文档，系统采用了**双模智能解析流**：

* **文本层提取 (PyMuPDF):** 优先通过 `fitz` 库提取 PDF 内置文本。若检测到有效字符数 > 100，系统将直接使用该文本流，秒级完成解析。
* **OCR 兜底识别 (EasyOCR):** 针对扫描版或图片类 PDF，系统自动启用 OCR 流程：
    * **处理:** 使用 `pdf2image` 将页面转为 **250 DPI** 的高清图像。
    * **模型:** 采用 `ch_sim` (简体中文) 与 `en` (英文) 模型，并按 Y 轴坐标排序以保证阅读顺序逻辑正确。
    * **资源释放:** 识别完成后，系统会主动调用 `gc.collect()` 和 `torch.cuda.empty_cache()` 释放显存，确保不与后续的 LLM 推理争抢 GPU 资源。
* **缓存机制:** 基于文件元数据（路径、大小、修改时间）生成 MD5 哈希缓存，结果存储在 `/app/data/.ocr_cache`，大幅缩短二次加载时间。

---

## 2. 知识库构建流程
系统通过以下链路完成从原始文档到语义向量的转化：

* **分块策略 (Chunking):** 使用 `RecursiveCharacterTextSplitter`，设置 `chunk_size=500` 字，`chunk_overlap=80` 字，在保持语义连贯性的同时，适应 Embedding 模型的上下文长度。
* **Embedding 模型:** 选用 **BAAI/bge-m3** (通过 Ollama 加载)。该模型向量维度为 **1024**，是目前开源社区对中文语义理解最强的模型之一。
* **向量数据库选型:** 选用 **Milvus v2.6.5 (Standalone)**。
    * 使用 `IVF_FLAT` 索引搭配 `L2` 距离度量，确保在大规模文档下的检索性能。
    * 通过 **Attu** 提供可视化管理界面。
* **混合检索 (Hybrid Search):** 结合了 **jieba 分词关键词过滤**。
    * 检索公式：`Score = Vector_Dist * 0.6 + Keyword_Match * 0.4`。
    * 这种策略极大提升了针对“GBT 1568”等特定技术编号或工业专有名词的召回准确率。

---

## 3. 如何构建和运行 Docker 容器

### 准备工作
确保宿主机已安装 Docker 和 Docker Compose。

### 部署步骤
1.  **启动基础设施:**
    ```bash
    docker-compose up -d
    ```
2.  **下载模型 (关键步骤):**
    由于 Ollama 容器启动后默认是空的，需手动执行拉取命令：
    ```bash
    # 拉取推理模型
    docker exec -it ollama-service ollama pull qwen2:7b
    # 拉取嵌入模型
    docker exec -it ollama-service ollama pull bge-m3
    ```

### 服务端口说明
* **API 接口:** `http://localhost:8000`
* **API 文档:** `http://localhost:8000/docs`
* **Milvus 可视化 (Attu):** `http://localhost:3000`

---

## 4. API 调用示例

### 智能对话 (/chat)
系统支持多轮对话记忆，通过 `conversation_id` 隔离不同用户的会话。

```bash
curl -X 'POST' \
  'http://localhost:8000/chat' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "GBT 1568-2008 技术条件中对于键的硬度要求是什么？",
    "conversation_id": "user_01"
  }'
```

### 清除会话记忆
```bash
curl -X 'DELETE' 'http://localhost:8000/conversation/standard_user_01'
```

## 5. 实际完成用时
本项目从环境搭建到核心逻辑调试，共耗时约 3 小时。

## 6. 已知问题或待改进项
并发加载压力: 目前 Ollama 在同时加载 Embedding 和 LLM 模型时，对显存（建议 8G+）有一定要求。

复杂表格解析: 当前 OCR 对 PDF 中的嵌套表格识别率有待提升。

动态更新: 知识库目前在服务启动时初始化，后续需支持动态上传 PDF 并实时增量更新向量库。