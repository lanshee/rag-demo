import os
import logging
import jieba

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from langchain_ollama import OllamaEmbeddings, ChatOllama   # ✅ LLM 也走 Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.pdf_parser import PDFParserOCR
from src.conversation_memory import ConversationMemory

logger = logging.getLogger(__name__)

VECTOR_DIM = 1024
COLLECTION_NAME = "knowledge_base_standard"


class RAGEngine:
    def __init__(self, pdf_path: str) -> None:
        self.pdf_path = pdf_path
        self.memory = ConversationMemory(window_size=5)

        self.milvus_host: str = os.getenv("MILVUS_HOST", "milvus")
        self.milvus_port: str = os.getenv("MILVUS_PORT", "19530")
        ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

        self.embeddings = OllamaEmbeddings(
            base_url=ollama_base_url,
            model=os.getenv("EMBEDDING_MODEL", "bge-m3"),
        )

        self.llm = ChatOllama(
            base_url=ollama_base_url,
            model=os.getenv("LLM_MODEL", "qwen2:7b"),
            temperature=0.1,
        )

        connections.connect(
            alias="default",
            host=self.milvus_host,
            port=self.milvus_port,
        )
        logger.info(f"Milvus 连接成功: {self.milvus_host}:{self.milvus_port}")

        self._verify_ollama()
        self.col: Collection = self._init_store()

    # 验证 Ollama
    def _verify_ollama(self) -> None:
        logger.info("验证 Ollama embedding 服务...")
        try:
            vec = self.embeddings.embed_query("测试")
            logger.info(f"Ollama embedding 正常，向量维度: {len(vec)}")
        except Exception as e:
            raise RuntimeError(f"Ollama embedding 异常: {e}") from e

    # 建 Collection + 索引
    def _ensure_collection(self) -> Collection:
        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME)
            logger.info(f"已删除旧 collection: {COLLECTION_NAME}")

        fields = [
            FieldSchema(name="pk",     dtype=DataType.INT64,        is_primary=True, auto_id=True),
            FieldSchema(name="text",   dtype=DataType.VARCHAR,       max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR,  dim=VECTOR_DIM),
            FieldSchema(name="page",   dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields=fields, description="RAG knowledge base")
        col = Collection(name=COLLECTION_NAME, schema=schema)
        logger.info(f"Collection 创建成功: {COLLECTION_NAME}")

        col.create_index(
            field_name="vector",
            index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
        )
        logger.info("向量索引创建成功 (IVF_FLAT, L2)")

        utility.wait_for_index_building_complete(COLLECTION_NAME)
        col.load()
        logger.info("Collection 已加载到内存")
        return col

    # 初始化：OCR → 切块 → 向量化 → 直接用 pymilvus 插入
    def _init_store(self) -> Collection:
        col = self._ensure_collection()

        parser = PDFParserOCR()
        pages: list[dict] = parser.parse_pdf(self.pdf_path)

        documents: list[Document] = [
            Document(page_content=p["content"], metadata={"page": p["page"]})
            for p in pages
            if p["content"].strip()
        ]
        if not documents:
            raise ValueError(f"PDF 解析结果为空: {self.pdf_path}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        chunks: list[Document] = splitter.split_documents(documents)
        logger.info(f"文本切分完成，共 {len(chunks)} 个块，开始向量化...")

        texts = [c.page_content for c in chunks]
        pages_ = [c.metadata.get("page", 0) for c in chunks]

        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_pages = pages_[i:i + batch_size]
            batch_vecs = self.embeddings.embed_documents(batch_texts)
            col.insert([batch_texts, batch_vecs, batch_pages])
            logger.info(f"写入进度: {min(i + batch_size, len(texts))}/{len(texts)}")

        col.flush()
        logger.info(f"全部写入完成，共 {len(texts)} 条")

        # ✅ 删除 MARISA_TRIE 相关代码，VARCHAR 字段无需额外索引
        return col

    # Hybrid Search（向量 + 关键词 expr 过滤）
    def _extract_keywords(self, text: str, top_k: int = 5) -> list[str]:
        words = jieba.cut(text)
        return [w for w in words if len(w) > 1][:top_k]

    def _vector_search(
        self, query_vec: list[float], k: int, expr: str | None = None
    ) -> list[tuple[str, int, float]]:
        """返回 [(text, page, score), ...]"""
        search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
        kwargs = dict(
            data=[query_vec],
            anns_field="vector",
            param=search_params,
            limit=k,
            output_fields=["text", "page"],
        )
        if expr:
            kwargs["expr"] = expr
        try:
            results = self.col.search(**kwargs)
            hits = results[0]
            return [(h.entity.get("text"), h.entity.get("page"), h.distance) for h in hits]
        except Exception as e:
            logger.warning(f"向量检索失败 (expr={expr}): {e}")
            return []

    def _hybrid_search(self, question: str, k: int = 6) -> list[tuple[Document, float]]:
        query_vec = self.embeddings.embed_query(question)

        # 路径 A：纯向量检索
        vector_hits = self._vector_search(query_vec, k=k)

        # 路径 B：关键词过滤 + 向量检索
        keywords = self._extract_keywords(question)
        keyword_hits: list[tuple[str, int, float]] = []
        if keywords:
            expr = " || ".join([f'text like "%{kw}%"' for kw in keywords[:3]])
            keyword_hits = self._vector_search(query_vec, k=k, expr=expr)
            logger.info(f"关键词命中 {len(keyword_hits)} 条，关键词: {keywords[:3]}")

        # 合并去重加权
        score_map: dict[str, float] = {}
        meta_map:  dict[str, int]   = {}

        for text, page, dist in vector_hits:
            score_map[text] = score_map.get(text, 0.0) + dist * 0.6
            meta_map[text]  = page
        for text, page, dist in keyword_hits:
            score_map[text] = score_map.get(text, 0.0) + dist * 0.4
            meta_map[text]  = page

        ranked = sorted(score_map.items(), key=lambda x: x[1])[:3]
        return [
            (Document(page_content=text, metadata={"page": meta_map[text]}), score)
            for text, score in ranked
        ]

    # 查询
    def query(self, question: str, conversation_id: str = "default") -> dict:
        if not question.strip():
            raise ValueError("问题不能为空")

        history = self.memory.get_formatted(conversation_id)
        results = self._hybrid_search(question)

        if not results:
            answer = "未在文档中找到相关内容。"
            self.memory.add(conversation_id, "user", question)
            self.memory.add(conversation_id, "assistant", answer)
            return {"answer": answer, "sources": [], "confidence": 0.0}

        context = "\n\n".join([r[0].page_content for r in results])
        sources = [
            {"page": r[0].metadata.get("page", 0), "content_preview": r[0].page_content[:50]}
            for r in results
        ]

        history_section = f"历史对话：\n{history}\n\n" if history else ""
        prompt = (
            f"{history_section}"
            f"参考文档：\n{context}\n\n"
            f"问题：{question}\n\n"
            f"请基于文档内容作答，如有历史上下文请结合参考。"
        )

        try:
            response = self.llm.invoke(prompt)
            answer: str = response.content
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise RuntimeError(f"语言模型调用失败: {e}") from e

        self.memory.add(conversation_id, "user", question)
        self.memory.add(conversation_id, "assistant", answer)

        confidence = round(max(0.0, 1 - (results[0][1] / 2)), 2)
        return {"answer": answer, "sources": sources, "confidence": confidence}

    def clear_conversation(self, conversation_id: str) -> None:
        self.memory.clear(conversation_id)
        logger.info(f"已清空会话: {conversation_id}")