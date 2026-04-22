import os
import logging

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    Function,
    FunctionType,
    AnnSearchRequest,
    RRFRanker,
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.pdf_parser import PDFParserOCR
from src.conversation_memory import ConversationMemory

logger = logging.getLogger(__name__)

VECTOR_DIM = 1024
COLLECTION_NAME = "knowledge_base_v2"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class RAGEngine:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.memory = ConversationMemory(window_size=5)

        self.milvus_host = os.getenv("MILVUS_HOST", "milvus")
        self.milvus_port = os.getenv("MILVUS_PORT", "19530")

        api_key = os.getenv("QWEN_API_KEY", "")
        if not api_key:
            raise RuntimeError("QWEN_API_KEY 未设置")

        # Embedding
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3"),
            api_key=api_key,
            base_url=QWEN_BASE_URL,
            tiktoken_enabled=False,  # 你已经设置了这个，但还需要下面这个参数
            check_embedding_ctx_length=False,  # ✅ 核心修复：禁用本地分词器长度检查
        )

        # LLM
        self.llm = ChatOpenAI(
            model=os.getenv("QWEN_MODEL", "qwen-plus"),
            api_key=api_key,
            base_url=QWEN_BASE_URL,
            temperature=0.1,
        )

        connections.connect(
            host=self.milvus_host,
            port=self.milvus_port
        )

        self.col = self._init_store()
        self._verify_embedding()

    # =========================
    # embedding check
    # =========================
    def _verify_embedding(self):
        vec = self.embeddings.embed_query("测试")
        if len(vec) != VECTOR_DIM:
            raise RuntimeError(f"Embedding维度错误: {len(vec)} != {VECTOR_DIM}")

    def _ensure_collection(self):
        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME)

        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            # ✅ 修改 1：文本字段必须设置 enable_analyzer=True，否则无法被 BM25 函数解析
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]

        bm25_function = Function(
            name="bm25_func",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse"],
        )

        schema = CollectionSchema(
            fields,
            description="Hybrid RAG (Milvus 2.6.5)",
            # ✅ 修改 2：正确的参数名是 functions，而不是 function_groups
            functions=[bm25_function],
        )

        col = Collection(
            name=COLLECTION_NAME,
            schema=schema,
        )

        # Dense index
        col.create_index(
            field_name="vector",
            index_params={
                "index_type": "HNSW",
                "metric_type": "IP",
                "params": {"M": 16, "efConstruction": 200},
            },
        )

        # Sparse/BM25 index (保持我们上次改的正确索引类型)
        col.create_index(
            field_name="sparse",
            index_params={
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25",
            },
        )

        col.load()
        return col

    # =========================
    # build vector store
    # =========================
    def _init_store(self):
        col = self._ensure_collection()
        parser = PDFParserOCR()
        pages = parser.parse_pdf(self.pdf_path)

        # 1. 转换为 LangChain Document 对象
        raw_docs = [
            Document(
                page_content=p["content"],
                metadata={"page": int(p["page"])}  # 确保页码存入 metadata
            )
            for p in pages if p["content"].strip()
        ]

        # 2. 切分文档（注意：splitter 会自动传播 metadata）
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 稍微调小一点，增加检索粒度
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "；"]
        )
        chunks = splitter.split_documents(raw_docs)

        # 3. 提取对齐的数据列表
        # 这一步非常关键：必须确保每个 chunk 的文本、向量、页码在索引上完全一一对应
        texts = [c.page_content for c in chunks]
        # 从 chunk 的 metadata 中提取页码，如果没有则默认为 0
        pages_to_insert = [c.metadata.get("page", 0) for c in chunks]

        logger.info(f"开始生成向量，总片段数: {len(texts)}")
        dense_vecs = self.embeddings.embed_documents(texts)

        # 4. 批量插入
        # Milvus 插入时，列表的顺序必须与 Field 定义顺序一致：text, vector, page
        # 我们使用一个循环，以确保数据块对齐
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            end = i + batch_size
            col.insert([
                texts[i:end],
                dense_vecs[i:end],
                pages_to_insert[i:end],
            ])

        col.flush()
        logger.info("数据入库完成并已持久化")
        return col

    def _hybrid_search(self, question: str, k=20):
        dense_vec = self.embeddings.embed_query(question)

        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="vector",
            param={
                "metric_type": "IP",
                "params": {"ef": 64},
            },
            limit=k,
        )

        bm25_req = AnnSearchRequest(
            data=[question],
            anns_field="sparse",
            param={
                "metric_type": "BM25",
            },
            limit=k,
        )

        res = self.col.hybrid_search(
            reqs=[dense_req, bm25_req],
            rerank=RRFRanker(k=60),  # RRF 的 k 保持 60 是对的
            limit=k,
            output_fields=["text", "page"],
        )

        docs = []
        for hit in res[0]:
            # 从 entity 中获取 page 字段
            page_num = hit.entity.get("page")
            docs.append(
                Document(
                    page_content=hit.entity.get("text"),
                    metadata={"page": page_num}  # ✅ 必须写回 metadata
                )
            )
        return docs

    # =========================
    # rerank (LLM simple)
    # =========================
    def _rerank(self, question, docs):
        if not docs:
            return []

        joined = "\n".join(
            [f"ID {i + 1}: {d.page_content}" for i, d in enumerate(docs)]
        )

        prompt = f"""
任务：从以下候选文档中选出最能回答问题的3个文档ID。
问题：{question}

候选文档：
{joined}

请只输出数字编号，用空格分隔，例如：1 3 5。
如果候选文档都无关，请输出：0
"""
        try:
            resp = self.llm.invoke(prompt).content
            import re
            # 使用正则表达式提取所有数字，更稳健
            idxs = [int(i) - 1 for i in re.findall(r'\d+', resp)]
            # 过滤掉无效索引和 0
            selected_docs = [docs[i] for i in idxs if 0 <= i < len(docs)]

            if not selected_docs:
                logger.warning("Reranker 没选出任何结果，回退到原始搜索前3条")
                return docs[:3]
            return selected_docs[:3]
        except Exception as e:
            logger.error(f"Rerank 出错: {e}")
            return docs[:3]

    # =========================
    # query entry
    # =========================
    def query(self, question, conversation_id="default"):
        if not question.strip():
            raise ValueError("问题不能为空")

        # 1. 混合检索（拿到 k=10 个候选）
        initial_docs = self._hybrid_search(question)
        logger.info(f"检索到 {len(initial_docs)} 条片段")
        for i, d in enumerate(initial_docs):
            logger.info(f"片段{i + 1} (页码:{d.metadata['page']}): {d.page_content[:50]}...")
        # 2. LLM 重排（从 10 个中选出真正相关的 3 个）
        docs = self._rerank(question, initial_docs)

        # 3. 构造上下文
        context = "\n\n".join([d.page_content for d in docs]) if docs else "未找到相关参考资料。"
        history = self.memory.get_formatted(conversation_id)

        # 4. 最终回答 Prompt
        final_prompt = f"""{history}
【参考资料】：
{context}

1. 请基于上述参考资料回答问题。如果资料中没写，请回答不知道，不要瞎猜。
2. 绝对严禁根据你自身的知识库回答。
3. 如果资料中有相关条款（如“第3.1条”），请在回答中指明。
问题：{question}
"""
        response = self.llm.invoke(final_prompt)
        answer = response.content

        # 5. 更新记忆
        self.memory.add(conversation_id, "user", question)
        self.memory.add(conversation_id, "assistant", answer)

        return {
            "answer": answer,
            "sources": [
                {
                    # 这里的 d.metadata["page"] 就是从 Milvus 查出来的原始页码
                    "page": d.metadata.get("page", 0) + 1,
                    "content_preview": d.page_content[:100]
                }
                for d in docs
            ],
            "confidence": 0.9,
        }

    # 在 RAGEngine 类中添加此方法
    async def query_stream(self, question: str, conversation_id="default"):
        if not question.strip():
            raise ValueError("问题不能为空")

        # 1. 检索与重排序 (这部分通常很快，保持同步或改异步均可)
        docs = self._hybrid_search(question)
        docs = self._rerank(question, docs)
        context = "\n\n".join([d.page_content for d in docs]) if docs else ""
        history = self.memory.get_formatted(conversation_id)

        prompt = f"{history}\n\n参考资料：\n{context}\n\n问题：\n{question}"

        # 2. 准备完整回答的容器，用于最后存入记忆
        full_answer = ""

        # 3. 使用 astream 发起流式调用
        async for chunk in self.llm.astream(prompt):
            content = chunk.content
            if content:
                full_answer += content
                yield content  # 逐个片段返回

        # 4. 回答完成后，异步存入记忆
        self.memory.add(conversation_id, "user", question)
        self.memory.add(conversation_id, "assistant", full_answer)