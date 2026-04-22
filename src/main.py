import os
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag_engine import RAGEngine

# 1. 必须先配置 logging 并定义 logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 2. 全局变量声明
rag_engine: RAGEngine | None = None

# 3. 路径配置
base_dir = os.path.dirname(os.path.abspath(__file__))
PDF_FILE_PATH = os.path.abspath(
    os.path.join(base_dir, "..", "data", "GBT 1568-2008 键 技术条件.pdf")
)


# ---------- 请求 / 响应模型 ----------
class SourceItem(BaseModel):
    page: int
    content_preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    confidence: float


class QueryRequest(BaseModel):
    question: str
    conversation_id: str = "default_user"


# ---------- 生命周期管理 ----------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """服务启动时初始化引擎"""
    global rag_engine
    # 此时 logger 已经在全局作用域定义，可以安全使用
    logger.info("正在启动服务并初始化 RAGEngine...")

    try:
        if not os.path.exists(PDF_FILE_PATH):
            logger.error(f"找不到 PDF 文件: {PDF_FILE_PATH}")
            raise FileNotFoundError(f"文件不存在: {PDF_FILE_PATH}")

        rag_engine = RAGEngine(pdf_path=PDF_FILE_PATH)
        logger.info("RAGEngine 初始化成功，API 服务就绪")
    except Exception as e:
        logger.error(f"RAGEngine 初始化期间发生致命错误: {str(e)}")
        # 注意：在 lifespan 中 raise 异常会导致 FastAPI 启动失败并退出
        raise e

    yield
    logger.info("正在关闭服务...")


# ---------- 应用实例 ----------
app = FastAPI(title="Smart Document QA Agent", lifespan=lifespan)


# ---------- 路由 ----------
@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest) -> QueryResponse:
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG 引擎未就绪")

    try:
        # 调用引擎获取结果
        result = rag_engine.query(
            question=request.question,
            conversation_id=request.conversation_id,
        )

        # 严格按照你要求的格式构建返回对象
        return QueryResponse(
            answer=result["answer"],
            sources=[SourceItem(**s) for s in result["sources"]],
            confidence=result.get("confidence", 0.0)
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat 接口错误: {str(e)}")
        raise HTTPException(status_code=500, detail="服务器内部错误")


@app.get("/health")
async def health_check():
    return {
        "status": "ready" if rag_engine is not None else "initializing",
        "pdf_path": PDF_FILE_PATH
    }