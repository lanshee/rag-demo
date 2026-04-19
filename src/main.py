import os
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局引擎实例
rag_engine: RAGEngine | None = None

base_dir = os.path.dirname(os.path.abspath(__file__))
PDF_FILE_PATH = os.path.abspath(
    os.path.join(base_dir, "..", "data", "GBT 1568-2008 键 技术条件.pdf")
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """服务启动时初始化引擎，uvicorn 可以立刻绑定端口"""
    global rag_engine
    logger.info("正在初始化 RAGEngine...")
    try:
        rag_engine = RAGEngine(pdf_path=PDF_FILE_PATH)
        logger.info("RAGEngine 初始化完成，服务就绪")
    except Exception as e:
        logger.error(f"RAGEngine 初始化失败: {e}")
        raise
    yield
    logger.info("服务关闭")


app = FastAPI(title="Smart Document QA Agent", lifespan=lifespan)


# ---------- 请求 / 响应模型 ----------
class QueryRequest(BaseModel):
    question: str
    conversation_id: str = "default_user"


class SourceItem(BaseModel):
    page: int
    content_preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    confidence: float


# ---------- 路由 ----------
@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest) -> QueryResponse:
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="服务正在初始化，请稍后重试")
    try:
        result = rag_engine.query(
            question=request.question,
            conversation_id=request.conversation_id,
        )
        return QueryResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.error(f"未知错误: {e}")
        raise HTTPException(status_code=500, detail="服务内部错误")


@app.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str) -> dict[str, str]:
    """清除指定会话的历史记忆"""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="服务正在初始化")
    rag_engine.clear_conversation(conversation_id)
    return {"message": f"会话 {conversation_id} 历史已清除"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    status = "ready" if rag_engine is not None else "initializing"
    return {"status": status}