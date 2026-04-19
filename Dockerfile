# 使用稳定的 Python 3.10 基础镜像
FROM python:3.10-slim-bookworm

WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # 增加 PYTHONPATH 确保 src 目录下的模块能被正确识别
    PYTHONPATH=/app

# 安装系统级依赖
# 补充说明：
# 1. poppler-utils: pdf2image 必需
# 2. libgl1, libglib2.0-0, libgomp1: PaddleOCR/OpenCV 必需
# 3. g++: 某些 Python 依赖包（如 pymilvus 的某些扩展）在编译安装时可能需要
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt /app/
# 建议先升级 pip，然后一次性安装依赖
RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目文件
# 注意：确保 src 目录结构在容器内是 /app/src/main.py
COPY src/ /app/src/
COPY data/ /app/data/

EXPOSE 8000

# 建议在启动命令中增加 --proxy-headers（如果以后用 Nginx 代理）
# 同时确保 app 的路径与目录结构一致
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]