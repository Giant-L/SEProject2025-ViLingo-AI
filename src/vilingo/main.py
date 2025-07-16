from fastapi import FastAPI
from contextlib import asynccontextmanager
from .core import models
from .api import analysis

# Lifespan上下文管理器，用于处理应用启动和关闭时的事件
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 应用启动时: 加载所有AI模型
    print("应用启动中...")
    models.load_all_models()
    print("模型加载完毕，应用准备就绪。")
    yield
    # 应用关闭时: 清理资源 (可选)
    print("应用关闭中...")
    models.MODEL_REGISTRY.clear()

# 创建FastAPI应用实例，并应用lifespan管理器
app = FastAPI(
    title="Vilingo AI Pipeline Service",
    description="一个集成了语音识别、文本概括和语义对比的AI服务。",
    version="1.0.0",
    lifespan=lifespan
)

# 挂载分析API路由
app.include_router(analysis.router, prefix="/api/v1", tags=["Analysis"])

# 创建一个根路由用于健康检查
@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "message": "Welcome to Vilingo AI Service!"}