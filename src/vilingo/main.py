# src/vilingo/main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager

# 从core和api模块导入所需组件
from .core import models
from .api import analysis # <--- 导入我们刚刚修正的 analysis.py

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

# 创建FastAPI应用实例
app = FastAPI(
    title="Vilingo AI Pipeline Service",
    description="一个实现了异步任务处理的AI服务。",
    version="1.2.0",
    lifespan=lifespan
)

# --- 【解决404错误的关键】挂载API路由 ---
# 这行代码会将 analysis.py 文件中定义的所有API接口（/analyze 和 /results/{job_id}）
# 统一注册到应用的 /api/v1 路径下。
app.include_router(analysis.router, prefix="/api/v1", tags=["Analysis"])

# 创建一个根路由用于健康检查
@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "message": "Welcome to Vilingo AI Service!"}