# src/vilingo/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, status
from pydantic import BaseModel
from contextlib import asynccontextmanager
import shutil
import os
import uuid

from .core import models, pipeline

# --- 内存数据库，用于存储任务状态和结果 ---
# 这是一个简单的字典，在生产环境中可替换为Redis等
# key: job_id, value: {"status": "...", "result": ... / "error": ...}
JOB_RESULTS_DB = {}

# --- 应用生命周期管理 ---
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

# --- API响应模型 ---
class JobResponse(BaseModel):
    job_id: str
    message: str

class StatusResponse(BaseModel):
    status: str
    stage: str | None = None
    error: str | None = None
    result: dict | None = None


# --- FastAPI 应用实例 ---
app = FastAPI(
    title="Vilingo AI Pipeline Service",
    description="一个实现了异步任务处理的AI服务。",
    version="1.1.0",
    lifespan=lifespan
)


# --- API Endpoints ---

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "message": "Welcome to Vilingo AI Service!"}

@app.post("/api/v1/analyze", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Analysis"])
async def start_analysis(
    background_tasks: BackgroundTasks,
    # --- 主要变更在这里 ---
    srt_file: UploadFile = File(..., description="用户上传的字幕文件 (.srt)"),
    # ----------------------
    user_audio: UploadFile = File(..., description="用户的录音文件")
):
    """
    接收字幕和音频文件，启动一个后台分析任务，并立即返回任务ID。
    """
    job_id = str(uuid.uuid4())
    temp_dir = f"temp_{job_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # --- 变更文件保存逻辑 ---
        srt_path = os.path.join(temp_dir, "source_subtitle.srt")
        user_audio_path = os.path.join(temp_dir, "user_audio.wav")

        with open(srt_path, "wb") as f:
            shutil.copyfileobj(srt_file.file, f)
        # ------------------------
        with open(user_audio_path, "wb") as f:
            shutil.copyfileobj(user_audio.file, f)
            
        JOB_RESULTS_DB[job_id] = {"status": "processing", "stage": "Initializing..."}
        
        # --- 变更后台任务的参数 ---
        background_tasks.add_task(
            pipeline.execute_analysis_pipeline, 
            job_id, 
            srt_path,  # <--- 传递 srt 文件路径
            user_audio_path,
            JOB_RESULTS_DB
        )
        # ------------------------
        
        return {"job_id": job_id, "message": "Analysis task has been accepted and is running in the background."}

    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Failed to start job: {e}")


@app.get("/api/v1/results/{job_id}", response_model=StatusResponse, tags=["Analysis"])
async def get_analysis_result(job_id: str):
    """
    根据任务ID查询分析结果。
    前端可以轮询调用此接口来获取最新状态。
    """
    result = JOB_RESULTS_DB.get(job_id)
    
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job ID not found.")
        
    return result