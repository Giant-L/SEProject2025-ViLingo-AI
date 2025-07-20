# src/vilingo/api/analysis.py

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, status
from pydantic import BaseModel
import shutil
import os
import uuid

# 从我们项目内部的其他模块导入所需组件
from ..core import pipeline
from ..core.state import JOB_RESULTS_DB

# --- API响应模型，用于定义数据结构 ---
class JobResponse(BaseModel):
    job_id: str
    message: str

class StatusResponse(BaseModel):
    status: str
    stage: str | None = None
    error: str | None = None
    result: dict | None = None

# 创建一个API路由器，所有与分析相关的接口都定义在这里
router = APIRouter()

# --- 接口一：启动分析任务 ---
@router.post("/analyze", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def start_analysis(
    background_tasks: BackgroundTasks,
    summary_txt_file: UploadFile = File(..., description="包含英文内容摘要的文本文件 (.txt)"),
    user_audio: UploadFile = File(..., description="用户的录音文件")
):
    """
    接收摘要文本和音频文件，启动一个后台分析任务。
    """
    job_id = str(uuid.uuid4())
    temp_dir = f"temp_{job_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        summary_path = os.path.join(temp_dir, "summary.txt")
        user_audio_path = os.path.join(temp_dir, "user_audio.mp3")

        with open(summary_path, "wb") as f:
            shutil.copyfileobj(summary_txt_file.file, f)
        with open(user_audio_path, "wb") as f:
            shutil.copyfileobj(user_audio.file, f)
            
        JOB_RESULTS_DB[job_id] = {"status": "processing", "stage": "Initializing..."}
        
        background_tasks.add_task(
            pipeline.execute_analysis_pipeline, 
            job_id, 
            summary_path,
            user_audio_path,
            JOB_RESULTS_DB
        )
        
        return {"job_id": job_id, "message": "Analysis task has been accepted and is running in the background."}

    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Failed to start job: {e}")

# --- 接口二：查询任务结果 ---
@router.get("/results/{job_id}", response_model=StatusResponse)
def get_analysis_result(job_id: str):
    """
    根据任务ID查询分析结果。
    """
    result = JOB_RESULTS_DB.get(job_id)
    
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job ID not found.")
        
    return result