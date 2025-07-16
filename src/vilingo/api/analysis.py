from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
import uuid
from ..core import pipeline

# 创建一个API路由器
router = APIRouter()

@router.post("/analyze")
async def analyze_content(
    video_file: UploadFile = File(..., description="用户上传的视频文件"),
    user_audio: UploadFile = File(..., description="用户的录音文件")
):
    """
    接收视频和音频文件，执行完整的AI分析流程，并返回结果。
    """
    # 创建唯一的临时工作目录
    request_id = str(uuid.uuid4())
    temp_dir = f"temp_{request_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        video_path = os.path.join(temp_dir, "source_video")
        user_audio_path = os.path.join(temp_dir, "user_audio")

        # 保存上传的文件到临时目录
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video_file.file, f)
        with open(user_audio_path, "wb") as f:
            shutil.copyfileobj(user_audio.file, f)
        
        # 调用核心流水线处理
        result = pipeline.run_analysis_pipeline(video_path, user_audio_path)
        return result

    except ValueError as e:
        # 捕获流水线中可预见的错误 (例如，音频为空)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 捕获其他意外错误
        print(f"处理请求 {request_id} 时发生意外错误: {e}")
        raise HTTPException(status_code=500, detail="服务器内部发生未知错误。")
    finally:
        # 无论成功或失败，都清理临时文件
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)