from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, Optional
import time
from pathlib import Path
import dataclasses

from utils import parse_json_to_pose12, download_video, extract_frame_as_base64
from compare_service import PoseComparator
from report_service import ReportGenerator

app = FastAPI(title="Compare Pose API", description="API chấm điểm tư thế yoga/thể dục")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# No static output directory needed anymore

class CompareRequest(BaseModel):
    ref_url: str
    ref_height: int
    stu_url: str
    stu_height: int
    submission_data: Dict[str, Any]
    n_keyframes: Optional[int] = 30
    tolerance: Optional[float] = 0.985
    max_errors_per_pair: Optional[int] = 3

@app.post("/compare-pose")
def compare_pose(req: CompareRequest):
    try:
        cache_root = Path("d:/compare-pose-project/cache")
        cache_root.mkdir(parents=True, exist_ok=True)

        print("[API] Đang xử lý dữ liệu JSON...")
        ref_pose12_seq = parse_json_to_pose12(req.submission_data, data_key="standardData")
        stu_pose12_seq = parse_json_to_pose12(req.submission_data, data_key="studentData")

        if len(ref_pose12_seq) == 0 or len(stu_pose12_seq) == 0:
            raise HTTPException(status_code=400, detail="Dữ liệu JSON không chứa frames hợp lệ.")

        print(f"[API] Đã đọc xong JSON. Tải video... (nếu cần)")
        ref_video_path = download_video(req.ref_url, cache_root)
        stu_video_path = download_video(req.stu_url, cache_root)

        print(f"[API] Bắt đầu so sánh...")
        report_data = PoseComparator.compare_videos(
            ref_video=ref_video_path,
            ref_height=req.ref_height,
            ref_pose12_seq=ref_pose12_seq,
            stu_video=stu_video_path,
            stu_height=req.stu_height,
            stu_pose12_seq=stu_pose12_seq,
            n_keyframes=req.n_keyframes,
            tolerance=req.tolerance,
        )

        # Nhúng ảnh Base64
        extracted_refs = {}
        extracted_stus = {}
        
        for error in report_data.errors:
            ref_frame = error["ref_frame"]
            stu_frame = error["student_frame"]
            
            if ref_frame not in extracted_refs:
                extracted_refs[ref_frame] = extract_frame_as_base64(ref_video_path, ref_frame)
                
            if stu_frame not in extracted_stus:
                extracted_stus[stu_frame] = extract_frame_as_base64(stu_video_path, stu_frame)
                
            error["ref_image_path"] = extracted_refs[ref_frame]
            error["stu_image_path"] = extracted_stus[stu_frame]
        
        html_content = ReportGenerator.render_html_report(
            ref_url=req.ref_url,
            stu_url=req.stu_url,
            report_data=report_data,
            output_path=None,
            max_errors_per_pair=req.max_errors_per_pair,
        )
            
        return HTMLResponse(content=html_content)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
