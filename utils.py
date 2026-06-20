import hashlib
import sys
import urllib.request
from pathlib import Path
import json
import numpy as np
from typing import Union
from config import Config
from typing import Union, Optional

def parse_json_to_pose12(json_data: Union[str, dict], data_key: str = "standardData") -> np.ndarray:
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
        
    if "data" in data and "studentSubmissionResponse" in data["data"]:
        data_str = data["data"]["studentSubmissionResponse"].get(data_key)
    elif data_key in data:
        data_str = data.get(data_key)
    else:
        data_str = None

    if data_str:
        frames_data = json.loads(data_str) if isinstance(data_str, str) else data_str
    elif "frames" in data:
        frames_data = data
    else:
        frames_data = {"frames": data} if isinstance(data, list) else data

    frames = frames_data.get("frames", [])
    
    joint_names = [
        "right_shoulder", "right_elbow", "right_wrist",
        "left_shoulder", "left_elbow", "left_wrist",
        "right_hip", "right_knee", "right_ankle",
        "left_hip", "left_knee", "left_ankle"
    ]
    
    num_frames = len(frames)
    pose12 = np.zeros((num_frames, 12, 3), dtype=np.float32)
    
    for i, frame in enumerate(frames):
        kps = frame.get("keypoints_3d", {})
        for j, name in enumerate(joint_names):
            if name in kps:
                pose12[i, j] = kps[name]
                
    return pose12

def extract_key_frames_3d(pose12_seq: np.ndarray, n_clusters: int = 30) -> np.ndarray:
    from sklearn.cluster import KMeans
    from compare_service import PoseComparator
    
    F = pose12_seq.shape[0]
    if F == 0:
        return np.array([], dtype=np.int64)

    vectors = []
    for i in range(F):
        norm_pose, _ = PoseComparator.normalize_pose12(pose12_seq[i])
        vectors.append(norm_pose.flatten())
    vectors = np.array(vectors)

    k = min(n_clusters, F)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(vectors)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    rep_indices = []
    for cluster_id in range(k):
        mask = labels == cluster_id
        if not mask.any():
            continue
        cluster_vectors = vectors[mask]
        cluster_orig_idx = np.where(mask)[0]
        centroid = centroids[cluster_id]
        dists = np.linalg.norm(cluster_vectors - centroid, axis=1)
        best = cluster_orig_idx[int(np.argmin(dists))]
        rep_indices.append(int(best))

    return np.array(sorted(set(rep_indices)), dtype=np.int64)

def download_video(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    
    ext = ".mp4"
    base_url = url.split("?")[0]
    if "." in base_url.split("/")[-1]:
        possible_ext = "." + base_url.split("/")[-1].split(".")[-1]
        if possible_ext.lower() in Config.VIDEO_EXTS:
            ext = possible_ext.lower()
            
    video_path = cache_dir / f"video_{url_hash[:10]}{ext}"
    if not video_path.exists():
        print(f"[download] Đang tải video từ: {url}")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(video_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"[download] Đã lưu tại: {video_path}")
        except Exception as e:
            raise RuntimeError(f"Lỗi khi tải video từ {url}: {e}")
    else:
        print(f"[download] Video đã có trong cache: {video_path}")
    return video_path

def count_video_frames(path: Path) -> Optional[int]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return total if total > 0 else None

def extract_frame_as_image(video_path: Path, frame_idx: int, output_path: Path) -> None:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(str(output_path), frame)
    cap.release()

def extract_frame_as_base64(video_path: Path, frame_idx: int) -> str:
    import cv2
    import base64
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Giảm kích thước ảnh xuống một nửa để chuỗi Base64 không quá dài
        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (width // 2, height // 2))
        
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        b64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{b64_str}"
    return ""

