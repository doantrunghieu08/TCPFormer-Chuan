import numpy as np
from pathlib import Path
from typing import List, Tuple
from config import Config
from dto import ErrorDetail, CompareReport
from pose2d_service import Pose2DService
from pose3d_service import Pose3DService

class PoseComparator:
    @staticmethod
    def calculate_move_distance(bone_idx: int, angle_diff_deg: float, student_height_cm: float) -> float:
        # Tỉ lệ chiều dài các đoạn xương so với tổng chiều cao cơ thể
        bone_ratios = {
            0: 0.19,   # Trái: Vai -> Khuỷu tay
            1: 0.15,   # Trái: Khuỷu tay -> Cổ tay
            2: 0.19,   # Phải: Vai -> Khuỷu tay
            3: 0.15,   # Phải: Khuỷu tay -> Cổ tay
            4: 0.245,  # Trái: Hông -> Đầu gối
            5: 0.246,  # Trái: Đầu gối -> Mắt cá
            6: 0.245,  # Phải: Hông -> Đầu gối
            7: 0.246,  # Phải: Đầu gối -> Mắt cá
            8: 0.29,   # Trái: Vai -> Hông (Thân)
            9: 0.29,   # Phải: Vai -> Hông (Thân)
        }
        
        ratio = bone_ratios.get(bone_idx, 0.2)
        bone_len_cm = student_height_cm * ratio
        
        # Tính cung tròn (khoảng cách di chuyển của child joint)
        import math
        theta_rad = math.radians(angle_diff_deg)
        move_cm = bone_len_cm * theta_rad
        
        return round(move_cm, 2)

    @staticmethod
    def normalize_pose12(pose12_xyz: np.ndarray) -> Tuple[np.ndarray, float]:
        pose = np.asarray(pose12_xyz, dtype=np.float32)[:, :3]
        right_hip = pose[6]
        left_hip = pose[9]
        pelvis = (right_hip + left_hip) / 2.0

        centered = pose - pelvis
        mins = centered.min(axis=0)
        maxs = centered.max(axis=0)
        scale = float(np.linalg.norm(maxs - mins))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        return centered / scale, scale

    @staticmethod
    def align_poses_3d(ref_pose: np.ndarray, stu_pose: np.ndarray) -> np.ndarray:
        """
        Sử dụng thuật toán Kabsch (Procrustes Analysis) để tìm ma trận xoay tối ưu R
        nhằm xoay stu_pose khớp với ref_pose.
        Cả hai pose đều phải được chuẩn hóa (normalize) và có cùng kích thước.
        """
        # H = stu_pose^T * ref_pose
        H = np.dot(stu_pose.T, ref_pose)
        
        # SVD
        U, S, Vt = np.linalg.svd(H)
        
        # R = U * V^T (với Vt là V^T từ numpy svd)
        R = np.dot(U, Vt)
        
        # Đảm bảo không bị lật ngược (Reflection)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(U, Vt)
            
        # Xoay student pose
        stu_aligned = np.dot(stu_pose, R)
        return stu_aligned

    @staticmethod
    def compare_pose_frame(
        ref_pose12_xyz: np.ndarray,
        stu_pose12_xyz: np.ndarray,
        stu_height_cm: float,
        tolerance: float,
    ) -> List[ErrorDetail]:
        ref_norm, _ = PoseComparator.normalize_pose12(ref_pose12_xyz)
        stu_norm, _ = PoseComparator.normalize_pose12(stu_pose12_xyz)
        
        # XOAY (ALIGN) sinh viên khớp với góc quay của người mẫu
        stu_norm = PoseComparator.align_poses_3d(ref_norm, stu_norm)

        errors: List[ErrorDetail] = []

        for i, (parent_idx, child_idx, bone_name, child_name) in enumerate(Config.BONES_12):
            v_ref = ref_norm[child_idx] - ref_norm[parent_idx]
            v_stu = stu_norm[child_idx] - stu_norm[parent_idx]

            norm_ref = float(np.linalg.norm(v_ref))
            norm_stu = float(np.linalg.norm(v_stu))
            if norm_ref < 1e-8 or norm_stu < 1e-8:
                continue

            cos_sim = float(np.dot(v_ref, v_stu) / (norm_ref * norm_stu))
            cos_sim = float(np.clip(cos_sim, -1.0, 1.0))

            if cos_sim < tolerance:
                import math
                angle_rad = math.acos(cos_sim)
                angle_deg = math.degrees(angle_rad)
                move_cm = PoseComparator.calculate_move_distance(i, angle_deg, stu_height_cm)

                errors.append(ErrorDetail(
                    bone=bone_name,
                    child_joint=child_name,
                    cosine=round(cos_sim, 4),
                    move_cm=move_cm
                ))

        return errors

    @staticmethod
    def align_poses_dtw(ref_pose12_seq: np.ndarray, stu_pose12_seq: np.ndarray) -> dict:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        from collections import defaultdict
        
        def get_norm_seq(pose12_seq):
            seq = []
            for i in range(len(pose12_seq)):
                norm, _ = PoseComparator.normalize_pose12(pose12_seq[i])
                seq.append(norm.flatten())
            return np.array(seq)
            
        ref_seq = get_norm_seq(ref_pose12_seq)
        stu_seq = get_norm_seq(stu_pose12_seq)
        
        print(f"[dtw] Aligning ref ({len(ref_seq)}) and stu ({len(stu_seq)})")
        _, path = fastdtw(ref_seq, stu_seq, dist=euclidean)
        
        mapping_dict = defaultdict(list)
        for r_idx, s_idx in path:
            mapping_dict[r_idx].append(s_idx)
            
        final_mapping = {}
        for r_idx, s_list in mapping_dict.items():
            final_mapping[r_idx] = s_list[len(s_list) // 2]
            
        return final_mapping

    @staticmethod
    def compare_videos(
        ref_video: Path, ref_height: int, ref_pose12_seq: np.ndarray,
        stu_video: Path, stu_height: int, stu_pose12_seq: np.ndarray,
        n_keyframes: int, tolerance: float
    ) -> CompareReport:
        print(f"\n=== SO SÁNH VIDEO ===")
        print(f"[video ref] {ref_video.name}")
        print(f"[video stu] {stu_video.name}")

        from utils import extract_key_frames_3d
        ref_keyframes = extract_key_frames_3d(ref_pose12_seq, n_clusters=n_keyframes)

        frame_mapping = PoseComparator.align_poses_dtw(ref_pose12_seq, stu_pose12_seq)
        m = len(ref_keyframes)
        
        stu_keyframes_mapped = [frame_mapping.get(int(r), 0) for r in ref_keyframes]

        errors: List[dict] = []

        for pair_index, ref_frame in enumerate(ref_keyframes):
            ref_frame = int(ref_frame)
            stu_frame = frame_mapping.get(ref_frame, 0)

            ref_pose12 = ref_pose12_seq[ref_frame]
            stu_pose12 = stu_pose12_seq[stu_frame]

            frame_errors = PoseComparator.compare_pose_frame(ref_pose12, stu_pose12, float(stu_height), tolerance=tolerance)
            for e in frame_errors:
                e.pair_index = pair_index
                e.ref_frame = ref_frame
                e.student_frame = stu_frame
                errors.append(e.to_dict())

        print(f"[compare] ref_keyframes={len(ref_keyframes)} stu_keyframes={len(stu_keyframes_mapped)} m={m} errors={len(errors)}")
        return CompareReport(
            ref_height=ref_height,
            stu_height=stu_height,
            ref_keyframes=ref_keyframes.tolist(),
            stu_keyframes=stu_keyframes_mapped,
            m=m,
            errors=errors,
            use_pkl_joints=False
        )
