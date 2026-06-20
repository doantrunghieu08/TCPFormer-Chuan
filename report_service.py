from collections import defaultdict
from html import escape
from pathlib import Path
from typing import List, Dict, Any
from dto import CompareReport

class ReportGenerator:
    @staticmethod
    def group_by_pair(errors: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for e in errors:
            groups[int(e["pair_index"])].append(e)
        return dict(sorted(groups.items(), key=lambda x: x[0]))

    @staticmethod
    def render_side_table(side_title: str, errors: List[Dict[str, Any]], max_errors_per_pair: int) -> str:
        if not errors:
            return f"""
            <h3>{escape(side_title)}</h3>
            <div class="success-box">Không phát hiện lỗi đáng kể.</div>
            """

        groups = ReportGenerator.group_by_pair(errors)
        rows: List[str] = []

        for stt, (_, group) in enumerate(groups.items(), start=1):
            group = sorted(group, key=lambda e: float(e.get("move_cm", 0)), reverse=True)[:max_errors_per_pair]
            first = group[0]
            detail_rows: List[str] = []

            for e in group:
                detail_rows.append(f"""
                <tr>
                    <td>{escape(str(e["bone"]))}</td>
                    <td>{escape(str(e["child_joint"]))}</td>
                    <td><b>{escape(str(e["move_cm"]))} cm</b></td>
                </tr>
                """)

            rows.append(f"""
            <tr>
                <td>{stt}</td>
                <td>
                    <div style="margin-bottom: 5px;">Frame học viên: <b>{escape(str(first["student_frame"]))}</b></div>
                    <img src="{escape(str(first.get('stu_image_path', '')))}" style="width: 150px; border-radius: 8px; border: 1px solid #ccc;" alt="Student frame {escape(str(first['student_frame']))}" />
                </td>
                <td>
                    <div style="margin-bottom: 5px;">Frame chuẩn: <b>{escape(str(first["ref_frame"]))}</b></div>
                    <img src="{escape(str(first.get('ref_image_path', '')))}" style="width: 150px; border-radius: 8px; border: 1px solid #ccc;" alt="Reference frame {escape(str(first['ref_frame']))}" />
                </td>
                <td><b style="color:#b00020">Chưa đạt</b></td>
                <td>
                    <div class="explain-box">
                        <b>Frame học viên {escape(str(first["student_frame"]))}</b>
                        so với <b>frame chuẩn {escape(str(first["ref_frame"]))}</b>.
                        <table class="joint-table">
                            <thead>
                                <tr>
                                    <th>Bộ phận / xương bị lệch</th>
                                    <th>Khớp cần di chuyển</th>
                                    <th>Khoảng cách cần di chuyển</th>
                                </tr>
                            </thead>
                            <tbody>{''.join(detail_rows)}</tbody>
                        </table>
                    </div>
                </td>
            </tr>
            """)

        return f"""
        <h3>{escape(side_title)}</h3>
        <table>
            <thead>
                <tr>
                    <th style="width:90px">STT frame</th>
                    <th style="width:150px">Frame học viên</th>
                    <th style="width:150px">Frame chuẩn</th>
                    <th style="width:100px">Kết quả</th>
                    <th>Danh sách khớp cần di chuyển</th>
                </tr>
            </thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
        """

    @staticmethod
    def render_html_report(
        ref_url: str, stu_url: str,
        report_data: CompareReport,
        output_path: Optional[Path] = None, max_errors_per_pair: int = 3
    ) -> str:
        errors = report_data.errors

        def count_display_errors(errors_list: List[Dict[str, Any]]) -> int:
            return sum(min(len(g), max_errors_per_pair) for g in ReportGenerator.group_by_pair(errors_list).values())

        pair_count = len(ReportGenerator.group_by_pair(errors))
        display_count = count_display_errors(errors)

        html_table = ReportGenerator.render_side_table("Chi tiết lỗi", errors, max_errors_per_pair)

        html = f"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Báo cáo chấm pose</title>
<style>
body {{ font-family: Arial, Helvetica, sans-serif; margin: 22px; line-height: 1.52; color:#1f1f1f; background:#fafafa; }}
h1, h2, h3 {{ color:#174b25; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 16px; background:#fff; }}
th, td {{ border: 1px solid #d0d0d0; padding: 10px; vertical-align: top; }}
th {{ background: #f1f7f1; color:#153f1e; }}
.summary-box {{ background: #f7fff7; border: 1px solid #9bd49b; padding: 18px; border-radius: 12px; margin-bottom: 20px; }}
.badge {{ display: inline-block; padding: 5px 12px; border-radius: 999px; background: #e8f6e8; border:1px solid #b9dfb9; margin-right: 8px; margin-bottom:6px; }}
.small {{ color: #555; font-size: 14px; margin-top: 8px; }}
.explain-box {{ background: #fffdf5; border: 1px solid #ffe08a; padding: 12px 14px; border-radius: 10px; margin-top: 8px; }}
.joint-table {{ margin-top: 8px; width: 100%; border-collapse: collapse; }}
.joint-table th, .joint-table td {{ border: 1px solid #e0d4aa; padding: 6px 8px; }}
.success-box {{ background:#f4fff4; border:1px solid #9ed79e; padding:14px 16px; border-radius:12px; }}
</style>
</head>
<body>
<h1>Báo cáo chấm pose</h1>
<div class="summary-box">
    <div>
        <span class="badge"><b>Video chuẩn:</b> <a href="{escape(ref_url)}" target="_blank">Link</a></span>
        <span class="badge"><b>Video học viên:</b> <a href="{escape(stu_url)}" target="_blank">Link</a></span>
        <span class="badge"><b>Chiều cao chuẩn:</b> {report_data.ref_height} cm</span>
        <span class="badge"><b>Chiều cao học viên:</b> {report_data.stu_height} cm</span>
    </div>
    <p>
        <b>Tổng lỗi chi tiết hiển thị:</b> {display_count} lỗi / {pair_count} frame
    </p>
    <p class="small">
        Báo cáo so sánh tư thế của hai người có chiều cao khác nhau. Chương trình tìm keyframe từ video,
        lấy pose tương ứng, so sánh hướng các vector xương bằng cosine similarity và chỉ hiển thị
        khoảng cách cần di chuyển của khớp người học viên để đạt tư thế tương đồng hơn với người chuẩn.
    </p>
</div>
<h2>Danh sách lỗi</h2>
<div>{html_table}</div>
</body>
</html>
"""
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html, encoding="utf-8")
            
        return html
