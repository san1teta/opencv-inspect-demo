import csv
from openpyxl import Workbook
import logging
from typing import Any

logger = logging.getLogger(__name__)

# CSV 和 Excel 共用一份表头，免得两处各写一遍然后改漏一边
# 「圆度」是数值、「圆度判定」是对照门限的结论 —— 两列都要。曾经只有结论没有数值，
# 于是真样本的圆度分布根本拿不到，min_circularity 这条临时门限就永远没有重定的依据。
# 「径向校核」是第二路独立读数，只用于对照不用于判定；「轮廓可疑」是两路分叉的告警位。
# 加这两列的理由：j 那枚被高边圈的内台阶骗，静默读小 1mm（圆度 0.966 看着完全健康），
# 任何质量判据都发现不了。互检把它变成**报表里看得见的一列** —— 只写日志等于没写。
HEADER = ["序号", "图片名称", "面额", "直径(mm)", "圆度", "圆度判定", "偏差(mm)", "缺损(mm)",
          "方向", "严重程度", "判定依据", "径向校核(mm)", "轮廓可疑"]


def _suspect(det: dict[str, Any]) -> str:
    return '是' if det.get('edge_suspect') or det.get('edge_closed') else ''


def _export_data(all_results:list[dict[str, Any]], coin_name: str) -> list[list[Any]]:
    rows = []
    for frame in all_results:
        if not frame['detections']:
            rows.append([frame['index'], frame['image_name'], coin_name] + [None] * 9 + ['未检出'])
            continue
        for det in frame['detections']:
            rad = det.get('radial_mm')
            rows.append([
                frame['index'],
                frame['image_name'],
                coin_name,
                round(det['diameter_mm'], 3),
                round(det['circularity'], 3) if det.get('circularity') is not None else None,
                det['result']['circularity'],
                det['result']['deviation'],
                det['result']['deficit_mm'],
                det['result']['direction'],
                det['result']['severity'],
                det['result']['basis'],
                round(rad, 3) if rad is not None else None,
                _suspect(det),
            ])
    return rows

def export_to_csv(all_results:list[dict[str,Any]], filepath:str, coin_name:str) -> None: 
    if not all_results:
        logger.warning("没有结果可导出")
        return
    
    rows = _export_data(all_results, coin_name)
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)
        logger.info(f"检测报告已成功导出到{filepath}")

def export_to_excel(all_results:list[dict[str,Any]], filepath:str, coin_name:str) -> None:
    if not all_results:
        logger.warning("没有结果可导出")
        return
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.append(HEADER)
    for row in _export_data(all_results, coin_name):
        ws.append(row)
    wb.save(filepath)
    logger.info(f"检测报告已成功导出到{filepath}")
    