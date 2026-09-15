import csv
from openpyxl import Workbook
import logging
from typing import Any

logger = logging.getLogger(__name__)

def _export_data(all_results:list[list[dict[str, Any]]], coin_name: str) -> list[list[Any]]:
    rows = []
    for i ,frame in enumerate(all_results):
        for item in frame:
            rows.append([
                i+1,
                coin_name,
                item['diameter_mm'],
                item['result']['circularity'],
                item['result']['deviation'],
                item['result']['direction'],
                item['result']['severity']
            ])
    return rows

def export_to_csv(all_results:list[list[dict[str,Any]]], filepath:str, coin_name:str) -> None: 
    if not all_results:
        logger.warning("没有结果可导出")
        return
    
    rows = _export_data(all_results, coin_name)
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["图片编号", "面额", "直径(mm)", "圆度", "偏差(mm)", "方向", "严重程度"])
        writer.writerows(rows)
        logger.info(f"检测报告已成功导出到{filepath}")

def export_to_excel(all_results:list[list[dict[str,Any]]], filepath:str, coin_name:str) -> None:
    if not all_results:
        logger.warning("没有结果可导出")
        return
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.append(["图片编号", "面额", "直径(mm)", "圆度", "偏差(mm)", "方向", "严重程度"])
    for row in _export_data(all_results, coin_name):
        ws.append(row)
    wb.save(filepath)
    logger.info(f"检测报告已成功导出到{filepath}")
    