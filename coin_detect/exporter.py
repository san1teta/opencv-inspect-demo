import csv
from openpyxl import Workbook

def export_to_csv(all_results, filepath, coin_name):
    if not all_results:
        print("No results to export.")
        return

    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["图片编号", "面额", "直径(mm)", "圆度", "偏差(mm)", "方向", "严重程度"])
        for i, frame in enumerate(all_results):
            for item in frame:
                writer.writerow([
                    i + 1,
                    coin_name,
                    item['diameter_mm'],
                    item['result']['circularity'],
                    item['result']['deviation'],
                    item['result']['direction'],
                    item['result']['severity']
                ])
        print(f"检测报告已成功导出到{filepath}")

def export_to_excel(all_results, filepath, coin_name):
    if not all_results:
        print("No results to export.")
        return
    wb = Workbook()
    ws = wb.active
    ws.append(["图片编号", "面额", "直径(mm)", "圆度", "偏差(mm)", "方向", "严重程度"])
    for i, frame in enumerate(all_results):
        for item in frame:
            ws.append([
                    i + 1,
                    coin_name,
                    item['diameter_mm'],
                    item['result']['circularity'],
                    item['result']['deviation'],
                    item['result']['direction'],
                    item['result']['severity']
                ])
    wb.save(filepath)
    print(f"检测报告已成功导出到{filepath}")
    