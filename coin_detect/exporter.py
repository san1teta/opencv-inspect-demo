import csv
from openpyxl import Workbook

def export_to_csv(all_results,filepath):
    if not all_results:
        print("No results to export.")
        return

    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["图片编号", "硬币类型", "面积", "周长", "圆度", "直径(像素)", "缺陷"])
        for i, result in enumerate(all_results):
            for item in result:
                defects_str = ','.join(item['defects']) if item['defects'] else '无'
                writer.writerow([
                    i + 1,
                    item['coin_type'],
                    item['features']['area'],
                    item['features']['perimeter'],
                    round(item['features']['circularity'], 3),
                    item['features']['diameter_pixels'],
                    defects_str
                ])
        print(f"检测报告已成功导出到{filepath}")

def export_to_excel(all_results, filepath):
    if not all_results:
        print("No results to export.")
        return
    wb = Workbook()
    ws = wb.active
    ws.append(["图片编号", "硬币类型", "面积", "周长", "圆度", "直径(像素)", "缺陷"])
    for i, result in enumerate(all_results):
        for item in result:
            defects_str = ','.join(item['defects']) if item['defects'] else '无'
            ws.append([i + 1, 
                    item['coin_type'], 
                    item['features']['area'], 
                    item['features']['perimeter'], 
                    round(item['features']['circularity'], 3), 
                    item['features']['diameter_pixels'], 
                    defects_str])
    wb.save(filepath)
    print(f"检测报告已成功导出到{filepath}")