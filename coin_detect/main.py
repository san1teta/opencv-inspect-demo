import argparse
import preprocess
import detector
import defect_analyzer
import visualizer
import exporter
from batch_loader import load_images_from_folder
from camera_capture import start_camera_loop
import cv2
import os

img_path = r"D:\vscode_workplace\pre_pics"
output_path = r"D:\vscode_workplace\pre_pics\result"
csv_path = os.path.join(output_path, "coin_detect_report.csv")
excel_path = os.path.join(output_path, "coin_detect_report.xlsx") 

def process_single_frame(frame):
    """单帧定义"""
    gray_img = preprocess.gray(frame)
    blur_img = preprocess.gaussian_blur(gray_img)
    edge_img = preprocess.canny(blur_img)
    contours = detector.find_contours(edge_img)
    all_features = []
    for cnt in contours:
        features = detector.calculate_features(cnt)
        defects, coin_type = defect_analyzer.analyze_defects(features)
        all_features.append({
            'contour': cnt,
            'features': features,
            'defects': defects,
            'coin_type': coin_type, 
        })
        if defects:
            print(f"检测到缺陷:{defects}, 该硬币类型为:{coin_type}")
        else:
            print("未检测到缺陷")
    detect_visualize = visualizer.draw_result(frame,all_features)

    return detect_visualize, all_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="batch")
    args = parser.parse_args()

    if args.mode == "batch":
        print("进入批量模式...")
        imgs = load_images_from_folder(img_path)
        os.makedirs(output_path, exist_ok=True)
        all_results = []
        for i, img in enumerate(imgs):
            print(f"开始处理第{i+1}张图片...") 
            result_img, all_features = process_single_frame(img)
            cv2.imwrite(os.path.join(output_path, f"result_{i}.jpg"), result_img) 
            all_results.append(all_features) 
            print(f"处理完成第{i+1}张图片")
        exporter.export_to_csv(all_results, csv_path)
        exporter.export_to_excel(all_results, excel_path)
    else:
        print("进入实时模式...")
        start_camera_loop(lambda frame: process_single_frame(frame)[0])
