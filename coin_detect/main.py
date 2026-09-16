import exporter
from camera_capture import start_camera_loop
from pipeline import run_batch, process_single_frame
import os
import sys
import logging
import argparse

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="batch", choices=["batch", "camera"])
    parser.add_argument("--coin", type=str, required=True, choices=["1元", "5角", "1角"])
    parser.add_argument("--input", "-i", type=str, help="待检测图片文件夹")
    parser.add_argument("--output", "-o", type=str, help="输出图片路径")
    args = parser.parse_args()

    if args.mode == "batch":
        if not args.input:
            sys.exit("请输入图片路径")
        output_path = args.output or os.path.join(args.input, "result")
        logger.info("进入批量模式...")
        all_results = run_batch(args.input, output_path, args.coin)
        exporter.export_to_csv(all_results, os.path.join(output_path, "coin_detect_report.csv"),args.coin)
        exporter.export_to_excel(all_results, os.path.join(output_path, "coin_detect_report.xlsx"), args.coin)
    else:
        logger.info("进入实时模式...")
        start_camera_loop(lambda frame: process_single_frame(frame, args.coin)[0])
