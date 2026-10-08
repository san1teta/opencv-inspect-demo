"""合成硬币自检：绕开相机，单独验证「像素 -> 毫米 -> 分级」这条链本身准不准。

真硬币的直径是未知的，所以拿真硬币只能看重复性，看不出准确度。这里画一个
直径已知的圆，走完全相同的管线，测出来的值必须等于真值，否则问题一定在代码里
（ppm 用错、取整、分级刀口算反），跟拍摄无关。
"""
import argparse
import io
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))

import config_setting  # noqa: E402
import pipeline  # noqa: E402

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(errors='replace')   # GBK 控制台别被一个字符带走整份报告


def make_coin(diameter_mm: float, ppm: float, w: int, h: int,
              nick_mm: float = 0.0) -> tuple[np.ndarray, float]:
    """深色背景上画一枚边缘干净的圆币，尺寸按标定 ppm 换算。

    半径只能取整，所以真值是「画出来的」那个直径 —— 拿名义值当基准，
    会把最多 ±0.05mm 的画图量化算到测量链头上。
    nick_mm 是缺口深度：咬痕圆的圆心压在轮廓上，半径 = 深度，所以啃进去正好 nick_mm。
    """
    img = np.full((h, w, 3), 40, np.uint8)
    r = int(round(diameter_mm * ppm / 2))
    cv2.circle(img, (w // 2, h // 2), r, (200, 200, 200), -1)
    if nick_mm > 0:
        cv2.circle(img, (w // 2 + r, h // 2), int(round(nick_mm * ppm)), (40, 40, 40), -1)
    return img, 2 * r / ppm


def main() -> None:
    ap = argparse.ArgumentParser(description="合成硬币测量链自检")
    ap.add_argument("--coin", default="1元")
    ap.add_argument("--diameter", type=float, action="append",
                    help="要画的直径(mm)，可重复；默认覆盖四档分级")
    args = ap.parse_args()

    spec = config_setting.coin_specs[args.coin]
    std, tol = spec['diameter'], spec['tol']
    # 默认四个点：合格、轻度、中度、重度各命中一次（excess = |d-std| - tol）
    diams = args.diameter or [std, std + tol + 0.15, std + tol + 0.45, std + tol + 0.75]

    w, h = config_setting.calib_image_size
    if (w, h) == (0, 0):
        sys.exit("calib_image_size 未标定，先跑 tools/measure_ppm.py --write")
    ppm = config_setting.ppm
    print(f"标定 ppm {ppm:.4f} -> 1px = {1000 / ppm:.1f} 微米 | {args.coin} 标准 {std}mm 允差 ±{tol}mm")
    print(f"尺寸档：|偏差|<= {tol:.2f} 合格，<= {tol + config_setting.grade_light:.2f} 轻度，"
          f"<= {tol + config_setting.grade_mid:.2f} 中度，否则重度")
    print(f"缺损档：径向凹陷 <= {config_setting.defect_light} 合格，<= {config_setting.defect_mid} 轻度，"
          f"<= {config_setting.defect_heavy} 中度，否则重度\n")

    grades = set()

    print('== 1) 直径真值扫描（尺寸路）==')
    worst = 0.0
    for d in diams:
        img, drawn = make_coin(d, ppm, w, h)
        dets = pipeline.process_single_frame(img, args.coin)[1]
        if len(dets) != 1:
            print(f"真值 {drawn:6.3f}mm -> 检出 {len(dets)} 个轮廓，应为 1 个  [FAIL]")
            continue
        got = dets[0]['diameter_mm']
        res = dets[0]['result']
        err = got - drawn
        worst = max(worst, abs(err))
        grades.add(res['severity'])
        flag = 'PASS' if abs(err) < 0.05 else 'FAIL'
        print(f"名义 {d:5.2f} 真值 {drawn:6.3f} -> 测得 {got:6.3f}mm  误差 {err:+.3f}mm [{flag}]  "
              f"分级 {res['severity']} 依据 {res['basis']}")
    print(f"   测量链最大误差 {worst:.3f}mm（阈值 0.05mm）\n")

    print('== 2) 缺口深度扫描（缺损路）==')
    print('   这一节证明为什么不能只看直径：缺口是局部事件，面积等效直径是全局量，抓不住')
    base = pipeline.process_single_frame(make_coin(std, ppm, w, h)[0], args.coin)[1][0]
    for nick in (0.0, 0.5, 1.0, 2.0, 3.0):
        img, _ = make_coin(std, ppm, w, h, nick_mm=nick)
        det = pipeline.process_single_frame(img, args.coin)[1][0]
        res = det['result']
        grades.add(res['severity'])
        dd = det['diameter_mm'] - base['diameter_mm']
        flag = 'PASS' if abs(res['deficit_mm'] - nick) < 0.15 else 'FAIL'
        print(f"缺口 {nick:4.1f}mm -> 测得凹陷 {res['deficit_mm']:5.3f}mm [{flag}]  "
              f"直径只变 {dd:+.3f}mm  分级 {res['severity']} 依据 {res['basis']}")

    missing = {'合格', '轻度', '中度', '重度'} - grades
    print()
    print("四档分级全部触发" if not missing else f"未触发的分级：{'、'.join(sorted(missing))}")
    if worst >= 0.05 or missing:
        sys.exit(1)



if __name__ == "__main__":
    main()
