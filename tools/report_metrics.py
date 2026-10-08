"""把一批实拍硬币跑成 demo 指标：检出率 / 耗时 / 重复性 / 三面额线性度 / SPC 建议控制限。

目录约定（零配置，按子目录名自动识别）：

    shots/coins/1元/     coin01.jpg ...     一个面额一叠，算分布和线性度
    shots/coins/5角/     ...
    shots/coins/1角/     ...
    shots/coins/repeat/  r01.jpg ...        同一枚挪开再放回重拍，算重复性

线性度为什么能代替卡尺：三个面额的标称直径是已知值（25.0/20.5/19.0）。测得均值之间的
「差」对上标称差，就说明比例尺在量程内是线性的 —— 这个结论不依赖任何绝对基准。
绝对偏移则单独列出，如实标注为「相对标称值」。
"""
import argparse
import io
import math
import os
import re
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))

import config_setting  # noqa: E402
import defect_analyzer  # noqa: E402
import detector  # noqa: E402
import image_io  # noqa: E402
import preprocess  # noqa: E402
from batch_loader import is_sample  # noqa: E402  工具产物不算样本，理由见 batch_loader

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(errors='replace')   # GBK 控制台别被一个字符带走整份报告


def _one(img: np.ndarray, coin: str) -> tuple[dict | None, float]:
    """返回（最优检测, 本张耗时 ms）。最优 = 面积最大的轮廓，即硬币本体。"""
    t0 = time.perf_counter()
    gray = preprocess.gray(img)
    blur = preprocess.gaussian_blur(gray, config_setting.gaussian_kernel)
    edges = preprocess.canny(blur, config_setting.canny_low, config_setting.canny_high)
    contours = detector.find_contours(edges, config_setting.min_area)
    ms = (time.perf_counter() - t0) * 1000
    if not contours:
        return None, ms
    f = detector.calculate_features(max(contours, key=cv2.contourArea))
    ppm = config_setting.ppm
    d = f['diameter_pixels'] / ppm
    deficit = f['deficit_pixels'] / ppm
    return ({'d': d, 'deficit': deficit,
             'result': defect_analyzer.judge_quality(d, deficit, f['circularity'], coin)}, ms)


def _part_id(stem: str) -> str:
    """从文件名剥掉尾部序号得到「哪一枚」：a01/a02 -> a；r07 -> r。

    所以命名规则是「前缀 = 第几枚，后缀 = 第几次重拍」。全部用 coin01..coin12 这种
    名字会被归成一枚 —— 报告会原样列出分组结果，看一眼就能发现命名不对。
    """
    return re.sub(r'[\d_\-\.]+$', '', stem) or stem


def _stats(folder: str, coin: str) -> dict | None:
    names = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png')) and is_sample(f))
    ds, defs, ms, grades, parts = [], [], [], {}, {}
    for name in names:
        det, this_ms = _one(image_io.read_image(os.path.join(folder, name)), coin)
        ms.append(this_ms)
        if det is not None:
            ds.append(det['d'])
            defs.append(det['deficit'])
            parts.setdefault(_part_id(os.path.splitext(name)[0]), []).append(det['d'])
            g = det['result']['severity']
            grades[g] = grades.get(g, 0) + 1
    if not names:
        return None
    return {'n': len(names), 'hit': len(ds), 'ds': ds, 'defs': defs, 'parts': parts,
            'ms': ms, 'grades': grades, 'coin': coin}


def _spread(s: dict) -> list[str]:
    """把散布拆成「重复性」和「枚间差异」—— 这两个不能混在一个 σ 里报。"""
    out = []
    reps = {k: v for k, v in s['parts'].items() if len(v) > 1}
    if reps:
        # 合并样本方差：同一枚内的散布才是测量重复性，跨枚平均会混进币的差异
        num = sum(statistics.variance(v) * (len(v) - 1) for v in reps.values())
        den = sum(len(v) - 1 for v in reps.values())
        lo, hi = min(len(v) for v in reps.values()), max(len(v) for v in reps.values())
        out.append(f"      重复性 σ = {math.sqrt(num / den):.3f}mm"
                   f"（{len(reps)} 枚 x 各 {lo if lo == hi else f'{lo}~{hi}'} 次，自由度 {den}）")
    else:
        out.append("      重复性 σ 算不出：没有一枚被重拍过两次以上（命名没区分枚，或没放 repeat）")
    means = [statistics.fmean(v) for v in s['parts'].values()]
    if len(means) > 1:
        out.append(f"      枚间差异：{len(means)} 枚，均值极差 {max(means) - min(means):.3f}mm"
                   f"  ← n={len(means)} 太小，只能报观测极差，不能当群体 σ 用")
    return out



def _fmt(s: dict) -> str:
    mean = statistics.fmean(s['ds']) if s['ds'] else float('nan')
    sd = statistics.stdev(s['ds']) if len(s['ds']) > 1 else 0.0
    nom = config_setting.coin_specs[s['coin']]['diameter']
    return (f"  {s['coin']:<3} n={s['hit']}/{s['n']} 检出  直径均值 {mean:7.3f}  σ={sd:.3f}mm  "
            f"相对标称 {mean - nom:+.3f}mm  缺损max {max(s['defs']) if s['defs'] else float('nan'):.2f}  "
            f"耗时 {statistics.fmean(s['ms']):.0f}ms/张  分级 {s['grades']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="硬币批次指标报告")
    ap.add_argument("input", help="含面额子目录的根目录")
    args = ap.parse_args()

    if tuple(config_setting.calib_image_size) == (0, 0):
        sys.exit("尚未标定 ppm，先跑 tools/measure_ppm.py --write")
    subs = [d for d in sorted(os.listdir(args.input)) if os.path.isdir(os.path.join(args.input, d))]
    if not subs:
        sys.exit(f"{args.input} 下没有子目录；按面额建 1元/5角/1角，重复性放 repeat")

    w, h = config_setting.calib_image_size
    print(f"标定 ppm {config_setting.ppm:.4f}（1px = {1000 / config_setting.ppm:.1f}µm）| 标定图 {w}x{h}\n")

    batches, repeat = {}, None
    for sub in subs:
        path = os.path.join(args.input, sub)
        if sub.lower() == 'repeat':
            repeat = _stats(path, '1元')
        elif sub in config_setting.coin_specs:
            s = _stats(path, sub)
            if s:
                batches[sub] = s
        else:
            print(f"跳过 {sub}：不是已知面额（{list(config_setting.coin_specs)}）")
    if not batches:
        sys.exit("没有可统计的面额目录")

    print('== 各面额 ==')
    for s in batches.values():
        print(_fmt(s))
        for line in _spread(s):
            print(line)
        n_parts = len(s['parts'])
        if n_parts >= 8 and len(s['ds']) > 1:
            mean, sd = statistics.fmean(s['ds']), statistics.stdev(s['ds'])
            out = [round(d, 3) for d in s['ds'] if abs(d - mean) > 3 * sd]
            if out:
                print(f"      离群点(>3σ) {out}  ← 先确认不是拍糊、放偏，或那枚真有缺口")
            print(f"      SPC 控制限 {mean - 3 * sd:.3f} ~ {mean + 3 * sd:.3f} mm")
        else:
            print(f"      SPC 控制限不出：只有 {n_parts} 枚（需 >=8 枚不同硬币），"
                  f"重复拍的张数不能顶这个数 —— 同一枚的固定偏差不会随重拍平均掉")

    print('\n== 线性度（不需要卡尺的准确度证据）==')
    order = [c for c in ('1角', '5角', '1元') if len(batches.get(c, {}).get('ds', [])) > 1]
    for a, b in zip(order, order[1:]):
        got = statistics.fmean(batches[b]['ds']) - statistics.fmean(batches[a]['ds'])
        want = config_setting.coin_specs[b]['diameter'] - config_setting.coin_specs[a]['diameter']
        flag = 'PASS' if abs(got - want) < 0.1 else 'FAIL'
        print(f"  {a}→{b} 实测差 {got:+.3f}mm  标称差 {want:+.1f}mm  残差 {got - want:+.3f}mm [{flag}]")
    if len(order) < 2:
        print('  至少需要两个面额才能做线性度校验')

    print('\n== 重复性（同一枚挪开再放回，Type A 不确定度）==')
    if repeat and len(repeat['ds']) > 1:
        sd = statistics.stdev(repeat['ds'])
        print(f"  n={repeat['hit']}/{repeat['n']}  直径 σ = {sd:.3f}mm  "
              f"极差 = {max(repeat['ds']) - min(repeat['ds']):.3f}mm")
        print(f"  判据：σ 要远小于 tol({config_setting.coin_specs['1元']['tol']}mm) 和 "
              f"defect_light({config_setting.defect_light}mm)，否则分级会被噪声主导")
    elif repeat:
        print('  repeat 目录样本不足')
    else:
        print('  没有 repeat 目录 → 算不出重复性，README 的精度声明会缺一条腿')

    print('\n== 缺损底噪（完好币凹陷分布，用来定 defect_light）==')
    for s in batches.values():
        if s['defs']:
            print(f"  {s['coin']}  中位 {statistics.median(s['defs']):.3f}  最大 {max(s['defs']):.3f}mm"
                  f"  ← 最大若已接近门限 {config_setting.defect_light}mm，说明齿边抬高了底噪，门限要重定")


if __name__ == "__main__":
    main()

