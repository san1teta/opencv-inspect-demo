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

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))

import config_setting  # noqa: E402
import defect_analyzer  # noqa: E402
import detector  # noqa: E402
import image_io  # noqa: E402
import preprocess  # noqa: E402
from batch_loader import is_sample  # noqa: E402  工具产物不算样本，理由见 batch_loader
from calib_guard import stale_paths  # noqa: E402  新旧装置照片混批 = 报表作废，判据见该模块
import pipeline  # noqa: E402  检出走 find_coins，与流水线共用一份判据

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(errors='replace')   # GBK 控制台别被一个字符带走整份报告


def _one(img: np.ndarray, coin: str) -> tuple[dict | None, float]:
    """返回（最优检测, 本张耗时 ms）。最优 = 过了尺寸闸门之后面积最大的轮廓。

    必须先过闸门再挑最大。原来的写法是「直接取面积最大的轮廓」，两个后果：
    1) 币没入画时它选中背景团，照样给出一个直径 —— 于是「检出率」恒为 100%，
       数字好看但全是假的，而这恰恰是要拿去写 README 精度声明的那个数；
    2) 背景里比币大的东西（本装置画面下方那个失焦的桶）会把币顶掉。

    检出本身改走 `pipeline.find_coins`：以前这里自己 canny + find_contours + 窗口过滤
    走了一遍，等于和流水线各写一套判据 —— 2026-10-10 补的「币缘断口闭运算兜底」
    就只有流水线有、报表没有，同一个坑换个位置再踩。现在三处共用一份。
    耗时口径不变，仍只量到 find_contours 为止，好让新旧 ms/张 可比。
    """
    t0 = time.perf_counter()
    pairs = pipeline.find_coins(img, coin)[0]
    ms = (time.perf_counter() - t0) * 1000
    if not pairs:
        return None, ms
    f = max((f for _, f in pairs), key=lambda x: x['area'])
    ppm = config_setting.ppm
    d = f['diameter_pixels'] / ppm
    deficit = f['deficit_pixels'] / ppm
    return ({'d': d, 'deficit': deficit, 'circ': f['circularity'],
             'caliper': f['caliper_diameter_pixels'] / ppm,
             # 互检/兜底标记：这两类帧的读数不该和正常帧混在同一个 σ 里
             'flag': ('可疑' if f.get('edge_suspect') else
                      ('兜底' if f.get('edge_closed') else '')),
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
    # 报表最怕的就是「新旧装置的照片混在一批」：旧照片会被当前 ppm 量出一个看着正常的直径，
    # 均值和 σ 一起废掉，而检出率还是 100%。判据与验收工具共用 calib_guard，见该模块。
    stale = stale_paths([os.path.join(folder, n) for n in names],
                        getattr(config_setting, 'calib_stamp', ''))
    if stale:
        print(f"  [警告] {coin}：{len(stale)}/{len(names)} 张比标定"
              f"（{config_setting.calib_stamp}）旧，当前 ppm 量它们没有意义，"
              f"下面的均值/σ 已被污染："
              + '、'.join(os.path.basename(p) for p in stale[:8])
              + ('……' if len(stale) > 8 else ''))
    ds, defs, cs, ms, grades, parts, flags, ptot = [], [], [], [], {}, {}, [], {}
    s_defs: dict[str, list[float]] = {}   # 按枚存缺损，才能只统计采信枚
    s_circ: dict[str, list[float]] = {}   # 同上，圆度是定 min_circularity 的依据
    s_sev: dict[str, list[str]] = {}      # 同上，分级也要能只算采信枚
    for name in names:
        pid = _part_id(os.path.splitext(name)[0])
        # 分母要按「这一枚实际有几张」算，不是「检出了几张」—— 漏检的那几张正是
        # 拒收依据（见 config_setting.coin_spread_max_mm 那段），不记就永远看不见。
        ptot[pid] = ptot.get(pid, 0) + 1
        det, this_ms = _one(image_io.read_image(os.path.join(folder, name)), coin)
        ms.append(this_ms)
        if det is not None:
            ds.append(det['d'])
            defs.append(det['deficit'])
            cs.append(det['circ'])
            flags.append(det['flag'])
            parts.setdefault(pid, []).append(det['d'])
            s_defs.setdefault(pid, []).append(det['deficit'])
            s_circ.setdefault(pid, []).append(det['circ'])
            g = det['result']['severity']
            grades[g] = grades.get(g, 0) + 1
            s_sev.setdefault(pid, []).append(g)
    if not names:
        return None
    return {'n': len(names), 'hit': len(ds), 'ds': ds, 'defs': defs, 'cs': cs, 'parts': parts,
            'pdefs': s_defs, 'pcirc': s_circ, 'psev': s_sev, 'ptot': ptot, 'ms': ms,
            'grades': grades, 'flags': flags, 'coin': coin}


def _accept(s: dict) -> list[tuple[str, float, bool, str]]:
    """枚级拒收判定。判据与来由全在 config_setting 的注释里，这里只执行。

    返回 [(枚, 枚均值, 是否采信, 拒收原因)]。这一条是本项目第一次把「哪几枚的数
    能要」写成显式规则 —— 之前报表把所有帧混进同一个 σ，等于默认每张都可信。
    """
    nom = config_setting.coin_specs[s['coin']]['diameter']
    rows = []
    for k in sorted(s['parts']):
        v = s['parts'][k]
        tot = s['ptot'].get(k, len(v))
        why = []
        if len(v) < tot:
            why.append(f'{tot - len(v)}/{tot} 张未检出')
        if len(v) > 1 and max(v) - min(v) > config_setting.coin_spread_max_mm:
            why.append(f'组内极差 {max(v) - min(v):.3f}mm'
                       f'>{config_setting.coin_spread_max_mm}mm')
        mean = statistics.fmean(v)
        if abs(mean - nom) > config_setting.coin_nominal_max_mm:
            why.append(f'离标称 {mean - nom:+.3f}mm'
                       f'>{config_setting.coin_nominal_max_mm}mm')
        rows.append((k, mean, not why, '；'.join(why)))
    return rows


def _spread(s: dict, ok: set[str] | None = None) -> list[str]:
    """把散布拆成「重复性」和「枚间差异」—— 这两个不能混在一个 σ 里报。

    ok 给定时只用**枚级判定采信**的枚。这一步不是洁癖：b/j 那种整枚读错的组，
    它自己的组内极差能到 0.24mm，混进来会把重复性 σ 从 0.01 抬到 0.09，
    于是「这台机器重复性 0.09mm」成了假话 —— 真实重复性是被错样本拖累出来的。
    """
    parts = {k: v for k, v in s['parts'].items() if ok is None or k in ok}
    out = []
    reps = {k: v for k, v in parts.items() if len(v) > 1}
    if ok is not None and len(ok) < len(s['parts']):
        out.append(f"      （枚级判定剔掉 {len(s['parts']) - len(ok)} 枚，"
                   f"下面只统计采信的那 {len(ok)} 枚）")
    if reps:
        # 合并样本方差：同一枚内的散布才是测量重复性，跨枚平均会混进币的差异
        num = sum(statistics.variance(v) * (len(v) - 1) for v in reps.values())
        den = sum(len(v) - 1 for v in reps.values())
        lo, hi = min(len(v) for v in reps.values()), max(len(v) for v in reps.values())
        out.append(f"      重复性 σ = {math.sqrt(num / den):.3f}mm"
                   f"（{len(reps)} 枚 x 各 {lo if lo == hi else f'{lo}~{hi}'} 次，自由度 {den}）")
    else:
        out.append("      重复性 σ 算不出：没有一枚被重拍过两次以上（命名没区分枚，或没放 repeat）")
    means = [statistics.fmean(v) for v in parts.values()]
    if len(means) > 1:
        out.append(f"      枚间差异：{len(means)} 枚，均值极差 {max(means) - min(means):.3f}mm"
                   f"  ← n={len(means)} 太小，只能报观测极差，不能当群体 σ 用")
    return out



def _fmt(s: dict) -> str:
    """头一行优先用**枚级判定采信**后的读数。

    这行是最容易被直接抄进 README 的数，让它带着 b/e/j 那种整枚读错的样本，
    等于把 0.42mm 的假 σ 当成机器能力去写精度声明。
    """
    vals = s.get('ads') or s['ds']
    dv = s.get('adefs') or s['defs']
    mean = statistics.fmean(vals) if vals else float('nan')
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    nom = config_setting.coin_specs[s['coin']]['diameter']
    nf = sum(1 for x in s.get('flags', []) if x)
    tag = '（仅采信枚）' if s.get('ads') else ''
    return (f"  {s['coin']:<3} n={s['hit']}/{s['n']} 检出  直径均值 {mean:7.3f}{tag}  σ={sd:.3f}mm  "
            f"相对标称 {mean - nom:+.3f}mm  缺损max {max(dv) if dv else float('nan'):.2f}  "
            f"耗时 {statistics.fmean(s['ms']):.0f}ms/张  分级 {s.get('agrades') or s['grades']}"
            + (f"\n      ⚠ {nf}/{s['hit']} 张读数可疑（互检分叉或闭运算兜底）"
               f" —— 上面的 σ 与控制限被这些帧污染，别直接引用" if nf else ''))


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
        rows = _accept(s)
        ok = {k for k, _, good, _ in rows if good}
        s['ads'] = [d for k in ok for d in s['parts'][k]]
        s['adefs'] = [d for k in ok for d in s['pdefs'].get(k, [])]
        s['acs'] = [d for k in ok for d in s['pcirc'].get(k, [])]
        gc: dict[str, int] = {}
        for k in ok:
            for g in s['psev'].get(k, []):
                gc[g] = gc.get(g, 0) + 1
        s['agrades'] = gc
        print(_fmt(s))
        for k, mean, good, why in rows:
            print(f"      枚 {k}: {mean:7.3f}mm  {'采信' if good else '拒收'}"
                  + (f'  ← {why}' if why else ''))
        for line in _spread(s, ok):
            print(line)
        ads = s['ads']
        n_parts = len(ok)
        if n_parts >= 8 and len(ads) > 1:
            mean, sd = statistics.fmean(ads), statistics.stdev(ads)
            out = [round(d, 3) for d in ads if abs(d - mean) > 3 * sd]
            if out:
                print(f"      离群点(>3σ) {out}  ← 先确认不是拍糊、放偏，或那枚真有缺口")
            print(f"      SPC 控制限 {mean - 3 * sd:.3f} ~ {mean + 3 * sd:.3f} mm")
        else:
            print(f"      SPC 控制限不出：采信枚数 {n_parts}（需 >=8 枚不同硬币），"
                  f"重复拍的张数不能顶这个数 —— 同一枚的固定偏差不会随重拍平均掉")

    print('\n== 线性度（不需要卡尺的准确度证据）==')
    def _val(s: dict) -> list[float]:
        """线性度用**采信枚**的读数。混进 b 那种整枚读错 0.6mm 的样本，
        面额差就跟着错 —— 而线性度正是我们唯一不依赖卡尺的准确度证据。"""
        return s.get('ads') or s['ds']

    order = [c for c in ('1角', '5角', '1元') if len(_val(batches.get(c, {'ds': []}))) > 1]
    for a, b in zip(order, order[1:]):
        got = statistics.fmean(_val(batches[b])) - statistics.fmean(_val(batches[a]))
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
        # 只用采信枚：b/e/j 那种闭到错结构上的帧会算出 2.4mm 的假缺损，
        # 拿它去抬 defect_light 等于把检测器的错误校准成「正常底噪」。
        dv = s.get('adefs') or s['defs']
        if dv:
            print(f"  {s['coin']}  中位 {statistics.median(dv):.3f}  最大 {max(dv):.3f}mm"
                  f"  ← 最大若已接近门限 {config_setting.defect_light}mm，说明齿边抬高了底噪，门限要重定")

    print('\n== 圆度分布（门限按面额，见 coin_specs[min_circ]）==')
    for s in batches.values():
        cv_ = s.get('acs') or s['cs']
        thr = config_setting.min_circ_for(s['coin'])
        if cv_:
            lo = min(cv_)
            below = sum(1 for x in cv_ if x < thr)
            print(f"  {s['coin']}  最小 {lo:.3f}  中位 {statistics.median(cv_):.3f}"
                  f"  门限 {thr}  余量 {lo - thr:+.3f}  低于门限 {below}/{len(cv_)} 帧"
                  + ('  ← ⚠ 采信币被门限判成形状不合格，误杀良品' if below else ''))
        else:
            print(f"  {s['coin']}  无检出，拿不到圆度")


if __name__ == "__main__":
    main()

