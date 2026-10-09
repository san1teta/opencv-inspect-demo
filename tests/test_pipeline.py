"""流水线的尺寸闸门：背景纹理不能被当成币。

回归的是 2026-10-09 的第一张实拍。鼠标垫织纹 + 画面下方一个失焦的桶，batch 模式
检出 51 个「币」（1.3~4.8mm 的噪声团，每个在 CSV 里占一行），均值、σ、检出率全被
这 50 行垃圾灌掉；同一张图 tools/check_placement.py 却只报 1 个目标 —— 因为闸门
当初只长在工具里，没长在流水线上，而报表走的是流水线。

此前的合成测试背景一律是纯色，所以这条必须显式造一个「有纹理的背景」才守得住。
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))

import cv2
import numpy as np

import config_setting
import detector
import pipeline
import preprocess

# 噪声团要过 min_area(100px²) 才会进候选，所以半径不能小于 6px。
# 实测这个尺寸段量出 1.4~1.8mm，与实拍那 50 个垃圾行同源同尺寸。
NOISE = [((60, 80), 6), ((640, 90), 8), ((90, 430), 7),
         ((610, 420), 6), ((350, 60), 8), ((350, 450), 7)]

# 颜色必须是三元组。cv2.circle 在 3 通道图上喂标量整数（如 235）只写进蓝通道，
# preprocess.gray 的 BGR2GRAY 之后就剩 0.114*235≈27，与背景 60 的对比度被压到 33，
# Canny 高门限 150 下直接一个像素都不剩 —— 场景看着"画好了"，实际是张空白图。
BG = (60, 60, 60)
NOISE_FILL = (235, 235, 235)
COIN_FILL = (215, 215, 215)


def _scene(with_coin: bool = True, diameter_mm: float | None = None) -> np.ndarray:
    img = np.full((500, 700, 3), BG, np.uint8)
    for (x, y), r in NOISE:
        cv2.circle(img, (x, y), r, NOISE_FILL, -1)
    if with_coin:
        mm = diameter_mm if diameter_mm is not None else config_setting.coin_specs['1元']['diameter']
        r = int(round(mm * config_setting.ppm / 2))
        cv2.circle(img, (350, 250), r, COIN_FILL, -1)
    return img


def _edges(img: np.ndarray) -> np.ndarray:
    gray = preprocess.gray(img)
    return preprocess.canny(preprocess.gaussian_blur(gray, config_setting.gaussian_kernel),
                            config_setting.canny_low, config_setting.canny_high)


class TestDiameterGate(unittest.TestCase):
    def setUp(self):
        # 闸门按 config 的 ppm 换算，测试也按同一个 ppm 画圆，两边不能各用各的
        self.ppm = config_setting.ppm

    def test_background_alone_yields_no_coin(self):
        """没有币就是没有币 —— 不能因为背景里有圆形纹理就报「检出」。

        先断言背景团确实进了候选：少了这一句，一张空白图也能让这个用例「通过」，
        而它什么也没守（夹具踩过的真实坑见 NOISE_FILL 上方注释）。
        """
        feats = [detector.calculate_features(c)
                 for c in detector.find_contours(_edges(_scene(with_coin=False)), config_setting.min_area)]
        self.assertGreaterEqual(len(feats), len(NOISE), '背景噪声团没进候选，夹具失效了')
        _, dets = pipeline.process_single_frame(_scene(with_coin=False), '1元')
        self.assertEqual(dets, [], '背景噪声团被当成币检出，报表会被垃圾行灌满')

    def test_one_coin_one_row_despite_noise(self):
        _, dets = pipeline.process_single_frame(_scene(), '1元')
        self.assertEqual(len(dets), 1, '一枚币就该是一行；多出来的每一行都是背景')
        spec = config_setting.coin_specs['1元']
        self.assertLess(abs(dets[0]['diameter_mm'] - spec['diameter']), 0.25)

    def test_circularity_alone_cannot_do_this_job(self):
        """为什么闸门是直径而不是圆度：实拍里圆度低的币是真缺陷，判据不能吞掉它。

        这个场景确实存在圆度高于门限的噪声团 —— 只按 circularity 过滤会放它过去，
        而只按圆度过滤的 check_placement 之所以没出事，是因为那张图运气好。
        """
        feats = [detector.calculate_features(c)
                 for c in detector.find_contours(_edges(_scene(with_coin=False)), config_setting.min_area)]
        self.assertTrue(feats, '噪声团一个都没检出，这个场景没在测东西')
        self.assertGreater(max(f['circularity'] for f in feats), config_setting.min_circularity)
        lo, hi = detector.diameter_window(config_setting.coin_specs['1元']['diameter'],
                                         self.ppm, config_setting.diameter_gate_mm)
        self.assertEqual([f for f in feats if lo <= f['diameter_pixels'] <= hi], [])

    def test_worn_coin_is_reported_not_swallowed(self):
        """闸门只管「是不是币」，「合不合格」归 defect_analyzer。

        20.6mm 比标称小 1.65mm，超出允差 0.25mm 很多，必须如实出现在报表里判重度；
        要是拿允差当检出条件，这种币就会变成「未检出」——最该报警的那类反而查不到。
        """
        img = _scene(with_coin=False)
        cv2.circle(img, (350, 250), int(round(20.6 * self.ppm / 2)), COIN_FILL, -1)
        _, dets = pipeline.process_single_frame(img, '1元')
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]['result']['severity'], '重度')

    def test_other_denomination_does_not_pass_the_wrong_gate(self):
        """1角(19.0) 放错到 1元 批次里，离标称 3.25mm，超出闸门 2mm → 不认。"""
        img = _scene(with_coin=False)
        cv2.circle(img, (350, 250), int(round(19.0 * self.ppm / 2)), COIN_FILL, -1)
        _, dets = pipeline.process_single_frame(img, '1元')
        self.assertEqual(dets, [])

    def test_numeric_circularity_reaches_the_report(self):
        """CSV 的「圆度」列要拿到数值。以前只有「合格/不合格」，
        min_circularity=0.90 这条临时门限就永远等不到真样本分布来重定。"""
        _, dets = pipeline.process_single_frame(_scene(), '1元')
        self.assertIsInstance(dets[0]['circularity'], float)
        self.assertGreater(dets[0]['circularity'], config_setting.min_circularity)


if __name__ == '__main__':
    unittest.main()
