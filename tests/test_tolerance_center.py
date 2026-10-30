"""允差中心拍板（2026-10-30）：保持「中心 = 标称」，本测试锁死这条口径。

背景（HANDOFF-TO-DSH.md §八）：相机读完好币系统性偏小约 0.12~0.13mm
（1元 22.162 / 5角 20.362 / 1角 18.768，2026-10-10 批，ppm=8.6510，采信枚 7/3/3）。
卡尺记录里两条指示互相冲突：34 行「不许为了消掉偏差去移动允差中心」、
35 行「允差中心取相机读完好币的均值」。拍板采 34 行 —— 把中心挪到相机自己的均值
= 用同一台机器给自己定基准，准确度就再没有外部锚了。偏差只作文档常量，不进判定链。

所以将来任何人想「优化」以下任何一条，都必须先推翻 §八 的论证，而不是改完测试了事：
  1) coin_specs 的 diameter 是**国家标称**，不是相机均值，也不是卡尺均值；
  2) judge_quality 的 deviation 按「读数 − 标称」算，不减任何偏差修正；
  3) 已声明的误判（1角 c 挂「轻度」）保持原样 —— 那是口径的代价，不是 bug。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))

import config_setting                       # noqa: E402
from defect_analyzer import judge_quality   # noqa: E402

# 2026-10-10 批实测均值（ppm=8.6510）。中心绝不允许挪到这些数上。
CAMERA_MEANS = {'1元': 22.162, '5角': 20.362, '1角': 18.768}
CALIPER_MEANS = {'1元': 22.28, '5角': 20.35, '1角': 18.90}


class TestToleranceCenter(unittest.TestCase):
    def test_中心必须是国标标称而不是相机或卡尺均值(self):
        nominal = {'1元': 22.25, '5角': 20.5, '1角': 19.0}
        for coin, dia in nominal.items():
            self.assertEqual(config_setting.coin_specs[coin]['diameter'], dia,
                             f'{coin} 的中心被动过 —— 见 §八 拍板，不许挪')
            self.assertNotAlmostEqual(config_setting.coin_specs[coin]['diameter'],
                                      CAMERA_MEANS[coin], delta=0.02)
            self.assertNotAlmostEqual(config_setting.coin_specs[coin]['diameter'],
                                      CALIPER_MEANS[coin], delta=0.02)

    def test_偏差按读数减标称算不做偏差修正(self):
        """相机读「1元 采信均值 22.162」时偏差必须是 −0.09（离标称），不是 0。

        若有人给判定链加了 −0.12mm 的「修正」，这里会变 0 或同号 —— 那就是
        把测量缺陷校准成正常，卡尺记录第 34 行警告的事。
        """
        r = judge_quality(22.162, 0.10, 0.99, '1元')
        self.assertAlmostEqual(r['deviation'], -0.09, places=2)
        self.assertEqual(r['severity'], '合格')  # 0.09 < tol 0.25，仍在合格区

    def test_已声明的误判保持原样(self):
        """1角 c（相机 18.711）离标称 0.289 > 0.25 ⇒ 轻度。

        卡尺说那批 4 枚全是 18.9，币大概率没问题 —— 但按 §八 口径它就是「轻度」，
        这是**写进 README 的代价**，谁也不许用挪中心或修读数的方式把它悄悄洗白。
        """
        r = judge_quality(18.711, 0.10, 0.99, '1角')
        self.assertAlmostEqual(r['deviation'], -0.29, places=2)
        self.assertEqual(r['severity'], '轻度')
        self.assertIn('尺寸偏小', r['basis'])

    def test_允差是能力带宽不是国标(self):
        """三面额 tol 一律 0.25：它是按本系统能力算的最小可判带宽，不是查来的标准。

        能力构成（README 允差段）：比例尺梯度 0.093 + 测量链 0.014 + 像素噪声 0.10
        ≈ 0.14，留余量到 0.25。若有人「查到了国标」想替换，先读 §三 的警告再动手。
        """
        for coin in CAMERA_MEANS:
            self.assertEqual(config_setting.coin_specs[coin]['tol'], 0.25)


if __name__ == '__main__':
    unittest.main()
