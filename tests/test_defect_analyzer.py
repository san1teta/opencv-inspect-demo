"""分级判据。所有直径入参一律写成「标称 + 偏移」，不写死绝对值。

为什么：标称直径属于版别而非面额（1元 2019 版 22.25mm，1999~2018 版 25.0mm）。
2026-10-09 把 coin_specs 里的 1元 从 25.0 改成 22.25 时，本文件 9 个用例全红 ——
因为它们把 25.x 当成「合格附近」写死了。绝对值写进测试，等于把测试和一次
版本假设焊死；改成相对标称后，标称再动这些用例照样成立。
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))

from defect_analyzer import judge_quality          # noqa: E402
import config_setting                              # noqa: E402

OK_DEFICIT = 0.10   # 完整圆的径向凹陷噪声底（实测值）
STD = config_setting.coin_specs['1元']['diameter']    # 标称随版别走，不在这里写死


def _q(diameter, deficit=OK_DEFICIT, circularity=0.95, coin='1元'):
    return judge_quality(diameter, deficit, circularity, coin)


class TestSizePath(unittest.TestCase):
    """尺寸路：超出量 = ||直径 - 标称| - 允差(0.25)|，再按 0.3 / 0.6 分档。"""

    def test_合格(self):
        r = _q(STD)
        self.assertEqual((r['circularity'], r['severity'], r['basis']), ('合格', '合格', '无'))

    def test_允差内不算超差(self):
        self.assertEqual(_q(STD + 0.2)['severity'], '合格')

    def test_偏大轻度(self):
        r = _q(STD + 0.3)
        self.assertEqual((r['severity'], r['direction'], r['basis']), ('轻度', '偏大', '尺寸偏大'))

    def test_偏大中度(self):
        self.assertEqual(_q(STD + 0.6)['severity'], '中度')

    def test_偏大重度(self):
        self.assertEqual(_q(STD + 0.9)['severity'], '重度')

    def test_偏小(self):
        r = _q(STD - 0.5)
        self.assertEqual((r['severity'], r['direction']), ('轻度', '偏小'))

    def test_5角用5角的标称(self):
        std5 = config_setting.coin_specs['5角']['diameter']
        self.assertEqual(_q(std5, coin='5角')['severity'], '合格')
        self.assertEqual(_q(std5 + 1.0, coin='5角')['severity'], '重度')


class TestEdgePath(unittest.TestCase):
    """边缘路：径向最大凹陷按 0.4 / 0.8 / 1.5mm 分档。"""

    def test_噪声底不误报(self):
        self.assertEqual(_q(STD, deficit=0.10)['severity'], '合格')

    def test_三档依次触发(self):
        self.assertEqual(_q(STD, deficit=0.50)['severity'], '轻度')
        self.assertEqual(_q(STD, deficit=1.00)['severity'], '中度')
        self.assertEqual(_q(STD, deficit=2.00)['severity'], '重度')

    def test_缺口直径看不出来但凹陷能看出来(self):
        """这条是换判据的全部理由：1mm 深的缺口只让等效直径动 0.035mm（尺寸路必然漏）。"""
        r = _q(STD + 0.035, deficit=1.001)
        self.assertEqual(r['severity'], '中度')
        self.assertEqual(r['basis'], '刀口1.00mm')
        self.assertEqual(r['direction'], '缺损')


class TestCombination(unittest.TestCase):
    def test_取更差的一档(self):
        """尺寸轻度 + 刀口重度 = 重度，不能相加也不能取先到的一条。"""
        r = _q(STD + 0.3, deficit=2.0)
        self.assertEqual(r['severity'], '重度')
        self.assertEqual(r['basis'], '刀口2.00mm')

    def test_同档两路都写进依据(self):
        r = _q(STD + 0.9, deficit=2.0)
        self.assertEqual(r['severity'], '重度')
        self.assertEqual(r['basis'], '尺寸偏大+刀口2.00mm')

    def test_形状不合格不再清空其他列(self):
        """形状不合格是**一路判据**，不是「这张图能不能测」的开关。

        原来这里是 `return`，于是 deviation/deficit/severity 全被清空。实拍里同一枚
        完好 1角 因为轮廓上一根一像素毛刺把外接圆撑大 1.7mm，圆度从 0.977 掉到 0.824，
        一张好端端的数据在报表里就成了「未检测」，量出来的直径和缺损一起丢掉 ——
        而 min_circularity 这条临时门限恰恰要靠这些分布来重定。
        """
        r = _q(STD, deficit=3.0, circularity=0.5)
        self.assertEqual(r['circularity'], '不合格')
        self.assertEqual(r['severity'], '重度', '刀口 3mm 是重度，不能被形状一路盖成未检测')
        self.assertEqual(r['deficit_mm'], 3.0, '缺损数值必须留下')
        self.assertEqual(r['deviation'], 0.0, '偏差数值必须留下')

    def test_形状单独命中至少轻度(self):
        """尺寸和边缘都在限内、只有形状不合格 → 至少轻度，且依据写明是形状。

        圆度入参必须**从门限推导**，不写死。本文件开头立了「不写绝对值」这条规矩，
        可这里原来写的是 0.85 —— 那是「刚好低于当时 0.9 门限」的意思。2026-10-10 门限
        按面额改成 1元 0.80 之后，0.85 变成合格，这条用例就会**静默地不再测它想测的东西**
        （断言全绿但方向反了），比直接红掉更危险。
        """
        below = config_setting.min_circ_for('1元') - 0.05
        r = _q(STD, deficit=0.10, circularity=below)
        self.assertEqual(r['severity'], '轻度')
        self.assertEqual(r['direction'], '形状')
        self.assertIn('形状', r['basis'])

    def test_形状门限必须按面额分开(self):
        """5角/1角 边部齿更密，把 面积/外接圆面积 结构性拉低 —— 这不是缺陷。

        真样本实测采信币最低圆度：1元 0.841 / 5角 0.738 / 1角 0.745。
        一个全局 0.9 会同时误杀三个面额；一个全局 0.80 仍会误杀 5角 和 1角。
        所以门限必须挂面额，而且必须留在这三个观测值之下。
        """
        obs = {'1元': 0.841, '5角': 0.738, '1角': 0.745}
        for coin, lo in obs.items():
            thr = config_setting.min_circ_for(coin)
            self.assertLess(thr, lo, f'{coin} 门限 {thr} 高于实测最低圆度 {lo}，会误杀良品')
            self.assertGreater(thr, lo - 0.10,
                               f'{coin} 门限 {thr} 离实测最低 {lo} 太远，形状判据形同虚设')
        self.assertNotEqual(config_setting.min_circ_for('1元'),
                            config_setting.min_circ_for('5角'),
                            '三个面额的可信币圆度差到 0.1，不该共用一个门限')

    def test_缺损量进报表(self):
        self.assertEqual(_q(STD, deficit=1.234)['deficit_mm'], 1.234)


if __name__ == '__main__':
    unittest.main()

