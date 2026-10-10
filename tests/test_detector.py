"""detector 的估计量必须对着「已知真值的圆」来量，光看跑不跑得通没用。

一律走真实预处理链（高斯 + Canny），因为偏差恰恰出在 Canny 环的外多边形上，
直接喂一个理想轮廓是测不出来的。
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))

import cv2
import numpy as np

import config_setting
import detector
import preprocess

PPM = 9.9749   # 2026-10-08 A4 实测标定值，1px ≈ 0.1mm


def _measure(diameter_mm: float, nick_mm: float = 0.0) -> tuple[dict, float]:
    """画一枚直径已知的硬币，返回（最大轮廓特征, 实际画出来的直径 mm）。

    cv2.circle 的半径只能取整，所以真值是「画出来的」那个直径，不是传进去的。
    nick_mm 是从边缘啃掉的缺口深度（咬痕圆的半径 = 缺口深度）。
    """
    img = np.full((700, 700), 40, np.uint8)
    r = int(round(diameter_mm * PPM / 2))
    cv2.circle(img, (350, 350), r, 200, -1)
    if nick_mm > 0:
        # 圆心压在轮廓上、半径 = 深度，所以啃进去正好 nick_mm（与 selftest_measure 同一约定）
        cv2.circle(img, (350 + r, 350), int(round(nick_mm * PPM)), 40, -1)
    edges = preprocess.canny(
        preprocess.gaussian_blur(img, config_setting.gaussian_kernel),
        config_setting.canny_low, config_setting.canny_high)
    cnt = max(detector.find_contours(edges, config_setting.min_area), key=cv2.contourArea)
    return detector.calculate_features(cnt), 2 * r / PPM


def _bucket(depth_mm: float) -> str:
    """缺损分级档位 —— 真正要守住的不是小数点，是落在哪一档。"""
    for name, lim in (('重度', config_setting.defect_heavy),
                      ('中度', config_setting.defect_mid),
                      ('轻度', config_setting.defect_light)):
        if depth_mm >= lim:
            return name
    return '合格'


def _spiked(diameter_mm: float, spike_px: int) -> tuple[dict, float]:
    """画一枚边缘带毛刺（向外凸起、不吞材料）的币，走同一条真实预处理链。"""
    img = np.full((700, 700), 40, np.uint8)
    r = int(round(diameter_mm * PPM / 2))
    cv2.circle(img, (350, 350), r, 200, -1)
    cv2.line(img, (350 + r - 4, 350), (350 + r + spike_px, 350), 200, 3)
    edges = preprocess.canny(preprocess.gaussian_blur(img, config_setting.gaussian_kernel),
                             config_setting.canny_low, config_setting.canny_high)
    cnt = max(detector.find_contours(edges, config_setting.min_area), key=cv2.contourArea)
    return detector.calculate_features(cnt), 2 * r / PPM


class TestCalculateFeatures(unittest.TestCase):
    def test_diameter_has_no_systematic_bias(self):
        """曾经用外接圆直径，19/20.5/25mm 一律 +1.21px（+0.12mm），吃光整个精度预算。"""
        for d in (19.0, 20.5, 25.0):
            with self.subTest(diameter=d):
                f, drawn = _measure(d)
                self.assertLess(abs(f['diameter_pixels'] / PPM - drawn), 0.05)

    def test_perfect_circle_clears_the_gate(self):
        """完美圆的 4πA/P² 只有 0.899，低于门槛 0.9 → 任何硬币都会被判不合格。"""
        f, _ = _measure(25.0)
        self.assertGreater(f['circularity'], config_setting.min_circularity)

    def test_intact_coin_sits_on_the_deficit_noise_floor(self):
        """完整圆的径向凹陷必须贴着噪声底，否则刀口判据会误报成真币缺陷。"""
        for d in (19.0, 25.0):
            with self.subTest(diameter=d):
                f, _ = _measure(d)
                self.assertLess(f['deficit_pixels'] / PPM, 0.15)

    def test_deficit_measures_notch_depth(self):
        """径向凹陷要定量：报出来的数就该落在缺口真值附近，而且必须落进同一档。

        容差按相对误差 15% 给，不按绝对 0.15mm —— 质心会被啃掉的材料推向缺口反向，
        缺口越深低估越多（3mm 真值报 2.59mm）。分档才是这套判据真正要守住的东西。
        """
        for depth in (0.5, 1.0, 2.0, 3.0):
            with self.subTest(depth=depth):
                f, _ = _measure(25.0, nick_mm=depth)
                got = f['deficit_pixels'] / PPM
                self.assertLessEqual(abs(got - depth) / depth, 0.15)
                self.assertEqual(_bucket(got), _bucket(depth))

    def test_outward_burr_does_not_invent_a_notch(self):
        """回归锁：边缘一处向外毛刺不得凭空造出缺口。

        这是真币上踩到的坑 —— 完好 1元 量出 deficit 0.690mm，越过 defect_light=0.4，
        每一枚良品都会被误判成有缺口。根因是 deficit 曾绕着 minEnclosingCircle 的圆心算，
        而外接圆只由 2~3 个极端点决定，一处毛刺就把圆心拖走 5px。改用质心后同一枚币
        降到 0.148mm。合成完美圆测不出这条，因为完美圆的外接圆圆心恰好就是真圆心。
        """
        for spike in (0, 8, 16):
            with self.subTest(spike_px=spike):
                f, _ = _spiked(25.0, spike)
                self.assertLess(f['deficit_pixels'] / PPM, config_setting.defect_light)

    def test_center_is_the_centroid_not_the_envelope(self):
        """'center' 要落在币的几何中心：放位检查和缺陷径向都靠它，不能被极端点带偏。"""
        f, _ = _spiked(25.0, 16)
        self.assertLess(abs(f['center'][0] - 350), 3)
        self.assertLess(abs(f['center'][1] - 350), 3)


    def test_notch_is_invisible_to_size_but_not_to_edge(self):
        """换判据的全部理由：2mm 深的缺口只让等效直径小 0.15mm —— 比允差 0.25mm 还小，
        尺寸路必然判合格；而缺损路直接报出 2mm 并给档。阈值挂在允差上，不挂在我随手写的数上。
        """
        intact, _ = _measure(25.0)
        nicked, _ = _measure(25.0, nick_mm=2.0)
        tol = config_setting.coin_specs['1元']['tol']
        self.assertLess(abs(intact['diameter_pixels'] - nicked['diameter_pixels']) / PPM, tol)
        self.assertGreater(nicked['deficit_pixels'] / PPM, config_setting.defect_mid)

    def test_overall_wear_is_not_reported_as_a_notch(self):
        """整体磨小是中位半径一起缩，凹陷仍是噪声底 —— 尺寸路和缺损路不能互相污染。"""
        big, _ = _measure(25.0)
        worn, _ = _measure(23.5)
        self.assertLess(worn['diameter_pixels'], big['diameter_pixels'])
        self.assertLess(worn['deficit_pixels'], 0.15 * PPM)


class TestDiameterWindow(unittest.TestCase):
    """闸门本身：像素窗口要等于「标称 ± gate」，两边都要算准。

    换算写在 detector 里，所以在这里锁死；写反一个符号（比如把 gate 加到直径上而不是
    标称两侧）在实拍上表现为「币全被毙掉」，而日志只说未检出，很难往回查。
    """

    def test_window_is_symmetric_around_nominal(self):
        lo, hi = detector.diameter_window(22.25, PPM, 2.0)
        self.assertAlmostEqual(lo / PPM, 20.25, places=6)
        self.assertAlmostEqual(hi / PPM, 24.25, places=6)

    def test_nominal_coin_falls_inside(self):
        f, _ = _measure(22.25)
        lo, hi = detector.diameter_window(22.25, PPM, config_setting.diameter_gate_mm)
        self.assertTrue(lo <= f['diameter_pixels'] <= hi,
                        f"标称尺寸的币被自己的闸门毙了：{f['diameter_pixels'] / PPM:.3f}mm")

    def test_reeding_shrinkage_still_clears_the_window(self):
        """实拍完好 1元 量出 22.07mm（比标称小 0.18mm，疑似齿边把等效直径拉到齿根侧）。
        闸门必须对这种系统性小偏完全无感，否则正常照片会被当成背景丢掉。"""
        f, _ = _measure(22.25 - 0.18)
        lo, hi = detector.diameter_window(22.25, PPM, config_setting.diameter_gate_mm)
        self.assertTrue(lo <= f['diameter_pixels'] <= hi)


class TestCircularityIsSpikeProof(unittest.TestCase):
    """圆度不能被一根毛刺摧毁 —— 2026-10-10 实拍：同一枚完好 1角 的三张连拍，
    面积只差 0.1%，外接圆被撑大 1.7mm，圆度 0.977 → 0.824，好币被判成形状不合格。
    """

    def test_spike_does_not_collapse_circularity(self):
        for spike in (0, 8, 16, 24):
            with self.subTest(spike_px=spike):
                f, _ = _spiked(25.0, spike)
                self.assertGreater(f['circularity'], config_setting.min_circularity,
                                   f'毛刺 {spike}px 把圆度压到 {f["circularity"]:.3f}')

    def test_a_real_notch_still_lowers_circularity(self):
        """反过来也得成立：真缺口是整段弧缺失，不能被稳健半径当成离群点滤掉。"""
        intact, _ = _measure(25.0)
        nicked, _ = _measure(25.0, nick_mm=3.0)
        self.assertLess(nicked['circularity'], intact['circularity'] - 0.01)


class TestCaliperDiameter(unittest.TestCase):
    """齿顶对径：与卡尺同口径的量。面积等效把齿槽算成「少掉的面积」，读数落在齿根侧。

    ⚠ 1元 实测比国标标称 22.25 小 0.21mm，但**这批样本没有卡尺实测值**，所以这 0.21mm 里
    混着「齿槽几何效应」「样本本身偏小」「标称值公差」三种成分，分不开。
    因此本组测试只锁得住「齿顶对径与面积等效是两个不同的量」，**锁不住「谁更接近真值」**
    —— 后者要等独立的长度基准，别拿这两条断言当它已经成立。
    """

    def test_bias_is_constant_not_proportional(self):
        """齿顶直径带一个 **+0.12mm 的恒定外扩**（Canny 环外多边形比真实边缘外扩 1.2px，
        而扇区取的是最外缘半径，把这个外扩全收了）。

        关键不是它有没有偏，而是**偏得不随尺寸变**：19/20.5/25mm 三档实测
        +0.121 / +0.130 / +0.122mm。常数偏移可以被「按面额用良品均值定允差中心」一口吃掉；
        如果是比例误差（随直径变大而变大），换面额就得重标，那才可怕。
        面积等效直径没有这个问题（它靠面积，带宽的外扩被平均掉了，实测 +0.014mm），
        但它被齿槽拉向齿根（-0.21mm 且逐枚不同）—— 两个量各有一种偏，所以两个都留着。
        """
        errs = []
        for d in (19.0, 20.5, 25.0):
            with self.subTest(diameter=d):
                f, drawn = _measure(d)
                e = f['caliper_diameter_pixels'] / PPM - drawn
                self.assertGreater(e, 0.0, '外扩方向必须为正（Canny 外多边形）')
                self.assertLess(e, 0.20)
                errs.append(e)
        self.assertLess(max(errs) - min(errs), 0.05,
                        f'偏移随尺寸漂移了 {max(errs) - min(errs):.3f}mm，不能当常数吸收')

    def test_notch_barely_moves_it(self):
        """卡尺夹的是两端齿顶，中间啃掉一块它几乎不缩 —— 所以尺寸路和缺损路必须分开。
        这条同时说明：换上齿顶直径不会把真缺陷洗成合格，缺损另有 deficit 判据兜着。
        """
        intact, _ = _measure(25.0)
        nicked, _ = _measure(25.0, nick_mm=2.0)
        self.assertLess(abs(intact['caliper_diameter_pixels'] - nicked['caliper_diameter_pixels'])
                        / PPM, 0.25)


class TestBrokenRimRecovery(unittest.TestCase):
    """币缘断成开口弧时，闭运算是唯一能找回它的手段（膨胀会把读数撑大）。

    实拍：5角 a 那枚三张连拍全部「未检出」，而币缘梯度中位 236 = Canny 高门限的 1.6 倍，
    说明边缘强得很，只是圆周上某段没过高门限导致环断开。开口弧被 RETR_EXTERNAL
    当成细长轮廓，面积≈0，等效直径只剩 4.8mm，于是整个落在尺寸窗外。
    """

    @staticmethod
    def _broken_ring(gap_px: int = 3) -> np.ndarray:
        edges = np.zeros((240, 240), np.uint8)
        cv2.circle(edges, (120, 120), 80, 255, 1)
        cv2.rectangle(edges, (118, 36), (118 + gap_px, 46), 0, -1)   # 顶部切一个口
        return edges

    def test_open_arc_reads_almost_no_area(self):
        """断口先要能被复现：开口弧被当成细带，enclosed area≈0，等效直径只剩几毫米。

        这里必须绕过 `find_contours` 的 min_area 过滤直接取轮廓 —— 实拍那个签名恰恰是
        「面积小到被 min_area 滤光，于是窗内一个候选都没有」，用过滤后的列表反而看不到。
        """
        raw = cv2.findContours(self._broken_ring(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)[0]
        cnt = max(raw, key=lambda c: cv2.arcLength(c, True))
        self.assertLess(cv2.contourArea(cnt), 0.05 * np.pi * 80 ** 2,
                        f'夹具没造出开口弧：enclosed area={cv2.contourArea(cnt):.0f}')

    def test_closing_bridges_the_gap(self):
        closed = preprocess.close_edges(self._broken_ring())
        cnt = max(detector.find_contours(closed, 100), key=cv2.contourArea)
        d = detector.calculate_features(cnt)['diameter_pixels'] / PPM
        self.assertAlmostEqual(d, 160 / PPM, delta=0.15,
                               msg=f'闭合后应读出 ~{160 / PPM:.2f}mm，实测 {d:.2f}')

    def test_dilation_would_bias_the_reading(self):
        """锁住「为什么是 CLOSE 不是 DILATE」：同一个缺口，膨胀读出来的直径系统性偏大。"""
        ring = self._broken_ring()
        d_close = detector.calculate_features(
            max(detector.find_contours(preprocess.close_edges(ring), 100),
                key=cv2.contourArea))['diameter_pixels'] / PPM
        d_dilate = detector.calculate_features(
            max(detector.find_contours(cv2.dilate(ring, np.ones((5, 5), np.uint8)), 100),
                key=cv2.contourArea))['diameter_pixels'] / PPM
        self.assertGreater(d_dilate - d_close, 0.05,
                           f'膨胀只比闭合大 {d_dilate - d_close:.3f}mm，夹具或算法变了')


class TestCrossCheckDetectsDoubleEdge(unittest.TestCase):
    """两路读数必须**在双边缘场景下分叉** —— 这是互检存在的唯一理由。

    ⚠ 边界要说清楚：这个夹具（币面 170 亮于边圈 90 → 内台阶是强下降沿）证明的是
    「双边缘几何会让多边形静默读小」这个**通用失效模式**，不是 j 那枚的机制——
    真 j 的实测亮度是边圈 92 > 币面 81，顺序相反。所以别拿这条测试当作「j 已解释」。
    j 为什么读 21.188 仍未定，需要定向实验（转 90°/翻面/换照明）才能判是光学还是币本身。
    """

    C = 250

    @staticmethod
    def _double_edge(rim_r=100, face_r=88, face=170, rim=90, bg=40, gap_deg=100) -> np.ndarray:
        C = TestCrossCheckDetectsDoubleEdge.C
        img = np.full((500, 500), bg, np.uint8)
        cv2.circle(img, (C, C), rim_r, int(rim), -1)
        # 先挖扇形、再补币面：顺序反了就变成「整个扇形没了」，那是缺口而不是外缘断裂
        if gap_deg:
            cv2.ellipse(img, (C, C), (rim_r + 2, rim_r + 2), 0,
                        -gap_deg / 2, gap_deg / 2, int(bg), -1)
        if face_r:
            cv2.circle(img, (C, C), face_r, int(face), -1)
        return img

    def test_polygon_and_radial_diverge_on_double_edge(self):
        # 夹具本身就是单通道灰度图，不能再过 preprocess.gray（cvtColor 对 1 通道会报错）
        blur = preprocess.gaussian_blur(self._double_edge(), config_setting.gaussian_kernel)
        edges = cv2.Canny(blur, config_setting.canny_low, config_setting.canny_high)
        cnt = max(detector.find_contours(edges, config_setting.min_area), key=cv2.contourArea)
        poly = detector.calculate_features(cnt)['diameter_pixels']
        rad = detector.radial_measure(blur, (self.C, self.C), poly / 2)
        self.assertIsNotNone(rad.get('radial_diameter_pixels'), '径向法没出数')
        gap = abs(rad['radial_diameter_pixels'] - poly) / config_setting.ppm
        self.assertGreater(gap, config_setting.edge_check_mm,
                           f'两路只差 {gap:.3f}mm，夹具没复现出分叉，互检将形同虚设')

    def test_plain_coin_agrees_within_threshold(self):
        """干净币上两路必须**不**分叉，否则阈值一开就是满屏假告警。"""
        C = self.C
        img = np.full((500, 500), 40, np.uint8)
        cv2.circle(img, (C, C), 100, 120, -1)
        blur = preprocess.gaussian_blur(img, config_setting.gaussian_kernel)
        edges = cv2.Canny(blur, config_setting.canny_low, config_setting.canny_high)
        cnt = max(detector.find_contours(edges, config_setting.min_area), key=cv2.contourArea)
        poly = detector.calculate_features(cnt)['diameter_pixels']
        rad = detector.radial_measure(blur, (C, C), poly / 2)
        gap = abs(rad['radial_diameter_pixels'] - poly) / config_setting.ppm
        self.assertLess(gap, config_setting.edge_check_mm,
                        f'干净币两路差 {gap:.3f}mm，超过阈值 {config_setting.edge_check_mm}')


if __name__ == '__main__':
    unittest.main()
