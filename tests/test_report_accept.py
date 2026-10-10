"""枚级拒收规则的判定必须逐条锁死。

这条规则的裁判不是我的推断，是卡尺：2026-10-10 那批 10 枚 1元，卡尺证明**每一枚
都在 22.2~22.3**，而相机把 b 读成 21.62、e 整组检不出、j 在同组内一半 22.22 一半
21.14。所以「拦住 b/e/j、放过其余七枚」是有真值支撑的验收标准，不是调参调出来的。

为什么不去修 b/e/j：六个机制（样本特殊 / Otsu 区域分割 / Canny 高门限 / 币缘反差 /
币被亮区盖住 / 碎弧圆拟合）全部被测量否掉，见 config_setting 里那段。在机制不明时
**让失败可识别**是唯一站得住的工程落点。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))

from report_metrics import _accept                      # noqa: E402
import config_setting                                   # noqa: E402


def _s(vals_by_part, totals=None, coin='1元'):
    """造一个 _stats 的返回值形状。totals 记「这一枚实际有几张」，缺省=检出张数。"""
    return {'coin': coin, 'parts': {k: list(v) for k, v in vals_by_part.items()},
            'ptot': dict(totals or {k: len(v) for k, v in vals_by_part.items()}),
            'ds': [d for v in vals_by_part.values() for d in v]}


def _verdicts(s):
    return {k: (good, why) for k, _, good, why in _accept(s)}


class TestCoinAccept(unittest.TestCase):
    NOM = 22.25

    def test_干净币六张全过应当采信(self):
        """a 组实测：6/6 检出、极差 0.008mm、均值 22.246。"""
        v = _verdicts(_s({'a': [22.242, 22.246, 22.250, 22.244, 22.245, 22.249]}))
        self.assertTrue(v['a'][0], f'干净币被拒收了：{v["a"][1]}')

    def test_同组内在两个结构间跳变必须拒收(self):
        """j 组实测：前三张 22.22、后两张 21.14 —— 同一枚币没动过，读数却差 1.08mm。"""
        v = _verdicts(_s({'j': [22.222, 22.225, 22.227, 21.136, 21.143]}))
        self.assertFalse(v['j'][0])
        self.assertIn('组内极差', v['j'][1])

    def test_整枚读偏必须拒收即使组内很稳(self):
        """b 组实测：检出的几张彼此只差 0.008mm，但一起闭到了内台阶上（21.155）。

        这条最重要 —— 组内一致**不等于**读数正确。它闭的是同一道错边，所以照样稳。
        只有「离标称」这一条能拦住它。
        """
        v = _verdicts(_s({'b': [21.155, 21.158, 21.150]}, {'b': 6}))
        self.assertFalse(v['b'][0])
        self.assertIn('未检出', v['b'][1])
        self.assertIn('离标称', v['b'][1])

    def test_漏检本身就该拒收(self):
        """e 组实测：6 张里只有 1 张检出，且那张是闭运算兜底出来的 23.4mm 垃圾。"""
        v = _verdicts(_s({'e': [23.395]}, {'e': 6}))
        self.assertFalse(v['e'][0])
        self.assertIn('5/6 张未检出', v['e'][1])

    def test_只有一张时不能靠极差混过判定(self):
        """单张算不出极差，但「离标称」仍然有效 —— 否则漏检组里剩一张就自动采信了。"""
        v = _verdicts(_s({'x': [21.000]}))
        self.assertFalse(v['x'][0])
        self.assertIn('离标称', v['x'][1])

    def test_卡尺真值下的整批判定(self):
        """七枚采信、三枚拒收 —— 与卡尺「10 枚全在 22.2~22.3」零冲突。"""
        s = _s({
            'a': [22.242, 22.246, 22.250, 22.244, 22.245, 22.249],
            'c': [22.071, 22.075, 22.080, 22.073, 22.076, 22.078],
            'd': [22.105, 22.112, 22.127, 22.108, 22.110, 22.115],
            'f': [22.022, 22.026, 22.029, 22.023, 22.025, 22.027],
            'g': [22.245, 22.248, 22.251, 22.246, 22.247, 22.250],
            'h': [22.209, 22.212, 22.215, 22.210, 22.211, 22.213],
            'i': [22.208, 22.213, 22.218, 22.209, 22.211, 22.216],
            'b': [21.155, 21.158, 21.150],
            'e': [23.395],
            'j': [22.222, 22.225, 22.227, 21.136, 21.143],
        }, {'a': 6, 'c': 6, 'd': 6, 'f': 6, 'g': 6, 'h': 6, 'i': 6,
            'b': 6, 'e': 6, 'j': 6})
        v = _verdicts(s)
        self.assertEqual(sorted(k for k, (good, _) in v.items() if good),
                         ['a', 'c', 'd', 'f', 'g', 'h', 'i'])
        self.assertEqual(sorted(k for k, (good, _) in v.items() if not good),
                         ['b', 'e', 'j'])

    def test_互检阈值必须高于干净币上的实测分叉(self):
        """0.3mm 的老阈值低于径向法自身噪声底，把 g 这种好币 6/6 标成可疑。

        实测干净币两路差到 0.56mm。阈值若再回到 0.56 以下，互检就退化成狼来了，
        反而没人信它抓到的真分叉（b/j 是 1.66mm）。
        """
        self.assertGreater(config_setting.edge_check_mm, 0.56,
                           '互检阈值低于干净币上的实测分叉，会产生假警报')
        self.assertLess(config_setting.edge_check_mm, 1.66,
                        '互检阈值高于内台阶错读的分叉，抓不到真问题')


if __name__ == '__main__':
    unittest.main()
