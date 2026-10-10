"""标定时刻与照片时刻的关系 —— 判据只写一份，验收工具和批量流水线共用。

为什么必须在 `coin_detect` 里而不是 `tools/check_placement.py` 里：
「照片比标定旧」这道闸门原先**只长在验收工具上**。重标定之后，旧装置拍的照片留在
样本目录里，`main.py --mode batch` 和 `tools/report_metrics.py` 会用**新 ppm** 去量它，
得到一个看起来完全正常的直径，混进均值和标准差 —— 全程不报任何错。
这和 2026-10-09「尺寸闸门只长在工具里、没长在流水线上」是同一个根因：
**工具和流水线各写一套判据，就一定只会在其中一条上长闸门。**
判据放这儿，两边都只是给它配不同的语气（验收 = NG，批量 = 警告）。
"""
import datetime
import os


def photo_vs_calibration(path: str, calib_stamp: str) -> tuple[datetime.datetime, datetime.datetime] | None:
    """照片早于标定时刻时返回 (拍摄时刻, 标定时刻)，否则返回 None。

    用文件修改时间而不是 EXIF：Pillow 不在 requirements.txt 里，不想为一个检查
    加未声明依赖（同 visualizer 不引 Pillow 的取舍）。经数据线/微信「原图」进来的文件，
    mtime 要么保留拍摄时间（判得准）、要么被改成传输时间（更晚，判不出但不会冤枉新图）。
    **只会漏判、不会冤枉新照片**，这个方向可接受。
    """
    if not calib_stamp:
        return None
    try:
        calib = datetime.datetime.fromisoformat(calib_stamp)
        shot = datetime.datetime.fromtimestamp(os.path.getmtime(path))
    except (ValueError, OSError):
        return None
    return (shot, calib) if shot < calib else None


def stale_paths(paths: list[str], calib_stamp: str) -> list[str]:
    """批量筛出「比标定旧」的照片，保持传入顺序。"""
    return [p for p in paths if photo_vs_calibration(p, calib_stamp) is not None]
