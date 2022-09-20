import numpy as np

import bezier
from base.base import BaseModel
from base.math import MVector2D

# MMDでの補間曲線の最大値
INTERPOLATION_MMD_MAX = 127


class Interpolation(BaseModel):
    def __init__(
        self,
        begin: MVector2D = None,
        start: MVector2D = None,
        end: MVector2D = None,
        finish: MVector2D = None,
    ):
        """
        補間曲線

        Parameters
        ----------
        start : MVector2D, optional
            補間曲線開始, by default None
        end : MVector2D, optional
            補間曲線終了, by default None
        """
        self.begin: MVector2D = begin or MVector2D(0, 0)
        self.start: MVector2D = start or MVector2D(20, 20)
        self.end: MVector2D = end or MVector2D(107, 107)
        self.finish: MVector2D = finish or MVector2D(127, 127)


def get_infections(values: list[float], threshold) -> list[int]:
    extract_idxs = np.where(np.abs(np.round(np.diff(values), 1)) > threshold)[0]
    if not extract_idxs.any():
        return []

    extracts = np.array(values)[extract_idxs]
    f_prime = np.gradient(extracts)
    infections = extract_idxs[np.where(np.diff(np.sign(f_prime)))[0]]

    return infections


def create_interpolation(values: list[float]):
    if len(values) <= 2 or abs(np.max(values) - np.min(values)) < 0.0001:
        return Interpolation()

    # Xは次数（フレーム数）分移動
    xs = np.arange(0, len(values))

    # YはXの移動分を許容範囲とする
    ys = np.array(values)

    # https://github.com/dhermes/bezier/issues/242
    s_vals = np.linspace(0, 1, len(values))
    representative = bezier.Curve.from_nodes(np.eye(4))
    transform = representative.evaluate_multi(s_vals).T
    nodes = np.vstack([xs, ys])
    reduced_t, residuals, rank, _ = np.linalg.lstsq(transform, nodes.T, rcond=None)
    reduced = reduced_t.T
    joined_curve = bezier.Curve.from_nodes(reduced)

    nodes = joined_curve.nodes

    # 次数を減らしたベジェ曲線をMMD用補間曲線に変換
    joined_org_bz = scale_bezier(
        MVector2D(nodes[0, 0], nodes[1, 0]),
        MVector2D(nodes[0, 1], nodes[1, 1]),
        MVector2D(nodes[0, 2], nodes[1, 2]),
        MVector2D(nodes[0, 3], nodes[1, 3]),
    )


# http://d.hatena.ne.jp/edvakf/20111016/1318716097
# https://pomax.github.io/bezierinfo
# https://shspage.hatenadiary.org/entry/20140625/1403702735
# https://bezier.readthedocs.io/en/stable/python/reference/bezier.curve.html#bezier.curve.Curve.evaluate
def evaluate(
    interpolation: Interpolation, start: int, now: int, end: int
) -> tuple[float, float, float]:
    """
    補間曲線を求める

    Parameters
    ----------
    interpolation : Interpolation
        補間曲線
    start : int
        開始キーフレ
    now : int
        計算キーフレ
    end : int
        終端キーフレ

    Returns
    -------
    tuple[float, float, float]
        x（計算キーフレ時点のX値）, y（計算キーフレ時点のY値）, t（計算キーフレまでの変化量）
    """
    if (now - start) == 0 or (end - start) == 0:
        return 0, 0, 0

    x = (now - start) / (end - start)
    x1 = interpolation.start.x / INTERPOLATION_MMD_MAX
    y1 = interpolation.start.y / INTERPOLATION_MMD_MAX
    x2 = interpolation.end.x / INTERPOLATION_MMD_MAX
    y2 = interpolation.end.y / INTERPOLATION_MMD_MAX

    t = 0.5
    s = 0.5

    # 二分法
    for i in range(15):
        ft = (3 * (s * s) * t * x1) + (3 * s * (t * t) * x2) + (t * t * t) - x

        if ft > 0:
            t -= 1 / (4 << i)
        else:
            t += 1 / (4 << i)

        s = 1 - t

    y = (3 * (s * s) * t * y1) + (3 * s * (t * t) * y2) + (t * t * t)

    return x, y, t
