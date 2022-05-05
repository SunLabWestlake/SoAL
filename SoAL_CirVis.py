"""
Circling visualization
Author: Jing Ning @ SunLab
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PolyCollection, LineCollection

from SoAL_Utils import load_dfs, load_dict

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['Arial']

MALE = 1
FEMALE = 2


def triangle_for_vector(x, y, px, py, r=4):
    return ((px+y/r)-x, (py-x/r)-y), ((px-y/r)-x, (py+x/r)-y), ((px+x), (py+y))

def triangle_for_angle(a, px, py, l):
    return triangle_for_vector(np.cos(np.deg2rad(a)) * l, np.sin(np.deg2rad(a)) * l, px, py)

def points_to_line_collection(xs, ys, cmap, linewidth):
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    t = np.arange(0, len(segments))
    norm = plt.Normalize(t.min(), t.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(t[::-1])  # t
    lc.set_linewidth(linewidth)
    return lc

WRONG_POS_MOVE_MIN = 2 #mm
def filter_wrong_pos(xs, ys, ds):
    # 2 tracks, use the longest one
    rxs, rys, rds = [], [], []
    xs = xs.tolist()
    ys = ys.tolist()
    lx, ly = xs[0], ys[0]
    rxs2, rys2, rds2 = [], [], []
    lx2, ly2 = 0, 0
    s_d_max = WRONG_POS_MOVE_MIN ** 2
    for x, y, d in zip(xs, ys, ds):
        if (x-lx)**2+(y-ly)**2 < s_d_max:
            rxs.append(x)
            rys.append(y)
            rds.append(d)
            lx, ly = x, y
        else:
            if rxs2:
                if (x-lx2)**2+(y-ly2)**2 < s_d_max:
                    rxs2.append(x)
                    rys2.append(y)
                    rds2.append(d)
                    lx2, ly2 = x, y
            else:
                rxs2.append(x)
                rys2.append(y)
                rds2.append(d)
                lx2, ly2 = x, y
    if len(rxs2) > len(rxs):
        rxs = rxs2
        rys = rys2
        rds = rds2
    return rxs, rys, rds

def plot_overlap_time(ax, xs, ys, dirs, fly, inter_frame=3, need_filter=True, cmap=None):
    if need_filter:
        xs, ys, dirs = filter_wrong_pos(xs, ys, dirs)
    else:
        xs, ys, dirs = xs.tolist(), ys.tolist(), dirs.tolist()
    if not cmap:
        cmap = "viridis" if fly == 1 else "autumn"
    ax.add_collection(points_to_line_collection(xs, ys, cmap, 2))

    if len(dirs) == 0:
        return
    verts = []
    j = 0
    for x, y in zip(xs, ys):
        if j % inter_frame == 0:
            d = dirs[j]
            verts.append(triangle_for_angle(d, x, y, 0.5))
        j += 1
    t = np.arange(0, len(verts))
    norm = plt.Normalize(t.min(), t.max())
    pc = PolyCollection(verts, cmap=cmap, norm=norm, alpha=1)
    pc.set_array(t[::-1])
    ax.add_collection(pc)
    # need_colorbar and plot_colorbar(ax, im=pc)

def plot_traj(dfs, cir_bouts):
    rel_x = dfs[FEMALE]["rel_pos:x"]
    rel_y = dfs[FEMALE]["rel_pos:y"]
    # male_x = dfs[MALE]["pos:x"]
    # male_y = dfs[MALE]["pos:y"]
    rel_dir = dfs[MALE]["dir"] - dfs[FEMALE]["dir"] + 90

    plt.figure(figsize=(3, 4))
    ax = plt.gca()
    rect = plt.Polygon(triangle_for_angle(90, 0, 0, 0.5), color="r", alpha=1, linewidth=0)
    ax.add_patch(rect)
    for s, e in cir_bouts:
        xs = rel_x.iloc[s: e+1]
        ys = rel_y.iloc[s: e+1]
        dirs = rel_dir.iloc[s: e+1]
        plot_overlap_time(ax, xs, ys, dirs, MALE)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 6)
    ax.set_xlabel("Relative male position x (mm)")
    ax.set_ylabel("Relative male position y (mm)")
    plt.axis("equal")
    # plt.show()
    plt.savefig("traj.pdf")

CENTER_RANGE = [[-3, 3], [-1.2, 6]]
BODY_SIZE = [2.53, 0.87]
COLOR_FOR_FLY = {0: "y", 1: "b", 2: "r", 3: "gray"}
def plot_fly_body(ax, fly, range_xy, body_size, line_color="w", lw=2, zorder=None):
    ax.add_line(plt.Line2D((range_xy[0][0], range_xy[0][1]), (0, 0), linewidth=2, color=line_color, linestyle="--", zorder=zorder))
    ax.add_line(plt.Line2D((0, 0), (range_xy[1][0], range_xy[1][1]), linewidth=2, color=line_color, linestyle="--", zorder=zorder))
    body_len = body_size[0]
    body_sh = 0.4 * body_len
    ell1 = patches.Ellipse(xy=(0, 0), width=body_sh, height=body_len, facecolor=COLOR_FOR_FLY[fly], alpha=0.5, edgecolor="w", linewidth=lw)
    ell2 = patches.Ellipse(xy=(0, body_len / 2 - body_sh * 0.3), width=body_sh, height=body_sh * 0.8, facecolor=COLOR_FOR_FLY[fly], alpha=0.5, edgecolor="w", linewidth=lw)
    ax.add_patch(ell1)
    ax.add_patch(ell2)

def plot_we(dfs):
    rel_x = dfs[MALE]["rel_pos_h:x"]
    rel_y = dfs[MALE]["rel_pos_h:y"]
    wing_l = dfs[MALE]["wing_l"]
    wing_r = dfs[MALE]["wing_r"]
    we_idx = dfs[MALE]["we"] > 0
    plt.figure(figsize=(3, 4))
    ax = plt.gca()
    cmap = LinearSegmentedColormap.from_list("mycmap", ["#b70e5e", "#6AC7E2"])
    ax.scatter(rel_x[we_idx], rel_y[we_idx], c=(-wing_l > wing_r)[we_idx], cmap=cmap, s=1, alpha=0.2)
    plot_fly_body(ax, 1, CENTER_RANGE, BODY_SIZE, line_color="k")

    plt.axis("equal")
    ax.set_xlim(CENTER_RANGE[0])
    ax.set_ylim(CENTER_RANGE[1])
    ax.set_xlabel("Relative female head position x (mm)")
    ax.set_ylabel("Relative female head position y (mm)")
    # plt.show()
    plt.savefig("we.pdf")

if __name__ == '__main__':
    mot0 = sys.argv[1]
    dfs = load_dfs(mot0)
    cir_meta = load_dict(mot0.replace("mot_para0.pickle", "config_circl.json"))
    # cir_meta = load_dict(mot0.replace("stat0.pickle", "cir_meta.txt"))
    plot_traj(dfs, cir_meta["cir_bouts1"])
    plot_we(dfs)
