"""数据根目录解析 —— 本地项目 + 远程服务器双环境（2026-10-29 定稿）。

三个权威地址（用户 2026-10-29 确认）：
  · 本地项目（代码编辑，PyCharm 打开这里）：D:\\ELE\\2026-07-13-13-59-39
  · 服务器数据根（图片）：/media/yczxserver/SSD-windows/wjf/worm_Sclass_pipeline_f/data_new
    （组目录 control/5mg/8mg/10mg 直接位于其下）
  · 服务器解释器（PyCharm SSH 解释器）：
    /home/yczxserver/.virtualenvs/worm_5class_pipeline/bin/python

解析优先级：
  1. 环境变量 WORM_DATA_ROOT（显式指定永远优先）
  2. 自动探测：第一个"看起来像数据根"（至少含一个组目录 control/5mg/8mg/10mg）
     的候选目录——服务器数据根优先，其次项目内/上一级 data_new（兼容
     data_new/images 子结构）、旧本机 D:/data_new/images
  3. 回退服务器数据根（当前唯一实际数据位置）

2026-11-13 修正：用户实测隔离区在 `data_new/images/_noise_quarantine`，
说明组目录实际位于 `data_new/images` 这一层（本地 E 盘同构）。原候选表缺
`<root>/data_new/images`，导致所有候选都不含组目录时静默回退到错层。
现在把两种 `images` 子层都纳入候选；`_looks_like_data_root` 仍以「含组目录」为准，
因此新旧两种布局都能自动判对。

用法：
  python cluster_atlas.py              # 不带 -i/--root，自动解析
  python cluster_atlas.py -i /path    # 显式指定，永远优先（argparse 原样保留）
  服务器一键设置（bash）：
    export WORM_DATA_ROOT=/media/yczxserver/SSD-windows/wjf/worm_Sclass_pipeline_f/data_new
  （PyCharm 运行配置的 Environment variables 里设同理）
"""
import os

# 服务器数据根（2026-10-29 确认，当前唯一实际数据位置）
SERVER_DATA_ROOT = '/media/yczxserver/SSD-windows/wjf/worm_Sclass_pipeline_f/data_new'
# 旧 Windows 本机数据根（仅自动探测兼容，不再是默认值）
LEGACY_LOCAL_ROOT = 'D:/data_new/images'

KNOWN_GROUPS = ('control', '5mg', '8mg', '10mg')


def _looks_like_data_root(path):
    """候选目录存在且至少含一个组目录，才认为是数据根。"""
    if not os.path.isdir(path):
        return False
    return any(os.path.isdir(os.path.join(path, g)) for g in KNOWN_GROUPS)


def _candidates():
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    return [
        SERVER_DATA_ROOT,                              # 服务器数据根（首选）
        os.path.join(SERVER_DATA_ROOT, 'images'),      # 服务器 images 子层（2026-11-13 实测）
        os.path.join(parent, 'data_new'),              # 项目同级 data_new
        os.path.join(parent, 'data_new', 'images'),    # 兼容 data_new/images 子结构
        os.path.join(here, 'data_new'),                # 脚本同级 data_new
        os.path.join(here, 'data_new', 'images'),      # 脚本同级 images 子层（新增）
        os.path.join(here, 'images'),
        LEGACY_LOCAL_ROOT,                             # 旧本机路径（兼容）
    ]


def get_data_root():
    """返回图片数据根目录（含 group/date/row/worm 五维结构的那一层）。"""
    env = os.environ.get('WORM_DATA_ROOT')
    if env:
        return env
    for cand in _candidates():
        if _looks_like_data_root(cand):
            return cand
    return SERVER_DATA_ROOT


DATA_ROOT = get_data_root()
