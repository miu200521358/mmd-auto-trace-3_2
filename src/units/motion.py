import csv
import json
import math
import os
from datetime import datetime
from glob import glob

import numpy as np
from base.logger import MLogger
from base.math import MVector3D
from mmd.pmx.collection import PmxModel
from mmd.pmx.part import Bone, Ik, IkLink
from mmd.pmx.reader import PmxReader
from mmd.vmd.collection import VmdMotion
from mmd.vmd.part import VmdBoneFrame
from mmd.vmd.writer import VmdWriter
from tqdm import tqdm

logger = MLogger(__name__)

MIKU_METER = 12.5 / 10


def execute(args):
    try:
        logger.info(
            "モーション生成処理開始: {img_dir}",
            img_dir=args.img_dir,
            decoration=MLogger.DECORATION_BOX,
        )

        if not os.path.exists(args.img_dir):
            logger.error(
                "指定された処理用ディレクトリが存在しません。: {img_dir}",
                img_dir=args.img_dir,
                decoration=MLogger.DECORATION_BOX,
            )
            return False

        motion_dir_path = os.path.join(args.img_dir, "motion")
        os.makedirs(motion_dir_path, exist_ok=True)

        # モデルをCSVから読み込む
        process_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

        # トレース用モデルを読み込む
        trace_mov_model = PmxReader().read_by_filepath(args.trace_mov_model_config)

        smooth_person_file_paths = sorted(
            glob(os.path.join(args.img_dir, "smooth", "*.json"))
        )

        # 全人物分の順番別ファイル
        start_z = 0
        for oidx, person_file_path in enumerate(smooth_person_file_paths):
            logger.info(
                "【No.{oidx}】モーション結果位置計算開始",
                f"{oidx:02d}",
                decoration=MLogger.DECORATION_LINE,
            )

            trace_mov_motion = VmdMotion()

            frame_joints = {}
            with open(person_file_path, "r", encoding="utf-8") as f:
                frame_joints = json.load(f)

            target_bone_vecs = {}
            for jname in PMX_CONNECTIONS.keys():
                target_bone_vecs[jname] = {}

            max_fno = 0
            for fidx, frames in tqdm(frame_joints.items(), desc="Read Json ..."):
                fno = int(fidx)
                for jname, joint in frames.items():
                    if jname not in target_bone_vecs:
                        target_bone_vecs[jname] = {}
                    target_bone_vecs[jname][fno] = np.array(
                        [
                            float(joint["x"]) * MIKU_METER / 30,
                            float(joint["y"]) * MIKU_METER / 30,
                            float(joint["z"]) * MIKU_METER * 10,
                        ]
                    )
                    if oidx == 0 and fno == 0 and jname == "root":
                        start_z = target_bone_vecs[jname][fno][2]
                    target_bone_vecs[jname][fno][2] -= start_z

                # 下半身
                target_bone_vecs["pelvis"][fno] = np.copy(target_bone_vecs["root"][fno])

                # 上半身2
                target_bone_vecs["spine2"][fno] = np.mean(
                    [
                        target_bone_vecs["root"][fno],
                        target_bone_vecs["head_bottom"][fno],
                    ],
                    axis=0,
                )

                # 左肩
                target_bone_vecs["left_collar"][fno] = np.mean(
                    [
                        target_bone_vecs["left_shoulder"][fno],
                        target_bone_vecs["right_shoulder"][fno],
                    ],
                    axis=0,
                )

                # 右肩
                target_bone_vecs["right_collar"][fno] = np.copy(
                    target_bone_vecs["left_collar"][fno]
                )

                # 下半身先
                target_bone_vecs["pelvis2"][fno] = target_bone_vecs["pelvis"][fno] + (
                    (
                        target_bone_vecs["head_bottom"][fno]
                        - target_bone_vecs["root"][fno]
                    )
                    / 3
                )

                max_fno = fno

            logger.info(
                "【No.{oidx}】モーション(移動)計算開始",
                f"{oidx:02d}",
                decoration=MLogger.DECORATION_LINE,
            )

            with tqdm(
                total=(len(target_bone_vecs) * max_fno),
                desc="Create Move BoneFrame ...",
            ) as pchar:
                for jname, bone_vecs in target_bone_vecs.items():
                    pconn = PMX_CONNECTIONS[jname]
                    jmmd_name = PMX_CONNECTIONS[jname]["mmd"]

                    bone = trace_mov_model.bones[jmmd_name]

                    parent_bone = (
                        trace_mov_model.bones[bone.parent_index]
                        if bone.parent_index >= 0
                        else None
                    )

                    # 親ボーンのモデルの位置
                    parent_bone_position = (
                        parent_bone.position if parent_bone else MVector3D()
                    )

                    for fno, bone_vec in bone_vecs.items():
                        bf = VmdBoneFrame(
                            name=jmmd_name,
                            index=fno,
                            regist=True,
                        )

                        # 親ボーンのモーションの位置
                        parent_vec = MVector3D(
                            target_bone_vecs.get(pconn["parent"], {}).get(
                                fno, np.array([0, 0, 0])
                            )
                        )

                        bf.position = (
                            MVector3D(bone_vec)
                            - trace_mov_model.bones[jmmd_name].position
                        ) - (parent_vec - parent_bone_position)

                        trace_mov_motion.bones.append(bf)
                        pchar.update(1)

            trace_mov_motion_path = os.path.join(
                motion_dir_path, f"trace_{process_datetime}_mov_no{oidx:02d}.vmd"
            )
            logger.info(
                "【No.{oidx}】モーション(移動)生成開始【{path}】",
                oidx=f"{oidx:02d}",
                path=os.path.basename(trace_mov_motion_path),
                decoration=MLogger.DECORATION_LINE,
            )
            VmdWriter.write(
                trace_mov_model.name, trace_mov_motion, trace_mov_motion_path
            )

        logger.info(
            "モーション結果保存完了: {motion_dir_path}",
            motion_dir_path=motion_dir_path,
            decoration=MLogger.DECORATION_BOX,
        )

        return True
    except Exception as e:
        logger.critical("姿勢推定で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False


PMX_CONNECTIONS = {
    "全ての親": {
        "mmd": "全ての親",
        "parent": None,
    },
    "センター": {
        "mmd": "センター",
        "parent": "全ての親",
    },
    "グルーブ": {
        "mmd": "グルーブ",
        "parent": "センター",
    },
    "root": {
        "mmd": "上半身",
        "parent": "グルーブ",
    },
    "spine2": {
        "mmd": "上半身2",
        "parent": "root",
    },
    "head_bottom": {
        "mmd": "首",
        "parent": "spine2",
    },
    "nose": {
        "mmd": "頭",
        "parent": "head_bottom",
    },
    "left_collar": {
        "mmd": "左肩",
        "parent": "spine2",
    },
    "right_collar": {
        "mmd": "右肩",
        "parent": "spine2",
    },
    "left_shoulder": {
        "mmd": "左腕",
        "parent": "left_collar",
    },
    "right_shoulder": {
        "mmd": "右腕",
        "parent": "right_collar",
    },
    "left_elbow": {
        "mmd": "左ひじ",
        "parent": "left_shoulder",
    },
    "right_elbow": {
        "mmd": "右ひじ",
        "parent": "right_shoulder",
    },
    "left_wrist": {
        "mmd": "左手首",
        "parent": "left_elbow",
    },
    "right_wrist": {
        "mmd": "右手首",
        "parent": "right_elbow",
    },
    "pelvis": {
        "mmd": "下半身",
        "parent": "グルーブ",
    },
    "pelvis2": {
        "mmd": "下半身先",
        "parent": "pelvis",
    },
    "left_hip": {
        "mmd": "左足",
        "parent": "pelvis",
    },
    "right_hip": {
        "mmd": "右足",
        "parent": "pelvis",
    },
    "left_knee": {
        "mmd": "左ひざ",
        "parent": "left_hip",
    },
    "right_knee": {
        "mmd": "右ひざ",
        "parent": "right_hip",
    },
    "left_ankle": {
        "mmd": "左足首",
        "parent": "left_knee",
    },
    "right_ankle": {
        "mmd": "右足首",
        "parent": "right_knee",
    },
}
