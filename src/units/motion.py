import csv
import json
import math
import os
from datetime import datetime
from glob import glob

import numpy as np
from base.logger import MLogger, get_file_encoding
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
        miku_model = read_bone_csv(args.bone_config)
        process_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

        # トレース用モデルを読み込む
        trace_model = PmxReader().read_by_filepath(args.trace_mov_model_config)

        smooth_person_file_paths = sorted(
            glob(os.path.join(args.img_dir, "smooth", "*.json"))
        )
        # 全人物分の順番別ファイル
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
            for jname in [
                "全ての親",
                "センター",
                "グルーブ",
                "root",
                "nose",
                "head_bottom",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
                "pelvis",
                "pelvis2",
                "spine2",
                "left_collar",
                "right_collar",
            ]:
                target_bone_vecs[jname] = {}

            max_fno = 0
            for fidx, frames in tqdm(frame_joints.items(), desc="Read Json ..."):
                fno = int(fidx)
                for jname, joint in frames.items():
                    if jname not in target_bone_vecs:
                        target_bone_vecs[jname] = {}
                    target_bone_vecs[jname][fno] = np.array(
                        [
                            float(joint["x"]) * MIKU_METER / 40,
                            float(joint["y"]) * MIKU_METER / 40,
                            -float(joint["z"]) * MIKU_METER * 6,
                        ]
                    )

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
                    for fno, bone_vec in bone_vecs.items():
                        pconn = PMX_CONNECTIONS[jname]
                        bf = VmdBoneFrame(
                            name=pconn["mmd"],
                            index=fno,
                            regist=True,
                        )

                        parent_vec = MVector3D(
                            target_bone_vecs.get(pconn["parent"], {}).get(
                                fno, np.array([0, 0, 0])
                            )
                        )
                        parent_bone_name = PMX_CONNECTIONS.get(pconn["parent"], {}).get(
                            "mmd", None
                        )
                        parent_bone_position = MVector3D()
                        if parent_bone_name in trace_model.bones:
                            parent_bone_position = trace_model.bones[
                                parent_bone_name
                            ].position

                        bf.position = (
                            MVector3D(bone_vec)
                            - parent_vec
                            - (
                                trace_model.bones[pconn["mmd"]].position
                                - parent_bone_position
                            )
                        )

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
            VmdWriter.write(trace_model.name, trace_mov_motion, trace_mov_motion_path)

        return True
    except Exception as e:
        logger.critical("姿勢推定で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False


def read_bone_csv(bone_csv_path: str):
    model = PmxModel()
    model.name = os.path.splitext(os.path.basename(bone_csv_path))[0]

    with open(bone_csv_path, "r", encoding=get_file_encoding(bone_csv_path)) as f:
        reader = csv.reader(f)

        for ridx, row in enumerate(reader):
            if row[0] == "Bone":
                bone = Bone(
                    name=row[1],
                    english_name=row[2],
                    layer=int(row[3]),
                    position=MVector3D(float(row[5]), float(row[6]), float(row[7])),
                    bone_flg=(
                        int(row[14]) * 0x0001
                        | int(row[8]) * 0x0002
                        | int(row[9]) * 0x0004
                        | int(row[10]) * 0x0020
                        | int(row[11]) * 0x0008
                        | int(row[12]) * 0x0010
                        | int(row[19]) * 0x0080
                        | int(row[20]) * 0x0100
                        | int(row[21]) * 0x0200
                        | int(row[24]) * 0x0400
                        | int(row[28]) * 0x0800
                        | int(row[35]) * 0x2000
                    ),
                    tail_index=row[15],
                    fixed_axis=MVector3D(
                        float(row[25]), float(row[26]), float(row[27])
                    ),
                    parent_index=row[13],
                    effect_factor=float(row[22]),
                    effect_index=row[23],
                    local_x_vector=MVector3D(
                        float(row[29]), float(row[30]), float(row[31])
                    ),
                    local_z_vector=MVector3D(
                        float(row[32]), float(row[33]), float(row[34])
                    ),
                    tail_position=MVector3D(
                        float(row[16]), float(row[17]), float(row[18])
                    ),
                )

                if len(row[37]) > 0:
                    # IKターゲットがある場合、IK登録
                    bone.ik = Ik(
                        model.bones[row[37]].index,
                        int(row[38]),
                        math.radians(float(row[39])),
                    )

                model.bones.append(bone)
            elif row[0] == "IKLink":
                iklink = IkLink(
                    bone_index=model.bones[row[2]].index,
                    angle_limit=int(row[3]),
                    min_angle_limit_radians=MVector3D(
                        float(row[4]), float(row[6]), float(row[8])
                    ),
                    max_angle_limit_radians=MVector3D(
                        float(row[5]), float(row[7]), float(row[9])
                    ),
                )
                model.bones[row[1]].ik.links.append(iklink)

    for bone in model.bones:
        # 親ボーンINDEXの設定
        if bone.parent_index and bone.parent_index in model.bones:
            bone.parent_index = model.bones[bone.parent_index].index
        else:
            bone.parent_index = -1

        if bone.tail_index and bone.tail_index in model.bones:
            bone.tail_index = model.bones[bone.tail_index].index
        else:
            bone.tail_index = -1

    return model


PMX_CONNECTIONS = {
    "root": {
        "mmd": "上半身",
        "parent": None,
        "tail": "spine2",
        "display": "体幹",
        "axis": None,
    },
    "spine2": {
        "mmd": "上半身2",
        "parent": "root",
        "tail": "head_bottom",
        "display": "体幹",
        "axis": None,
    },
    "head_bottom": {
        "mmd": "首",
        "parent": "spine2",
        "tail": "head",
        "display": "体幹",
        "axis": None,
    },
    "nose": {
        "mmd": "頭",
        "parent": "head_bottom",
        "tail": "head_tail",
        "display": "体幹",
        "axis": MVector3D(1, 0, 0),
        "parent_axis": MVector3D(1, 0, 0),
    },
    "left_collar": {
        "mmd": "左肩",
        "parent": "spine2",
        "tail": "left_shoulder",
        "display": "左手",
        "axis": MVector3D(1, 0, 0),
    },
    "right_collar": {
        "mmd": "右肩",
        "parent": "spine2",
        "tail": "right_shoulder",
        "display": "右手",
        "axis": MVector3D(-1, 0, 0),
    },
    "left_shoulder": {
        "mmd": "左腕",
        "parent": "left_collar",
        "tail": "left_elbow",
        "display": "左手",
        "axis": MVector3D(1, 0, 0),
    },
    "right_shoulder": {
        "mmd": "右腕",
        "parent": "right_collar",
        "tail": "right_elbow",
        "display": "右手",
        "axis": MVector3D(-1, 0, 0),
    },
    "left_elbow": {
        "mmd": "左ひじ",
        "parent": "left_shoulder",
        "tail": "left_wrist",
        "display": "左手",
        "axis": MVector3D(1, 0, 0),
    },
    "right_elbow": {
        "mmd": "右ひじ",
        "parent": "right_shoulder",
        "tail": "right_wrist",
        "display": "右手",
        "axis": MVector3D(-1, 0, 0),
    },
    "left_wrist": {
        "mmd": "左手首",
        "parent": "left_elbow",
        "tail": "left_wrist_tail",
        "display": "左手",
        "axis": MVector3D(1, 0, 0),
    },
    "right_wrist": {
        "mmd": "右手首",
        "parent": "right_elbow",
        "tail": "right_wrist_tail",
        "display": "右手",
        "axis": MVector3D(-1, 0, 0),
    },
    "pelvis": {
        "mmd": "下半身",
        "parent": None,
        "tail": "pelvis2",
        "display": "体幹",
        "axis": None,
    },
    "pelvis2": {
        "mmd": "下半身先",
        "parent": "pelvis",
        "tail": "",
        "display": "体幹",
        "axis": None,
    },
    "left_hip": {
        "mmd": "左足",
        "parent": "pelvis",
        "tail": "left_knee",
        "display": "左足",
        "axis": None,
    },
    "right_hip": {
        "mmd": "右足",
        "parent": "pelvis",
        "tail": "right_knee",
        "display": "右足",
        "axis": None,
    },
    "left_knee": {
        "mmd": "左ひざ",
        "parent": "left_hip",
        "tail": "left_ankle",
        "display": "左足",
        "axis": None,
    },
    "right_knee": {
        "mmd": "右ひざ",
        "parent": "right_hip",
        "tail": "right_ankle",
        "display": "右足",
        "axis": None,
    },
    "left_ankle": {
        "mmd": "左足首",
        "parent": "left_knee",
        "tail": "left_foot",
        "display": "左足",
        "axis": None,
    },
    "right_ankle": {
        "mmd": "右足首",
        "parent": "right_knee",
        "tail": "right_foot",
        "display": "右足",
        "axis": None,
    },
}
