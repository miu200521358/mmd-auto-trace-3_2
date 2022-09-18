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

MIKU_METER = 12.5
# 頭ボーンまでの高さ
HEAD_HEIGHT = 150.0
# 画素数->ミクセル変換（横）
PIXEL_RATIO_HORIZONAL = 60.0


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
        trace_rot_model = PmxReader().read_by_filepath(args.trace_rot_model_config)

        logger.info(
            "モーション中心位置計算開始",
            decoration=MLogger.DECORATION_LINE,
        )

        # 全人物分の順番別ファイル
        max_fno = 0
        all_root_pos = MVector3D()
        all_frame_joints: dict[int, dict[int, MVector3D]] = {}
        for person_file_path in sorted(
            glob(os.path.join(args.img_dir, "mix", "*.json"))
        ):
            pname, _ = os.path.splitext(os.path.basename(person_file_path))

            frame_joints = {}
            with open(person_file_path, "r", encoding="utf-8") as f:
                frame_joints = json.load(f)

            for fidx, frames in frame_joints.items():
                fno = int(fidx)

                if fno not in all_frame_joints:
                    all_frame_joints[fno] = {}

                PIXEL_RATIO_VERTICAL = PIXEL_RATIO_HORIZONAL * (
                    frames["image"]["height"] / frames["image"]["width"]
                )
                all_frame_joints[fno][pname] = MVector3D(
                    frames["root"]["x"]
                    * (PIXEL_RATIO_HORIZONAL / frames["image"]["width"]),
                    frames["root"]["y"]
                    * (PIXEL_RATIO_VERTICAL / frames["image"]["height"]),
                    frames["root"]["z"],
                )
                break

        # 開始キーフレ
        start_fno = list(sorted(list(all_frame_joints.keys())))[0]
        all_root_pos = MVector3D(99999999, 0, 0)
        root_pname = -1
        for pname, rpos in all_frame_joints[start_fno].items():
            if abs(all_root_pos.x) > abs(rpos.x):
                # よりセンターに近い方がrootとなる
                all_root_pos = rpos
                root_pname = pname
        all_root_pos.x = 0

        logger.info(
            "モーション中心位置: 人物INDEX [{root_pname}], 中心位置 {root_pos}",
            root_pname=root_pname,
            root_pos=all_root_pos.to_log(),
            decoration=MLogger.DECORATION_LINE,
        )

        for person_file_path in sorted(
            glob(os.path.join(args.img_dir, "mix", "*.json"))
        ):
            pname, _ = os.path.splitext(os.path.basename(person_file_path))

            logger.info(
                "【No.{pname}】モーション結果位置計算開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            trace_abs_mov_motion = VmdMotion()
            trace_rel_mov_motion = VmdMotion()

            frame_joints = {}
            with open(person_file_path, "r", encoding="utf-8") as f:
                frame_joints = json.load(f)

            for fidx, frames in tqdm(frame_joints.items(), desc=f"No.{pname} ... "):
                fno = int(fidx)
                PIXEL_RATIO_VERTICAL = PIXEL_RATIO_HORIZONAL * (
                    frames["image"]["height"] / frames["image"]["width"]
                )
                root_pos = (
                    MVector3D(
                        frames["root"]["x"]
                        * (PIXEL_RATIO_HORIZONAL / frames["image"]["width"]),
                        frames["root"]["y"]
                        * (PIXEL_RATIO_VERTICAL / frames["image"]["height"]),
                        frames["root"]["z"],
                    )
                    - all_root_pos
                )

                for jname, joint in frames["body"].items():
                    if jname not in PMX_CONNECTIONS:
                        continue
                    bf = VmdBoneFrame(name=PMX_CONNECTIONS[jname], index=fno)
                    bf.position = (
                        MVector3D(
                            float(joint["x"]) * MIKU_METER,
                            float(joint["y"]) * MIKU_METER,
                            float(joint["z"]) * MIKU_METER,
                        )
                        - root_pos
                    )
                    trace_abs_mov_motion.bones.append(bf)

                max_fno = fno

            logger.info(
                "【No.{pname}】モーション(移動)計算開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            with tqdm(
                total=(len(trace_abs_mov_motion.bones) * max_fno),
                desc=f"No.{pname} ... ",
            ) as pchar:
                for bone_name in PMX_CONNECTIONS.values():
                    # 処理対象ボーン
                    bone = trace_mov_model.bones[bone_name]
                    # 処理対象の親ボーン
                    parent_bone = (
                        trace_mov_model.bones[bone.parent_index]
                        if bone.parent_index in trace_mov_model.bones
                        else None
                    )
                    # 親ボーンの絶対座標
                    parent_pos = parent_bone.position if parent_bone else MVector3D()

                    for abs_bf in trace_abs_mov_motion.bones[bone_name]:
                        # 処理対象ボーンの親ボーンキーフレ絶対位置
                        abs_parent_bf = (
                            trace_abs_mov_motion.bones[parent_bone.name][abs_bf.index]
                            if parent_bone
                            else VmdBoneFrame()
                        )

                        bf = VmdBoneFrame(name=abs_bf.name, index=abs_bf.index)
                        bf.position = (abs_bf.position - abs_parent_bf.position) - (
                            bone.position - parent_pos
                        )
                        trace_rel_mov_motion.bones.append(bf)

                        pchar.update(1)

            trace_mov_motion_path = os.path.join(
                motion_dir_path, f"trace_{process_datetime}_mov_no{pname}.vmd"
            )
            logger.info(
                "【No.{pname}】モーション(移動)生成開始【{path}】",
                pname=pname,
                path=os.path.basename(trace_mov_motion_path),
                decoration=MLogger.DECORATION_LINE,
            )
            VmdWriter.write(
                trace_mov_model.name, trace_rel_mov_motion, trace_mov_motion_path
            )

            # # -------------------------------------------------
            # # 回転

            # logger.info(
            #     "【No.{pname}】モーション(回転)計算開始",
            #     pname=pname,
            #     decoration=MLogger.DECORATION_LINE,
            # )

            # trace_rot_motion = VmdMotion()

            # with tqdm(
            #     total=(len(trace_rel_mov_motion.bones) * max_fno),
            #     desc=f"No.{pname} ... ",
            # ) as pchar:
            #     for fno in range(max_fno):
            #         upper_mov_bf = trace_rel_mov_motion.bones["下半身"][fno]
            #         upper_abs_pos: MVector3D = (
            #             upper_mov_bf.position + trace_mov_model.bones["下半身"].position
            #         )
            #         upper_abs_pos.y += trace_rot_model.bones["下半身"].position.y

            #         center_bf = VmdBoneFrame("センター", fno)
            #         center_bf.position = MVector3D(upper_abs_pos.x, 0, upper_abs_pos.z)
            #         trace_rot_motion.bones.append(center_bf)

            #         groove_bf = VmdBoneFrame("グルーブ", fno)
            #         groove_bf.position = MVector3D(
            #             0, upper_abs_pos.y - trace_rot_model.bones["グルーブ"].position.y, 0
            #         )
            #         trace_rot_motion.bones.append(groove_bf)

            #         # for bone_name in trace_rel_mov_motion.bones.names():
            #         #     # 処理対象ボーン
            #         #     bone = trace_rot_model.bones[bone_name]
            #         #     # 処理対象の親ボーン
            #         #     parent_bone = (
            #         #         trace_rot_model.bones[bone.parent_index]
            #         #         if bone.parent_index in trace_rot_model.bones
            #         #         else None
            #         #     )
            #         #     # 親ボーンの絶対座標
            #         #     parent_pos = (
            #         #         parent_bone.position if parent_bone else MVector3D()
            #         #     )

            #         #     # 処理対象ボーンキーフレ絶対位置
            #         #     abs_bf = trace_rel_mov_motion.bones[bone_name][fno]
            #         #     # 処理対象ボーンの親ボーンキーフレ絶対位置
            #         #     abs_parent_bf = (
            #         #         trace_rel_mov_motion.bones[parent_bone.name][fno]
            #         #         if parent_bone
            #         #         else VmdBoneFrame()
            #         #     )

            #         #     bf = VmdBoneFrame(name=bone_name, index=fno)
            #         #     bf.position = (abs_bf.position - abs_parent_bf.position) - (
            #         #         bone.position - parent_pos
            #         #     )
            #         #     trace_rel_mov_motion.bones.append(bf)

            #         #     pchar.update(1)

            # trace_rot_motion_path = os.path.join(
            #     motion_dir_path, f"trace_{process_datetime}_rot_no{pname}.vmd"
            # )
            # logger.info(
            #     "【No.{pname}】モーション(回転)生成開始【{path}】",
            #     pname=pname,
            #     path=os.path.basename(trace_rot_motion_path),
            #     decoration=MLogger.DECORATION_LINE,
            # )
            # VmdWriter.write(
            #     trace_rot_model.name, trace_rot_motion, trace_rot_motion_path
            # )

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
    "spine": "上半身",
    "neck": "首",
    "nose": "鼻",
    "head": "頭",
    "right_eye": "右目",
    "left_eye": "左目",
    "right_ear": "右耳",
    "left_ear": "左耳",
    "pelvis": "下半身",
    "left_hip": "左足",
    "right_hip": "右足",
    "left_knee": "左ひざ",
    "right_knee": "右ひざ",
    "left_ankle": "左足首",
    "right_ankle": "右足首",
    "left_foot_index": "左つま先",
    "right_foot_index": "右つま先",
    "left_heel": "左かかと",
    "right_heel": "右かかと",
    "left_collar": "左肩",
    "right_collar": "右肩",
    "left_shoulder": "左腕",
    "right_shoulder": "右腕",
    "left_elbow": "左ひじ",
    "right_elbow": "右ひじ",
    "body_left_wrist": "左手首",
    "body_right_wrist": "右手首",
    "body_right_pinky": "右小指１",
    "body_left_pinky": "左小指１",
    "body_right_index": "右人指１",
    "body_left_index": "左人指１",
    "body_right_thumb": "右親指０",
    "body_left_thumb": "左親指０",
}
