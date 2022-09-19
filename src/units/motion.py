import csv
import json
import math
import os
from datetime import datetime
from glob import glob

from base.logger import MLogger
from base.math import MMatrix4x4, MQuaternion, MVector3D
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

        process_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

        # トレース用モデルを読み込む
        trace_rot_model = PmxReader().read_by_filepath(args.trace_rot_model_config)

        logger.info(
            "モーション中心位置計算開始",
            decoration=MLogger.DECORATION_LINE,
        )

        # 全人物分の合成済みファイル
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

                pixel_ratio_vertical = PIXEL_RATIO_HORIZONAL * (
                    frames["image"]["height"] / frames["image"]["width"]
                )
                all_frame_joints[fno][pname] = MVector3D(
                    frames["root"]["x"]
                    * (PIXEL_RATIO_HORIZONAL / frames["image"]["width"]),
                    frames["root"]["y"]
                    * (pixel_ratio_vertical / frames["image"]["height"]),
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

            frame_joints = {}
            with open(person_file_path, "r", encoding="utf-8") as f:
                frame_joints = json.load(f)

            max_fno = 0
            for fidx, frames in tqdm(frame_joints.items(), desc=f"No.{pname} ... "):
                fno = int(fidx)
                pixel_ratio_vertical = PIXEL_RATIO_HORIZONAL * (
                    frames["image"]["height"] / frames["image"]["width"]
                )
                root_pos = (
                    MVector3D(
                        frames["root"]["x"]
                        * (PIXEL_RATIO_HORIZONAL / frames["image"]["width"]),
                        frames["root"]["y"]
                        * (pixel_ratio_vertical / frames["image"]["height"]),
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

                if fno > max_fno:
                    max_fno = fno

            logger.info(
                "【No.{pname}】モーション(センター)計算開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            trace_rot_motion = VmdMotion()

            for mov_bf in tqdm(
                trace_abs_mov_motion.bones["下半身"], desc=f"No.{pname} ... "
            ):
                center_abs_pos: MVector3D = mov_bf.position

                center_pos: MVector3D = center_abs_pos - (
                    trace_rot_model.bones["下半身"].position
                    - trace_rot_model.bones["腰"].position
                )
                center_pos.y = 0

                center_bf = VmdBoneFrame(name="センター", index=mov_bf.index)
                center_bf.position = center_pos
                trace_rot_motion.bones.append(center_bf)

                groove_pos: MVector3D = center_abs_pos - (
                    trace_rot_model.bones["下半身"].position
                    - trace_rot_model.bones["腰"].position
                )
                groove_pos.x = 0
                groove_pos.z = 0

                groove_bf = VmdBoneFrame(name="グルーブ", index=mov_bf.index)
                groove_bf.position = groove_pos
                trace_rot_motion.bones.append(groove_bf)

            logger.info(
                "【No.{pname}】モーション(回転)計算開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            with tqdm(
                total=(len(VMD_CONNECTIONS) * max_fno),
                desc=f"No.{pname} ... ",
            ) as pchar:
                for target_bone_name, vmd_params in VMD_CONNECTIONS.items():
                    direction_from_name = vmd_params["direction"][0]
                    direction_to_name = vmd_params["direction"][1]
                    up_from_name = vmd_params["up"][0]
                    up_to_name = vmd_params["up"][1]
                    cross_from_name = (
                        vmd_params["cross"][0]
                        if "cross" in vmd_params
                        else vmd_params["direction"][0]
                    )
                    cross_to_name = (
                        vmd_params["cross"][1]
                        if "cross" in vmd_params
                        else vmd_params["direction"][1]
                    )
                    cancel_names = vmd_params["cancel"]

                    for mov_bf in trace_abs_mov_motion.bones[target_bone_name]:
                        bone_direction = (
                            trace_rot_model.bones[direction_to_name].position
                            - trace_rot_model.bones[direction_from_name].position
                        ).normalized()

                        bone_up = (
                            trace_rot_model.bones[up_to_name].position
                            - trace_rot_model.bones[up_from_name].position
                        ).normalized()

                        bone_cross = (
                            trace_rot_model.bones[cross_to_name].position
                            - trace_rot_model.bones[cross_from_name].position
                        ).normalized()

                        bone_cross_vec: MVector3D = bone_up.cross(
                            bone_cross
                        ).normalized()

                        initial_qq = MQuaternion.from_direction(
                            bone_direction, bone_cross_vec
                        )

                        direction_from_abs_pos = trace_abs_mov_motion.bones[
                            direction_from_name
                        ][mov_bf.index].position

                        direction_to_abs_pos = trace_abs_mov_motion.bones[
                            direction_to_name
                        ][mov_bf.index].position

                        direction: MVector3D = (
                            direction_to_abs_pos - direction_from_abs_pos
                        ).normalized()

                        up_from_abs_pos = trace_abs_mov_motion.bones[up_from_name][
                            mov_bf.index
                        ].position

                        up_to_abs_pos = trace_abs_mov_motion.bones[up_to_name][
                            mov_bf.index
                        ].position

                        up: MVector3D = (up_to_abs_pos - up_from_abs_pos).normalized()

                        cross_from_abs_pos = trace_abs_mov_motion.bones[
                            cross_from_name
                        ][mov_bf.index].position

                        cross_to_abs_pos = trace_abs_mov_motion.bones[cross_to_name][
                            mov_bf.index
                        ].position

                        cross: MVector3D = (
                            cross_to_abs_pos - cross_from_abs_pos
                        ).normalized()

                        motion_cross_vec: MVector3D = up.cross(cross).normalized()

                        motion_qq = MQuaternion.from_direction(
                            direction, motion_cross_vec
                        )

                        cancel_qq = MQuaternion()
                        for cancel_name in cancel_names:
                            cancel_qq *= trace_rot_motion.bones[cancel_name][
                                mov_bf.index
                            ].rotation

                        bf = VmdBoneFrame(name=target_bone_name, index=mov_bf.index)
                        bf.rotation = (
                            cancel_qq.inverse() * motion_qq * initial_qq.inverse()
                        )
                        trace_rot_motion.bones.append(bf)

                        pchar.update(1)

            logger.info(
                "【No.{pname}】モーション(IK)計算開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            with tqdm(
                total=(2 * max_fno),
                desc=f"No.{pname} ... ",
            ) as pchar:
                for direction in ["左", "右"]:

                    leg_ik_bone_name = f"{direction}足ＩＫ"
                    leg_bone_name = f"{direction}足"
                    knee_bone_name = f"{direction}ひざ"
                    ankle_bone_name = f"{direction}足首"
                    toe_bone_name = f"{direction}つま先"

                    leg_link_names = [
                        "全ての親",
                        "センター",
                        "グルーブ",
                        "腰",
                        "下半身",
                        f"腰キャンセル{direction}",
                        leg_bone_name,
                        knee_bone_name,
                        ankle_bone_name,
                        toe_bone_name,
                    ]

                    leg_bone_direction = (
                        trace_rot_model.bones[ankle_bone_name].position
                        - trace_rot_model.bones[knee_bone_name].position
                    ).normalized()

                    leg_bone_up = (
                        trace_rot_model.bones[toe_bone_name].position
                        - trace_rot_model.bones[ankle_bone_name].position
                    ).normalized()

                    leg_bone_cross_vec: MVector3D = leg_bone_up.cross(
                        leg_bone_direction
                    ).normalized()

                    leg_initial_qq = MQuaternion.from_direction(
                        leg_bone_direction, leg_bone_cross_vec
                    )

                    for lower_bf in trace_rot_motion.bones["下半身"]:
                        fno = lower_bf.index
                        mats = {}
                        for bidx, bname in enumerate(leg_link_names):
                            mat = MMatrix4x4(identity=True)
                            bf = trace_rot_motion.bones[bname][fno]
                            # キーフレの相対位置
                            relative_pos = (
                                trace_rot_model.bones[bname].position + bf.position
                            )
                            if bidx > 0:
                                # 子ボーンの場合、親の位置をさっぴく
                                relative_pos -= trace_rot_model.bones[
                                    leg_link_names[bidx - 1]
                                ].position

                            mat.translate(relative_pos)
                            mat.rotate(bf.rotation)
                            mats[bname] = (
                                mats[leg_link_names[bidx - 1]] * mat
                                if bidx > 0
                                else mat
                            )

                        # 足IKの角度

                        knee_abs_pos = mats[knee_bone_name] * MVector3D()
                        ankle_abs_pos = mats[ankle_bone_name] * MVector3D()
                        toe_abs_pos = mats[toe_bone_name] * MVector3D()

                        leg_direction_pos = (ankle_abs_pos - knee_abs_pos).normalized()
                        leg_up_pos = (toe_abs_pos - ankle_abs_pos).normalized()
                        leg_cross_pos: MVector3D = leg_up_pos.cross(
                            leg_direction_pos
                        ).normalized()

                        leg_ik_qq = (
                            MQuaternion.from_direction(leg_direction_pos, leg_cross_pos)
                            * leg_initial_qq.inverse()
                        )

                        leg_ik_bf = VmdBoneFrame(name=leg_ik_bone_name, index=fno)
                        leg_ik_bf.position = (
                            ankle_abs_pos
                            - trace_rot_model.bones[ankle_bone_name].position
                        )
                        leg_ik_bf.rotation = leg_ik_qq
                        trace_rot_motion.bones.append(leg_ik_bf)

                        pchar.update(1)

            trace_rot_motion_path = os.path.join(
                motion_dir_path, f"trace_{process_datetime}_rot_no{pname}.vmd"
            )
            logger.info(
                "【No.{pname}】モーション(回転)生成開始【{path}】",
                pname=pname,
                path=os.path.basename(trace_rot_motion_path),
                decoration=MLogger.DECORATION_LINE,
            )
            VmdWriter.write(
                trace_rot_model.name, trace_rot_motion, trace_rot_motion_path
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

VMD_CONNECTIONS = {
    "下半身": {"direction": ("首", "上半身"), "up": ("左足", "右足"), "cancel": ()},
    "上半身": {"direction": ("下半身", "首"), "up": ("左腕", "右腕"), "cancel": ()},
    "首": {"direction": ("首", "頭"), "up": ("左腕", "右腕"), "cancel": ("上半身",)},
    "頭": {
        "direction": ("首", "頭"),
        "up": ("左耳", "右耳"),
        "cancel": (
            "上半身",
            "首",
        ),
    },
    "左肩": {
        "direction": ("左肩", "左腕"),
        "up": ("上半身", "首"),
        "cancel": ("上半身",),
    },
    "左腕": {
        "direction": ("左腕", "左ひじ"),
        "up": ("左肩", "左腕"),
        "cancel": (
            "上半身",
            "左肩",
        ),
    },
    "左ひじ": {
        "direction": ("左ひじ", "左手首"),
        "up": ("左腕", "左ひじ"),
        "cancel": (
            "上半身",
            "左肩",
            "左腕",
        ),
    },
    "左手首": {
        "direction": ("左手首", "左人指１"),
        "up": ("左ひじ", "左手首"),
        "cancel": (
            "上半身",
            "左肩",
            "左腕",
            "左ひじ",
        ),
    },
    "左足": {"direction": ("左足", "左ひざ"), "up": ("左足", "右足"), "cancel": ("下半身",)},
    "左ひざ": {
        "direction": ("左ひざ", "左足首"),
        "up": ("左足", "右足"),
        "cancel": (
            "下半身",
            "左足",
        ),
    },
    "左足首": {
        "direction": ("左足首", "左つま先"),
        "up": ("左足", "右足"),
        "cancel": (
            "下半身",
            "左足",
            "左ひざ",
        ),
    },
    "右肩": {
        "direction": ("右肩", "右腕"),
        "up": ("上半身", "首"),
        "cancel": ("上半身",),
    },
    "右腕": {
        "direction": ("右腕", "右ひじ"),
        "up": ("右肩", "右腕"),
        "cancel": (
            "上半身",
            "右肩",
        ),
    },
    "右ひじ": {
        "direction": ("右ひじ", "右手首"),
        "up": ("右腕", "右ひじ"),
        "cancel": (
            "上半身",
            "右肩",
            "右腕",
        ),
    },
    "右手首": {
        "direction": ("右手首", "右人指１"),
        "up": ("右ひじ", "右手首"),
        "cancel": (
            "上半身",
            "右肩",
            "右腕",
            "右ひじ",
        ),
    },
    "右足": {"direction": ("右足", "右ひざ"), "up": ("右足", "左足"), "cancel": ("下半身",)},
    "右ひざ": {
        "direction": ("右ひざ", "右足首"),
        "up": ("右足", "左足"),
        "cancel": (
            "下半身",
            "右足",
        ),
    },
    "右足首": {
        "direction": ("右足首", "右つま先"),
        "up": ("右足", "左足"),
        "cancel": (
            "下半身",
            "右足",
            "右ひざ",
        ),
    },
}
