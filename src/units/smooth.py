import json
import os
import re
from glob import glob

import numpy as np
from base.logger import MLogger
from mmd.vmd.filter import OneEuroFilter
from tqdm import tqdm

logger = MLogger(__name__, level=MLogger.DEBUG)


def execute(args):
    try:
        logger.info(
            "関節スムージング処理開始: {img_dir}",
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

        if not os.path.exists(os.path.join(args.img_dir, "mediapipe", "json")):
            logger.error(
                "指定されたmediapipe姿勢推定ディレクトリが存在しません。\n姿勢推定が完了していない可能性があります。: {img_dir}",
                img_dir=os.path.join(args.img_dir, "mediapipe", "json"),
                decoration=MLogger.DECORATION_BOX,
            )
            return False

        os.makedirs(os.path.join(args.img_dir, "smooth"), exist_ok=True)

        os.makedirs(os.path.join(args.img_dir, "mix"), exist_ok=True)

        for mediapipe_file_path in sorted(
            glob(os.path.join(args.img_dir, "mediapipe", "json", "*.json"))
        ):
            pname, _ = os.path.splitext(os.path.basename(mediapipe_file_path))

            logger.info(
                "【No.{pname}】関節スムージング準備開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            frame_joints = {}
            with open(mediapipe_file_path, "r") as f:
                frame_joints = json.load(f)

            max_fno = 0
            joint_datas = {}
            for fidx, frame_json_data in tqdm(
                frame_joints.items(), desc=f"No.{pname} ... "
            ):
                fno = int(fidx)
                width = int(frame_json_data["snipper"]["image"]["width"])
                height = int(frame_json_data["snipper"]["image"]["height"])
                for jtype in [
                    "snipper",
                    "mp_body_joints",
                    "mp_body_world_joints",
                    "mp_left_hand_joints",
                    "mp_right_hand_joints",
                ]:
                    if jtype not in frame_json_data:
                        continue

                    for jname, joint in frame_json_data[jtype]["joints"].items():
                        if float(joint.get("score", 1.0)) < 0.9 and jname != "root":
                            # scoreが低い時はスルー
                            continue
                        for axis in ["x", "y", "z"]:
                            if (jname, jtype, axis) not in joint_datas:
                                joint_datas[(jname, jtype, axis)] = {}
                        if jtype == "snipper":
                            joint_datas[(jname, jtype, "x")][fno] = (width / 2) - float(
                                joint["x"]
                            )
                            joint_datas[(jname, jtype, "y")][fno] = (
                                height / 2
                            ) - float(joint["y"])
                            joint_datas[(jname, jtype, "z")][fno] = -float(joint["z"])
                        else:
                            for axis in ["x", "y", "z"]:
                                joint_datas[(jname, jtype, axis)][fno] = float(
                                    joint[axis]
                                )

                max_fno = fno

            logger.info(
                "【No.{pname}】関節スムージング開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            # スムージング
            for (jname, jtype, axis), joints in tqdm(
                joint_datas.items(), desc=f"No.{pname} ... "
            ):
                filter = OneEuroFilter(freq=30, mincutoff=0.3, beta=0.01, dcutoff=0.25)
                for fno, joint in joints.items():
                    joint_datas[(jname, jtype, axis)][fno] = filter(joint, fno)

            logger.info(
                "【No.{pname}】関節スムージング出力開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            # ジョイントグローバル座標を保存
            smooth_joints = {}
            with tqdm(
                total=(len(joint_datas) * max_fno),
                desc=f"No.{pname} ... ",
            ) as pchar:
                for (jname, jtype, axis), frame_json_data in joint_datas.items():
                    for fno, smooth_value in frame_json_data.items():
                        fidx = str(fno)
                        if fidx not in smooth_joints:
                            smooth_joints[fidx] = {
                                "image": {
                                    "width": float(
                                        frame_joints[fidx]["snipper"]["image"]["width"]
                                    ),
                                    "height": float(
                                        frame_joints[fidx]["snipper"]["image"]["height"]
                                    ),
                                },
                                "bbox": {
                                    "x": float(
                                        frame_joints[fidx]["snipper"]["bbox"]["x"]
                                    ),
                                    "y": float(
                                        frame_joints[fidx]["snipper"]["bbox"]["y"]
                                    ),
                                    "width": float(
                                        frame_joints[fidx]["snipper"]["bbox"]["width"]
                                    ),
                                    "height": float(
                                        frame_joints[fidx]["snipper"]["bbox"]["height"]
                                    ),
                                },
                            }
                            if "mp_face_joints" in frame_joints[fidx]:
                                smooth_joints[fidx]["mp_face_joints"] = frame_joints[
                                    fidx
                                ]["mp_face_joints"]
                        if jtype not in smooth_joints[fidx]:
                            smooth_joints[fidx][jtype] = {"joints": {}}
                        if jname not in smooth_joints[fidx][jtype]["joints"]:
                            smooth_joints[fidx][jtype]["joints"][jname] = {}

                        smooth_joints[fidx][jtype]["joints"][jname][axis] = float(
                            smooth_value
                        )
                        pchar.update(1)

            logger.info(
                "【No.{pname}】関節スムージング結果保存",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            with open(
                os.path.join(args.img_dir, "smooth", f"{pname}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(smooth_joints, f, indent=4)

            logger.info(
                "【No.{pname}】関節スムージング合成開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            mix_joints = {}
            for fidx, smooth_json_data in tqdm(
                smooth_joints.items(), desc=f"No.{pname} ... "
            ):
                mix_joints[fidx] = {
                    "image": smooth_json_data["image"],
                    "root": smooth_json_data["snipper"]["joints"]["root"],
                    "body": {},
                }

                if "mp_body_world_joints" not in smooth_json_data:
                    continue

                mix_joints[fidx]["body"] = smooth_json_data["mp_body_world_joints"][
                    "joints"
                ]

                for jname in [
                    "pelvis",
                    "spine",
                    "neck",
                    "head",
                    "left_collar",
                    "right_collar",
                ]:
                    mix_joints[fidx]["body"][jname] = {}

                for axis in ["x", "y", "z"]:
                    # 下半身
                    mix_joints[fidx]["body"]["pelvis"][axis] = np.mean(
                        [
                            mix_joints[fidx]["body"]["left_hip"][axis],
                            mix_joints[fidx]["body"]["right_hip"][axis],
                        ]
                    )
                    # 上半身
                    mix_joints[fidx]["body"]["spine"][axis] = np.mean(
                        [
                            mix_joints[fidx]["body"]["left_hip"][axis],
                            mix_joints[fidx]["body"]["right_hip"][axis],
                        ]
                    )
                    # 首
                    mix_joints[fidx]["body"]["neck"][axis] = np.mean(
                        [
                            mix_joints[fidx]["body"]["left_shoulder"][axis],
                            mix_joints[fidx]["body"]["right_shoulder"][axis],
                        ]
                    )
                    # 左肩
                    mix_joints[fidx]["body"]["left_collar"][axis] = np.mean(
                        [
                            mix_joints[fidx]["body"]["left_shoulder"][axis],
                            mix_joints[fidx]["body"]["right_shoulder"][axis],
                        ]
                    )
                    # 右肩
                    mix_joints[fidx]["body"]["right_collar"][axis] = np.mean(
                        [
                            mix_joints[fidx]["body"]["left_shoulder"][axis],
                            mix_joints[fidx]["body"]["right_shoulder"][axis],
                        ]
                    )
                    # 頭
                    mix_joints[fidx]["body"]["head"][axis] = np.mean(
                        [
                            mix_joints[fidx]["body"]["left_ear"][axis],
                            mix_joints[fidx]["body"]["right_ear"][axis],
                        ]
                    )

                for direction in ["left", "right"]:
                    body_wrist_jname = f"body_{direction}_wrist"
                    hand_jtype = f"mp_{direction}_hand_joints"
                    if (
                        body_wrist_jname
                        not in smooth_json_data["mp_body_world_joints"]["joints"]
                        or hand_jtype not in smooth_json_data
                    ):
                        continue

                    hand_root_jname = f"{direction}_hand"
                    hand_root_vec = {}
                    for axis in ["x", "y", "z"]:
                        hand_root_vec[axis] = float(
                            smooth_json_data["mp_body_world_joints"]["joints"][
                                body_wrist_jname
                            ][axis]
                            - smooth_json_data[hand_jtype]["joints"]["wrist"][axis]
                        )

                    if hand_root_jname not in mix_joints[fidx]:
                        mix_joints[fidx][hand_root_jname] = {}

                    for jname, jvalues in smooth_json_data[hand_jtype][
                        "joints"
                    ].items():
                        mix_joints[fidx][hand_root_jname][jname] = {}
                        for axis in ["x", "y", "z"]:
                            mix_joints[fidx][hand_root_jname][jname][axis] = float(
                                jvalues[axis]
                            ) + float(hand_root_vec[axis])

                if "mp_face_joints" in smooth_json_data:
                    mix_joints[fidx]["face"] = smooth_json_data["mp_face_joints"][
                        "joints"
                    ]

            logger.info(
                "【No.{pname}】関節スムージング合成結果保存",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            with open(
                os.path.join(args.img_dir, "mix", f"{pname}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(mix_joints, f, indent=4)

        logger.info(
            "関節スムージング処理終了: {img_dir}",
            img_dir=args.img_dir,
            decoration=MLogger.DECORATION_BOX,
        )

        return True
    except Exception as e:
        logger.critical(
            "関節スムージングで予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX
        )
        return False
