import json
import os
import re
from glob import glob

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

        for oidx, mediapipe_file_path in enumerate(
            sorted(glob(os.path.join(args.img_dir, "mediapipe", "json", "*.json")))
        ):
            logger.info(
                "【No.{oidx}】関節スムージング準備開始",
                oidx=f"{oidx:02}",
                decoration=MLogger.DECORATION_LINE,
            )

            frame_joints = {}
            with open(mediapipe_file_path, "r") as f:
                frame_joints = json.load(f)

            max_fno = 0
            joint_datas = {}
            for fidx, frame_json_data in tqdm(
                frame_joints.items(), desc=f"No.{oidx:02} ... "
            ):
                fno = int(fidx)
                for jtype in [
                    "snipper",
                    "mp_body_joints",
                    "mp_body_world_joints",
                    "mp_left_hand_joints",
                    "mp_right_hand_joints",
                    "mp_face_joints",
                ]:
                    if jtype not in frame_json_data:
                        continue

                    for jname, joint in frame_json_data[jtype]["joints"].items():
                        for axis in ["x", "y", "z"]:
                            if (jname, jtype, axis) not in joint_datas:
                                joint_datas[(jname, jtype, axis)] = {}

                            joint_datas[(jname, jtype, axis)][fno] = float(joint[axis])

                max_fno = fno

            logger.info(
                "【No.{oidx}】関節スムージング開始",
                oidx=f"{oidx:02}",
                decoration=MLogger.DECORATION_LINE,
            )

            # スムージング
            for (jname, jtype, axis), joints in tqdm(
                joint_datas.items(), desc=f"No.{oidx:02} ... "
            ):
                filter = OneEuroFilter(
                    freq=30, mincutoff=1, beta=0.00000000001, dcutoff=1
                )
                for fno, joint in joints.items():
                    joint_datas[(jname, jtype, axis)][fno] = filter(joint, fno)

            logger.info(
                "【No.{oidx}】関節スムージング出力開始",
                oidx=f"{oidx:02}",
                decoration=MLogger.DECORATION_LINE,
            )

            # ジョイントグローバル座標を保存
            with tqdm(
                total=(len(joint_datas) * max_fno),
                desc=f"No.{oidx:02} ... ",
            ) as pchar:
                for (jname, jtype, axis), frame_json_data in joint_datas.items():
                    for fno, smooth_value in frame_json_data.items():
                        fidx = str(fno)
                        if fidx not in frame_joints or jtype not in frame_joints[fidx]:
                            continue

                        frame_joints[fidx][jtype]["joints"][jname][axis] = float(
                            smooth_value
                        )
                        pchar.update(1)

            # 出力先ソート済みファイル
            smoothed_person_file_path = os.path.join(
                args.img_dir, "smooth", f"{oidx:02}.json"
            )

            with open(smoothed_person_file_path, "w", encoding="utf-8") as f:
                json.dump(frame_joints, f, indent=4)

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
