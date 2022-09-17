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

        # 全人物分の順番別フォルダ
        ordered_person_file_pathes = sorted(
            glob(os.path.join(args.img_dir, "trace", "mediapipe", "*.json"))
        )
        os.makedirs(os.path.join(args.img_dir, "smooth"), exist_ok=True)

        frame_pattern = re.compile(r"^(\d+)\.")

        for oidx, ordered_person_file_path in enumerate(ordered_person_file_pathes):
            logger.info(
                "【No.{oidx}】関節スムージング開始",
                oidx=f"{oidx:02}",
                decoration=MLogger.DECORATION_LINE,
            )

            all_joints = {}
            with open(ordered_person_file_path, "r") as f:
                frame_joints = json.load(f)

            # ジョイントグローバル座標を保持
            for fidx, frames in frame_joints.items():
                fno = int(fidx)
                for jname, joint in frames.items():
                    if (jname, "x") not in all_joints:
                        all_joints[(jname, "x")] = {}

                    if (jname, "y") not in all_joints:
                        all_joints[(jname, "y")] = {}

                    if (jname, "z") not in all_joints:
                        all_joints[(jname, "z")] = {}

                    if (jname, "score") not in all_joints:
                        all_joints[(jname, "score")] = {}

                    all_joints[(jname, "x")][fno] = float(joint["x"])
                    all_joints[(jname, "y")][fno] = float(joint["y"])
                    all_joints[(jname, "z")][fno] = float(joint["z"])
                    all_joints[(jname, "score")][fno] = float(joint["score"])

            # スムージング
            for (jname, axis), joints in tqdm(
                all_joints.items(), desc=f"Filter No.{oidx:02} ... "
            ):
                if axis == "score":
                    continue
                filter = OneEuroFilter(
                    freq=30, mincutoff=1, beta=0.00000000001, dcutoff=1
                )
                for fno, joint in joints.items():
                    all_joints[(jname, axis)][fno] = filter(joint, fno)

            # 出力先ソート済みファイル
            smoothed_person_file_path = os.path.join(
                args.img_dir, "smooth", f"{oidx:02}.json"
            )

            # ジョイントグローバル座標を保存
            for fidx, frames in frame_joints.items():
                fno = int(fidx)
                for jname, joint in frames.items():
                    frame_joints[fidx][jname]["x"] = str(all_joints[(jname, "x")][fno])
                    frame_joints[fidx][jname]["y"] = str(all_joints[(jname, "y")][fno])
                    frame_joints[fidx][jname]["z"] = str(all_joints[(jname, "z")][fno])
                    frame_joints[fidx][jname]["score"] = str(
                        all_joints[(jname, "score")][fno]
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
