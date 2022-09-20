import argparse
import json
import os
import random
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../Snipper")))

import cv2
import torch
from base.logger import MLogger
from PIL import Image
from Snipper.datasets.data_preprocess.dataset_util import Joint
from Snipper.inference_utils import associate_snippets, bbox_2d_padded, get_all_samples
from Snipper.models.model import build_model
from tqdm import tqdm

logger = MLogger(__name__)


def execute(args):
    try:
        logger.info(
            "人物姿勢推定開始: {img_dir}",
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

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        parser = get_args_parser()
        argv = parser.parse_args(args=[])
        argv.max_depth = 300

        device = torch.device(argv.device)

        model, criterion, postprocessors = build_model(argv)
        if argv.resume and os.path.exists(argv.resume):
            checkpoint = torch.load(argv.resume, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
        else:
            logger.error(
                "指定された学習モデルが存在しません\n{resume}",
                resume=argv.resume,
                decoration=MLogger.DECORATION_BOX,
            )
            return False
        model.eval()

        logger.info(
            "学習モデル準備完了: {resume}",
            resume=argv.resume,
            decoration=MLogger.DECORATION_LINE,
        )

        for frame_dir in sorted(glob(os.path.join(args.img_dir, "frames", "*"))):
            dir_name = os.path.basename(frame_dir)
            argv.data_dir = frame_dir
            argv.output_dir = os.path.join(
                args.img_dir, "snipper", os.path.basename(frame_dir)
            )
            os.makedirs(argv.output_dir, exist_ok=True)

            logger.info(
                "【No.{dir_name}】snipper姿勢推定開始",
                dir_name=dir_name,
                decoration=MLogger.DECORATION_LINE,
            )

            all_samples, frame_indices, all_filenames = get_all_samples(
                argv
            )  # snippet of images

            results = []
            with torch.set_grad_enabled(
                False
            ):  # deactivate autograd to reduce memory usage
                for samples in tqdm(all_samples):
                    imgs = samples["imgs"].to(device).unsqueeze(dim=0)  # batchsize = 1
                    input_size = (
                        samples["input_size"]
                        .to(device)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )  # [1, 1, 1, 2]
                    outputs, _ = model(imgs)

                    max_depth = argv.max_depth
                    bs, num_queries = outputs["pred_logits"].shape[:2]
                    for i in range(bs):
                        human_prob = outputs["pred_logits"][i].softmax(-1)[..., 1]

                        _out_kepts_depth = outputs["pred_depth"][
                            i
                        ]  # n x T x num_kpts x 1
                        # root + displacement
                        _out_kepts_depth[:, :, 1:, :] = (
                            _out_kepts_depth[:, :, 0:1, :]
                            + _out_kepts_depth[:, :, 1:, :] / max_depth
                        )
                        out_kepts_depth = (
                            max_depth * _out_kepts_depth
                        )  # scale to original depth

                        out_score = outputs["pred_kpts2d"][
                            i, :, :, :, 2:3
                        ]  # n x T x num_kpts x 1
                        out_kepts2d = outputs["pred_kpts2d"][
                            i, :, :, :, 0:2
                        ]  # n x T x num_kpts x 2
                        # root + displacement
                        out_kepts2d[:, :, 1:, :] = (
                            out_kepts2d[:, :, :1, :] + out_kepts2d[:, :, 1:, :]
                        )
                        out_kepts2d = (
                            out_kepts2d * input_size
                        )  # scale to original image size

                        inv_trans = samples["inv_trans"]
                        input_size = samples["input_size"]
                        img_size = samples["img_size"]
                        filenames = samples["filenames"]
                        results.append(
                            {
                                "human_score": human_prob.cpu().numpy(),  # [n]
                                "pred_kpt_scores": out_score.cpu().numpy(),  # [n, T, num_joints, 1]
                                "pred_kpts": out_kepts2d.cpu().numpy(),  # [n, T, num_kpts, 2]
                                "pred_depth": out_kepts_depth.cpu().numpy(),  # [n, T, num_kpts, 1]
                                "inv_trans": inv_trans.cpu().numpy(),  # [2, 3]
                                "filenames": filenames,  # [filename_{t}, filename_{t+gap}, ...]
                                "input_size": input_size.cpu().numpy(),  # (w, h)
                                "img_size": img_size.cpu().numpy(),  # (w, h)
                            }
                        )

            logger.info(
                "【No.{dir_name}】姿勢推定の関連付け",
                dir_name=dir_name,
                decoration=MLogger.DECORATION_LINE,
            )

            all_frames_results, max_pid = associate_snippets(
                results, frame_indices, all_filenames, argv
            )

            logger.info(
                "【No.{dir_name}】姿勢推定結果保存(3D)",
                dir_name=dir_name,
                decoration=MLogger.DECORATION_LINE,
            )

            save_results_3d(
                all_frames_results,
                all_filenames,
                argv.data_dir,
                argv.output_dir,
                max_pid,
                argv.max_depth,
                argv.seq_gap,
            )

        logger.info(
            "姿勢推定結果検証",
            decoration=MLogger.DECORATION_LINE,
        )

        mix_output_dir = os.path.join(args.img_dir, "snipper", "mix")
        os.makedirs(mix_output_dir, exist_ok=True)

        mix_output_json_dir = os.path.join(mix_output_dir, "json")
        os.makedirs(mix_output_json_dir, exist_ok=True)

        mix_output_track2d_dir = os.path.join(mix_output_dir, "track2d")
        os.makedirs(mix_output_track2d_dir, exist_ok=True)

        frame_count = 0
        for output_dir in sorted(glob(os.path.join(args.img_dir, "snipper", "*"))):
            frame_count += len(glob(os.path.join(output_dir, "json", "*.json"))) * 1000

        all_json_datas = {}
        with tqdm(
            total=frame_count,
        ) as pchar:

            for oidx, output_dir in enumerate(
                sorted(glob(os.path.join(args.img_dir, "snipper", "*")))
            ):
                for json_path in list(
                    sorted(glob(os.path.join(output_dir, "json", "*.json")))
                ):
                    file_name = os.path.basename(json_path)
                    person_idx, _ = file_name.split(".")
                    json_datas = {}
                    with open(json_path, "r") as f:
                        json_datas = json.load(f)

                    if oidx == 0:
                        # 最初はそのままコピー
                        all_json_datas[person_idx] = json_datas
                        continue

                    start_matchs = {}
                    for (
                        target_person_idx,
                        person_json_datas,
                    ) in all_json_datas.items():
                        start_matchs[target_person_idx] = {}

                        for sidx in list(json_datas.keys())[:200]:
                            start_matchs[target_person_idx][sidx] = 9999999999

                            bbox = json_datas[sidx]["snipper"]["bbox"]

                            bbox_x = int(bbox["x"])
                            bbox_y = int(bbox["y"])
                            bbox_w = int(bbox["width"])
                            bbox_h = int(bbox["height"])

                            if sidx not in person_json_datas:
                                continue

                            pbbox = person_json_datas[sidx]["snipper"]["bbox"]

                            pbbox_x = int(pbbox["x"])
                            pbbox_y = int(pbbox["y"])
                            pbbox_w = int(pbbox["width"])
                            pbbox_h = int(pbbox["height"])

                            # bboxの差異を図る
                            start_matchs[target_person_idx][sidx] = (
                                abs(pbbox_x - bbox_x)
                                + abs(pbbox_y - bbox_y)
                                + abs(pbbox_w - bbox_w)
                                + abs(pbbox_h - bbox_h)
                            )

                    match_idxs = {}
                    for pidx, start_match in start_matchs.items():
                        match_idxs[pidx] = np.mean(list(start_match.values()))

                    match_person_idx = list(match_idxs.keys())[
                        np.argmin(list(match_idxs.values()))
                    ]
                    # マッチしたのでも差異が大きければ新しくINDEX付与
                    if match_idxs[match_person_idx] > 80:
                        match_person_idx = (
                            f"{(int(list(all_json_datas.keys())[-1]) + 1):02d}"
                        )
                        all_json_datas[match_person_idx] = {}

                    for sidx, json_data in json_datas.items():
                        all_json_datas[match_person_idx][sidx] = json_data

                        pchar.update(1)

        random.seed(13)
        pids = list(all_json_datas.keys())
        cmap = plt.get_cmap("rainbow")
        pid_colors = [cmap(i) for i in np.linspace(0, 1, len(pids))]
        random.shuffle(pid_colors)
        pid_colors_opencv = [
            (np.array((c[2], c[1], c[0])) * 255).astype(int).tolist()
            for c in pid_colors
        ]

        logger.info("姿勢推定結果保存(2D)", decoration=MLogger.DECORATION_LINE)

        with tqdm(
            total=frame_count,
        ) as pchar:
            image_paths = {}
            for pidx, (pid, json_datas) in enumerate(all_json_datas.items()):
                for fidx, json_data in json_datas.items():
                    image_path = json_data["snipper"]["image"]["path"]
                    if image_path in image_paths:
                        image_path = image_paths[image_path]

                    process_path = os.path.join(
                        mix_output_track2d_dir, os.path.basename(image_path)
                    )
                    save_visual_results_2d(
                        image_path,
                        process_path,
                        pidx,
                        pid,
                        pid_colors_opencv,
                        json_data["snipper"]["joints"],
                        json_data["snipper"]["bbox"],
                    )

                    image_paths[image_path] = process_path

                    pchar.update(1)

                with open(
                    os.path.join(mix_output_json_dir, f"{pid}.json"), mode="w"
                ) as f:
                    json.dump(json_datas, f, indent=4)

        logger.info("姿勢推定結果保存(動画)", decoration=MLogger.DECORATION_LINE)

        frames = glob(os.path.join(mix_output_track2d_dir, "*.*"))
        img = Image.open(frames[0])

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            os.path.join(mix_output_dir, "snipper.mp4"),
            fourcc,
            30.0,
            (img.size[0], img.size[1]),
        )

        for process_img_path in tqdm(frames):
            # トラッキングmp4合成
            out.write(cv2.imread(process_img_path))

        out.release()
        cv2.destroyAllWindows()

        logger.info(
            "姿勢推定結果保存完了: {output_dir}",
            output_dir=argv.output_dir,
            decoration=MLogger.DECORATION_BOX,
        )

        return True
    except Exception as e:
        logger.critical("姿勢推定で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False


def save_visual_results_2d(
    image_path: str,
    process_path: str,
    pidx: int,
    pid: str,
    pid_colors_opencv: list,
    joints: dict,
    bbox: dict,
):
    img = cv2.imread(image_path)

    SKELETONS = [
        ("root", "left_hip"),
        ("root", "right_hip"),
        ("root", "head_bottom"),
        ("head_bottom", "left_shoulder"),
        ("head_bottom", "right_shoulder"),
        ("head_bottom", "nose"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]

    for l, (j1, j2) in enumerate(SKELETONS):
        joint1 = joints[j1]
        joint2 = joints[j2]

        joint1_x = int(joint1["x"])
        joint1_y = int(joint1["y"])
        joint2_x = int(joint2["x"])
        joint2_y = int(joint2["y"])

        if joint1["z"] > 0 and joint2["z"] > 0:
            t = 4
            r = 8
            cv2.line(
                img,
                (joint1_x, joint1_y),
                (joint2_x, joint2_y),
                color=tuple(pid_colors_opencv[pidx]),
                # color=tuple(sks_colors[l]),
                thickness=t,
            )
            cv2.circle(
                img,
                thickness=-1,
                center=(joint1_x, joint1_y),
                radius=r,
                color=tuple(pid_colors_opencv[pidx]),
            )
            cv2.circle(
                img,
                thickness=-1,
                center=(joint2_x, joint2_y),
                radius=r,
                color=tuple(pid_colors_opencv[pidx]),
            )

    bbox_x = int(bbox["x"])
    bbox_y = int(bbox["y"])
    bbox_w = int(bbox["width"])
    bbox_h = int(bbox["height"])
    bbx_thick = 3
    cv2.line(
        img,
        (bbox_x, bbox_y),
        (bbox_x + bbox_w, bbox_y),
        color=tuple(pid_colors_opencv[pidx]),
        thickness=bbx_thick,
    )
    cv2.line(
        img,
        (bbox_x, bbox_y),
        (bbox_x, bbox_y + bbox_h),
        color=tuple(pid_colors_opencv[pidx]),
        thickness=bbx_thick,
    )
    cv2.line(
        img,
        (bbox_x + bbox_w, bbox_y),
        (bbox_x + bbox_w, bbox_y + bbox_h),
        color=tuple(pid_colors_opencv[pidx]),
        thickness=bbx_thick,
    )
    cv2.line(
        img,
        (bbox_x, bbox_y + bbox_h),
        (bbox_x + bbox_w, bbox_y + bbox_h),
        color=tuple(pid_colors_opencv[pidx]),
        thickness=bbx_thick,
    )

    cv2.putText(
        img,
        pid,
        (bbox_x + bbox_w // 3, bbox_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color=tuple(pid_colors_opencv[pidx]),
        thickness=bbx_thick,
    )

    cv2.imwrite(process_path, img)


def save_results_3d(
    all_frames_results, all_filenames, data_dir, save_dir, max_pid, max_depth, gap
):

    result_dir = os.path.join(save_dir, "json")
    os.makedirs(result_dir, exist_ok=True)

    json_datas = {}
    for frame_idx in tqdm(all_frames_results.keys()):
        filename = all_filenames[frame_idx]
        # ファイル名をそのままフレーム番号として扱う
        fno = int(filename.split(".")[0])
        img = cv2.imread(os.path.join(data_dir, filename))
        h, w, _ = img.shape

        pids, poses = all_frames_results[frame_idx]
        for p, pid in enumerate(pids):
            kpt_3d = poses[p]
            if pid not in json_datas:
                json_datas[pid] = {}

            bbx = bbox_2d_padded(kpt_3d, 0.3, 0.3)

            json_datas[int(pid)][fno] = {
                "snipper": {
                    "image": {
                        "path": os.path.join(data_dir, filename),
                        "width": float(w),
                        "height": float(h),
                    },
                    "bbox": {
                        "x": float(bbx[0]),
                        "y": float(bbx[1]),
                        "width": float(bbx[2]),
                        "height": float(bbx[3]),
                    },
                    "joints": {},
                },
            }
            for n, (x, y, z, score) in enumerate(kpt_3d):
                json_datas[int(pid)][fno]["snipper"]["joints"][Joint.NAMES[n]] = {
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "score": float(score),
                }

    for pid, json_data in json_datas.items():
        with open(os.path.join(result_dir, f"{pid:02d}.json"), mode="w") as f:
            json.dump(json_data, f, indent=4)


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument(
        "--lr_backbone_names", default=["backbone.0"], type=str, nargs="+"
    )
    parser.add_argument(
        "--lr_linear_proj_names",
        default=["reference_points", "sampling_offsets"],
        type=str,
        nargs="+",
    )
    parser.add_argument("--lr_linear_proj_mult", default=0.1, type=float)

    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--lr_drop", default=1, type=int)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )
    parser.add_argument("--use_amp", default=0, type=int)
    parser.add_argument("--use_pytorch_deform", default=1, type=int)

    parser.add_argument(
        "--output_dir",
        default="C:/Users/shihaozou/Desktop/exps",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--data_dir",
        default="C:/Users/shihaozou/Desktop/exps/seq3",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--resume",
        default=os.path.abspath(
            os.path.join(__file__, "../../../data/model/12-06_20-17-34/checkpoint.pth")
        ),
        help="resume from checkpoint",
    )
    parser.add_argument(
        "--vis_heatmap_frame_name",
        default=None,
        help="visualize heatmap of a frame, None means all the sampled frames",
    )

    # * dataset parameters
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument(
        "--input_width", default=800, type=int, help="input image shape (H, W)"
    )
    parser.add_argument(
        "--input_height", default=600, type=int, help="input image shape (H, W)"
    )
    parser.add_argument("--max_depth", type=int, default=45)
    parser.add_argument("--num_frames", default=4, type=int, help="Number of frames")
    parser.add_argument(
        "--num_future_frames", default=0, type=int, help="Number of frames"
    )
    parser.add_argument(
        "--seq_gap", default=1, type=int, help="Number of maximum gap frames"
    )
    parser.add_argument("--num_workers", type=int, default=4)

    # * Backbone
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )
    parser.add_argument(
        "--num_feature_levels", default=3, type=int, help="number of feature levels"
    )

    # * transformer
    parser.add_argument(
        "--hidden_dim",
        default=384,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=1024,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument("--pre_norm", action="store_true")
    parser.add_argument(
        "--num_queries", default=60, type=int, help="Number of query slots"
    )
    parser.add_argument(
        "--num_kpts", default=15, type=int, help="Number of query slots"
    )
    parser.add_argument("--dec_n_points", default=4, type=int)
    parser.add_argument("--enc_n_points", default=4, type=int)

    # * matcher
    parser.add_argument("--set_cost_is_human", default=1, type=float)

    parser.add_argument("--set_cost_root", default=1, type=float)
    parser.add_argument("--set_cost_root_depth", default=1, type=float)
    parser.add_argument("--set_cost_root_vis", default=0.1, type=float)

    parser.add_argument("--set_cost_joint", default=1, type=float)
    parser.add_argument("--set_cost_joint_depth", default=1, type=float)
    parser.add_argument("--set_cost_joint_vis", default=0.1, type=float)

    # * Segmentation
    parser.add_argument(
        "--masks",
        default=False,
        type=bool,
        help="Train segmentation head if the flag is provided",
    )

    # Loss
    # parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
    #                     help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument("--aux_loss", default=True, type=bool)
    parser.add_argument("--two_stage", default=False, type=bool)

    # * Loss coefficients
    parser.add_argument("--is_human_loss_coef", default=1, type=float)

    parser.add_argument("--root_loss_coef", default=1, type=float)
    parser.add_argument("--root_depth_loss_coef", default=1, type=float)
    parser.add_argument("--root_vis_loss_coef", default=1, type=float)

    parser.add_argument("--joint_loss_coef", default=1, type=float)
    parser.add_argument("--joint_depth_loss_coef", default=1, type=float)
    parser.add_argument("--joint_vis_loss_coef", default=1, type=float)

    parser.add_argument("--joint_disp_loss_coef", default=1, type=float)
    parser.add_argument("--joint_disp_depth_loss_coef", default=1, type=float)

    parser.add_argument("--cont_loss_coef", default=0.1, type=float)
    parser.add_argument("--heatmap_loss_coef", default=0.01, type=float)

    parser.add_argument(
        "--eos_coef",
        default=0.25,
        type=float,
        help="Relative classification weight of the no-object class",
    )

    parser.add_argument("--local_rank", type=int, default=0)
    return parser
