import argparse
import json
import os
import random
import torch
import base64
import uuid
import cv2
import numpy as np
import hashlib
from pathlib import Path
from termcolor import colored
import llava
from llava import conversation as clib
from llava.media import Image, Video, Depth
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

######################## Flask
from flask import Flask, request, jsonify

app = Flask(__name__)
UPLOAD_FOLDER = './tmp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
######################## Flask

######################## Models Initialization
depth_anything = None
vlm_model = None
depth_input_size = 518
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# The research pilot compares RGB-only and RGB-D one-shot outputs.  Sampling
# would confound that paired comparison, so the API can explicitly lock greedy
# decoding and expose the active policy in every response.
generation_mode = 'checkpoint_default'
loaded_model_fingerprint = None
pilot_random_seed = None
########################


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_model(model_path):
    root = Path(model_path).expanduser().resolve()
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        entries.append({
            "path": str(path.relative_to(root)),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    canonical = (
        json.dumps(entries, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return {
        "model_name": root.name,
        "model_root": str(root),
        "file_count": len(entries),
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def init_models(depth_model_path, vlm_model_path, depth_encoder='vitl', load_depth_model=True):
    global depth_anything, vlm_model

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    if load_depth_model:
        depth_anything = DepthAnythingV2(**model_configs[depth_encoder])
        depth_anything.load_state_dict(torch.load(depth_model_path, map_location='cpu'))
        depth_anything = depth_anything.to(DEVICE).eval()
    else:
        # A robot already has a registered depth camera.  In that deployment
        # loading Depth Anything duplicates both work and GPU memory, so the
        # ROS adapter supplies a normalized depth image with every request.
        depth_anything = None

    vlm_conv_mode = 'auto'
    vlm_model = llava.load(vlm_model_path)
    clib.default_conversation = clib.conv_templates[vlm_conv_mode].copy()


def decode_base64_to_file(base64_str, prefix="image"):
    filename = f"{UPLOAD_FOLDER}/{prefix}_{uuid.uuid4().hex}.png"
    with open(filename, "wb") as f:
        f.write(base64.b64decode(base64_str))
    return filename

@app.route('/', methods=['GET'])
def index():
    return "Hello, World!"


@app.route('/query', methods=['POST'])
def query():

    data = request.get_json()

    image_urls = data.get("image_url", [])
    depth_urls = data.get("depth_url", [])
    enable_depth = data.get("enable_depth", 0)
    text = data.get("text", "")

    image_files = [decode_base64_to_file(img_b64, prefix="image") for img_b64 in image_urls]

    depth_files = []
    if enable_depth == 1:
        if len(depth_urls) > 0:
            assert len(depth_urls) == len(image_urls), "Depth URL's number is not equal to Image URL's number"
            depth_files = [decode_base64_to_file(dp_b64, prefix="depth") for dp_b64 in depth_urls]
        else:
            if depth_anything is None:
                return jsonify({
                    'result': 0,
                    'answer': '',
                    'error': (
                        'Depth input is required because the server was started '
                        'with --skip_depth_model'
                    ),
                }), 400
            for img_f in image_files:
                raw_image = cv2.imread(img_f)
                depth = depth_anything.infer_image(raw_image, input_size=depth_input_size, device=DEVICE)

                # 归一化并转为 8 bit
                depth_range = max(float(depth.max() - depth.min()), 1e-6)
                depth = (depth - depth.min()) / depth_range * 255.0
                depth = depth.astype(np.uint8)
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)

                depth_file = f"depth_{uuid.uuid4().hex}.png"
                cv2.imwrite(depth_file, depth)
                depth_files.append(depth_file)
                print(f"Depth file saved to {depth_file}")

    prompt = []
    for img_f in image_files:
        if any(img_f.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
            prompt.append(Image(img_f))
        elif any(img_f.endswith(ext) for ext in [".mp4", ".mkv", ".webm"]):
            prompt.append(Video(img_f))
        else:
            raise ValueError(f"Unsupported media type: {img_f}")

    # 如果需要深度图，则将深度图也放到 prompt
    if enable_depth == 1 and depth_files:
        for dp_f in depth_files:
            prompt.append(Depth(dp_f))

    if text:
        prompt.append(text)

    if generation_mode == 'greedy':
        generation_config = vlm_model.default_generation_config
        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
        answer = vlm_model.generate_content(
            prompt, generation_config=generation_config
        )
    else:
        answer = vlm_model.generate_content(prompt)

    print(colored(answer, "cyan", attrs=["bold"]))


    for img_f in image_files:
        os.remove(img_f)
    for dp_f in depth_files:
        os.remove(dp_f)

    response = jsonify({
        'result': 1,
        'answer': answer,
        'generation_mode': generation_mode,
        'generation_config': {
            'do_sample': False if generation_mode == 'greedy' else None,
            'temperature': None if generation_mode == 'greedy' else 'checkpoint',
            'top_p': None if generation_mode == 'greedy' else 'checkpoint',
            'top_k': None if generation_mode == 'greedy' else 'checkpoint',
        },
        'model_fingerprint': loaded_model_fingerprint,
        'random_seed': pilot_random_seed,
    })

    response.headers.set('Content-Type', 'application/json')

    return response

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=25547)
    parser.add_argument("--depth_model_path", type=str, default="/share/project/zhouenshen/hpfs/ckpt/depthanything/depth_anything_v2_vitl.pth")
    parser.add_argument("--vlm_model_path", type=str, default="/share/project/zhouenshen/hpfs/code/VILA/runs/train/RoboRefer-2B-SFT")
    parser.add_argument(
        "--skip_depth_model",
        action="store_true",
        help="Do not load Depth Anything; require depth_url in every RGB-D request.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=8132026,
        help="Recorded RNG seed; paired pilot still requires --generation_mode greedy.",
    )
    parser.add_argument(
        "--generation_mode",
        choices=("checkpoint_default", "greedy"),
        default="checkpoint_default",
        help=(
            "Decoding policy. Use greedy for paired scientific ablations; "
            "checkpoint_default preserves the upstream checkpoint behavior."
        ),
    )
    args = parser.parse_args()

    generation_mode = args.generation_mode
    pilot_random_seed = int(args.seed)
    random.seed(pilot_random_seed)
    np.random.seed(pilot_random_seed)
    torch.manual_seed(pilot_random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(pilot_random_seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    loaded_model_fingerprint = fingerprint_model(args.vlm_model_path)

    init_models(
        args.depth_model_path,
        args.vlm_model_path,
        load_depth_model=not args.skip_depth_model,
    )

    app.run(host='0.0.0.0', port=args.port)
