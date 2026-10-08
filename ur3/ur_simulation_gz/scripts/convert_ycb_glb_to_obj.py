#!/usr/bin/env python3
"""Convert a simple, embedded-texture YCB GLB into Gazebo 6 OBJ assets.

Ignition Gazebo 6 cannot render GLB files. The selected ai-habitat YCB assets
contain one triangle primitive with float positions/normals/UVs, uint16
indices and an embedded PNG, so a small dependency-free converter is enough.
"""

import argparse
from io import BytesIO
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

from PIL import Image


COMPONENTS = {
    5121: ("B", 1),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}
WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read_glb(path: Path):
    data = path.read_bytes()
    magic, version, total_length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or total_length != len(data):
        raise ValueError(f"{path} is not a valid GLB 2.0 file")
    offset = 12
    chunks = {}
    while offset < len(data):
        length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunks[chunk_type] = data[offset:offset + length]
        offset += length
    document = json.loads(chunks[0x4E4F534A].decode("utf-8").rstrip("\0 "))
    return document, chunks[0x004E4942]


def accessor_values(document, binary, accessor_index):
    accessor = document["accessors"][accessor_index]
    view = document["bufferViews"][accessor["bufferView"]]
    component_format, component_size = COMPONENTS[accessor["componentType"]]
    width = WIDTHS[accessor["type"]]
    packed_size = component_size * width
    stride = view.get("byteStride", packed_size)
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    unpack_format = "<" + component_format * width
    return [
        struct.unpack_from(unpack_format, binary, start + index * stride)
        for index in range(accessor["count"])
    ]


def decode_texture(
    texture_bytes: bytes,
    mime_type: str,
    ktx_command: str,
    basisu_command: str,
) -> Image.Image:
    if mime_type == "image/png":
        with Image.open(BytesIO(texture_bytes)) as texture:
            return texture.copy()
    if mime_type != "image/x-basis":
        raise ValueError(f"unsupported embedded texture MIME type: {mime_type}")

    def resolve_executable(command: str, description: str) -> str:
        resolved = shutil.which(command)
        if resolved is not None:
            return resolved
        candidate = Path(command).expanduser().resolve()
        if not candidate.is_file():
            raise ValueError(
                f"Basis texture requires {description}; pass its executable path"
            )
        return str(candidate)

    ktx2_magic = b"\xabKTX 20\xbb\r\n\x1a\n"
    with tempfile.TemporaryDirectory(prefix="ycb-texture-") as temporary_name:
        temporary = Path(temporary_name)
        if texture_bytes.startswith(ktx2_magic):
            resolved_ktx = resolve_executable(
                ktx_command, "Khronos `ktx extract`"
            )
            ktx_path = temporary / "embedded.ktx2"
            png_path = temporary / "decoded.png"
            ktx_path.write_bytes(texture_bytes)
            subprocess.run(
                [
                    resolved_ktx,
                    "extract",
                    "--transcode",
                    "rgba8",
                    str(ktx_path),
                    str(png_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
        elif texture_bytes.startswith(b"sB"):
            resolved_basisu = resolve_executable(
                basisu_command, "Basis Universal `basisu -unpack`"
            )
            basis_path = temporary / "embedded.basis"
            basis_path.write_bytes(texture_bytes)
            # ETC1_RGB is transcoder format 0 and produces ordinary RGB PNGs.
            # Restricting the unpacker to this native ETC1S-compatible format
            # avoids hundreds of diagnostic variants. Running with
            # cwd=temporary guarantees no mip images leak into the repository.
            subprocess.run(
                [
                    resolved_basisu,
                    "-unpack",
                    "-no_ktx",
                    "-format_only",
                    "0",
                    "-file",
                    basis_path.name,
                ],
                cwd=temporary,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            candidates = sorted(
                temporary.glob("embedded_unpacked_rgb_ETC1_RGB_0_*.png")
            )
            if len(candidates) != 1:
                raise ValueError(
                    "Basis Universal did not emit exactly one ETC1 RGB mip-0 image"
                )
            png_path = candidates[0]
        else:
            raise ValueError("image/x-basis payload is neither raw Basis nor KTX2")
        with Image.open(png_path) as texture:
            return texture.copy()


def convert(
    input_path: Path,
    output_directory: Path,
    max_texture_size: int,
    ktx_command: str = "ktx",
    basisu_command: str = "basisu",
    uv_v_mode: str = "flip",
) -> None:
    document, binary = read_glb(input_path)
    primitives = [
        primitive
        for mesh in document["meshes"]
        for primitive in mesh["primitives"]
    ]
    if len(primitives) != 1 or primitives[0].get("mode", 4) != 4:
        raise ValueError("converter expects exactly one triangle primitive")
    primitive = primitives[0]
    attributes = primitive["attributes"]
    positions = accessor_values(document, binary, attributes["POSITION"])
    normals = accessor_values(document, binary, attributes["NORMAL"])
    texcoords = accessor_values(document, binary, attributes["TEXCOORD_0"])
    indices = [value[0] for value in accessor_values(
        document, binary, primitive["indices"]
    )]
    if len(indices) % 3:
        raise ValueError("triangle index count is not divisible by three")

    output_directory.mkdir(parents=True, exist_ok=True)
    # Ignition Gazebo 6 may cache OBJ/MTL resources by their basename and
    # material name.  Reusing generic names (textured.obj, material.mtl,
    # texture.png and ycb_material) therefore makes every YCB mesh inherit the
    # texture of the first object loaded.  Keep every resource globally unique.
    asset_name = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in output_directory.name
    )
    material_name = f"ycb_{asset_name}_material"
    obj_filename = f"{asset_name}.obj"
    mtl_filename = f"{asset_name}.mtl"
    texture_filename = f"{asset_name}.png"
    image = document["images"][0]
    if "bufferView" not in image:
        raise ValueError("converter expects one embedded texture bufferView")
    image_view = document["bufferViews"][image["bufferView"]]
    image_start = image_view.get("byteOffset", 0)
    texture_path = output_directory / texture_filename
    texture_bytes = binary[image_start:image_start + image_view["byteLength"]]
    with decode_texture(
        texture_bytes,
        image.get("mimeType", ""),
        ktx_command,
        basisu_command,
    ) as texture:
        resampling = getattr(Image, "Resampling", Image)
        texture.thumbnail((max_texture_size, max_texture_size), resampling.LANCZOS)
        texture.save(texture_path, format="PNG", optimize=True)

    (output_directory / mtl_filename).write_text(
        f"newmtl {material_name}\n"
        "Ka 1.000000 1.000000 1.000000\n"
        "Kd 1.000000 1.000000 1.000000\n"
        "Ks 0.000000 0.000000 0.000000\n"
        "d 1.0\n"
        "illum 1\n"
        f"map_Kd {texture_filename}\n",
        encoding="utf-8",
    )

    lines = [f"mtllib {mtl_filename}", f"o {asset_name}", f"usemtl {material_name}"]
    lines.extend(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in positions)
    # Some GLBs embed an ordinary top-down PNG and need the conventional
    # glTF-to-OBJ V flip. The ai-habitat raw Basis payloads used by the newer
    # YCB assets already carry the vertical orientation expected after BasisU
    # decoding; flipping those UVs makes the mesh sample the atlas background.
    # Keep this explicit so regenerated legacy assets do not silently change.
    if uv_v_mode == "flip":
        output_texcoords = ((u, 1.0 - v) for u, v in texcoords)
    elif uv_v_mode == "preserve":
        output_texcoords = iter(texcoords)
    else:
        raise ValueError(f"unsupported uv_v_mode: {uv_v_mode}")
    lines.extend(f"vt {u:.9g} {v:.9g}" for u, v in output_texcoords)
    lines.extend(f"vn {x:.9g} {y:.9g} {z:.9g}" for x, y, z in normals)
    for index in range(0, len(indices), 3):
        face = [value + 1 for value in indices[index:index + 3]]
        lines.append("f " + " ".join(f"{value}/{value}/{value}" for value in face))
    (output_directory / obj_filename).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_glb", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--max-texture-size", type=int, default=1024)
    parser.add_argument(
        "--ktx-command",
        default="ktx",
        help="Khronos ktx executable used for embedded Basis/KTX2 textures",
    )
    parser.add_argument(
        "--basisu-command",
        default="basisu",
        help="Basis Universal executable used for embedded raw Basis textures",
    )
    parser.add_argument(
        "--uv-v-mode",
        choices=("flip", "preserve"),
        default="flip",
        help=(
            "flip for conventional embedded PNGs; preserve for ai-habitat "
            "raw Basis payloads whose decoded atlas is already Y-oriented"
        ),
    )
    args = parser.parse_args()
    if args.max_texture_size < 64:
        parser.error("--max-texture-size must be at least 64")
    convert(
        args.input_glb,
        args.output_directory,
        args.max_texture_size,
        args.ktx_command,
        args.basisu_command,
        args.uv_v_mode,
    )


if __name__ == "__main__":
    main()
