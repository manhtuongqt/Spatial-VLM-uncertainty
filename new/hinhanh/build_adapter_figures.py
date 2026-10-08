#!/usr/bin/env python3
"""Add Vietnamese figures for the user-selected P-CRA-U, preserving old figures.

Reuse the original figure geometry, colors and plotting rules. No training,
inference, threshold fitting or image synthesis. Each output has _adapter_vi.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / "new/outputs/pcrau_answerability_language_dev_20261003"
TEST = RUN / "test_iid_7p5"
SUFFIX = "_adapter_vi"
FOLDERS = [
    "10_reliability", "11_risk_coverage", "14_decision_cases", "15_failure_cases",
    "16_stratified_object", "17_stratified_relation", "18_stratified_variant",
    "19_answerability_confusion", "20_training_curves", "21_source_delta",
    "22_uncertainty_localization", "23_coverage_area",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    data = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    indexed = {row["sample_id"]: row for row in data}
    assert len(indexed) == len(data) == 1000
    return indexed


TRANSLATIONS = {
    "Ideal calibration": "Hiệu chuẩn lý tưởng",
    "10 equal-width bins": "10 khoảng có độ rộng bằng nhau",
    "Observed error rate": "Tỉ lệ lỗi tổng hợp quan sát",
    "Predicted grounding risk": "Risk tổng hợp đã hiệu chuẩn",
    "Samples": "Số mẫu",
    "Target mask (GT)": "Mask mục tiêu (nhãn thật)",
    "Executed P-CRA-U point": "Điểm P-CRA-U được chấp nhận",
    "(a) P-CRA-U hit · RoboRefer miss": "(a) P-CRA-U đúng · RoboRefer sai",
    "(b) RoboRefer hit · P-CRA-U miss": "(b) RoboRefer đúng · P-CRA-U sai",
    "(c) Both miss": "(c) Cả hai đều sai",
    "Point-in-target accuracy (%)": "Tỉ lệ điểm nằm trong mục tiêu (%)",
    "Grounding by target object group": "Định vị mục tiêu theo nhóm vật thể",
    "Grounding by relation": "Định vị mục tiêu theo quan hệ",
    "Grounding by Test-IID variant": "Định vị mục tiêu theo biến thể Test-IID",
    "fruit": "Hoa quả", "container": "Vật chứa", "mug": "Cốc",
    "box": "Hộp", "cube": "Khối lập phương",
    "direct": "Trực tiếp", "left of": "Bên trái", "right of": "Bên phải",
    "front of": "Phía trước", "behind": "Phía sau",
    "nearer than": "Gần hơn", "farther than": "Xa hơn",
    "between\nin depth": "Ở giữa\ntheo độ sâu",
    "nearer than\nboth": "Gần hơn\ncả hai",
    "clean": "Gốc",
    "semantic\ncounterfactual": "Phản thực\nngữ nghĩa",
    "relation\ncounterfactual": "Phản thực\nquan hệ",
    "depth\ncorruption": "Nhiễu\nđộ sâu",
    "occlusion/view\ncounterfactual": "Phản thực\nche khuất/góc nhìn",
    "P-CRA-U answerability · Test-IID": "P-CRA-U · Khả năng trả lời · Test-IID",
    "(a) Sample counts": "(a) Số lượng mẫu",
    "(b) Row-normalized percentages": "(b) Phần trăm theo hàng nhãn thật",
    "Predicted answerability": "Trạng thái dự đoán",
    "Ground-truth answerability": "Trạng thái nhãn thật",
    "Percentage within true class (%)": "Tỉ lệ trong lớp nhãn thật (%)",
    "FOUND": "Có thể\ntrả lời", "AMBIGUOUS": "Mơ hồ", "ABSENT": "Không có\nmục tiêu",
    "INSUFFICIENT\nEVIDENCE": "Chưa đủ\nbằng chứng",
    "Semantic CF": "Phản thực ngữ nghĩa",
    "Relation CF": "Phản thực quan hệ",
    "Depth corruption": "Nhiễu độ sâu",
    "Occlusion / view CF": "Phản thực che khuất/góc nhìn",
    "Semantic": "Ngữ nghĩa", "Relation": "Quan hệ", "Spatial*": "Không gian*",
    "Depth": "Độ sâu", "Occlusion": "Che khuất",
    "Predicted uncertainty source": "Nguồn bất định dự đoán",
    "Counterfactual variant paired with clean": "Biến thể ghép cặp với mẫu gốc",
    "P-CRA-U · paired source probability changes · 200 families": "P-CRA-U · Thay đổi xác suất nguồn · 200 family",
    "Mean probability difference (variant − clean)": "Chênh lệch xác suất trung bình (biến thể − gốc)",
    "Outlined cells: source matching the variant. Spatial* is weakly supervised.\nProbability differences, not accuracy; descriptive results without confidence intervals.":
        "Ô viền đen: nguồn tương ứng biến thể. Không gian* dùng nhãn yếu.\nĐây là chênh lệch xác suất; kết quả mô tả, chưa có khoảng tin cậy.",
    "Normalized spatial entropy": "Entropy không gian chuẩn hóa",
    "1 − spatial peak margin": "1 − chênh lệch hai đỉnh",
    "Calibrated risk": "Risk đã hiệu chuẩn",
    "P-CRA-U · uncertainty vs localization error · true FOUND only (n=420)":
        "P-CRA-U · Bất định và sai số định vị · nhãn thật có thể trả lời (n=420)",
    "Hit: distance = 0 (405)": "Đúng: khoảng cách = 0 (405)",
    "Miss: distance > 0 (15)": "Sai: khoảng cách > 0 (15)",
    "Distance from MAP to target mask (px; symlog)": "Khoảng cách MAP tới mask mục tiêu (pixel; symlog)",
    "All 420 samples retained; 405 zero-distance points overlap. No jitter or outlier removal.\nPearson uses raw pixel distance; Spearman uses average ranks for ties. Descriptive, no p-values.":
        "Giữ đủ 420 mẫu; 405 điểm khoảng cách 0 chồng nhau. Không dịch điểm hay bỏ ngoại lệ.\nPearson dùng khoảng cách pixel; Spearman dùng hạng trung bình khi bằng nhau. Thống kê mô tả.",
    "P-CRA-U · spatial region coverage–area · 835 target-present samples":
        "P-CRA-U · Độ phủ–diện tích vùng không gian · 835 mẫu có mục tiêu",
    "Evaluator score event: S ≤ mass": "Biến cố score của evaluator: S ≤ mass",
    "Direct region–mask intersection": "Vùng giao trực tiếp mask mục tiêu",
    "Nominal coverage: 90%": "Độ phủ danh nghĩa: 90%",
    "Empirical coverage (%)": "Độ phủ thực nghiệm (%)",
    "(a) Coverage as probability mass increases": "(a) Độ phủ theo khối lượng xác suất",
    "Requested highest-density probability mass": "Khối lượng xác suất vùng mật độ cao",
    "(b) Coverage versus mean region area": "(b) Độ phủ theo diện tích vùng trung bình",
    "Mean grid area (% of 768 cells; logarithmic axis)": "Diện tích trung bình (% của 768 ô; trục logarit)",
    "Descriptive test sweep only: the calibration-fitted mass is unchanged.":
        "Quét test để mô tả; giữ nguyên mass đã fit trên calibration của nhánh không gian đóng băng.",
}


def translate(text):
    if text in TRANSLATIONS:
        return TRANSLATIONS[text]
    # Translate labels with support counts without changing sample identifiers.
    if "\n(n=" in text:
        base, support = text.rsplit("\n(n=", 1)
        if base in TRANSLATIONS:
            return TRANSLATIONS[base] + "\n(n=" + support
    match = re.fullmatch(r"(.*)\nmatched-source increase: (\d+)/200", text)
    if match:
        return TRANSLATIONS[match[1]] + f"\nNguồn tương ứng tăng: {match[2]}/200"
    for old, new in [
        ("Normalized spatial entropy", "Entropy không gian chuẩn hóa"),
        ("1 − spatial peak margin", "1 − chênh lệch hai đỉnh"),
        ("Calibrated risk", "Risk đã hiệu chuẩn"),
        ("Frozen mass =", "Mass đã khóa ="),
        ("Frozen mean area =", "Diện tích đã khóa ="),
        ("score:", "score:"), ("intersection:", "giao mask:"),
    ]:
        if old in text:
            text = text.replace(old, new)
    if text.startswith("Test-IID risk reliability"):
        text = text.replace("Test-IID risk reliability", "Hiệu chuẩn risk P-CRA-U · Test-IID")
    if "\nv211iid_family_" in text and " · " in text.split("\n", 1)[0]:
        prefix, sid = text.split("\n", 1)
        obj, relation = prefix.split(" · ", 1)
        rel = {"nearer_than": "gần hơn", "direct": "trực tiếp", "left_of": "bên trái"}
        text = TRANSLATIONS.get(obj, obj) + " · " + rel.get(relation, relation) + "\n" + sid
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-existing", action="store_true",
                        help="Allow rerendering only this script's _adapter_vi outputs")
    args = parser.parse_args()
    old_hashes = {p: sha(p) for folder in FOLDERS for p in (HERE / folder).iterdir()
                  if p.is_file() and p.suffix in (".png", ".pdf") and SUFFIX not in p.stem}
    active = read(ROOT / "new/outputs/active_experimental_profile.json")
    assert active["model_display_name"] == "P-CRA-U"
    assert Path(active["checkpoint"]) == RUN / "detail_cost/checkpoints/best/model.safetensors"
    for field in ("checkpoint", "config", "calibrator", "profiles"):
        assert sha(active[field]) == active[field + "_sha256"], field
    lock = read(TEST / "evaluation_lock.json")
    for path, expected in lock["frozen_files"].items():
        assert sha(path) == expected, path
    result = read(TEST / "full_metrics.json")
    assert sha(TEST / "inference/predictions.jsonl") == result["artifacts"]["predictions_sha256"]
    assert sha(TEST / "runtime_decisions.jsonl") == result["artifacts"]["runtime_decisions_sha256"]

    spec = importlib.util.spec_from_file_location("legacy_pcrau_figures", HERE / "build_figures.py")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    pred = rows(TEST / "inference/predictions.jsonl")
    runtime = rows(TEST / "runtime_decisions.jsonl")
    assert pred.keys() == runtime.keys() == legacy.predictions.keys()
    assert len({p["family_id"] for p in pred.values()}) == 200
    for sid in pred:
        for key in ("spatial", "source_probabilities", "relation_edge_probabilities", "evaluation"):
            assert pred[sid][key] == legacy.predictions[sid][key], (sid, key)
        xy = pred[sid]["spatial"]["map_pixel_xy"]
        assert xy == [float(legacy.paired[sid]["v2_x"]), float(legacy.paired[sid]["v2_y"])]

    # The unchanged spatial branch retains its historical calibrated mass.
    # The new logistic33 calibrates error risk only; it does not refit a region.
    combined = deepcopy(legacy.decisions)
    for sid, decision in runtime.items():
        combined[sid].update(decision)
        combined[sid]["calibrated_grounding_risk"] = decision["risk"]
    metrics = deepcopy(legacy.metrics)
    metrics["answerability"] = result["answerability_after"]
    metrics["risk_calibration_and_ranking"] = {
        **result["risk_metrics"],
        "frozen_threshold": active["risk_threshold"],
        "accepted_samples": result["policy"]["accepted"],
    }
    legacy.predictions, legacy.decisions, legacy.metrics = pred, combined, metrics

    # Recompute displayed stratified grounding from the adapter's saved outputs.
    for key, strata in legacy.comparison["stratified_target_present_grounding"].items():
        if key not in ("target_object_group", "relation", "variant"):
            continue
        for name, stats in strata.items():
            ids = [sid for sid in pred if pred[sid]["evaluation"]["target_exists"]
                   and legacy.paired[sid][key] == name]
            assert len(ids) == stats["samples"]
            for field in ("b1_hit", "v2_hit"):
                value = np.mean([int(legacy.paired[sid][field]) if field == "b1_hit"
                                 else pred[sid]["evaluation"]["map_inside_target"] for sid in ids])
                assert np.isclose(value, stats["family_cluster_bootstrap_95ci"][field]["estimate"])
                stats["family_cluster_bootstrap_95ci"][field]["estimate"] = float(value)

    generated = []
    def save(fig, folder, basename):
        assert folder in FOLDERS
        target = HERE / folder
        paths = [target / (basename + SUFFIX + ext) for ext in (".pdf", ".png")]
        if any(path.exists() for path in paths) and not args.refresh_existing:
            raise FileExistsError(f"Refusing to overwrite: {paths}")
        fig.canvas.draw()
        # Tick formatters recreate Text at draw time: update the formatter labels
        # themselves so Vietnamese categorical labels survive PNG/PDF rendering.
        for ax in fig.axes:
            for axis in ("x", "y"):
                tick_text = [label.get_text() for label in getattr(ax, f"get_{axis}ticklabels")()]
                translated = [translate(label) for label in tick_text]
                if tick_text != translated:
                    getattr(ax, f"set_{axis}ticks")(getattr(ax, f"get_{axis}ticks")(), translated)
        for artist in fig.findobj(match=Text):
            artist.set_text(translate(artist.get_text()))
        fig.canvas.draw()
        fig.savefig(paths[0], bbox_inches="tight", pad_inches=0.12)
        fig.savefig(paths[1], dpi=220, bbox_inches="tight", pad_inches=0.12)
        generated.extend(paths)
        plt.close(fig)
        print(f"Đã tạo {paths[1].relative_to(ROOT)}", flush=True)
    legacy.save = save

    legacy.fig10_reliability()
    risk, error = legacy.test_risk_arrays()
    order = np.argsort(risk, kind="stable")
    coverage = np.arange(1, 1001) / 1000
    curve = np.cumsum(error[order]) / np.arange(1, 1001)
    assert np.isclose(curve.mean(), result["risk_metrics"]["aurc"], atol=1e-12)
    accepted = np.array([runtime[sid]["action"] == "EXECUTE" for sid in
                         [row["sample_id"] for row in legacy.manifest["entries"]]])
    assert accepted.sum() == 361 and error[accepted].sum() == 34
    x, y = accepted.mean(), error[accepted].mean()
    fig, ax = plt.subplots(figsize=(7.0, 4.9), constrained_layout=True)
    ax.plot(coverage, curve, color=legacy.PURPLE, linewidth=2.0,
            label="Xếp hạng theo risk P-CRA-U đã hiệu chuẩn")
    ax.axhline(error.mean(), color=legacy.GRAY, linestyle="--", linewidth=1,
               label=f"Lỗi trên toàn bộ mẫu = {error.mean():.1%}")
    ax.scatter([x], [y], s=75, color=legacy.ORANGE, zorder=5)
    ax.annotate(f"Ngưỡng đã khóa = {active['risk_threshold']:.4f}\n"
                f"Nhận {x:.1%}, lỗi {y:.2%}\nDự đoán có thể trả lời + cổng risk",
                xy=(x,y), xytext=(x+0.09, y+0.16),
                arrowprops={"arrowstyle":"-", "color":legacy.ORANGE, "lw":1},
                color=legacy.ORANGE, fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Tỉ lệ chấp nhận trong 1.000 mẫu")
    ax.set_ylabel("Tỉ lệ lỗi trong tập chấp nhận")
    ax.set_title(f"Test-IID · Rủi ro–tỉ lệ chấp nhận · AURC={curve.mean():.4f}",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "11_risk_coverage", "fig11_risk_coverage_test_iid")

    # Same four samples, image panels, colors and placement as historical Fig.14.
    cases = [("000127__clean", "EXECUTE"), ("000003__depth_corruption", "REOBSERVE"),
             ("000031__clean", "ASK_USER"), ("000005__clean", "ABSTAIN")]
    actions = {"EXECUTE":"CHẤP NHẬN", "REOBSERVE":"QUAN SÁT LẠI",
               "ASK_USER":"HỎI NGƯỜI DÙNG", "ABSTAIN":"TỪ CHỐI"}
    states = {"FOUND":"Có thể trả lời", "AMBIGUOUS":"Mơ hồ", "ABSENT":"Không có mục tiêu",
              "INSUFFICIENT_EVIDENCE":"Chưa đủ bằng chứng"}
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.7), constrained_layout=True)
    for i, ((short, action), ax) in enumerate(zip(cases, axes.ravel())):
        sid = "v211iid_family_" + short
        assert runtime[sid]["action"] == action
        legacy.image_axis(ax, sid, f"({chr(97+i)}) {actions[action]}", mask=True)
        if action == "EXECUTE":
            legacy.mark_points(ax, sid, b1=False)
        truth = pred[sid]["evaluation"]["answerability_state"]
        guessed = runtime[sid]["predicted_answerability"]
        label = (f"Nhãn thật: {states[truth]}\nDự đoán: {states[guessed]}"
                 f" · risk: {runtime[sid]['risk']:.3f}\n{sid}")
        ax.text(0.02, 0.025, label, transform=ax.transAxes, va="bottom", fontsize=7.4,
                linespacing=1.25, bbox={"facecolor":"white", "edgecolor":legacy.ACTION_COLORS[action],
                                      "alpha":0.90, "pad":4})
    fig.legend(handles=[legacy.Line2D([],[],color=legacy.GREEN,linewidth=1.8,label="Mask mục tiêu (nhãn thật)"),
                        legacy.Line2D([],[],marker="x",linestyle="none",color=legacy.ORANGE,markersize=8,
                                      label="Điểm P-CRA-U được chấp nhận")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5,-0.035))
    save(fig, "14_decision_cases", "fig14_four_policy_decisions")
    legacy.fig15_failures()
    legacy.strata_figure("target_object_group", "16_stratified_object", "fig16_grounding_by_object",
                        "Grounding by target object group", ["fruit","container","mug","box","cube"])
    legacy.strata_figure("relation", "17_stratified_relation", "fig17_grounding_by_relation",
                        "Grounding by relation", ["direct","left_of","right_of","front_of","behind","nearer_than",
                        "farther_than","between_in_depth","nearer_than_both"])
    legacy.strata_figure("variant", "18_stratified_variant", "fig18_grounding_by_variant",
                        "Grounding by Test-IID variant", ["clean","semantic_counterfactual","relation_counterfactual",
                        "depth_corruption","occlusion_view_counterfactual"])
    legacy.fig19_answerability_confusion()

    history = [json.loads(line) for line in (RUN / "detail_cost/history.jsonl").read_text().splitlines() if line]
    selected = read(RUN / "detail_cost/checkpoints/best/metadata.json")
    epoch = selected["epoch"]
    epochs = np.array([r["epoch"] for r in history])
    assert epochs.tolist() == list(range(25)) and epoch == 13 and selected["global_step"] == 1120
    eligible = [r for r in history if r["eligible"]]
    best = max(eligible, key=lambda r:(r["dev"]["macro_f1"],r["dev"]["per_class"][0]["recall"],-r["dev"]["false_found"]))
    assert best["epoch"] == epoch
    loss = np.array([r["train_loss"] for r in history])
    f1 = np.array([r["dev"]["macro_f1"] for r in history]) * 100
    accuracy = np.array([r["dev"]["accuracy"] for r in history]) * 100
    assert np.isclose(f1[epoch],87.25120101083764)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), constrained_layout=True)
    fig.suptitle("P-CRA-U · Huấn luyện Adapter · chọn theo macro-F1 dev và điều kiện kiểm soát lỗi",
                 fontsize=12, fontweight="bold")
    axes[0].plot(epochs, loss, color=legacy.BLUE, marker="o", markersize=3, linewidth=1.7,
                 label="Loss huấn luyện Adapter")
    axes[0].set_title("(a) Loss khả năng trả lời: trọng số và phạt", loc="left", fontweight="bold")
    axes[0].set_ylabel("Loss huấn luyện")
    axes[0].set_ylim(0, loss.max()*1.08)
    axes[0].annotate(f"Loss tại epoch chọn = {loss[epoch]:.4f}", xy=(epoch,loss[epoch]),
                     xytext=(7.2,loss.max()*0.76), arrowprops={"arrowstyle":"-","color":legacy.ORANGE,"lw":1},
                     color=legacy.ORANGE, fontsize=9)
    axes[0].legend(frameon=False,loc="upper right")
    axes[1].plot(epochs, f1, color=legacy.GREEN, marker="o", markersize=3, linewidth=1.7,
                 label="Macro-F1 khả năng trả lời dev")
    axes[1].plot(epochs, accuracy, color=legacy.ORANGE, marker="s", markersize=3, linewidth=1.7,
                 label="Độ chính xác khả năng trả lời dev")
    axes[1].set_title("(b) Khả năng trả lời trên dev · n=400 mẫu", loc="left", fontweight="bold",fontsize=9.5)
    axes[1].set_ylabel("Chỉ số khả năng trả lời (%)")
    axes[1].set_ylim(75, 100)
    axes[1].annotate(f"Macro-F1 = {f1[epoch]:.2f}% tại epoch {epoch}", xy=(epoch,f1[epoch]),
                     xytext=(8.3,79), arrowprops={"arrowstyle":"-","color":legacy.GREEN,"lw":1},
                     color=legacy.GREEN, fontsize=9)
    axes[1].legend(frameon=False,loc="upper right")
    for ax in axes:
        ax.axvline(epoch,color=legacy.GRAY,linestyle=":",linewidth=1.2,zorder=0)
        ax.set_xlabel("Epoch (bắt đầu từ 0; epoch 13 là lượt thứ 14)")
        ax.set_xlim(-0.4,24.4); ax.set_xticks([0,4,8,13,16,20,24])
        ax.set_axisbelow(True); ax.grid(axis="y",color=legacy.LIGHT,linewidth=0.7)
    save(fig, "20_training_curves", "fig20_pcrau_training_curves")

    legacy.fig21_source_delta()
    legacy.fig22_uncertainty_localization()
    legacy.fig23_coverage_area()
    assert len(generated) == 24
    for path, expected in old_hashes.items():
        assert sha(path) == expected, f"Old figure changed: {path}"
    write_index(generated, active, result, old_hashes, args.refresh_existing)
    print("PASS: 12 hình mới (PNG + PDF); toàn bộ hình cũ và artifact đã khóa giữ nguyên.")


def write_index(paths, active, result, old_hashes, refresh):
    index = HERE / "ADAPTER_FIGURES.md"
    if index.exists() and not refresh:
        raise FileExistsError(index)
    lines = ["# Hình tiếng Việt cho P-CRA-U được chọn (V2 + Adapter)", "",
             "Dựng từ prediction Test-IID đã lưu, không huấn luyện, suy luận lại hoặc fit ngưỡng.",
             "Mỗi thư mục được thêm một PNG và một PDF với hậu tố `_adapter_vi`; hình cũ giữ nguyên.", "",
             "| Hình | PNG | PDF |", "|---|---|---|"]
    for pdf, png in zip(paths[0::2], paths[1::2]):
        number = png.name[3:5]
        lines.append(f"| {number} | [{png.name}]({png.relative_to(HERE)}) | [PDF]({pdf.relative_to(HERE)}) |")
    lines += ["", "## Nguồn và phạm vi số liệu", "",
              "- Checkpoint: `" + active["checkpoint"] + "`.",
              "- SHA-256 checkpoint: `" + active["checkpoint_sha256"] + "`.",
              "- Calibrator logistic33: `" + active["calibrator"] + "`.",
              "- SHA-256 calibrator: `" + active["calibrator_sha256"] + "`.",
              "- Prediction: `" + str(TEST / "inference/predictions.jsonl") + "`.",
              "- Quyết định: `" + str(TEST / "runtime_decisions.jsonl") + "`.",
              "- Báo cáo: `" + str(TEST / "full_metrics.json") + "`.",
              "- Hình 20: `" + str(RUN / "detail_cost/history.jsonl") + "`, metadata của checkpoint best và freeze lock.",
              "- RGB, mask hậu kiểm và điểm RoboRefer: manifest Test-IID cùng paired_samples.csv của phép so sánh gốc.", "",
              "## Đối chiếu nội dung", "",
              "- Hình 10: ECE risk = 0,03108; 10 khoảng độ rộng bằng nhau, kích cỡ điểm theo số mẫu.",
              "- Hình 11: AURC = 0,255748; policy nhận 361/1000 mẫu, có 34 lỗi (9,42%). Ngưỡng risk = 0,261193.",
              "  Đường cong xếp hạng risk trên toàn bộ mẫu; điểm cam là policy còn yêu cầu dự đoán FOUND.",
              "  Chỉ lọc risk cho 362 mẫu, khác 361 mẫu được policy nhận. Không ép điểm policy lên đường xếp hạng.",
              "- Hình 14: giữ đúng bốn sample cũ; cập nhật answerability, risk và quyết định từ output Adapter.",
              "- Hình 15–18: grounding giống hệt V2 lịch sử trên 1000 sample đã đối chiếu; giữ số và mẫu minh họa.",
              "- Hình 19: ma trận Adapter, accuracy 81,10%, macro-F1 0,809686; nhãn thật có thể trả lời 420 mẫu.",
              "- Hình 20: giữ bố cục hai panel, dùng loss train Adapter và accuracy/macro-F1 answerability dev.",
              "  Log không lưu loss dev theo epoch. Không dựng đường loss dev giả hoặc dùng lịch sử train V2 cho Adapter.",
              "  Epoch chọn 13, step 1120 theo macro-F1 dev dưới các điều kiện kiểm soát recall/false FOUND; không chọn bằng dev total loss.",
              "- Hình 21: source head đóng băng, paired delta không đổi; xác nhận từ prediction Adapter.",
              "- Hình 22: giữ 420 nhãn thật FOUND (405 MAP đúng, 15 sai); panel risk dùng logistic33 mới.",
              "- Hình 23: spatial distribution không đổi; giữ mass đã fit 0,6041832 của nhánh không gian V2.",
              "  Score coverage 86,83%, direct intersection 95,81%, diện tích trung bình 0,5194%.",
              "  Logistic33 không fit lại spatial mass; vùng ô lưới được kiểm tra với quyết định spatial gốc.", "",
              "Bộ IID này đã được phân tích trong lịch sử. Ngân sách chọn ngưỡng calibration 7,5% chưa đạt trên test;",
              "risk thực nghiệm 9,42% là mức đánh đổi người dùng chấp nhận. Quyết định trên ảnh chưa phải robot motion/task success.",
              "Chưa sửa LaTeX hoặc các hình nằm ngoài 12 thư mục được yêu cầu.", "",
              "## Tái tạo và kiểm tra", "",
              "```bash", ".conda-roborefer/bin/python3.10 new/hinhanh/build_adapter_figures.py", "```", "",
              "Script mặc định từ chối ghi đè output đã tồn tại; `--refresh-existing` chỉ dựng lại bản `_adapter_vi`.",
              "Đã kiểm tra hash của checkpoint/config/calibrator/profiles,",
              "hash prediction/decision, cùng sample ID và các đầu ra đóng băng; đối chiếu ma trận, ECE và AURC.",
              f"Đã kiểm tra {len(old_hashes)} PNG/PDF cũ trong 12 thư mục giữ nguyên SHA-256.", "",
              "| Output mới | SHA-256 |", "|---|---|"]
    lines += [f"| `{path.relative_to(HERE)}` | `{sha(path)}` |" for path in paths]
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
