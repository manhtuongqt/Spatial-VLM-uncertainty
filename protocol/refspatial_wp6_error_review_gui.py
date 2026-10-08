#!/usr/bin/env python3
"""Fast Tkinter review GUI for every error in the WP6 B0/B1 error union.

The reviewer sees RGB and instruction without overlays first.  Source, B0 and
B1 points are revealed only on request.  Clicking the image records a corrected
human target.  Decisions are saved atomically to CSV and a progress JSON.

Reviewed WP6 examples remain evaluation-only.  Their decisions may diagnose a
label/evaluator/model failure and guide fixes on disjoint training families,
but these examples must not be copied into training data.
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "results/spatial_vlm_refspatial_v1/wp6_unbiased_challenge/error_review"
DEFAULT_QUEUE = DEFAULT_DIR / "wp6_error_review_queue.jsonl"
DEFAULT_DECISIONS = DEFAULT_DIR / "wp6_error_review_decisions.csv"
DEFAULT_PROGRESS = DEFAULT_DIR / "wp6_error_review_progress.json"

FIELDNAMES = [
    "sample_id", "scene_id", "error_category", "relation", "challenge_stratum",
    "reviewer", "reviewed_at", "review_origin", "target_correct", "object_set_correct",
    "ordinal_direction_correct", "reference_frame", "label_ambiguity", "source_target_valid",
    "b0_error_confirmed", "b1_error_confirmed", "decision", "human_target_x",
    "human_target_y", "correction_action", "completed", "notes",
]

CHOICES = {
    "target_correct": ("", "yes", "no", "unsure"),
    "object_set_correct": ("", "yes", "no", "unsure"),
    "ordinal_direction_correct": ("", "yes", "no", "unsure", "na"),
    "reference_frame": ("", "image_viewer_left_to_right", "other", "ambiguous", "unsure"),
    "label_ambiguity": ("", "no", "tie", "multiple_valid", "instruction_unclear", "unsure"),
    "source_target_valid": ("", "yes", "no", "unsure"),
    "b0_error_confirmed": ("", "yes", "no", "unsure", "na"),
    "b1_error_confirmed": ("", "yes", "no", "unsure", "na"),
    "decision": ("", "model_error", "source_label_error", "ambiguous", "unsure"),
    "correction_action": ("", "keep_source_target", "replace_source_target", "exclude_from_eval", "needs_second_review"),
}


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_decisions(path, queue):
    base = {row["sample_id"]: {key: "" for key in FIELDNAMES} for row in queue}
    if path.exists():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("sample_id") in base:
                    base[row["sample_id"]].update({key: row.get(key, "") for key in FIELDNAMES})
    for row in queue:
        saved = base[row["sample_id"]]
        for key in ("sample_id", "scene_id", "error_category", "relation", "challenge_stratum"):
            saved[key] = row.get(key, "")
    return base


def atomic_csv(path, rows):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in FIELDNAMES} for row in rows)
    temp.replace(path)


class ReviewApp:
    def __init__(self, root, queue_path, decisions_path, progress_path):
        self.root = root
        self.queue_path = queue_path
        self.decisions_path = decisions_path
        self.progress_path = progress_path
        self.queue = load_jsonl(queue_path)
        if not self.queue:
            raise RuntimeError(f"Empty review queue: {queue_path}")
        self.decisions = load_decisions(decisions_path, self.queue)
        self.index = 0
        self.original_image = None
        self.tk_image = None
        self.image_box = (0, 0, 1, 1)
        self.loading = False

        root.title("WP6 B0/B1 — Human Error Review")
        root.geometry("1500x920")
        root.minsize(1100, 720)
        root.attributes("-topmost", True)
        root.after(1800, lambda: root.attributes("-topmost", False))
        root.after(100, root.lift)
        root.after(150, root.focus_force)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.reviewer = tk.StringVar()
        self.show_source = tk.BooleanVar(value=False)
        self.show_b0 = tk.BooleanVar(value=False)
        self.show_b1 = tk.BooleanVar(value=False)
        self.status = tk.StringVar()
        self.point_status = tk.StringVar(value="Human target: chưa chọn")
        self.vars = {key: tk.StringVar() for key in CHOICES}
        self.notes_var = None

        self.build_ui()
        self.load_case(0)
        root.bind("<Control-s>", lambda _event: self.save_current(False))
        root.bind("<Control-Right>", lambda _event: self.save_current(True))
        root.bind("<Control-Left>", lambda _event: self.previous())

    def build_ui(self):
        header = ttk.Frame(self.root, padding=(10, 8))
        header.pack(fill="x")
        ttk.Label(header, text="WP6 – review toàn bộ lỗi B0/B1", font=("TkDefaultFont", 16, "bold")).pack(side="left")
        ttk.Label(header, text="Reviewer:").pack(side="left", padx=(30, 5))
        ttk.Entry(header, textvariable=self.reviewer, width=24).pack(side="left")
        ttk.Label(header, textvariable=self.status, font=("TkDefaultFont", 11, "bold")).pack(side="right")

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        left = ttk.Frame(body)
        right_outer = ttk.Frame(body, width=430)
        body.add(left, weight=4)
        body.add(right_outer, weight=2)

        self.meta = ttk.Label(left, anchor="w", font=("TkDefaultFont", 10, "bold"))
        self.meta.pack(fill="x", pady=(0, 4))
        self.instruction = tk.Text(left, height=5, wrap="word", font=("TkDefaultFont", 11), background="#f3f5f7")
        self.instruction.pack(fill="x", pady=(0, 6))
        self.instruction.configure(state="disabled")
        self.canvas = tk.Canvas(left, background="#202124", cursor="crosshair", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.image_click)
        self.canvas.bind("<Configure>", lambda _event: self.redraw())

        overlay = ttk.Frame(left, padding=(0, 7))
        overlay.pack(fill="x")
        ttk.Checkbutton(overlay, text="Hiện SOURCE (xanh lá)", variable=self.show_source, command=self.redraw).pack(side="left")
        ttk.Checkbutton(overlay, text="Hiện B0 (đỏ)", variable=self.show_b0, command=self.redraw).pack(side="left", padx=12)
        ttk.Checkbutton(overlay, text="Hiện B1 (xanh dương)", variable=self.show_b1, command=self.redraw).pack(side="left")
        ttk.Label(overlay, textvariable=self.point_status, foreground="#8a5a00").pack(side="right")

        right_canvas = tk.Canvas(right_outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(right_outer, orient="vertical", command=right_canvas.yview)
        controls = ttk.Frame(right_canvas, padding=(10, 0, 12, 10))
        controls.bind("<Configure>", lambda e: right_canvas.configure(scrollregion=right_canvas.bbox("all")))
        right_canvas.create_window((0, 0), window=controls, anchor="nw")
        right_canvas.configure(yscrollcommand=scrollbar.set)
        right_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        ttk.Label(controls, text="1. Xem ảnh + câu lệnh trước", font=("TkDefaultFont", 11, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        ttk.Label(controls, text="2. Click vào tâm object đúng nếu cần sửa target.\n3. Bật overlay để so sánh.\n4. Chọn quyết định rồi Save & Next.", wraplength=390).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 10))

        row_num = 2
        labels = {
            "target_correct": "Target mô tả đúng?",
            "object_set_correct": "Tập object đúng?",
            "ordinal_direction_correct": "Hướng/thứ tự đúng?",
            "reference_frame": "Reference frame",
            "label_ambiguity": "Mức mơ hồ",
            "source_target_valid": "Điểm source hợp lệ?",
            "b0_error_confirmed": "B0 thực sự sai?",
            "b1_error_confirmed": "B1 thực sự sai?",
            "decision": "Kết luận chính",
            "correction_action": "Hành động",
        }
        for key, label in labels.items():
            ttk.Label(controls, text=label).grid(row=row_num, column=0, sticky="w", pady=3)
            combo = ttk.Combobox(controls, textvariable=self.vars[key], values=CHOICES[key], state="readonly", width=26)
            combo.grid(row=row_num, column=1, sticky="ew", pady=3)
            row_num += 1

        ttk.Label(controls, text="Ghi chú").grid(row=row_num, column=0, sticky="nw", pady=3)
        self.notes = tk.Text(controls, width=36, height=5, wrap="word")
        self.notes.grid(row=row_num, column=1, sticky="ew", pady=3)
        row_num += 1

        ttk.Separator(controls).grid(row=row_num, column=0, columnspan=2, sticky="ew", pady=9)
        row_num += 1
        ttk.Label(controls, text="Preset nhanh", font=("TkDefaultFont", 11, "bold")).grid(row=row_num, column=0, columnspan=2, sticky="w")
        row_num += 1
        ttk.Button(controls, text="Nhãn đúng → xác nhận lỗi model", command=self.preset_model_error).grid(row=row_num, column=0, columnspan=2, sticky="ew", pady=3)
        row_num += 1
        ttk.Button(controls, text="Nhãn sai → click điểm đúng", command=self.preset_label_error).grid(row=row_num, column=0, columnspan=2, sticky="ew", pady=3)
        row_num += 1
        ttk.Button(controls, text="Mơ hồ/tie → loại khỏi eval", command=self.preset_ambiguous).grid(row=row_num, column=0, columnspan=2, sticky="ew", pady=3)
        row_num += 1

        ttk.Separator(controls).grid(row=row_num, column=0, columnspan=2, sticky="ew", pady=9)
        row_num += 1
        nav = ttk.Frame(controls)
        nav.grid(row=row_num, column=0, columnspan=2, sticky="ew")
        ttk.Button(nav, text="← Trước", command=self.previous).pack(side="left")
        ttk.Button(nav, text="Lưu", command=lambda: self.save_current(False)).pack(side="left", padx=6)
        ttk.Button(nav, text="Lưu & tiếp →", command=lambda: self.save_current(True)).pack(side="right")
        controls.columnconfigure(1, weight=1)

        footer = ttk.Label(self.root, text="Evaluation-only: quyết định review dùng chẩn đoán/sửa pipeline trên dữ liệu train tách biệt; không copy các case này vào train.", foreground="#8b0000", padding=(10, 2))
        footer.pack(fill="x")

    def current(self):
        return self.queue[self.index]

    def load_case(self, index):
        self.loading = True
        self.index = max(0, min(index, len(self.queue) - 1))
        row = self.current()
        saved = self.decisions[row["sample_id"]]
        self.reviewer.set(saved.get("reviewer", "") or self.reviewer.get())
        for key, variable in self.vars.items():
            variable.set(saved.get(key, ""))
        self.notes.delete("1.0", "end")
        self.notes.insert("1.0", saved.get("notes", ""))
        self.show_source.set(False)
        self.show_b0.set(False)
        self.show_b1.set(False)

        image_path = ROOT / row["image"]
        self.original_image = Image.open(image_path).convert("RGB")
        self.meta.configure(text=f"Case {self.index + 1}/{len(self.queue)}  |  {row['error_category']}  |  {row['relation']}  |  {row['sample_id']}")
        self.instruction.configure(state="normal")
        self.instruction.delete("1.0", "end")
        self.instruction.insert("1.0", row["instruction"])
        self.instruction.configure(state="disabled")
        self.update_point_label(saved)
        self.loading = False
        self.update_status()
        self.redraw()

    def update_point_label(self, saved):
        x, y = saved.get("human_target_x", ""), saved.get("human_target_y", "")
        self.point_status.set(f"Human target: ({x}, {y})" if x and y else "Human target: chưa chọn")

    def redraw(self):
        if self.original_image is None or self.canvas.winfo_width() < 10:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.original_image.size
        scale = min(cw / iw, ch / ih)
        dw, dh = max(1, int(iw * scale)), max(1, int(ih * scale))
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        rendered = self.original_image.resize((dw, dh), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(rendered)
        row = self.current()
        saved = self.decisions[row["sample_id"]]

        def marker(point, color, label):
            if not point or len(point) != 2:
                return
            x, y = float(point[0]) * dw, float(point[1]) * dh
            radius = max(8, min(dw, dh) // 55)
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=4)
            draw.line((x-radius-5, y, x+radius+5, y), fill=color, width=3)
            draw.line((x, y-radius-5, x, y+radius+5), fill=color, width=3)
            draw.text((x+radius+5, y-radius-5), label, fill=color, stroke_width=2, stroke_fill="black")

        if self.show_source.get():
            marker(row["source_target_xy"], "#32e875", "SOURCE")
        if self.show_b0.get():
            marker(row["model_outputs"]["b0"]["prediction_xy"], "#ff4545", "B0")
        if self.show_b1.get():
            marker(row["model_outputs"]["b1"]["prediction_xy"], "#45a3ff", "B1")
        if saved.get("human_target_x") and saved.get("human_target_y"):
            marker([saved["human_target_x"], saved["human_target_y"]], "#ffe45c", "HUMAN")

        self.tk_image = ImageTk.PhotoImage(rendered)
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor="nw", image=self.tk_image)
        self.image_box = (ox, oy, dw, dh)

    def image_click(self, event):
        ox, oy, dw, dh = self.image_box
        if not (ox <= event.x <= ox + dw and oy <= event.y <= oy + dh):
            return
        x = min(1.0, max(0.0, (event.x - ox) / dw))
        y = min(1.0, max(0.0, (event.y - oy) / dh))
        saved = self.decisions[self.current()["sample_id"]]
        saved["human_target_x"] = f"{x:.4f}"
        saved["human_target_y"] = f"{y:.4f}"
        self.update_point_label(saved)
        self.redraw()

    def preset_model_error(self):
        row = self.current()
        values = {
            "target_correct": "yes", "object_set_correct": "yes",
            "ordinal_direction_correct": "yes", "reference_frame": "image_viewer_left_to_right",
            "label_ambiguity": "no", "source_target_valid": "yes",
            "b0_error_confirmed": "yes" if not row["model_outputs"]["b0"]["hit_at_008"] else "no",
            "b1_error_confirmed": "yes" if not row["model_outputs"]["b1"]["hit_at_008"] else "no",
            "decision": "model_error", "correction_action": "keep_source_target",
        }
        for key, value in values.items():
            self.vars[key].set(value)
        self.show_source.set(True)
        self.show_b0.set(True)
        self.show_b1.set(True)
        self.redraw()

    def preset_label_error(self):
        values = {
            "target_correct": "no", "source_target_valid": "no", "label_ambiguity": "no",
            "decision": "source_label_error", "correction_action": "replace_source_target",
            "b0_error_confirmed": "unsure", "b1_error_confirmed": "unsure",
        }
        for key, value in values.items():
            self.vars[key].set(value)
        self.show_source.set(True)
        self.redraw()
        messagebox.showinfo("Chọn target", "Hãy click vào tâm object đúng trên ảnh (marker HUMAN màu vàng).")

    def preset_ambiguous(self):
        values = {
            "target_correct": "unsure", "object_set_correct": "unsure",
            "ordinal_direction_correct": "unsure", "reference_frame": "ambiguous",
            "label_ambiguity": "multiple_valid", "source_target_valid": "unsure",
            "b0_error_confirmed": "unsure", "b1_error_confirmed": "unsure",
            "decision": "ambiguous", "correction_action": "exclude_from_eval",
        }
        for key, value in values.items():
            self.vars[key].set(value)

    def collect(self):
        row = self.current()
        saved = self.decisions[row["sample_id"]]
        saved["reviewer"] = self.reviewer.get().strip()
        saved["review_origin"] = "human" if saved["reviewer"] else ""
        for key, variable in self.vars.items():
            saved[key] = variable.get()
        saved["notes"] = self.notes.get("1.0", "end-1c").strip()
        return saved

    def validate(self, saved):
        required = ["reviewer", *CHOICES.keys()]
        missing = [key for key in required if not saved.get(key)]
        if saved.get("correction_action") == "replace_source_target":
            if not saved.get("human_target_x") or not saved.get("human_target_y"):
                missing.append("human_target_(click on image)")
        return missing

    def save_current(self, move_next):
        saved = self.collect()
        missing = self.validate(saved)
        if missing:
            saved["completed"] = "no"
            if not messagebox.askyesno("Thiếu thông tin", "Case chưa hoàn chỉnh:\n- " + "\n- ".join(missing) + "\n\nVẫn lưu bản nháp?"):
                return
        else:
            saved["completed"] = "yes"
            saved["reviewed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        self.write_all()
        self.update_status()
        if move_next:
            if self.index + 1 < len(self.queue):
                self.load_case(self.index + 1)
            else:
                messagebox.showinfo("Hoàn tất", "Đã đến case cuối. Tiến độ đã được lưu.")

    def write_all(self):
        rows = [self.decisions[row["sample_id"]] for row in self.queue]
        atomic_csv(self.decisions_path, rows)
        completed = [row for row in rows if row.get("completed") == "yes"]
        progress = {
            "status": "HUMAN_REVIEW_COMPLETE" if len(completed) == len(rows) else "HUMAN_REVIEW_IN_PROGRESS",
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "queue_count": len(rows), "completed_count": len(completed),
            "pending_count": len(rows) - len(completed),
            "decision_counts": dict(Counter(row.get("decision") for row in completed)),
            "correction_action_counts": dict(Counter(row.get("correction_action") for row in completed)),
            "review_origins": dict(Counter(row.get("review_origin") for row in completed)),
            "evaluation_only": True, "training_eligible": False, "sam2_used": False,
        }
        temp = self.progress_path.with_suffix(self.progress_path.suffix + ".tmp")
        temp.write_text(json.dumps(progress, indent=2, ensure_ascii=False) + "\n")
        temp.replace(self.progress_path)

    def update_status(self):
        complete = sum(self.decisions[row["sample_id"]].get("completed") == "yes" for row in self.queue)
        current_done = self.decisions[self.current()["sample_id"]].get("completed") == "yes"
        flag = " ✓" if current_done else ""
        self.status.set(f"Tiến độ: {complete}/{len(self.queue)} | case {self.index + 1}{flag}")

    def previous(self):
        self.collect()
        if self.index > 0:
            self.load_case(self.index - 1)

    def close(self):
        self.collect()
        self.write_all()
        self.root.destroy()


def check(queue_path, decisions_path):
    queue = load_jsonl(queue_path)
    required = {"sample_id", "scene_id", "instruction", "image", "source_target_xy", "model_outputs"}
    errors = []
    for row in queue:
        missing = sorted(required - row.keys())
        if missing:
            errors.append({"sample_id": row.get("sample_id"), "missing": missing})
        image = ROOT / row.get("image", "")
        if not image.is_file():
            errors.append({"sample_id": row.get("sample_id"), "missing_image": str(image)})
    saved = load_decisions(decisions_path, queue)
    result = {
        "status": "PASS" if not errors else "FAIL", "queue_count": len(queue),
        "b0_errors": sum(not row["model_outputs"]["b0"]["hit_at_008"] for row in queue),
        "b1_errors": sum(not row["model_outputs"]["b1"]["hit_at_008"] for row in queue),
        "completed": sum(row.get("completed") == "yes" for row in saved.values()),
        "errors": errors, "sam2_used": False,
    }
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--check", action="store_true", help="Validate inputs without opening the GUI.")
    args = parser.parse_args()
    if args.check:
        return check(args.queue, args.decisions)
    if not os.environ.get("DISPLAY") and sys.platform != "win32":
        parser.error("DISPLAY is not set; launch this script from the desktop session.")
    root = tk.Tk()
    ReviewApp(root, args.queue, args.decisions, args.progress)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
