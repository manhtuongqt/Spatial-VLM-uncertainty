#!/usr/bin/env python3
"""Independent human-review GUI for the 300-case WP1 RefSpatial audit."""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw, ImageTk


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "results/spatial_vlm_refspatial_v1/wp1_source_audit"
DEFAULT_QUEUE = DEFAULT_DIR / "manual_audit_queue.jsonl"
DEFAULT_DECISIONS = DEFAULT_DIR / "manual_audit_human_decisions.csv"
DEFAULT_PROGRESS = DEFAULT_DIR / "manual_audit_human_progress.json"
DEFAULT_AI_DECISIONS = DEFAULT_DIR / "manual_audit_decisions.csv"

FIELDS = [
    "sample_id", "scene_id", "split", "audit_strata", "reviewer", "reviewed_at",
    "target_correct", "relation_correct", "anchor_correct", "tabletop_appropriate",
    "reference_frame", "reasoning_depth", "decision", "notes", "review_origin",
    "human_target_x", "human_target_y", "correction_action", "completed",
]

CHOICES = {
    "target_correct": ("", "yes", "no", "unsure"),
    "relation_correct": ("", "yes", "no", "unsure", "na"),
    "anchor_correct": ("", "yes", "no", "unsure", "na"),
    "tabletop_appropriate": ("", "yes", "no", "unsure"),
    "reference_frame": (
        "", "image_viewer_left_to_right", "camera_depth_near_far",
        "object_anchor_relative", "not_applicable", "other", "unsure",
    ),
    "reasoning_depth": ("", "0", "1", "2", "3"),
    "decision": ("", "accept", "reject", "unsure"),
    "correction_action": (
        "", "keep_source_target", "replace_source_target", "exclude_from_release",
        "needs_second_review",
    ),
}


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def blank_rows(queue):
    rows = {item["sample_id"]: {key: "" for key in FIELDS} for item in queue}
    for item in queue:
        row = rows[item["sample_id"]]
        row.update({
            "sample_id": item["sample_id"],
            "scene_id": item["scene_id"],
            "split": item["split"],
            "audit_strata": "|".join(item.get("audit_strata", [])),
        })
    return rows


def load_saved(path, queue):
    rows = blank_rows(queue)
    if path.exists():
        with path.open(newline="") as handle:
            for saved in csv.DictReader(handle):
                sid = saved.get("sample_id")
                if sid in rows:
                    rows[sid].update({key: saved.get(key, "") for key in FIELDS})
    return rows


def load_ai_hints(path):
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def write_csv_atomic(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in FIELDS} for row in rows)
    temp.replace(path)


def suggested_frame(item):
    relations = set(item.get("relation_candidates", []))
    if item.get("anchor_candidates_offline"):
        return "object_anchor_relative"
    if relations & {"leftmost_ranking", "rightmost_ranking", "horizontal_ordinal_ranking"}:
        return "image_viewer_left_to_right"
    if relations & {"nearest_ranking", "farthest_ranking"}:
        return "camera_depth_near_far"
    return "not_applicable"


def suggested_depth(item):
    if item.get("anchor_candidates_offline"):
        return "2"
    if item.get("relation_candidates"):
        return "1"
    return "0"


class HumanReviewApp:
    def __init__(self, root, queue_path, decisions_path, progress_path, ai_path):
        self.root = root
        self.queue_path = queue_path
        self.decisions_path = decisions_path
        self.progress_path = progress_path
        self.queue = load_jsonl(queue_path)
        if not self.queue:
            raise RuntimeError(f"Empty queue: {queue_path}")
        self.saved = load_saved(decisions_path, self.queue)
        self.ai_hints = load_ai_hints(ai_path)
        self.index = 0
        self.original = self.tk_image = None
        self.image_box = (0, 0, 1, 1)

        self.reviewer = tk.StringVar()
        self.media_kind = tk.StringVar(value="rgb")
        self.show_source = tk.BooleanVar(value=False)
        self.show_anchor = tk.BooleanVar(value=False)
        self.show_ai = tk.BooleanVar(value=False)
        self.status = tk.StringVar()
        self.point_status = tk.StringVar()
        self.case_help = tk.StringVar()
        self.vars = {key: tk.StringVar() for key in CHOICES}

        root.title("WP1 — 300 Human Source Audit")
        root.geometry("1530x940")
        root.minsize(1150, 740)
        root.attributes("-topmost", True)
        root.after(1800, lambda: root.attributes("-topmost", False))
        root.after(100, root.lift)
        root.after(150, root.focus_force)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.build()
        self.load_case(self.first_pending())
        root.bind("<Control-s>", lambda _event: self.save(False))
        root.bind("<Control-Right>", lambda _event: self.save(True))
        root.bind("<Control-Left>", lambda _event: self.previous())

    def build(self):
        header = ttk.Frame(self.root, padding=(10, 8))
        header.pack(fill="x")
        ttk.Label(header, text="WP1 — human audit 300 case", font=("TkDefaultFont", 16, "bold")).pack(side="left")
        ttk.Label(header, text="Reviewer:").pack(side="left", padx=(25, 5))
        ttk.Entry(header, textvariable=self.reviewer, width=24).pack(side="left")
        ttk.Button(header, text="Tới case chưa làm", command=lambda: self.load_case(self.first_pending())).pack(side="left", padx=10)
        ttk.Label(header, textvariable=self.status, font=("TkDefaultFont", 11, "bold")).pack(side="right")

        pane = ttk.Panedwindow(self.root, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        left, right = ttk.Frame(pane), ttk.Frame(pane, width=490)
        pane.add(left, weight=4)
        pane.add(right, weight=2)

        self.meta = ttk.Label(left, font=("TkDefaultFont", 10, "bold"))
        self.meta.pack(fill="x", pady=(0, 4))
        self.instruction = tk.Text(left, height=5, wrap="word", background="#f3f5f7", font=("TkDefaultFont", 11))
        self.instruction.pack(fill="x", pady=(0, 6))
        self.instruction.configure(state="disabled")
        self.canvas = tk.Canvas(left, background="#202124", cursor="crosshair", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<Configure>", lambda _event: self.redraw())

        bottom = ttk.Frame(left, padding=(0, 7))
        bottom.pack(fill="x")
        ttk.Radiobutton(bottom, text="RGB", variable=self.media_kind, value="rgb", command=self.change_media).pack(side="left")
        ttk.Radiobutton(bottom, text="Depth", variable=self.media_kind, value="depth", command=self.change_media).pack(side="left", padx=(6, 12))
        ttk.Checkbutton(bottom, text="Hiện SOURCE (xanh lá)", variable=self.show_source, command=self.redraw).pack(side="left")
        ttk.Checkbutton(bottom, text="Hiện ANCHOR (tím)", variable=self.show_anchor, command=self.redraw).pack(side="left", padx=8)
        ttk.Label(bottom, textvariable=self.point_status, foreground="#8a5a00").pack(side="right")

        outer = tk.Canvas(right, highlightthickness=0)
        scroll = ttk.Scrollbar(right, orient="vertical", command=outer.yview)
        form = ttk.Frame(outer, padding=(10, 0, 12, 10))
        form.bind("<Configure>", lambda _event: outer.configure(scrollregion=outer.bbox("all")))
        outer.create_window((0, 0), window=form, anchor="nw")
        outer.configure(yscrollcommand=scroll.set)
        outer.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        ttk.Label(form, text="Đọc prompt + ảnh trước, sau đó mới bật SOURCE/ANCHOR.", wraplength=430, font=("TkDefaultFont", 11, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))
        ttk.Label(form, textvariable=self.case_help, wraplength=430, foreground="#5b3a00").grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))
        labels = {
            "target_correct": "SOURCE target đúng?",
            "relation_correct": "Quan hệ/thứ tự đúng?",
            "anchor_correct": "ANCHOR đúng?",
            "tabletop_appropriate": "Phù hợp tabletop?",
            "reference_frame": "Reference frame",
            "reasoning_depth": "Reasoning depth",
            "decision": "Quyết định",
            "correction_action": "Hành động",
        }
        n = 2
        for key, label in labels.items():
            ttk.Label(form, text=label).grid(row=n, column=0, sticky="w", pady=4)
            ttk.Combobox(form, textvariable=self.vars[key], values=CHOICES[key], state="readonly", width=29).grid(row=n, column=1, sticky="ew", pady=4)
            n += 1
        ttk.Label(form, text="Ghi chú").grid(row=n, column=0, sticky="nw", pady=4)
        self.notes = tk.Text(form, width=41, height=6, wrap="word")
        self.notes.grid(row=n, column=1, sticky="ew", pady=4)
        n += 1

        ttk.Checkbutton(form, text="Hiện gợi ý AI cũ (không sao chép vào phiếu người)", variable=self.show_ai, command=self.update_ai_hint).grid(row=n, column=0, columnspan=2, sticky="w", pady=(7, 2))
        n += 1
        self.ai_text = tk.Text(form, width=48, height=5, wrap="word", background="#eeeeee")
        self.ai_text.grid(row=n, column=0, columnspan=2, sticky="ew")
        self.ai_text.configure(state="disabled")
        n += 1

        ttk.Separator(form).grid(row=n, column=0, columnspan=2, sticky="ew", pady=10)
        n += 1
        ttk.Label(form, text="Preset nhanh", font=("TkDefaultFont", 11, "bold")).grid(row=n, column=0, columnspan=2, sticky="w")
        n += 1
        ttk.Button(form, text="Nhãn nguồn đúng theo ảnh/frame", command=self.preset_accept).grid(row=n, column=0, columnspan=2, sticky="ew", pady=3)
        n += 1
        ttk.Button(form, text="SOURCE sai → click target đúng", command=self.preset_wrong_source).grid(row=n, column=0, columnspan=2, sticky="ew", pady=3)
        n += 1
        ttk.Button(form, text="Không xác minh được / mơ hồ", command=self.preset_unsure).grid(row=n, column=0, columnspan=2, sticky="ew", pady=3)
        n += 1

        ttk.Separator(form).grid(row=n, column=0, columnspan=2, sticky="ew", pady=10)
        n += 1
        nav = ttk.Frame(form)
        nav.grid(row=n, column=0, columnspan=2, sticky="ew")
        ttk.Button(nav, text="← Trước", command=self.previous).pack(side="left")
        ttk.Button(nav, text="Lưu", command=lambda: self.save(False)).pack(side="left", padx=6)
        ttk.Button(nav, text="Lưu & tiếp →", command=lambda: self.save(True)).pack(side="right")
        form.columnconfigure(1, weight=1)

        ttk.Label(
            self.root,
            text="Human-only audit: không tự đổi dữ liệu train, không mở B2, không dùng SAM2. Depth PNG chỉ để đối chiếu tương đối, không phải metric GT.",
            foreground="#8b0000", padding=(10, 2),
        ).pack(fill="x")

    def current(self):
        return self.queue[self.index]

    def first_pending(self):
        for index, item in enumerate(self.queue):
            if self.saved[item["sample_id"]].get("completed") != "yes":
                return index
        return len(self.queue) - 1

    def load_case(self, index):
        self.index = max(0, min(index, len(self.queue) - 1))
        item = self.current()
        saved = self.saved[item["sample_id"]]
        self.reviewer.set(saved.get("reviewer") or self.reviewer.get())
        for key, var in self.vars.items():
            var.set(saved.get(key, ""))
        self.notes.delete("1.0", "end")
        self.notes.insert("1.0", saved.get("notes", ""))
        self.show_source.set(False)
        self.show_anchor.set(False)
        self.show_ai.set(False)
        self.media_kind.set("rgb")
        self.load_image()
        relations = ", ".join(item.get("relation_candidates", [])) or "không có relation candidate"
        anchor_count = len(item.get("anchor_candidates_offline", []))
        self.meta.configure(text=f"Case {self.index+1}/{len(self.queue)} | {' / '.join(item['audit_strata'])} | {item['split']} | {item['sample_id']}")
        self.case_help.set(f"Relation candidate: {relations} | Anchor candidate: {anchor_count}. Chỉ bật overlay sau khi tự xác định đáp án.")
        self.instruction.configure(state="normal")
        self.instruction.delete("1.0", "end")
        self.instruction.insert("1.0", item["instruction"])
        self.instruction.configure(state="disabled")
        self.update_point(saved)
        self.update_ai_hint()
        self.update_status()
        self.redraw()

    def load_image(self):
        key = "image" if self.media_kind.get() == "rgb" else "depth"
        self.original = Image.open(ROOT / self.current()[key]).convert("RGB")

    def change_media(self):
        self.load_image()
        self.redraw()

    def update_point(self, saved):
        x, y = saved.get("human_target_x", ""), saved.get("human_target_y", "")
        self.point_status.set(f"Human target: ({x}, {y})" if x and y else "Human target: chưa chọn")

    def redraw(self):
        if self.original is None or self.canvas.winfo_width() < 10:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.original.size
        scale = min(cw / iw, ch / ih)
        dw, dh = max(1, int(iw * scale)), max(1, int(ih * scale))
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        image = self.original.resize((dw, dh), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image)

        def marker(point, color, label):
            if not point or len(point) != 2:
                return
            x, y = float(point[0]) * dw, float(point[1]) * dh
            radius = max(8, min(dw, dh) // 55)
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=4)
            draw.line((x-radius-5, y, x+radius+5, y), fill=color, width=3)
            draw.line((x, y-radius-5, x, y+radius+5), fill=color, width=3)
            draw.text((x+radius+5, y-radius-5), label, fill=color, stroke_width=2, stroke_fill="black")

        item, saved = self.current(), self.saved[self.current()["sample_id"]]
        if self.show_source.get():
            marker(item.get("target_xy"), "#32e875", "SOURCE")
        if self.show_anchor.get():
            for number, anchor in enumerate(item.get("anchor_candidates_offline", []), 1):
                marker(anchor.get("xy"), "#da70d6", f"ANCHOR {number}")
        if saved.get("human_target_x") and saved.get("human_target_y"):
            marker([saved["human_target_x"], saved["human_target_y"]], "#ffe45c", "HUMAN")
        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor="nw", image=self.tk_image)
        self.image_box = (ox, oy, dw, dh)

    def click(self, event):
        ox, oy, dw, dh = self.image_box
        if not (ox <= event.x <= ox + dw and oy <= event.y <= oy + dh):
            return
        saved = self.saved[self.current()["sample_id"]]
        saved["human_target_x"] = f"{(event.x-ox)/dw:.4f}"
        saved["human_target_y"] = f"{(event.y-oy)/dh:.4f}"
        self.update_point(saved)
        self.redraw()

    def set_values(self, values):
        for key, value in values.items():
            self.vars[key].set(value)

    def preset_accept(self):
        item = self.current()
        self.set_values({
            "target_correct": "yes",
            "relation_correct": "yes" if item.get("relation_candidates") else "na",
            "anchor_correct": "yes" if item.get("anchor_candidates_offline") else "na",
            "tabletop_appropriate": "yes",
            "reference_frame": suggested_frame(item),
            "reasoning_depth": suggested_depth(item),
            "decision": "accept",
            "correction_action": "keep_source_target",
        })
        self.show_source.set(True)
        self.show_anchor.set(bool(item.get("anchor_candidates_offline")))
        self.redraw()

    def preset_wrong_source(self):
        item = self.current()
        self.set_values({
            "target_correct": "no",
            "relation_correct": "yes" if item.get("relation_candidates") else "na",
            "anchor_correct": "yes" if item.get("anchor_candidates_offline") else "na",
            "tabletop_appropriate": "yes",
            "reference_frame": suggested_frame(item),
            "reasoning_depth": suggested_depth(item),
            "decision": "reject",
            "correction_action": "replace_source_target",
        })
        self.show_source.set(True)
        self.show_anchor.set(bool(item.get("anchor_candidates_offline")))
        self.redraw()
        messagebox.showinfo("Chọn target", "Click vào tâm target đúng để tạo marker HUMAN màu vàng.")

    def preset_unsure(self):
        item = self.current()
        self.set_values({
            "target_correct": "unsure",
            "relation_correct": "unsure" if item.get("relation_candidates") else "na",
            "anchor_correct": "unsure" if item.get("anchor_candidates_offline") else "na",
            "tabletop_appropriate": "unsure",
            "reference_frame": "unsure",
            "reasoning_depth": suggested_depth(item),
            "decision": "unsure",
            "correction_action": "needs_second_review",
        })

    def update_ai_hint(self):
        self.ai_text.configure(state="normal")
        self.ai_text.delete("1.0", "end")
        if self.show_ai.get():
            hint = self.ai_hints.get(self.current()["sample_id"])
            if hint:
                fields = ("target_correct", "relation_correct", "anchor_correct", "tabletop_appropriate", "reference_frame", "reasoning_depth", "decision")
                summary = " | ".join(f"{key}={hint.get(key, '')}" for key in fields)
                self.ai_text.insert("1.0", summary + "\n" + hint.get("notes", ""))
            else:
                self.ai_text.insert("1.0", "Không có gợi ý AI cho case này.")
        else:
            self.ai_text.insert("1.0", "Gợi ý AI đang ẩn để tránh làm lệch đánh giá độc lập.")
        self.ai_text.configure(state="disabled")

    def collect(self):
        item, saved = self.current(), self.saved[self.current()["sample_id"]]
        saved["reviewer"] = self.reviewer.get().strip()
        saved["review_origin"] = "human" if saved["reviewer"] else ""
        for key, var in self.vars.items():
            saved[key] = var.get()
        saved["notes"] = self.notes.get("1.0", "end-1c").strip()
        saved.update({
            "sample_id": item["sample_id"], "scene_id": item["scene_id"],
            "split": item["split"], "audit_strata": "|".join(item.get("audit_strata", [])),
        })
        return saved

    def validate(self, saved):
        item = self.current()
        missing = [key for key in ("reviewer", *CHOICES.keys()) if not saved.get(key)]
        if item.get("relation_candidates") and saved.get("relation_correct") == "na":
            missing.append("relation_correct không được na")
        if item.get("anchor_candidates_offline") and saved.get("anchor_correct") == "na":
            missing.append("anchor_correct không được na")
        if saved.get("decision") == "accept":
            if saved.get("target_correct") != "yes" or saved.get("tabletop_appropriate") != "yes":
                missing.append("accept cần target/tabletop=yes")
            if saved.get("relation_correct") in {"no", "unsure"} or saved.get("anchor_correct") in {"no", "unsure"}:
                missing.append("accept cần relation/anchor áp dụng đã đúng")
            if saved.get("reference_frame") in {"other", "unsure"}:
                missing.append("accept cần reference frame đã xác minh")
        if saved.get("correction_action") == "replace_source_target" and not (saved.get("human_target_x") and saved.get("human_target_y")):
            missing.append("human target click")
        if saved.get("reference_frame") in {"other", "unsure"} and not saved.get("notes"):
            missing.append("ghi chú giải thích reference frame")
        return missing

    def save(self, move):
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
        if move:
            if self.index + 1 < len(self.queue):
                self.load_case(self.index + 1)
            else:
                messagebox.showinfo("Hoàn tất", "Đã tới case cuối và lưu tiến độ.")

    def write_all(self):
        rows = [self.saved[item["sample_id"]] for item in self.queue]
        write_csv_atomic(self.decisions_path, rows)
        complete = [row for row in rows if row.get("completed") == "yes"]
        progress = {
            "status": "HUMAN_REVIEW_COMPLETE" if len(complete) == len(rows) else "HUMAN_REVIEW_IN_PROGRESS",
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "queue_count": len(rows),
            "completed_count": len(complete),
            "pending_count": len(rows) - len(complete),
            "decision_counts": dict(Counter(row.get("decision") for row in complete)),
            "review_origins": dict(Counter(row.get("review_origin") for row in complete)),
            "independent_from_ai_decisions": True,
            "training_eligible": False,
            "b2_open": False,
            "sam2_used": False,
        }
        temp = self.progress_path.with_suffix(self.progress_path.suffix + ".tmp")
        temp.write_text(json.dumps(progress, indent=2, ensure_ascii=False) + "\n")
        temp.replace(self.progress_path)

    def update_status(self):
        count = sum(self.saved[item["sample_id"]].get("completed") == "yes" for item in self.queue)
        mark = " ✓" if self.saved[self.current()["sample_id"]].get("completed") == "yes" else ""
        self.status.set(f"Tiến độ: {count}/{len(self.queue)} | case {self.index+1}{mark}")

    def previous(self):
        self.collect()
        if self.index:
            self.load_case(self.index - 1)

    def close(self):
        self.collect()
        self.write_all()
        self.root.destroy()


def check(queue_path, decisions_path, ai_path):
    queue = load_jsonl(queue_path)
    ids = [item.get("sample_id") for item in queue]
    errors = []
    if len(queue) != 300:
        errors.append(f"expected 300 queue rows, found {len(queue)}")
    if len(ids) != len(set(ids)):
        errors.append("duplicate sample IDs")
    for item in queue:
        for key in ("image", "depth"):
            if not (ROOT / item.get(key, "")).is_file():
                errors.append(f"missing {key} for {item.get('sample_id')}")
        for anchor in item.get("anchor_candidates_offline", []):
            if not anchor.get("xy") or len(anchor["xy"]) != 2:
                errors.append(f"invalid anchor for {item.get('sample_id')}")
    saved = load_saved(decisions_path, queue)
    ai_hints = load_ai_hints(ai_path)
    completed = [row for row in saved.values() if row.get("completed") == "yes"]
    origins = Counter(row.get("review_origin") for row in completed)
    result = {
        "status": "PASS" if not errors else "FAIL",
        "queue_count": len(queue),
        "unique_sample_ids": len(set(ids)),
        "unique_families": len({item.get("family_id") for item in queue}),
        "completed_human_reviews": len(completed),
        "review_origins": dict(origins),
        "ai_hint_rows_loaded_separately": len(ai_hints),
        "errors": errors,
        "training_eligible": False,
        "b2_open": False,
        "sam2_used": False,
    }
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--ai-decisions", type=Path, default=DEFAULT_AI_DECISIONS)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        return check(args.queue, args.decisions, args.ai_decisions)
    if not os.environ.get("DISPLAY") and sys.platform != "win32":
        parser.error("DISPLAY is not set")
    root = tk.Tk()
    HumanReviewApp(root, args.queue, args.decisions, args.progress, args.ai_decisions)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
