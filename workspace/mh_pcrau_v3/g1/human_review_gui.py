"""Desktop Tkinter/Pillow reviewer for the locked G1 pilot audit queue.

Shows RGB/depth-view and prompt before optional candidate-point reveal. Saves
only human decisions; it never modifies dataset manifests or decides G1_PASS.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import tempfile
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

from .audit_development import ROOT, sha256
from .build_label_audit import BLIND_COLUMNS, DEFAULT_OUTPUT, SOURCE_CLASSES
from .validate_human_review import validate_rows


DEFAULT_QUEUE = DEFAULT_OUTPUT / "HUMAN_AUDIT_QUEUE.csv"
DEFAULT_DECISIONS = DEFAULT_OUTPUT / "HUMAN_AUDIT_FILLED.csv"
OUTPUT_COLUMNS = (*BLIND_COLUMNS, "human_target_u", "human_target_v", "reviewed_at")
CHOICES = {
    "review_relation": ("", "leftmost", "rightmost", "second_from_left", "second_from_right", "OUT_OF_SCOPE", "UNSURE"),
    "review_answerability": ("", "FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE", "UNSURE"),
    "review_target_correct": ("", "yes", "no", "unsure", "not_visible"),
    "review_reasoning_depth": ("", "0", "1", "2", "UNSURE"),
    "review_source": ("", *SOURCE_CLASSES, "NONE", "MULTIPLE", "UNSURE"),
    "review_single_source": ("", "yes", "no", "unsure"),
    "review_status": ("NOT_REVIEWED", "IN_PROGRESS", "DONE", "UNSURE"),
}


def load_queue(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({r["queue_id"] for r in rows}) != len(rows):
        raise ValueError("Queue empty or queue IDs duplicated")
    coverage_path = path.parent / "HEAD_COVERAGE.json"
    if coverage_path.is_file():
        expected = json.loads(coverage_path.read_text(encoding="utf-8"))["queue_file_sha256"]
        if sha256(path) != expected:
            raise ValueError("Queue SHA-256 differs from locked HEAD_COVERAGE")
    return rows


def load_progress(path: Path, queue):
    by_id = {r["queue_id"]: {key: r.get(key, "") for key in OUTPUT_COLUMNS} for r in queue}
    if not path.is_file():
        return by_id
    with path.open(newline="", encoding="utf-8-sig") as handle:
        saved = list(csv.DictReader(handle))
    if len(saved) != len(queue):
        raise ValueError("Saved review does not have the same number of queue rows")
    seen = set()
    for row in saved:
        qid = row.get("queue_id", "")
        if qid not in by_id or qid in seen:
            raise ValueError(f"Unknown/duplicate saved queue ID: {qid}")
        seen.add(qid)
        for key in ("dataset", "sample_id", "family_id"):
            if row.get(key) != by_id[qid][key]:
                raise ValueError(f"Saved review identity mismatch: {qid}/{key}")
        for key in OUTPUT_COLUMNS:
            if key in row and key not in {"queue_id", "dataset", "sample_id", "family_id"}:
                by_id[qid][key] = row[key]
    if len(seen) != len(queue):
        raise ValueError("Saved review is missing queue IDs")
    return by_id


def write_progress_atomic(path: Path, queue, progress):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".human_review_", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for item in queue:
                writer.writerow({key: progress[item["queue_id"]].get(key, "") for key in OUTPUT_COLUMNS})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def current_issues(saved, source_item):
    if saved.get("review_status") != "DONE":
        return []
    missing = []
    for key in ("reviewer_id", "review_relation", "review_answerability", "review_source", "review_single_source"):
        if not saved.get(key):
            missing.append(key)
    if source_item.get("candidate_target_uv") and not saved.get("review_target_correct"):
        missing.append("review_target_correct")
    source, single = saved.get("review_source"), saved.get("review_single_source")
    if source in SOURCE_CLASSES and single != "yes":
        missing.append("single_source must be yes for one named source")
    if source in SOURCE_CLASSES and not saved.get("review_notes", "").strip():
        missing.append("source needs evidence in notes")
    if source == "depth_invalid_noisy" and source_item.get("dataset") == "D_tabletop_clean_v1":
        missing.append("Tabletop relative depth cannot certify DEPTH source")
    if saved.get("review_answerability") != "FOUND" and saved.get("review_target_correct") == "yes":
        missing.append("non-FOUND cannot confirm one target point as correct")
    if source in {"NONE", "MULTIPLE"} and single == "yes":
        missing.append("NONE/MULTIPLE cannot have single_source=yes")
    return missing


def candidate_reasoning_depth(instruction: str) -> str:
    """Apply the locked prompt-complexity rubric; never infer hidden CoT."""
    text = " ".join(instruction.lower().split())
    text = re.split(r"your answer should be formatted|your answer should be|the coordinates should", text)[0]
    if not text:
        return "UNSURE"
    has_rank = bool(re.search(r"\b(second|third|fourth|fifth|sixth|seventh|eighth)\b", text))
    has_filter = bool(re.search(r"\b(among|red|blue|green|yellow|orange|purple|black|white|brown|pink|gray|grey|striped|wooden|metal)\b", text))
    relation_ops = len(re.findall(r"\b(leftmost|rightmost|closest|farthest|nearest|furthest|between|behind|front|left of|right of|above|below)\b", text))
    if has_filter and (has_rank or relation_ops >= 1):
        return "2"
    if has_rank or relation_ops >= 2:
        return "1"
    if relation_ops == 1:
        return "0"
    return "UNSURE"


def quick_confirmation_fields(item: dict, reviewer_id: str, answerability_override: str | None = None) -> dict:
    relation = (item.get("candidate_relation") or "").strip()
    state = answerability_override or (item.get("candidate_answerability") or "").strip()
    allowed_states = {"FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"}
    if state not in allowed_states:
        state = "FOUND"  # Tabletop point; accepted only on explicit human confirmation.
    relation = relation if relation in {"leftmost", "rightmost", "second_from_left", "second_from_right"} else "OUT_OF_SCOPE"
    target_correct = ("yes" if state == "FOUND" else "unsure") if item.get("candidate_target_uv") else ""
    depth = candidate_reasoning_depth(item.get("instruction", ""))
    if state == "FOUND":
        source, single = "NONE", "no"
        source_note = "source=NONE: reviewer explicitly confirmed a clear FOUND case"
    else:
        source, single = "UNSURE", "unsure"
        source_note = "source=UNSURE: not inferred from answerability"
    return {
        "reviewer_id": reviewer_id.strip(), "review_relation": relation,
        "review_answerability": state, "review_target_correct": target_correct,
        "review_reasoning_depth": depth, "review_source": source,
        "review_single_source": single,
        "review_notes": ("QUICK_CONFIRM_V2; reviewer confirmed candidate relation/answerability/target after image review; "
                         f"reasoning_depth={depth} from prompt rubric; {source_note}"),
        "review_status": "DONE",
    }


class G1HumanReviewApp:
    def __init__(self, root: tk.Tk, queue_path: Path, decisions_path: Path):
        self.root = root
        self.queue_path = queue_path
        self.decisions_path = decisions_path
        self.queue = load_queue(queue_path)
        self.progress = load_progress(decisions_path, self.queue)
        self.index = 0
        self.original = None
        self.tk_image = None
        self.image_box = (0, 0, 1, 1)
        self.reviewer = tk.StringVar()
        self.media_kind = tk.StringVar(value="RGB")
        self.show_candidate = tk.BooleanVar(value=False)
        self.status_text = tk.StringVar()
        self.point_text = tk.StringVar()
        self.fields = {key: tk.StringVar() for key in CHOICES}

        root.title("G1 — audit nhãn người MH-PCRA-U-v3")
        root.geometry("1500x920")
        root.minsize(1100, 700)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.build()
        self.load_case(self.first_pending())
        root.bind("<Control-s>", lambda _event: self.save(False))
        root.bind("<Control-Right>", lambda _event: self.save(True))
        root.bind("<Control-Left>", lambda _event: self.previous())
        root.bind("<Control-Return>", lambda _event: self.save(True))

    def build(self):
        header = ttk.Frame(self.root, padding=(10, 8))
        header.pack(fill="x")
        ttk.Label(header, text="G1 — 128 mẫu audit nhãn", font=("TkDefaultFont", 16, "bold")).pack(side="left")
        ttk.Label(header, text="Reviewer:").pack(side="left", padx=(22, 5))
        ttk.Entry(header, textvariable=self.reviewer, width=23).pack(side="left")
        ttk.Button(header, text="Tới case chưa làm", command=lambda: self.navigate(self.first_pending())).pack(side="left", padx=10)
        ttk.Label(header, textvariable=self.status_text, font=("TkDefaultFont", 11, "bold")).pack(side="right")

        pane = ttk.Panedwindow(self.root, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=10, pady=(0, 5))
        left, right = ttk.Frame(pane), ttk.Frame(pane, width=480)
        pane.add(left, weight=4)
        pane.add(right, weight=2)

        self.case_meta = ttk.Label(left, font=("TkDefaultFont", 10, "bold"))
        self.case_meta.pack(fill="x", pady=(0, 4))
        self.instruction = tk.Text(left, height=5, wrap="word", background="#f2f4f6", font=("TkDefaultFont", 11))
        self.instruction.pack(fill="x", pady=(0, 6))
        self.instruction.configure(state="disabled")
        self.canvas = tk.Canvas(left, background="#202124", cursor="crosshair", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.click_image)
        self.canvas.bind("<Configure>", lambda _event: self.redraw())
        media = ttk.Frame(left, padding=(0, 6))
        media.pack(fill="x")
        ttk.Radiobutton(media, text="RGB", variable=self.media_kind, value="RGB", command=self.change_media).pack(side="left")
        ttk.Radiobutton(media, text="Depth-view", variable=self.media_kind, value="Depth", command=self.change_media).pack(side="left", padx=8)
        ttk.Checkbutton(media, text="Hiện điểm nguồn (xanh lá) sau khi tự chấm", variable=self.show_candidate,
                        command=self.redraw).pack(side="left", padx=8)
        ttk.Label(media, textvariable=self.point_text, foreground="#8a5a00").pack(side="right")

        outer = tk.Canvas(right, highlightthickness=0)
        scroll = ttk.Scrollbar(right, orient="vertical", command=outer.yview)
        form = ttk.Frame(outer, padding=(10, 0, 10, 10))
        form.bind("<Configure>", lambda _event: outer.configure(scrollregion=outer.bbox("all")))
        outer.create_window((0, 0), window=form, anchor="nw")
        outer.configure(yscrollcommand=scroll.set)
        outer.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        ttk.Label(form, text="Xem ảnh + prompt, tự chấm rồi mới bật điểm nguồn.",
                  font=("TkDefaultFont", 11, "bold"), wraplength=430).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 9))
        ttk.Label(form, text=("Quy tắc v3: xem ONTOLOGY_AND_SOURCE_RULES.md trong thư mục ngay_03. "
                              "Bấm FOUND sẽ tự điền relation, depth, target và source; sau đó bấm Lưu."),
                  wraplength=430, foreground="#6b3e00").grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 9))
        labels = {
            "review_relation": "Quan hệ theo prompt",
            "review_answerability": "Answerability",
            "review_target_correct": "Điểm nguồn đúng?",
            "review_reasoning_depth": "Reasoning depth (pilot)",
            "review_source": "Nguồn bất định chính",
            "review_single_source": "Chỉ một nguồn?",
            "review_status": "Trạng thái",
        }
        row_num = 2
        for key, label in labels.items():
            ttk.Label(form, text=label).grid(row=row_num, column=0, sticky="w", pady=5)
            ttk.Combobox(form, textvariable=self.fields[key], values=CHOICES[key], state="readonly", width=39).grid(
                row=row_num, column=1, sticky="ew", pady=5)
            row_num += 1
        ttk.Label(form, text="Ghi chú").grid(row=row_num, column=0, sticky="nw", pady=5)
        self.notes = tk.Text(form, width=38, height=6, wrap="word")
        self.notes.grid(row=row_num, column=1, sticky="ew", pady=5)
        row_num += 1
        ttk.Separator(form).grid(row=row_num, column=0, columnspan=2, sticky="ew", pady=9)
        row_num += 1
        ttk.Label(form, text="Preset nhanh — FOUND tự điền toàn bộ trường audit", font=("TkDefaultFont", 11, "bold")).grid(
            row=row_num, column=0, columnspan=2, sticky="w")
        row_num += 1
        for label, state in (("FOUND — tự điền nhãn, rồi bấm Lưu", "FOUND"), ("AMBIGUOUS", "AMBIGUOUS"),
                             ("ABSENT", "ABSENT"), ("INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE")):
            ttk.Button(form, text=label, command=lambda value=state: self.preset(value)).grid(
                row=row_num, column=0, columnspan=2, sticky="ew", pady=2)
            row_num += 1
        ttk.Separator(form).grid(row=row_num, column=0, columnspan=2, sticky="ew", pady=9)
        row_num += 1
        nav = ttk.Frame(form)
        nav.grid(row=row_num, column=0, columnspan=2, sticky="ew")
        ttk.Button(nav, text="← Trước", command=self.previous).pack(side="left")
        ttk.Button(nav, text="Lưu", command=lambda: self.save(False)).pack(side="left", padx=5)
        ttk.Button(nav, text="Lưu & tiếp →", command=lambda: self.save(True)).pack(side="right")
        form.columnconfigure(1, weight=1)
        ttk.Label(self.root, text="Pilot development: nhãn người không tự cấp G1_PASS hoặc quyền train. Ctrl+S để lưu.",
                  foreground="#8b0000", padding=(10, 3)).pack(fill="x")

    def current(self):
        return self.queue[self.index]

    def first_pending(self):
        for idx, item in enumerate(self.queue):
            if self.progress[item["queue_id"]].get("review_status") != "DONE":
                return idx
        return len(self.queue) - 1

    def load_case(self, index):
        self.index = max(0, min(index, len(self.queue) - 1))
        item = self.current()
        saved = self.progress[item["queue_id"]]
        self.reviewer.set(saved.get("reviewer_id") or self.reviewer.get())
        for key, var in self.fields.items():
            var.set(saved.get(key) or ("NOT_REVIEWED" if key == "review_status" else ""))
        self.notes.delete("1.0", "end")
        self.notes.insert("1.0", saved.get("review_notes", ""))
        self.show_candidate.set(False)
        self.media_kind.set("RGB")
        self.load_image()
        self.case_meta.configure(text=f"Case {self.index+1}/{len(self.queue)} | {item['dataset']} | {item['sample_id']}")
        self.instruction.configure(state="normal")
        self.instruction.delete("1.0", "end")
        self.instruction.insert("1.0", item["instruction"])
        self.instruction.configure(state="disabled")
        self.update_status()
        self.update_point(saved)
        self.redraw()

    def load_image(self):
        key = "rgb_path" if self.media_kind.get() == "RGB" else "depth_view_path"
        path = (ROOT / self.current()[key]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            self.original = image.convert("RGB")

    def change_media(self):
        self.load_image()
        self.redraw()

    def redraw(self):
        if self.original is None or self.canvas.winfo_width() < 10:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.original.size
        scale = min(cw / iw, ch / ih)
        dw, dh = max(1, int(iw * scale)), max(1, int(ih * scale))
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        display = self.original.resize((dw, dh), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(display)

        def marker(point, color, label):
            if not point or len(point) != 2:
                return
            x, y = float(point[0]) * dw, float(point[1]) * dh
            radius = max(7, min(dw, dh) // 55)
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=4)
            draw.line((x-radius-4, y, x+radius+4, y), fill=color, width=3)
            draw.line((x, y-radius-4, x, y+radius+4), fill=color, width=3)
            draw.text((x+radius+5, y-radius), label, fill=color, stroke_width=2, stroke_fill="black")

        item = self.current()
        saved = self.progress[item["queue_id"]]
        if self.show_candidate.get() and item.get("candidate_target_uv"):
            marker(json.loads(item["candidate_target_uv"]), "#32e875", "SOURCE")
        if saved.get("human_target_u") and saved.get("human_target_v"):
            marker((saved["human_target_u"], saved["human_target_v"]), "#ffe45c", "HUMAN")
        self.tk_image = ImageTk.PhotoImage(display)
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor="nw", image=self.tk_image)
        self.image_box = (ox, oy, dw, dh)

    def click_image(self, event):
        ox, oy, dw, dh = self.image_box
        if not (ox <= event.x <= ox+dw and oy <= event.y <= oy+dh):
            return
        saved = self.progress[self.current()["queue_id"]]
        saved["human_target_u"] = f"{(event.x-ox)/dw:.4f}"
        saved["human_target_v"] = f"{(event.y-oy)/dh:.4f}"
        self.update_point(saved)
        self.redraw()

    def update_point(self, saved):
        u, v = saved.get("human_target_u", ""), saved.get("human_target_v", "")
        self.point_text.set(f"Human point: ({u}, {v})" if u and v else "Human point: chưa chọn")

    def preset(self, state):
        if state == "FOUND":
            item = self.current()
            saved = self.progress[item["queue_id"]]
            fields = quick_confirmation_fields(item, self.reviewer.get(), answerability_override="FOUND")
            saved.update(fields)
            for key, variable in self.fields.items():
                variable.set(saved.get(key, ""))
            self.notes.delete("1.0", "end")
            self.notes.insert("1.0", saved.get("review_notes", ""))
            return
        self.fields["review_answerability"].set(state)
        self.fields["review_status"].set("IN_PROGRESS")

    def collect(self):
        item = self.current()
        saved = self.progress[item["queue_id"]]
        saved["reviewer_id"] = self.reviewer.get().strip()
        for key, var in self.fields.items():
            saved[key] = var.get().strip()
        saved["review_notes"] = self.notes.get("1.0", "end-1c").strip()
        if saved["review_status"] == "DONE":
            saved["reviewed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        return saved

    def save(self, move):
        saved = self.collect()
        issues = current_issues(saved, self.current())
        if issues:
            messagebox.showwarning("Chưa thể đánh dấu DONE", "Thiếu hoặc mâu thuẫn:\n- " + "\n- ".join(issues))
            saved["review_status"] = "IN_PROGRESS"
            self.fields["review_status"].set("IN_PROGRESS")
        write_progress_atomic(self.decisions_path, self.queue, self.progress)
        self.update_status()
        if move:
            if self.index + 1 < len(self.queue):
                self.load_case(self.index + 1)
            else:
                messagebox.showinfo("Cuối hàng đợi", "Đã lưu case cuối.")

    def navigate(self, index):
        self.save(False)
        self.load_case(index)

    def previous(self):
        self.navigate(max(0, self.index-1))

    def update_status(self):
        complete = sum(self.progress[item["queue_id"]].get("review_status") == "DONE" for item in self.queue)
        self.status_text.set(f"Tiến độ: {complete}/{len(self.queue)} | case {self.index+1}")

    def close(self):
        saved = self.collect()
        if current_issues(saved, self.current()):
            saved["review_status"] = "IN_PROGRESS"
        write_progress_atomic(self.decisions_path, self.queue, self.progress)
        self.root.destroy()


def check(queue_path: Path, decisions_path: Path):
    queue = load_queue(queue_path)
    progress = load_progress(decisions_path, queue)
    missing = []
    for item in queue:
        for key in ("rgb_path", "depth_view_path"):
            if not (ROOT / item[key]).is_file():
                missing.append(f"{item['queue_id']}:{key}")
    rows = [progress[item["queue_id"]] for item in queue]
    audit = validate_rows(queue, rows)
    print(json.dumps({"status": "PASS" if not missing and not audit["issues"] else "FAIL",
                      "queue_rows": len(queue), "completed": audit["done_count"],
                      "missing_media": missing, "review_issues": audit["issues"]}, ensure_ascii=False, indent=2))
    return 0 if not missing and not audit["issues"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--check", action="store_true", help="Validate packet and saved review without opening a window")
    args = parser.parse_args()
    if args.check:
        return check(args.queue, args.decisions)
    root = tk.Tk()
    try:
        G1HumanReviewApp(root, args.queue, args.decisions)
    except Exception:
        root.destroy()
        raise
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
