import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import customtkinter as ctk
from tkinter import filedialog, messagebox

import photo_tool_v2 as pt


ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

OUTPUT_MODE_OPTIONS = [
    ("仅输出分类图", pt.OUTPUT_MODE_CLASSIFIED),
    ("仅输出识别效果图", pt.OUTPUT_MODE_EFFECTS),
    ("分类图 + 识别效果图", pt.OUTPUT_MODE_BOTH),
]
OUTPUT_MODE_LABEL_TO_VALUE = {label: value for label, value in OUTPUT_MODE_OPTIONS}
OUTPUT_MODE_VALUE_TO_LABEL = {value: label for label, value in OUTPUT_MODE_OPTIONS}


class StatCard(ctk.CTkFrame):
    def __init__(self, master, title: str, value_var: ctk.StringVar) -> None:
        super().__init__(
            master,
            corner_radius=20,
            fg_color=("#F7F3ED", "#1B2432"),
            border_width=1,
            border_color=("#E5DED3", "#243147"),
        )
        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self,
            text=title,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#765A4A", "#93A4BA"),
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(14, 2))

        ctk.CTkLabel(
            self,
            textvariable=value_var,
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=("#221A14", "#F6F0E8"),
        ).grid(row=1, column=0, sticky="w", padx=18, pady=(0, 14))


class PhotoToolGUI(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Photo Review Studio")
        self.geometry("1040x760")
        self.minsize(980, 720)
        self.configure(fg_color=("#EFE7DD", "#111827"))

        self.input_dir = ctk.StringVar(value="")
        self.output_dir = ctk.StringVar(value="")
        self.ear_value = ctk.StringVar(value="0.18")
        self.output_mode_label = ctk.StringVar(value=OUTPUT_MODE_OPTIONS[0][0])
        self.status_text = ctk.StringVar(value="准备就绪")
        self.current_photo_text = ctk.StringVar(value="当前照片：尚未开始")
        self.summary_text = ctk.StringVar(value="处理完成后会在这里显示结果统计。")
        self.runtime_text = ctk.StringVar(value="00:00")
        self.progress_text = ctk.StringVar(value="0 / 0")
        self.progress_percent_text = ctk.StringVar(value="0%")
        self.output_hint_text = ctk.StringVar(value="完成后可一键打开输出文件夹。")

        self._is_processing = False
        self._runtime_job = None
        self._run_started_at: float | None = None
        self._last_output_dir: Path | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        header = ctk.CTkFrame(
            self,
            corner_radius=26,
            fg_color=("#D8C4B2", "#162033"),
            border_width=1,
            border_color=("#B58F72", "#23314A"),
        )
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(24, 14))
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)

        ctk.CTkLabel(
            header,
            text="Photo Review Studio",
            font=ctk.CTkFont(family="Georgia", size=32, weight="bold"),
            text_color=("#251A14", "#F9F3EA"),
        ).grid(row=0, column=0, sticky="w", padx=24, pady=(18, 4))

        ctk.CTkLabel(
            header,
            text="闭眼、曝光和质量筛选的桌面工作台",
            font=ctk.CTkFont(size=15),
            text_color=("#4B372A", "#B9C6D7"),
        ).grid(row=1, column=0, sticky="w", padx=24, pady=(0, 18))

        self.status_badge = ctk.CTkLabel(
            header,
            textvariable=self.status_text,
            corner_radius=999,
            padx=18,
            pady=8,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=("#FFF4D6", "#243B53"),
            text_color=("#7B4B00", "#E6F0FF"),
        )
        self.status_badge.grid(row=0, column=1, rowspan=2, padx=24, pady=18, sticky="e")

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 24))
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(1, weight=1)

        left_column = ctk.CTkFrame(body, fg_color="transparent")
        left_column.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        left_column.grid_columnconfigure(0, weight=1)

        right_column = ctk.CTkFrame(body, fg_color="transparent")
        right_column.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(10, 0))
        right_column.grid_columnconfigure(0, weight=1)

        self._build_path_card(left_column)
        self._build_progress_card(left_column)
        self._build_control_card(right_column)
        self._build_summary_card(right_column)

    def _build_path_card(self, parent) -> None:
        card = ctk.CTkFrame(
            parent,
            corner_radius=22,
            fg_color=("#FAF7F2", "#161F2E"),
            border_width=1,
            border_color=("#E5DED3", "#243147"),
        )
        card.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            card,
            text="输入与输出",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#241A14", "#F6F0E8"),
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(18, 6))

        ctk.CTkLabel(
            card,
            text="选择照片目录与结果目录。默认建议把输出放在输入目录外侧。",
            font=ctk.CTkFont(size=13),
            text_color=("#6A5646", "#95A4B8"),
        ).grid(row=1, column=0, sticky="w", padx=20, pady=(0, 16))

        self.input_button = ctk.CTkButton(
            card,
            text="选择输入照片目录",
            height=40,
            corner_radius=14,
            command=self.select_input_dir,
        )
        self.input_button.grid(row=2, column=0, sticky="w", padx=20, pady=(0, 8))

        self.input_path_label = ctk.CTkLabel(
            card,
            textvariable=self.input_dir,
            wraplength=430,
            justify="left",
            anchor="w",
            font=ctk.CTkFont(size=13),
            text_color=("#5C4B3D", "#B7C4D4"),
        )
        self.input_path_label.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 14))

        self.output_button = ctk.CTkButton(
            card,
            text="选择输出目录",
            height=40,
            corner_radius=14,
            fg_color=("#C47B45", "#34567A"),
            hover_color=("#B16935", "#41698F"),
            command=self.select_output_dir,
        )
        self.output_button.grid(row=4, column=0, sticky="w", padx=20, pady=(0, 8))

        self.output_path_label = ctk.CTkLabel(
            card,
            textvariable=self.output_dir,
            wraplength=430,
            justify="left",
            anchor="w",
            font=ctk.CTkFont(size=13),
            text_color=("#5C4B3D", "#B7C4D4"),
        )
        self.output_path_label.grid(row=5, column=0, sticky="ew", padx=20, pady=(0, 18))

    def _build_progress_card(self, parent) -> None:
        card = ctk.CTkFrame(
            parent,
            corner_radius=22,
            fg_color=("#20150F", "#0F1726"),
            border_width=1,
            border_color=("#5B4335", "#22304A"),
        )
        card.grid(row=1, column=0, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)

        top_row = ctk.CTkFrame(card, fg_color="transparent")
        top_row.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 10))
        top_row.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            top_row,
            text="实时进度",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#FFF7EE", "#F5F7FB"),
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            top_row,
            textvariable=self.progress_percent_text,
            corner_radius=999,
            padx=14,
            pady=6,
            fg_color=("#3A261B", "#21324D"),
            text_color=("#F8CBA7", "#D8E7FF"),
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=0, column=1, sticky="e")

        self.progress_bar = ctk.CTkProgressBar(
            card,
            height=18,
            corner_radius=999,
            progress_color=("#D4864F", "#6AA9FF"),
            fg_color=("#4D3528", "#1D2A40"),
        )
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=20)
        self.progress_bar.set(0)

        meta_row = ctk.CTkFrame(card, fg_color="transparent")
        meta_row.grid(row=2, column=0, sticky="ew", padx=20, pady=(10, 16))
        meta_row.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            meta_row,
            textvariable=self.progress_text,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#D7C1B0", "#9FB3CB"),
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            meta_row,
            textvariable=self.runtime_text,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#D7C1B0", "#9FB3CB"),
        ).grid(row=0, column=1, sticky="e")

        ctk.CTkLabel(
            card,
            text="当前处理照片",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#A88672", "#90A5C2"),
        ).grid(row=3, column=0, sticky="w", padx=20)

        self.current_photo_label = ctk.CTkLabel(
            card,
            textvariable=self.current_photo_text,
            justify="left",
            anchor="w",
            wraplength=500,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#FFF1E5", "#EAF2FF"),
        )
        self.current_photo_label.grid(row=4, column=0, sticky="ew", padx=20, pady=(4, 18))

        stat_row = ctk.CTkFrame(card, fg_color="transparent")
        stat_row.grid(row=5, column=0, sticky="ew", padx=20, pady=(0, 20))
        stat_row.grid_columnconfigure((0, 1, 2), weight=1)

        self.processed_value = ctk.StringVar(value="0")
        self.total_value = ctk.StringVar(value="0")

        StatCard(stat_row, "已处理", self.processed_value).grid(
            row=0, column=0, sticky="ew", padx=(0, 8)
        )
        StatCard(stat_row, "总照片", self.total_value).grid(
            row=0, column=1, sticky="ew", padx=8
        )
        StatCard(stat_row, "运行时间", self.runtime_text).grid(
            row=0, column=2, sticky="ew", padx=(8, 0)
        )

    def _build_control_card(self, parent) -> None:
        card = ctk.CTkFrame(
            parent,
            corner_radius=22,
            fg_color=("#FAF7F2", "#161F2E"),
            border_width=1,
            border_color=("#E5DED3", "#243147"),
        )
        card.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            card,
            text="运行控制",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#241A14", "#F6F0E8"),
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(18, 6))

        ctk.CTkLabel(
            card,
            text="闭眼阈值越高，越容易判定为闭眼；导出模式可切换分类图和识别效果图。",
            font=ctk.CTkFont(size=13),
            text_color=("#6A5646", "#95A4B8"),
        ).grid(row=1, column=0, sticky="w", padx=20, pady=(0, 14))

        slider_row = ctk.CTkFrame(card, fg_color="transparent")
        slider_row.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 14))
        slider_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            slider_row,
            text="EAR 阈值",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("#3C2E24", "#D7E2F2"),
        ).grid(row=0, column=0, sticky="w")

        self.slider_ear = ctk.CTkSlider(
            slider_row,
            from_=0.10,
            to=0.30,
            number_of_steps=20,
            command=self._on_ear_changed,
        )
        self.slider_ear.set(0.18)
        self.slider_ear.grid(row=0, column=1, sticky="ew", padx=16)

        ctk.CTkLabel(
            slider_row,
            textvariable=self.ear_value,
            corner_radius=999,
            width=70,
            padx=12,
            pady=6,
            fg_color=("#F2E4D6", "#22324A"),
            text_color=("#5F4333", "#E4EEFF"),
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=0, column=2, sticky="e")

        mode_row = ctk.CTkFrame(card, fg_color="transparent")
        mode_row.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 14))
        mode_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            mode_row,
            text="导出模式",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("#3C2E24", "#D7E2F2"),
        ).grid(row=0, column=0, sticky="w")

        self.output_mode_menu = ctk.CTkOptionMenu(
            mode_row,
            values=[label for label, _ in OUTPUT_MODE_OPTIONS],
            variable=self.output_mode_label,
            height=36,
            corner_radius=12,
            dynamic_resizing=False,
            fg_color=("#D7C0A8", "#22324A"),
            button_color=("#C47B45", "#34567A"),
            button_hover_color=("#B16935", "#41698F"),
            dropdown_fg_color=("#FAF7F2", "#1B2432"),
            dropdown_hover_color=("#EEDBC8", "#26344A"),
            dropdown_text_color=("#2B211A", "#EAF2FF"),
            text_color=("#2B211A", "#EAF2FF"),
        )
        self.output_mode_menu.grid(row=0, column=1, sticky="ew", padx=(16, 0))

        button_row = ctk.CTkFrame(card, fg_color="transparent")
        button_row.grid(row=4, column=0, sticky="ew", padx=20, pady=(0, 20))
        button_row.grid_columnconfigure((0, 1), weight=1)

        self.start_button = ctk.CTkButton(
            button_row,
            text="开始处理",
            height=46,
            corner_radius=16,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.start_processing,
        )
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self.open_output_button = ctk.CTkButton(
            button_row,
            text="查看输出文件夹",
            height=46,
            corner_radius=16,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=("#C47B45", "#34567A"),
            hover_color=("#B16935", "#41698F"),
            state="disabled",
            command=self.open_output_folder,
        )
        self.open_output_button.grid(row=0, column=1, sticky="ew", padx=(8, 0))

    def _build_summary_card(self, parent) -> None:
        card = ctk.CTkFrame(
            parent,
            corner_radius=22,
            fg_color=("#14100E", "#0D1522"),
            border_width=1,
            border_color=("#4A382D", "#22304A"),
        )
        card.grid(row=1, column=0, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            card,
            text="结果摘要",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#FFF7EE", "#F5F7FB"),
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(18, 10))

        self.summary_label = ctk.CTkLabel(
            card,
            textvariable=self.summary_text,
            justify="left",
            anchor="nw",
            wraplength=420,
            font=ctk.CTkFont(size=15),
            text_color=("#EAD8CA", "#D9E6F8"),
        )
        self.summary_label.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 18))

        ctk.CTkLabel(
            card,
            textvariable=self.output_hint_text,
            justify="left",
            anchor="w",
            wraplength=420,
            font=ctk.CTkFont(size=13),
            text_color=("#B68E75", "#8CA4C2"),
        ).grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 20))

    def _on_ear_changed(self, value: float) -> None:
        self.ear_value.set(f"{value:.2f}")

    def select_input_dir(self) -> None:
        directory = filedialog.askdirectory(title="选择输入照片目录")
        if not directory:
            return
        self.input_dir.set(directory)

        if not self.output_dir.get():
            suggested_output = Path(directory).parent / "eye_review_output"
            self.output_dir.set(str(suggested_output))

    def select_output_dir(self) -> None:
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.output_dir.set(directory)

    def open_output_folder(self) -> None:
        target_dir = self._last_output_dir or Path(self.output_dir.get())
        if not target_dir or not target_dir.exists():
            messagebox.showwarning("无法打开", "输出目录不存在，请先完成一次处理。")
            return

        if sys.platform.startswith("win"):
            os.startfile(target_dir)
            return

        if sys.platform == "darwin":
            subprocess.Popen(["open", str(target_dir)])
            return

        subprocess.Popen(["xdg-open", str(target_dir)])

    def _get_output_mode(self) -> str:
        return OUTPUT_MODE_LABEL_TO_VALUE[self.output_mode_label.get()]

    def start_processing(self) -> None:
        if self._is_processing:
            return

        if not self.input_dir.get() or not self.output_dir.get():
            messagebox.showwarning("缺少目录", "请先选择输入目录和输出目录。")
            return

        self._last_output_dir = Path(self.output_dir.get())
        self._is_processing = True
        self._run_started_at = time.perf_counter()
        self._set_controls_enabled(False)
        self._set_status("正在准备", tone="info")
        self._set_progress(0, 0, "正在初始化...")
        self.summary_text.set("正在启动分析任务，请稍候。")
        self.output_hint_text.set("处理中，完成后这里会显示输出目录提示。")
        self.open_output_button.configure(state="disabled")
        self._start_runtime_clock()

        worker = threading.Thread(target=self.run_analysis_task, daemon=True)
        worker.start()

    def run_analysis_task(self) -> None:
        input_path = Path(self.input_dir.get())
        output_dir = Path(self.output_dir.get())
        ear_threshold = self.slider_ear.get()
        output_mode = self._get_output_mode()
        face_candidate_detector = None
        detector = None
        results = []

        try:
            self._run_on_ui(self._set_status, "扫描照片中", "info")
            image_paths = pt.iter_images(input_path, exclude_dirs=[output_dir])
            total_images = len(image_paths)
            self._run_on_ui(self._set_progress, 0, total_images, "扫描完成，准备加载模型...")

            if total_images == 0:
                raise RuntimeError("未找到支持的图片或 RAW 文件。")

            pt.ensure_raw_support(image_paths)
            pt.ensure_output_structure(output_dir, output_mode)

            self._run_on_ui(self._set_status, "加载模型中", "info")

            class DummyArgs:
                yolo_model_path = pt.DEFAULT_YOLO_MODEL_PATH
                yolo_score_threshold = 0.6
                yolo_nms_threshold = 0.3
                yolo_top_k = 5000
                min_face_pixels = 20
                max_candidates = 20

            face_candidate_detector = pt.create_face_candidate_detector(DummyArgs())
            detector = pt.create_detector(max_faces=12, model_path=pt.DEFAULT_MODEL_PATH)

            self._run_on_ui(self._set_status, "正在处理", "running")
            for index, image_path in enumerate(image_paths, start=1):
                self._run_on_ui(
                    self._set_progress,
                    index - 1,
                    total_images,
                    image_path.name,
                )
                result = pt.analyze_image_with_timing(
                    image_path=image_path,
                    face_candidate_detector=face_candidate_detector,
                    detector=detector,
                    max_faces=12,
                    ear_threshold=ear_threshold,
                    min_face_size=0.0025,
                    crop_scale=2.5,
                    landmark_min_crop_size=256,
                )
                results.append(result)
                pt.export_result_outputs(
                    image_path=image_path,
                    input_root=input_path,
                    output_dir=output_dir,
                    result=result,
                    output_mode=output_mode,
                )
                self._run_on_ui(self._set_progress, index, total_images, image_path.name)

            pt.write_csv(results, output_dir)
            elapsed = time.perf_counter() - (self._run_started_at or time.perf_counter())
            self._run_on_ui(
                self._finish_successfully,
                results,
                output_dir,
                elapsed,
                output_mode,
            )
        except SystemExit as exc:
            self._run_on_ui(self._finish_with_error, str(exc))
        except Exception as exc:
            self._run_on_ui(self._finish_with_error, str(exc))
        finally:
            if face_candidate_detector is not None:
                face_candidate_detector.close()
            if detector is not None:
                detector.close()

    def _set_status(self, text: str, tone: str) -> None:
        palette = {
            "info": (("#FFF4D6", "#243B53"), ("#7B4B00", "#E6F0FF")),
            "running": (("#E5F2E7", "#173126"), ("#21643A", "#BBF7D0")),
            "success": (("#E6F7EC", "#163126"), ("#1F6C3B", "#C8FACC")),
            "error": (("#FCE5E5", "#3A1D25"), ("#9B2C2C", "#FFD6DE")),
        }
        fg_color, text_color = palette[tone]
        self.status_text.set(text)
        self.status_badge.configure(fg_color=fg_color, text_color=text_color)

    def _set_progress(self, processed: int, total: int, photo_name: str) -> None:
        percentage = 0 if total == 0 else int((processed / total) * 100)
        self.progress_bar.set(0 if total == 0 else processed / total)
        self.progress_text.set(f"{processed} / {total}")
        self.progress_percent_text.set(f"{percentage}%")
        self.processed_value.set(str(processed))
        self.total_value.set(str(total))
        self.current_photo_text.set(f"当前照片：{photo_name}")

    def _start_runtime_clock(self) -> None:
        self._stop_runtime_clock()
        self.runtime_text.set("00:00")
        self._refresh_runtime_clock()

    def _refresh_runtime_clock(self) -> None:
        if self._run_started_at is None:
            return
        elapsed = time.perf_counter() - self._run_started_at
        self.runtime_text.set(self._format_duration(elapsed))
        if self._is_processing:
            self._runtime_job = self.after(250, self._refresh_runtime_clock)
        else:
            self._runtime_job = None

    def _stop_runtime_clock(self) -> None:
        if self._runtime_job is not None:
            self.after_cancel(self._runtime_job)
            self._runtime_job = None

    def _finish_successfully(
        self,
        results: list[pt.PhotoAssessment],
        output_dir: Path,
        elapsed_seconds: float,
        output_mode: str,
    ) -> None:
        total_images = len(results)
        reject_count = sum(result.decision == "reject" for result in results)
        review_count = sum(result.decision == "review" for result in results)
        keep_count = sum(result.decision == "keep" for result in results)
        best_count = sum(result.decision == "best" for result in results)
        non_person_count = sum(not result.is_person for result in results)
        output_mode_label = OUTPUT_MODE_VALUE_TO_LABEL.get(output_mode, output_mode)
        output_hint_lines = [f"输出模式：{output_mode_label}"]
        if pt.should_export_classified(output_mode):
            output_hint_lines.append(f"分类图目录：{output_dir}")
        if pt.should_export_effects(output_mode):
            output_hint_lines.append(
                f"识别效果图目录：{pt.get_recognition_effects_root(output_dir)}"
            )

        self._is_processing = False
        self._stop_runtime_clock()
        self.runtime_text.set(self._format_duration(elapsed_seconds))
        self._set_status("处理完成", tone="success")
        self._set_progress(total_images, total_images, "全部处理完成")
        self._set_controls_enabled(True)
        self.open_output_button.configure(state="normal")
        self._last_output_dir = output_dir

        self.summary_text.set(
            "\n".join(
                [
                    f"总照片：{total_images}",
                    f"reject：{reject_count}    review：{review_count}",
                    f"keep：{keep_count}    best：{best_count}",
                    f"非人像：{non_person_count}（已归入 reject）",
                    f"输出模式：{output_mode_label}",
                    f"总耗时：{elapsed_seconds:.1f} 秒",
                    f"平均耗时：{elapsed_seconds / max(total_images, 1):.2f} 秒/张",
                ]
            )
        )
        self.output_hint_text.set("\n".join(output_hint_lines))

        messagebox.showinfo(
            "处理完成",
            (
                f"本次共处理 {total_images} 张照片。\n"
                f"总耗时 {elapsed_seconds:.1f} 秒。\n\n"
                f"输出模式：{output_mode_label}\n\n"
                f"结果已保存到：\n{output_dir}"
            ),
        )

    def _finish_with_error(self, error_message: str) -> None:
        self._is_processing = False
        self._stop_runtime_clock()
        self._set_status("处理失败", tone="error")
        self._set_controls_enabled(True)
        self.current_photo_text.set("当前照片：处理中断")
        self.summary_text.set(f"运行失败：{error_message}")
        self.output_hint_text.set("修复问题后可重新运行。")
        messagebox.showerror("处理失败", error_message)

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.input_button.configure(state=state)
        self.output_button.configure(state=state)
        self.start_button.configure(
            state=state,
            text="开始处理" if enabled else "处理中...",
        )
        self.slider_ear.configure(state=state)
        self.output_mode_menu.configure(state=state)

    def _run_on_ui(self, callback, *args) -> None:
        self.after(0, lambda: callback(*args))

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = max(0, int(round(seconds)))
        minutes, seconds_remainder = divmod(total_seconds, 60)
        hours, minutes_remainder = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes_remainder:02d}:{seconds_remainder:02d}"
        return f"{minutes_remainder:02d}:{seconds_remainder:02d}"


if __name__ == "__main__":
    app = PhotoToolGUI()
    app.mainloop()
