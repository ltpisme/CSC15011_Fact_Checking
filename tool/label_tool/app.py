import customtkinter as ctk
import json
import os

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "output_dataset_hybrid.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "label_results.json")


class LabelApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Manual Label Tool")
        self.geometry("1100x950")
        self.minsize(900, 750)

        self.data = []
        self.total = 0
        self.current_index = 0
        self.results = {}  # {index: {"label": ..., "quote": ...}}

        self.build_ui()
        self.load_data()

    # ─────────────────────────── UI BUILD ────────────────────────────

    def build_ui(self):
        # ── Title ──────────────────────────────────────────────────
        self.title_label = ctk.CTkLabel(
            self, text="MANUAL LABEL TOOL", font=("Arial", 22, "bold")
        )
        self.title_label.pack(pady=(15, 5))

        # ── Navigation row ─────────────────────────────────────────
        self.nav_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.nav_frame.pack(pady=4, padx=30, fill="x")

        self.prev_button = ctk.CTkButton(
            self.nav_frame, text="◀ Trước", width=90, command=self.prev_item
        )
        self.prev_button.pack(side="left", padx=(0, 8))

        self.progress_label = ctk.CTkLabel(
            self.nav_frame, text="0/0", font=("Arial", 14, "bold")
        )
        self.progress_label.pack(side="left", padx=8)

        self.next_button = ctk.CTkButton(
            self.nav_frame, text="Tiếp ▶", width=90, command=self.next_item
        )
        self.next_button.pack(side="left", padx=(0, 8))

        self.labeled_count_label = ctk.CTkLabel(
            self.nav_frame, text="Đã nhãn: 0/0", font=("Arial", 12)
        )
        self.labeled_count_label.pack(side="left", padx=15)

        self.current_label_indicator = ctk.CTkLabel(
            self.nav_frame, text="", font=("Arial", 13, "bold")
        )
        self.current_label_indicator.pack(side="right", padx=5)

        # ── Claim ──────────────────────────────────────────────────
        self.claim_outer = ctk.CTkFrame(self, fg_color="#1e2a35", corner_radius=8)
        self.claim_outer.pack(pady=6, padx=30, fill="x")

        ctk.CTkLabel(
            self.claim_outer,
            text="CLAIM",
            font=("Arial", 13, "bold"),
            text_color="#f39c12",
        ).pack(anchor="w", padx=12, pady=(8, 2))

        self.claim_text = ctk.CTkTextbox(
            self.claim_outer,
            height=80,
            wrap="word",
            font=("Arial", 13),
            fg_color="#253545",
        )
        self.claim_text.pack(fill="x", padx=12, pady=(0, 10))
        self.claim_text.configure(state="disabled")

        # ── Evidence ───────────────────────────────────────────────
        evidence_header = ctk.CTkFrame(self, fg_color="#f39c12", corner_radius=6)
        evidence_header.pack(fill="x", padx=30, pady=(8, 0))
        ctk.CTkLabel(
            evidence_header,
            text="📄  EVIDENCE",
            font=("Arial", 17, "bold"),
            text_color="#1a252f",
        ).pack(anchor="w", padx=14, pady=6)

        self.evidence_frame = ctk.CTkScrollableFrame(
            self, fg_color="#1e2a35", height=340, corner_radius=8,
            border_width=2, border_color="#f39c12"
        )
        self.evidence_frame.pack(pady=(0, 4), padx=30, fill="both", expand=True)

        # ── AI Votes (reference) ───────────────────────────────────
        self.ai_toggle_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.ai_toggle_frame.pack(pady=(6, 0), padx=30, fill="x")

        self.ai_visible = ctk.BooleanVar(value=False)
        self.ai_toggle_btn = ctk.CTkButton(
            self.ai_toggle_frame,
            text="▶ Xem dự đoán AI (tham khảo)",
            width=230,
            height=28,
            font=("Arial", 11),
            fg_color="#2c3e50",
            hover_color="#3d5166",
            command=self.toggle_ai_votes,
        )
        self.ai_toggle_btn.pack(side="left")

        self.ai_votes_frame = ctk.CTkScrollableFrame(
            self, fg_color="#1a252f", height=100, corner_radius=6
        )
        # Not packed initially; shown on toggle

        # ── Quote input ────────────────────────────────────────────
        self.quote_outer = ctk.CTkFrame(self, fg_color="transparent")
        self.quote_outer.pack(pady=8, padx=30, fill="x")

        ctk.CTkLabel(
            self.quote_outer,
            text="Quote / Ghi chú của bạn:",
            font=("Arial", 12, "bold"),
        ).pack(anchor="w")

        self.quote_entry = ctk.CTkTextbox(
            self.quote_outer, height=65, wrap="word", font=("Arial", 12)
        )
        self.quote_entry.pack(fill="x", pady=(3, 0))

        # ── Label buttons ──────────────────────────────────────────
        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.pack(pady=10, padx=30, fill="x")

        self.supported_btn = ctk.CTkButton(
            self.btn_frame,
            text="✅  SUPPORTED",
            height=50,
            font=("Arial", 14, "bold"),
            fg_color="#27ae60",
            hover_color="#1e8449",
            command=lambda: self.label_item("SUPPORTED"),
        )
        self.supported_btn.pack(side="left", padx=(0, 10), expand=True, fill="x")

        self.nei_btn = ctk.CTkButton(
            self.btn_frame,
            text="❓  NEI",
            height=50,
            font=("Arial", 14, "bold"),
            fg_color="#7f8c8d",
            hover_color="#5d6d7e",
            command=lambda: self.label_item("NEI"),
        )
        self.nei_btn.pack(side="left", padx=(0, 10), expand=True, fill="x")

        self.refuted_btn = ctk.CTkButton(
            self.btn_frame,
            text="❌  REFUTED",
            height=50,
            font=("Arial", 14, "bold"),
            fg_color="#e74c3c",
            hover_color="#c0392b",
            command=lambda: self.label_item("REFUTED"),
        )
        self.refuted_btn.pack(side="left", expand=True, fill="x")

        # ── Status bar ─────────────────────────────────────────────
        self.status_label = ctk.CTkLabel(
            self, text="", text_color="#f39c12", font=("Arial", 11)
        )
        self.status_label.pack(pady=(4, 8))

    # ─────────────────────────── DATA ────────────────────────────────

    def load_data(self):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            self.total = len(self.data)
        except FileNotFoundError:
            self.update_status(f"Không tìm thấy file: {DATA_FILE}")
            return
        except json.JSONDecodeError as e:
            self.update_status(f"Lỗi đọc JSON: {e}")
            return

        # Load existing results
        if os.path.exists(OUTPUT_FILE):
            try:
                with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self.results = {int(k): v for k, v in saved.items()}
            except Exception:
                self.results = {}

        # Jump to first unlabeled item
        self.current_index = 0
        for i in range(self.total):
            if i not in self.results:
                self.current_index = i
                break

        self.update_ui()
        self.update_status(
            f"Đã tải {self.total} mục. Đã nhãn: {len(self.results)}/{self.total}"
        )

    def save_results(self):
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {str(k): v for k, v in self.results.items()},
                f,
                ensure_ascii=False,
                indent=2,
            )

    # ─────────────────────────── UI UPDATE ───────────────────────────

    def update_ui(self):
        if not self.data:
            return

        item = self.data[self.current_index]

        # Progress
        self.progress_label.configure(text=f"{self.current_index + 1}/{self.total}")
        self.labeled_count_label.configure(
            text=f"Đã nhãn: {len(self.results)}/{self.total}"
        )

        # Claim
        self.claim_text.configure(state="normal")
        self.claim_text.delete("1.0", "end")
        self.claim_text.insert("1.0", item.get("claim", ""))
        self.claim_text.configure(state="disabled")

        # Evidence
        for w in self.evidence_frame.winfo_children():
            w.destroy()

        for ev in item.get("evidence", []):
            container = ctk.CTkFrame(
                self.evidence_frame, fg_color="#253545", corner_radius=6
            )
            container.pack(fill="x", pady=4, padx=4)

            ctk.CTkLabel(
                container,
                text=f"🔗  {ev.get('source', '')}",
                font=("Arial", 13, "bold"),
                text_color="#5dade2",
            ).pack(anchor="w", padx=12, pady=(8, 0))

            tb = ctk.CTkTextbox(
                container,
                height=100,
                wrap="word",
                font=("Arial", 14),
                fg_color="#1a2632",
                text_color="#ecf0f1",
            )
            tb.pack(fill="x", padx=12, pady=(4, 10))
            tb.insert("1.0", ev.get("text", ""))
            tb.configure(state="disabled")

        # AI votes (rebuild hidden)
        for w in self.ai_votes_frame.winfo_children():
            w.destroy()

        for vd in item.get("voter_details", []):
            lbl_color_map = {
                "SUPPORTED": "#27ae60",
                "REFUTED": "#e74c3c",
                "NEI": "#7f8c8d",
            }
            vote = vd.get("vote", "")
            row = ctk.CTkFrame(self.ai_votes_frame, fg_color="#253545", corner_radius=4)
            row.pack(fill="x", pady=3, padx=4)
            ctk.CTkLabel(
                row,
                text=f"{vd.get('model', '')}",
                font=("Arial", 10, "bold"),
                text_color="#aab8c2",
            ).pack(side="left", padx=(8, 4), pady=4)
            ctk.CTkLabel(
                row,
                text=vote,
                font=("Arial", 10, "bold"),
                text_color=lbl_color_map.get(vote, "white"),
            ).pack(side="left", padx=4)
            ctk.CTkLabel(
                row,
                text=f"— {vd.get('quote', '')}",
                font=("Arial", 10),
                wraplength=600,
                justify="left",
            ).pack(side="left", padx=(4, 8), pady=4)

        # Quote & label indicator
        self.quote_entry.delete("1.0", "end")
        if self.current_index in self.results:
            res = self.results[self.current_index]
            lbl = res.get("label", "")
            self.quote_entry.insert("1.0", res.get("quote", ""))
            color_map = {
                "SUPPORTED": "#27ae60",
                "REFUTED": "#e74c3c",
                "NEI": "#f39c12",
            }
            self.current_label_indicator.configure(
                text=f"Đã nhãn: {lbl}",
                text_color=color_map.get(lbl, "white"),
            )
        else:
            self.current_label_indicator.configure(
                text="Chưa nhãn", text_color="#95a5a6"
            )

    def toggle_ai_votes(self):
        self.ai_visible.set(not self.ai_visible.get())
        if self.ai_visible.get():
            self.ai_votes_frame.pack(
                after=self.ai_toggle_frame, pady=(2, 4), padx=30, fill="x"
            )
            self.ai_toggle_btn.configure(text="▼ Ẩn dự đoán AI (tham khảo)")
        else:
            self.ai_votes_frame.pack_forget()
            self.ai_toggle_btn.configure(text="▶ Xem dự đoán AI (tham khảo)")

    # ─────────────────────────── ACTIONS ─────────────────────────────

    def label_item(self, label):
        quote = self.quote_entry.get("1.0", "end").strip()
        self.results[self.current_index] = {"label": label, "quote": quote}
        self.save_results()
        self.update_status(f"Đã lưu nhãn: {label}")

        color_map = {"SUPPORTED": "#27ae60", "REFUTED": "#e74c3c", "NEI": "#f39c12"}
        self.current_label_indicator.configure(
            text=f"Đã nhãn: {label}",
            text_color=color_map.get(label, "white"),
        )
        self.labeled_count_label.configure(
            text=f"Đã nhãn: {len(self.results)}/{self.total}"
        )

        # Auto advance to next item
        if self.current_index < self.total - 1:
            self.current_index += 1
            self.update_ui()

    def next_item(self):
        if self.current_index < self.total - 1:
            self.current_index += 1
            self.update_ui()

    def prev_item(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_ui()

    def update_status(self, message):
        self.status_label.configure(text=message)
        self.after(4000, lambda: self.status_label.configure(text=""))


if __name__ == "__main__":
    app = LabelApp()
    app.mainloop()
