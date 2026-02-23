import customtkinter as ctk
import json
import webbrowser
import os

# Cấu hình giao diện chung
ctk.set_appearance_mode("dark")  # Chế độ tối
ctk.set_default_color_theme("blue")  # Tông màu xanh


class NewsApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Finding links tool")
        self.geometry("800x600")

        # Biến lưu trữ
        self.data = []
        self.total_urls = 0
        self.current_index = 0
        self.current_file_type = "CT"  # CT hoặc TG
        self.pending_records = []  # Danh sách các record chờ lưu
        self.LOG_FILE = "log.json"  # File lưu trạng thái

        # 1. Title App
        self.label = ctk.CTkLabel(
            self, text="FINDING EVIDENCE FOR VNEXPRESS", font=("Arial", 20, "bold")
        )
        self.label.pack(pady=10)

        # 1.5 Chọn loại file để load (CT hoặc TG)
        self.file_select_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.file_select_frame.pack(pady=5, padx=30, fill="x")

        self.file_label = ctk.CTkLabel(self.file_select_frame, text="Chọn loại: ")
        self.file_label.pack(side="left")

        self.file_type_combo = ctk.CTkComboBox(
            self.file_select_frame,
            values=["CT (Chính trị)", "TG (Thế giới)"],
            width=150,
            command=self.on_file_type_change,
        )
        self.file_type_combo.set("CT (Chính trị)")
        self.file_type_combo.pack(side="left", padx=10)

        # 2 Tạo một Frame để chứa link nguồn:
        self.org_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.org_frame.pack(pady=10, padx=30, fill="x")
        # 2.1. Label thể hiện link nguồn
        self.org_label = ctk.CTkLabel(self.org_frame, text="Nguồn: ")
        self.org_label.pack(side="left", padx=(10, 0))
        self.org_id = ctk.CTkLabel(self.org_frame, text="ORG_12323")
        self.org_id.pack(side="left", padx=(10, 0))

        # Link có thể click
        self.org_title = ctk.CTkLabel(
            self.org_frame,
            text="Chưa tải dữ liệu...",
            text_color="#3498db",
            cursor="hand2",
        )
        self.org_title.pack(side="left", padx=(5, 0))
        self.org_title.bind("<Button-1>", self.open_current_link)

        # 3 Tạo một frame để chứa tracking process và nút
        self.nav_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.nav_frame.pack(pady=10, padx=30, fill="x")

        # 3.0 Tracking số lượng bài báo đã làm
        self.nav_label = ctk.CTkLabel(self.nav_frame, text="0/0")
        self.nav_label.pack(side="right")

        # 3.1 Nút back forward <- -> để chuyển qua nguồn kế tiếp
        self.forward_button = ctk.CTkButton(
            self.nav_frame, text="▶", width=40, command=self.next_article
        )
        self.forward_button.pack(side="right", padx=10, pady=5)

        # 3.2 Pack nút Backward tiếp theo với side="right" nó sẽ nằm kế bên nút Forward
        self.backward_button = ctk.CTkButton(
            self.nav_frame, text="◀", width=40, command=self.prev_article
        )
        self.backward_button.pack(side="right", padx=5, pady=5)

        # 4. Tạo một Frame để chứa cả Input và ComboBox trên 1 hàng
        self.input_row = ctk.CTkFrame(self, fg_color="#2b3e50")
        self.input_row.pack(pady=10, padx=30, fill="x")

        self.subframe1 = ctk.CTkFrame(self.input_row, fg_color="transparent")
        self.subframe1.pack(pady=5, padx=30, fill="x")
        # 4.1. Input Field để paste Link
        self.link_entry = ctk.CTkEntry(
            self.subframe1, placeholder_text="Dán link bài báo vào đây", width=350
        )
        self.link_entry.pack(side="left", padx=(0, 10))

        # 4.2. ComboBox để chọn nguồn (ngay kế bên Input)
        self.source_label = ctk.CTkLabel(self.subframe1, text="Nguồn: ")
        self.source_label.pack(side="left")
        self.source_combo = ctk.CTkComboBox(
            self.subframe1,
            values=[
                "thanhnien",
                "vtv",
                "tuoitre",
                "dantri",
                "vietnamnet",
                "laodong",
                "Khác",
            ],
            width=120,
            command=self.on_source_change,
        )
        self.source_combo.set("vtv")
        self.source_combo.pack(side="left", padx=(5, 0))

        # 4.2.1 Input Field cho nguồn tùy chỉnh (ẩn ban đầu)
        self.custom_source_entry = ctk.CTkEntry(
            self.subframe1, placeholder_text="Nhập nguồn...", width=120
        )
        # Không pack ngay, chỉ hiện khi chọn "Khác"

        # 4.3 Tạo nút Submit và nút Clear
        self.subframe2 = ctk.CTkFrame(self.input_row, fg_color="transparent")
        self.subframe2.pack(pady=5, padx=30, fill="x")
        self.submit_button = ctk.CTkButton(
            self.subframe2, text="SUBMIT", command=self.add_entry_to_table
        )
        self.submit_button.pack(side="right", padx=(0, 10))
        self.clear_button = ctk.CTkButton(
            self.subframe2, text="CLEAR", command=self.clear_input
        )
        self.clear_button.pack(side="right", padx=(0, 10))

        # 6. Lưu quá trình thủ công vào file
        self.saved_frame = ctk.CTkFrame(self, fg_color="transparent")
        # self.saved_frame.pack(pady=10, padx=30, fill="x")
        # 6.1 Nút saved
        self.save_button = ctk.CTkButton(
            self.saved_frame,
            text="💾 Save & Next",
            fg_color="#27ae60",
            hover_color="#1e8449",
            command=self.save_and_next,
        )
        self.save_button.pack(side="left", padx=(0, 10))

        # 6.2 Label hiển thị trạng thái
        self.status_label = ctk.CTkLabel(
            self.saved_frame, text="", text_color="#f39c12"
        )
        self.status_label.pack(side="left")

        # 5. Hiện thị các link đã chọn cùng với nguồn của nó:
        self.table_label = ctk.CTkLabel(
            self, text="📋 Danh sách nguồn đã thêm:", font=("Arial", 12, "bold")
        )
        self.table_label.pack(pady=(10, 5), padx=30, anchor="w")

        self.table_frame = ctk.CTkScrollableFrame(self, fg_color="#363c92", height=150)
        self.table_frame.pack(pady=5, padx=30, fill="both", expand=True)

        self.table_frame.columnconfigure(0, weight=1)
        self.table_frame.columnconfigure(1, weight=3)
        self.table_frame.columnconfigure(2, weight=1)

        self.row_counter = 0
        self.table_widgets = []  # Lưu trữ widgets của bảng

        # Load dữ liệu ban đầu
        self.load_data()

    # generate by gemini-pro
    def add_record(self, source, link):
        """Hàm này thêm một hàng mới vào bảng"""

        # Cột 0: Nguồn
        lbl_source = ctk.CTkLabel(
            self.table_frame, text=source, text_color="white", font=("Arial", 11)
        )
        lbl_source.grid(row=self.row_counter, column=0, sticky="w", pady=5, padx=5)

        # Cột 1: Link (Cắt bớt nếu quá dài để giữ form bảng)
        display_link = (link[:60] + "...") if len(link) > 60 else link
        lbl_link = ctk.CTkLabel(
            self.table_frame, text=display_link, text_color="#3498db", cursor="hand2"
        )
        lbl_link.grid(row=self.row_counter, column=1, sticky="w", pady=5)
        # Bind click để mở link
        lbl_link.bind("<Button-1>", lambda e, url=link: webbrowser.open(url))

        # Cột 2: Nút xóa
        current_row = self.row_counter
        btn_delete = ctk.CTkButton(
            self.table_frame,
            text="🗑",
            width=40,
            fg_color="#e74c3c",
            hover_color="#c0392b",
        )
        btn_delete.configure(command=lambda r=current_row: self.delete_row(r))
        btn_delete.grid(row=self.row_counter, column=2, sticky="e", pady=5, padx=5)

        # Lưu widgets và data
        record = {
            "row": self.row_counter,
            "source": source,
            "link": link,
            "widgets": [lbl_source, lbl_link, btn_delete],
        }
        self.table_widgets.append(record)
        self.pending_records.append({"source": source, "link": link})

        # Lưu vào log.json ngay lập tức
        self.save_to_log()
        print(
            f"Added record: {source} - pending_records now has {len(self.pending_records)} items"
        )

        self.row_counter += 1

    def delete_row(self, row_index):
        """Xóa một hàng khỏi bảng"""
        for i, record in enumerate(self.table_widgets):
            if record["row"] == row_index:
                # Xóa widgets
                for widget in record["widgets"]:
                    widget.destroy()
                # Xóa khỏi danh sách
                self.table_widgets.pop(i)
                # Xóa khỏi pending_records
                if i < len(self.pending_records):
                    self.pending_records.pop(i)
                break
        # Cập nhật log.json
        self.save_to_log()
        self.update_status(f"Đã xóa nguồn")

    def on_source_change(self, choice):
        """Xử lý khi thay đổi nguồn"""
        if choice == "Khác":
            self.custom_source_entry.pack(side="left", padx=(10, 0))
        else:
            self.custom_source_entry.pack_forget()

    def on_file_type_change(self, choice):
        """Xử lý khi thay đổi loại file (CT/TG)"""
        if "CT" in choice:
            self.current_file_type = "CT"
        else:
            self.current_file_type = "TG"

        self.clear_table_widgets_only()
        self.load_data()

    def open_current_link(self, event=None):
        """Mở link hiện tại trong trình duyệt"""
        if self.data and 0 <= self.current_index < len(self.data):
            url = self.data[self.current_index]["url"]
            webbrowser.open(url)

    def clear_input(self):
        """Xóa nội dung input"""
        self.link_entry.delete(0, "end")
        self.custom_source_entry.delete(0, "end")
        self.source_combo.set("vtv")
        self.custom_source_entry.pack_forget()

    def update_status(self, message):
        """Cập nhật thông báo trạng thái"""
        self.status_label.configure(text=message)
        # Tự động xóa sau 3 giây
        self.after(3000, lambda: self.status_label.configure(text=""))

    def update_ui(self):
        if not self.data or self.total_urls == 0:
            self.org_title.configure(text="Không có dữ liệu")
            self.nav_label.configure(text="0/0")
            return

        current_article = self.data[self.current_index]
        url = current_article["url"]
        id = current_article["id"]
        # Hiển thị link rút gọn
        display_url = url[:70] + "..." if len(url) > 70 else url
        self.org_title.configure(text=display_url)
        self.org_id.configure(text=id)
        self.nav_label.configure(text=f"{self.current_index + 1}/{self.total_urls}")

    def load_data(self):
        """Load dữ liệu từ file JSON tương ứng và lọc theo range ID"""
        filename = f"../link/org/org_{self.current_file_type}.json"

        # Định nghĩa range ID cho từng loại
        id_ranges = {"CT": (9331, 11284), "TG": (1, 3108)}

        try:
            with open(filename, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            # Lọc theo range ID
            min_id, max_id = id_ranges.get(self.current_file_type, (0, 999999))
            self.data = [
                item
                for item in raw_data
                if min_id <= int(item["id"].replace("ORG_", "")) <= max_id
            ]

            self.total_urls = len(self.data)
            if self.total_urls > 0:
                self.current_index = 0
                self.update_ui()
                # Load records từ log.json cho bài đầu tiên
                self.load_records_for_current_article()
                self.update_status(
                    f"Đã tải {self.total_urls}/{len(raw_data)} links từ {filename}"
                )
            else:
                self.update_ui()
        except FileNotFoundError:
            self.update_status(f"Không tìm thấy file: {filename}")
            self.data = []
            self.total_urls = 0
            self.update_ui()

    def next_article(self):
        if self.current_index < (self.total_urls - 1):
            self.current_index += 1
            self.load_records_for_current_article()
            self.update_ui()

    def prev_article(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_records_for_current_article()
            self.update_ui()

    def get_current_article_id(self):
        """Lấy ID của bài viết hiện tại"""
        if self.data and 0 <= self.current_index < len(self.data):
            return self.data[self.current_index].get("id", "")
        return ""

    def load_log(self):
        """Đọc log.json"""
        try:
            with open(self.LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"CT": {}, "TG": {}}

    def save_log(self, log_data):
        """Ghi vào log.json"""
        with open(self.LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

    def save_to_log(self):
        """Lưu pending_records vào log.json cho bài hiện tại"""
        article_id = self.get_current_article_id()
        if not article_id:
            return

        log_data = self.load_log()

        if self.pending_records:
            log_data[self.current_file_type][article_id] = [
                {"source": r["source"], "link": r["link"]} for r in self.pending_records
            ]
        elif article_id in log_data[self.current_file_type]:
            del log_data[self.current_file_type][article_id]

        self.save_log(log_data)
        print(f"Saved to log.json: {article_id} -> {len(self.pending_records)} records")

    def load_records_for_current_article(self):
        """Load records từ log.json cho bài hiện tại"""
        # Xóa bảng hiện tại
        self.clear_table_widgets_only()

        article_id = self.get_current_article_id()
        if not article_id:
            return

        log_data = self.load_log()

        # Load records nếu có trong log
        if article_id in log_data.get(self.current_file_type, {}):
            saved_records = log_data[self.current_file_type][article_id]
            print(
                f"Loading from log.json: {article_id} -> {len(saved_records)} records"
            )
            for record in saved_records:
                # Thêm vào bảng nhưng không save lại log (tránh loop)
                self.add_record_without_save(record["source"], record["link"])
        else:
            print(f"No records in log.json for: {article_id}")

    def add_record_without_save(self, source, link):
        """Thêm record vào bảng mà không save vào log (dùng khi load)"""
        # Cột 0: Nguồn
        lbl_source = ctk.CTkLabel(self.table_frame, text=source, font=("Arial", 11))
        lbl_source.grid(row=self.row_counter, column=0, sticky="w", pady=5, padx=5)

        # Cột 1: Link
        display_link = (link[:60] + "...") if len(link) > 60 else link
        lbl_link = ctk.CTkLabel(
            self.table_frame, text=display_link, text_color="#3498db", cursor="hand2"
        )
        lbl_link.grid(row=self.row_counter, column=1, sticky="w", pady=5)
        lbl_link.bind("<Button-1>", lambda e, url=link: webbrowser.open(url))

        # Cột 2: Nút xóa
        current_row = self.row_counter
        btn_delete = ctk.CTkButton(
            self.table_frame,
            text="🗑",
            width=40,
            fg_color="#e74c3c",
            hover_color="#c0392b",
        )
        btn_delete.configure(command=lambda r=current_row: self.delete_row(r))
        btn_delete.grid(row=self.row_counter, column=2, sticky="e", pady=5, padx=5)

        record = {
            "row": self.row_counter,
            "source": source,
            "link": link,
            "widgets": [lbl_source, lbl_link, btn_delete],
        }
        self.table_widgets.append(record)
        self.pending_records.append({"source": source, "link": link})
        self.row_counter += 1

    def remove_from_log(self):
        """Xóa bài hiện tại khỏi log.json sau khi đã save vào file txt"""
        article_id = self.get_current_article_id()
        if not article_id:
            return

        log_data = self.load_log()
        if article_id in log_data.get(self.current_file_type, {}):
            del log_data[self.current_file_type][article_id]
            self.save_log(log_data)
            print(f"Removed from log.json: {article_id}")

    def clear_table_widgets_only(self):
        """Chỉ xóa widgets, không xóa records_by_article"""
        for record in self.table_widgets:
            for widget in record["widgets"]:
                widget.destroy()
        self.table_widgets = []
        self.pending_records = []
        self.row_counter = 0

    def add_entry_to_table(self):
        """Thêm nguồn vào bảng"""
        link = self.link_entry.get().strip()
        source = self.source_combo.get()

        # Nếu chọn "Khác", lấy nguồn từ custom entry
        if source == "Khác":
            custom_source = self.custom_source_entry.get().strip()
            if not custom_source:
                self.update_status("Lỗi: Vui lòng nhập tên nguồn!")
                return
            source = custom_source.lower().replace(" ", "_")

        if not link:
            self.update_status("Lỗi: Link không được để trống!")
            return

        # Validate link
        if not link.startswith("http"):
            self.update_status("Lỗi: Link phải bắt đầu bằng http/https!")
            return

        # Thêm vào bảng
        self.add_record(source, link)

        # Xóa input sau khi thêm
        self.link_entry.delete(0, "end")
        self.update_status(f"Đã thêm link từ {source}")

    def save_records_to_file(self):
        """Lưu các record vào file tương ứng"""
        if not self.pending_records:
            self.update_status("Không có gì để lưu!")
            return False

        # Lấy thông tin bài viết hiện tại
        current_article_id = ""
        current_url = ""
        if self.data and 0 <= self.current_index < len(self.data):
            current_article = self.data[self.current_index]
            current_article_id = current_article.get("id", "")
            current_url = current_article.get("url", "")

        # Nhóm các record theo nguồn
        records_by_source = {}
        for record in self.pending_records:
            source = record["source"].lower()
            if source not in records_by_source:
                records_by_source[source] = []
            records_by_source[source].append(record["link"])

        # Lưu vào các file tương ứng
        for source, links in records_by_source.items():
            filename = f"{source}_van_{self.current_file_type}.txt"

            with open(filename, "a", encoding="utf-8") as f:
                # Chỉ ghi link, mỗi link một dòng
                for link in links:
                    f.write(f"{link}\n")

        saved_count = len(self.pending_records)
        self.update_status(f"Đã lưu {saved_count} nguồn vào file!")
        return True

    def save_and_next(self):
        """Lưu các record và chuyển sang bài tiếp theo"""
        if not self.pending_records:
            self.update_status("Vui lòng thêm ít nhất 1 nguồn!")
            return

        # Lưu vào file txt
        if self.save_records_to_file():
            # Xóa khỏi log.json sau khi đã save
            self.remove_from_log()
            # Xóa bảng
            self.clear_table_widgets_only()
            # Chuyển sang bài tiếp theo
            if self.current_index < (self.total_urls - 1):
                self.current_index += 1
                self.load_records_for_current_article()
                self.update_ui()

    def clear_table(self):
        """Xóa toàn bộ bảng"""
        self.clear_table_widgets_only()


if __name__ == "__main__":
    app = NewsApp()
    app.mainloop()
