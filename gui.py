import customtkinter as ctk
from src.bot import UniAssistBot


# ============================================================
# APPEARANCE
# ============================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ============================================================
# COLORS
# ============================================================

BG_COLOR = "#070B14"
CHAT_BG = "#0D1424"
BOT_BUBBLE = "#111827"
USER_BUBBLE = "#2563EB"
TEXT_COLOR = "#F8FAFC"
SECONDARY_TEXT = "#94A3B8"
BORDER_COLOR = "#1E293B"


# ============================================================
# MAIN APP
# ============================================================

class UniAssistGUI(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("UniAssist - University AI Assistant")
        self.geometry("850x650")
        self.minsize(700, 550)

        self.configure(fg_color=BG_COLOR)

        # Load existing chatbot brain
        self.bot = UniAssistBot()

        self._create_header()
        self._create_chat_area()
        self._create_input_area()

        # Initial message
        self.add_bot_message(
            "Assalam-o-Alaikum! 👋\n\n"
            "Main UniAssist hoon, aapka University AI Assistant.\n"
            "Aap mujhse students, CGPA, percentage, roll number, "
            "timetable, teachers aur university information ke "
            "baare mein pooch sakte hain."
        )

    # ========================================================
    # HEADER
    # ========================================================

    def _create_header(self):

        self.header = ctk.CTkFrame(
            self,
            height=75,
            corner_radius=0,
            fg_color=CHAT_BG,
            border_width=1,
            border_color=BORDER_COLOR
        )

        self.header.pack(
            fill="x",
            side="top"
        )

        self.header.pack_propagate(False)

        # Logo / icon
        self.logo = ctk.CTkLabel(
            self.header,
            text="🤖",
            font=("Segoe UI Emoji", 30)
        )

        self.logo.pack(
            side="left",
            padx=(25, 10)
        )

        # Title container
        title_frame = ctk.CTkFrame(
            self.header,
            fg_color="transparent"
        )

        title_frame.pack(
            side="left",
            pady=10
        )

        self.title_label = ctk.CTkLabel(
            title_frame,
            text="UniAssist",
            font=("Segoe UI", 24, "bold"),
            text_color=TEXT_COLOR
        )

        self.title_label.pack(
            anchor="w"
        )

        self.subtitle_label = ctk.CTkLabel(
            title_frame,
            text="University AI Assistant  •  Online",
            font=("Segoe UI", 12),
            text_color=SECONDARY_TEXT
        )

        self.subtitle_label.pack(
            anchor="w"
        )

    # ========================================================
    # CHAT AREA
    # ========================================================

    def _create_chat_area(self):

        self.chat_area = ctk.CTkScrollableFrame(
            self,
            fg_color=BG_COLOR,
            corner_radius=0
        )

        self.chat_area.pack(
            fill="both",
            expand=True,
            padx=0,
            pady=0
        )

    # ========================================================
    # INPUT AREA
    # ========================================================

    def _create_input_area(self):

        self.input_container = ctk.CTkFrame(
            self,
            height=85,
            corner_radius=0,
            fg_color=CHAT_BG,
            border_width=1,
            border_color=BORDER_COLOR
        )

        self.input_container.pack(
            fill="x",
            side="bottom"
        )

        self.input_container.pack_propagate(False)

        # Input box
        self.message_entry = ctk.CTkEntry(
            self.input_container,
            height=48,
            corner_radius=15,
            border_width=1,
            border_color=BORDER_COLOR,
            fg_color=BOT_BUBBLE,
            text_color=TEXT_COLOR,
            placeholder_text="Type your question here...",
            placeholder_text_color=SECONDARY_TEXT,
            font=("Segoe UI", 14)
        )

        self.message_entry.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(20, 10),
            pady=18
        )

        # Send button
        self.send_button = ctk.CTkButton(
            self.input_container,
            text="➤",
            width=55,
            height=48,
            corner_radius=15,
            fg_color=USER_BUBBLE,
            hover_color="#1D4ED8",
            font=("Segoe UI", 20, "bold"),
            command=self.send_message
        )

        self.send_button.pack(
            side="right",
            padx=(0, 20),
            pady=18
        )

        # Enter key
        self.message_entry.bind(
            "<Return>",
            self._enter_pressed
        )

        # Automatically focus input
        self.message_entry.focus()

    # ========================================================
    # ENTER KEY
    # ========================================================

    def _enter_pressed(self, event):

        self.send_message()

        return "break"

    # ========================================================
    # USER MESSAGE
    # ========================================================

    def add_user_message(self, message):

        row = ctk.CTkFrame(
            self.chat_area,
            fg_color="transparent"
        )

        row.pack(
            fill="x",
            padx=20,
            pady=(8, 4)
        )

        bubble = ctk.CTkLabel(
            row,
            text=message,
            fg_color=USER_BUBBLE,
            text_color=TEXT_COLOR,
            corner_radius=15,
            font=("Segoe UI", 14),
            wraplength=520,
            justify="left",
            padx=15,
            pady=10
        )

        bubble.pack(
            side="right",
            anchor="e"
        )

        self._scroll_to_bottom()

    # ========================================================
    # BOT MESSAGE
    # ========================================================

    def add_bot_message(self, message):

        row = ctk.CTkFrame(
            self.chat_area,
            fg_color="transparent"
        )

        row.pack(
            fill="x",
            padx=20,
            pady=(4, 8)
        )

        avatar = ctk.CTkLabel(
            row,
            text="🤖",
            font=("Segoe UI Emoji", 22)
        )

        avatar.pack(
            side="left",
            anchor="n",
            padx=(0, 8)
        )

        bubble = ctk.CTkLabel(
            row,
            text=message,
            fg_color=BOT_BUBBLE,
            text_color=TEXT_COLOR,
            corner_radius=15,
            font=("Segoe UI", 14),
            wraplength=600,
            justify="left",
            padx=15,
            pady=10
        )

        bubble.pack(
            side="left",
            anchor="w"
        )

        self._scroll_to_bottom()

    # ========================================================
    # SEND MESSAGE
    # ========================================================

    def send_message(self):

        message = self.message_entry.get().strip()

        if not message:
            return

        # Show user message
        self.add_user_message(message)

        # Clear input
        self.message_entry.delete(
            0,
            "end"
        )

        try:

            # Existing UniAssist backend
            result = self.bot.get_answer(message)

            answer = result.get(
                "answer",
                "Sorry, mujhe iska answer nahi mila."
            )

            self.add_bot_message(answer)

        except Exception as error:

            self.add_bot_message(
                "Sorry, ek error aa gaya:\n"
                f"{error}"
            )

        self.message_entry.focus()

    # ========================================================
    # SCROLL
    # ========================================================

    def _scroll_to_bottom(self):

        self.after(
            100,
            lambda: self.chat_area._parent_canvas.yview_moveto(1.0)
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app = UniAssistGUI()

    app.mainloop()