import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
from src.app_voice2text import AppSpeechRecognition
import time
import librosa
from data.icons import icon_to_image

class SpeechRecognitionApp:
    def __init__(self, root, speech_rec):
        self.root = root
        self.speech_rec = speech_rec
        self.recording = False
        self.scale_factor = 1.0  # Default scale factor
        # Set up the UI
        self.root.title("Speech Recognition")
        self.root.geometry("400x300")
        self.app_icon = icon_to_image("microphone", fill="#e2e2e9", scale_to_width=int(25*self.scale_factor))
        self.app_icon_red = icon_to_image("microphone", fill="#ff0000", scale_to_width=int(25*self.scale_factor))
        self.root.iconphoto(True, self.app_icon)

        
        self.mic_button = ttk.Button(
            root, 
            image=self.app_icon,
            command=self.toggle_recording,
            style='TButton'
        )
        self.mic_button.pack(pady=5)
        
        # Status label
        self.status_label = ttk.Label(root, text="Ready to record", font=('Arial', 8))
        self.status_label.pack()
        
        
        
        # Transcription display
        self.result_frame = ttk.LabelFrame(root, text="Transcription", padding=10)
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.result_text = tk.Text(
            self.result_frame, 
            height=5, 
            wrap=tk.WORD, 
            font=('Arial', 6),
            state=tk.DISABLED
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('TButton', font=('Arial', 8))
        self.style.configure('Red.TButton', foreground='red')
        
    def toggle_recording(self):
        if not self.recording:
            # Start recording
            self.recording = True
            # self.mic_button.configure(style='Red.TButton')
            self.mic_button.config(image=self.app_icon_red, text="")
            self.status_label.config(text="Recording...")
            
            # Start recording in a separate thread
            self.record_thread = threading.Thread(
                target=self.record_and_transcribe,
                daemon=True
            )
            self.record_thread.start()
        else:
             # Stop recording
            self.status_label.config(text="Recording stopped, processing...")
            self.root.update()
            time.sleep(0.1)  # Small delay to ensure recording stops cleanly
            self.speech_rec.stop_recording()
            self.recording = False
            # self.mic_button.configure(style='TButton')
            self.mic_button.config(image=self.app_icon)
            
            # Get transcription
            try:
                # Note: We don't need duration parameter here since we already recorded
                audio, sr = librosa.load(self.speech_rec.out_put_audio_path, sr=16000)
                inputs = self.speech_rec.processor(audio, sampling_rate=16000, return_tensors="pt")
                predicted_ids = self.speech_rec.model.generate(inputs["input_features"])
                transcription = self.speech_rec.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                self.show_results(transcription)
            except Exception as e:
                self.status_label.config(text=f"Error: {str(e)}")
                self.mic_button.configure(style='TButton')

    def record_and_transcribe(self):
        
        # Start recording and timer simultaneously
        if not self.speech_rec.start_recording():
            self.status_label.config(text="Failed to start recording")
            self.recording = False
            self.mic_button.configure(style='TButton')
            return
        
        # Record for the specified duration
        start_time = time.time()
        while self.recording:
            elapsed = time.time() - start_time
            elapsed = int(elapsed)
            self.status_label.config(text=f"Recording {elapsed}s")
            self.root.update()
            time.sleep(0.1)  # Small sleep to prevent CPU overuse
    
    def show_results(self, transcription):
        self.status_label.config(text="Process complete")
        
        # Clear the existing result frame and recreate it with more features
        self.result_frame.destroy()
        self.result_frame = ttk.LabelFrame(self.root, text="Transcription", padding=10)
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Add scrollbar to the text widget
        scrollbar = ttk.Scrollbar(self.result_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Make the text widget editable
        self.result_text = tk.Text(
            self.result_frame, 
            height=5, 
            wrap=tk.WORD, 
            font=('Arial', 14),
            yscrollcommand=scrollbar.set,
            padx=5,
            pady=5
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.result_text.yview)
        
        # Insert the transcription
        self.result_text.insert(tk.END, transcription)
        
        # Add a context menu for copy/paste etc.
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="Copy", command=self.copy_text)
        self.context_menu.add_command(label="Cut", command=self.cut_text)
        self.context_menu.add_command(label="Paste", command=self.paste_text)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Select All", command=self.select_all)
        self.context_menu.add_command(label="Save", command=self.save_to_file)
        
        # Bind right-click to show context menu
        self.result_text.bind("<Button-3>", self.show_context_menu)
        
        # Add a button frame at the bottom
        button_frame = ttk.Frame(self.result_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        

    def show_context_menu(self, event):
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def copy_text(self):
        self.root.clipboard_clear()
        text = self.result_text.get("sel.first", "sel.last")
        self.root.clipboard_append(text)

    def cut_text(self):
        self.copy_text()
        self.result_text.delete("sel.first", "sel.last")

    def paste_text(self):
        self.result_text.insert(tk.INSERT, self.root.clipboard_get())

    def select_all(self):
        self.result_text.tag_add(tk.SEL, "1.0", tk.END)
        self.result_text.mark_set(tk.INSERT, "1.0")
        self.result_text.see(tk.INSERT)
        return 'break'

    def clear_text(self):
        self.result_text.delete(1.0, tk.END)

    def save_to_file(self):
        from tkinter import filedialog
        text = self.result_text.get(1.0, tk.END)
        if not text.strip():
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                self.status_label.config(text=f"Saved to {file_path}")
            except Exception as e:
                self.status_label.config(text=f"Error saving file: {str(e)}")

if __name__ == "__main__":
    # Initialize speech recognition
    speech_rec = AppSpeechRecognition()
    root = tk.Tk()
    app = SpeechRecognitionApp(root, speech_rec)
    root.mainloop()