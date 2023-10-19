import tkinter as tk
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Загрузка модели и токенизатора
model = GPT2LMHeadModel.from_pretrained("./jokes_model")
tokenizer = GPT2Tokenizer.from_pretrained("./jokes_model")
tokenizer.pad_token = tokenizer.eos_token

class JokeGeneratorApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Joke Generator")
        self.geometry("500x400")

        self.input_text = tk.Text(self, height=5, width=50)
        self.input_text.pack(pady=10)

        self.generate_button = tk.Button(self, text="Generate Joke", command=self.generate_joke)
        self.generate_button.pack(pady=10)

        self.output_text = tk.Text(self, height=10, width=50, state=tk.DISABLED)
        self.output_text.pack(pady=10)

    def generate_joke(self):
        input_text = self.input_text.get("1.0", "end-1c")
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        output = model.generate(input_ids, max_length=100, num_beams=5, temperature=0.7)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, generated_text)
        self.output_text.config(state=tk.DISABLED)

def main():
    app = JokeGeneratorApp()
    app.mainloop()

if __name__ == "__main__":
    main()
