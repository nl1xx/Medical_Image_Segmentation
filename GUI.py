import numpy as np
import torch
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from torchvision import transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

model = torch.load('./model/ResUNet.pt')

class MedicalSegmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("医学图像分割")

        self.image_path = None
        self.image_label = None
        self.result_label = None

        # 创建按钮和标签
        self.load_button = ttk.Button(root, text="加载图像", command=self.load_image)
        self.load_button.pack(pady=10)

        self.segment_button = ttk.Button(root, text="运行分割", command=self.run_segmentation)
        self.segment_button.pack(pady=10)

        self.image_label = ttk.Label(root, text="原始图像")
        self.image_label.pack(pady=10)

        self.result_label = ttk.Label(root, text="分割结果")
        self.result_label.pack(pady=10)

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            image = Image.open(self.image_path).convert('L')
            image = image.resize((256, 256))
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

    def run_segmentation(self):
        if self.image_path:
            image = Image.open(self.image_path).convert('L')
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image)
                output = torch.sigmoid(output).cpu().numpy()[0, 0] * 255
                output = output.astype(np.uint8)

            # 将numpy数组转换为图像对象
            result_image = Image.fromarray(output)
            result_photo = ImageTk.PhotoImage(result_image)
            self.result_label.config(image=result_photo)
            self.result_label.image = result_photo
        else:
            messagebox.showerror("错误", "请先加载图像！")

if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalSegmentationGUI(root)
    root.mainloop()
