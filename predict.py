import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import os

# ===================== 模型和配置 =====================
# 交通标志名称映射
TSRD_CLASSES = {
    0: "限速5km/h", 1: "限速15km/h", 2: "限速30km/h", 3: "限速40km/h", 4: "限速50km/h",
    5: "限速60km/h", 6: "限速70km/h", 7: "限速80km/h", 8: "禁止直行与左转", 9: "禁止直行与右转",
    10: "禁止直行", 11: "禁止左转", 12: "禁止左转与右转", 13: "禁止右转", 14: "禁止超车",
    15: "禁止掉头", 16: "禁止机动车通行", 17: "禁止鸣笛", 18: "解除限速40km/h", 19: "解除限速50km/h",
    20: "直行和右转", 21: "直行", 22: "左转", 23: "左转和右转", 24: "右转",
    25: "靠左侧道路行驶", 26: "靠右侧道路行驶", 27: "环岛行驶", 28: "机动车通行", 29: "鸣喇叭",
    30: "非机动车行驶", 31: "允许掉头", 32: "左右绕行", 33: "注意信号灯", 34: "注意危险",
    35: "注意行人", 36: "注意非机动车", 37: "注意儿童", 38: "向右急转弯", 39: "向左急转弯",
    40: "下陡坡", 41: "上陡坡", 42: "注意慢行", 43: "注意左侧交叉路口", 44: "注意右侧交叉路口",
    45: "注意村庄", 46: "反向转弯", 47: "无人看守铁路道口", 48: "施工路段", 49: "连续弯路",
    50: "有人看守铁路道口", 51: "事故易发路段", 52: "停车让行", 53: "禁止通行", 54: "禁止停车",
    55: "禁止驶入", 56: "减速让行", 57: "停车检查"
}


# 模型结构
class TrafficNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )

        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 2 * 2, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, len(TSRD_CLASSES))
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)


# 图像预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 全局变量
model = None
device = None
current_image = None


# ===================== 模型加载 =====================
def load_model():
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "best_traffic_model.pth")

    try:
        model = TrafficNet()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return True
    except Exception as e:
        messagebox.showerror("错误", f"模型加载失败：{str(e)}")
        return False


# ===================== 识别函数 =====================
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probability, 1)
        predicted_idx = predicted_idx.item()
        confidence = confidence.item() * 100

    return TSRD_CLASSES[predicted_idx], confidence


# ===================== GUI函数 =====================
def select_image():
    global current_image
    # 打开文件选择对话框
    file_path = filedialog.askopenfilename(
        title="选择交通标志图片",
        filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
    )

    if not file_path:
        return

    try:
        # 加载并显示图片
        original_image = Image.open(file_path)
        # 缩放图片以适应显示区域（保持比例）
        max_size = 400
        ratio = min(max_size / original_image.width, max_size / original_image.height)
        new_size = (int(original_image.width * ratio), int(original_image.height * ratio))
        resized_image = original_image.resize(new_size, Image.Resampling.LANCZOS)

        current_image = ImageTk.PhotoImage(resized_image)
        image_label.config(image=current_image)
        image_label.image = current_image

        # 进行识别
        sign_name, confidence = predict_image(file_path)
        result_label.config(text=f"识别结果：{sign_name}\n置信度：{confidence:.2f}%")

    except Exception as e:
        messagebox.showerror("错误", f"图片处理失败：{str(e)}")


# ===================== 主窗口 =====================
if __name__ == "__main__":
    # 创建主窗口
    root = tk.Tk()
    root.title("交通标志识别系统")
    root.geometry("800x600")
    root.resizable(True, True)

    # 加载模型
    if not load_model():
        root.destroy()
        exit()

    # 顶部标题
    title_label = tk.Label(root, text="🚗 交通标志识别系统", font=("微软雅黑", 20, "bold"), pady=20)
    title_label.pack()

    # 选择图片按钮
    select_btn = tk.Button(
        root,
        text="📁 选择图片",
        font=("微软雅黑", 14),
        command=select_image,
        width=15,
        height=2,
        bg="#4CAF50",
        fg="white"
    )
    select_btn.pack(pady=10)

    # 图片显示区域
    image_frame = tk.Frame(root, width=400, height=400, bd=2, relief=tk.SUNKEN)
    image_frame.pack(pady=10)
    image_frame.pack_propagate(False)

    image_label = tk.Label(image_frame, text="请选择一张交通标志图片", font=("微软雅黑", 12))
    image_label.pack(expand=True)

    # 结果显示区域
    result_label = tk.Label(
        root,
        text="识别结果将显示在这里",
        font=("微软雅黑", 14),
        pady=20
    )
    result_label.pack()

    # 底部状态栏
    status_label = tk.Label(
        root,
        text=f"运行设备：{'GPU (NVIDIA RTX 4070)' if torch.cuda.is_available() else 'CPU'}",
        font=("微软雅黑", 10),
        fg="gray"
    )
    status_label.pack(side=tk.BOTTOM, pady=10)

    # 运行主循环
    root.mainloop()