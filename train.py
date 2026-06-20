import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
from data_loader import get_dataloader

# 交通标志名称映射（58类）
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
CLASS_NUM = len(TSRD_CLASSES)


# 优化后的CNN模型
class TrafficNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, CLASS_NUM)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)


def train_model():
    # 自动检测GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"✅ GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB\n")

    train_loader, val_loader, test_loader = get_dataloader(batch_size=64)
    model = TrafficNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 每20轮学习率减半

    best_val_acc = 0.0
    # 记录曲线数据
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    # 100轮训练
    for epoch in range(100):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # 计算指标
        train_loss_avg = train_loss / train_total
        val_loss_avg = val_loss / val_total
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        # 记录历史
        train_loss_history.append(train_loss_avg)
        val_loss_history.append(val_loss_avg)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # 更新学习率
        scheduler.step()

        # 打印结果
        print(f"【轮次 {epoch + 1}/100】")
        print(f"训练损失: {train_loss_avg:.4f} | 训练准确率: {train_acc:.2f}%")
        print(f"验证损失: {val_loss_avg:.4f} | 验证准确率: {val_acc:.2f}%")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}\n")

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_traffic_model.pth")
            print(f"🏆 保存最优模型，验证准确率: {best_val_acc:.2f}%\n")

    print("=" * 60)
    print(f"✅ 训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    print("=" * 60)

    # 绘制损失曲线
    plt.figure(figsize=(12, 7))
    plt.plot(train_loss_history, label="训练损失", linewidth=2, color="#1f77b4")
    plt.plot(val_loss_history, label="验证损失", linewidth=2, color="#ff7f0e")
    plt.xlabel("训练轮次", fontsize=12)
    plt.ylabel("损失值", fontsize=12)
    plt.title("训练与验证损失曲线（100轮）", fontsize=14, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig("loss_curve_100.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 绘制准确率曲线
    plt.figure(figsize=(12, 7))
    plt.plot(train_acc_history, label="训练准确率", linewidth=2, color="#1f77b4")
    plt.plot(val_acc_history, label="验证准确率", linewidth=2, color="#ff7f0e")
    plt.xlabel("训练轮次", fontsize=12)
    plt.ylabel("准确率 (%)", fontsize=12)
    plt.title("训练与验证准确率曲线（100轮）", fontsize=14, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig("accuracy_curve_100.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n🔍 开始在独立测试集上评估最优模型...")
    # 加载最优模型
    model.load_state_dict(torch.load("best_traffic_model.pth", map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 计算测试集准确率
    test_acc = (all_preds == all_labels).mean() * 100
    print(f"📊 测试集最终准确率: {test_acc:.2f}%")

    # 绘制PR曲线和混淆矩阵
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

    # PR曲线（展示前12个类别，避免拥挤）
    for i in range(min(12, CLASS_NUM)):
        precision, recall, _ = precision_recall_curve(all_labels == i, all_probs[:, i])
        ap = average_precision_score(all_labels == i, all_probs[:, i])
        ax1.plot(recall, precision, linewidth=2, label=f"{TSRD_CLASSES[i]} (AP={ap:.3f})")

    ax1.set_xlabel("召回率", fontsize=12)
    ax1.set_ylabel("精确率", fontsize=12)
    ax1.set_title("精确率-召回率曲线（前12类）", fontsize=14, pad=20)
    ax1.legend(fontsize=10, loc="lower left")
    ax1.grid(True, alpha=0.3)

    # 混淆矩阵（展示前15个类别）
    cm = confusion_matrix(all_labels, all_preds, labels=range(min(15, CLASS_NUM)))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[TSRD_CLASSES[i] for i in range(min(15, CLASS_NUM))]
    )
    disp.plot(ax=ax2, cmap="Blues", xticks_rotation="vertical", values_format="d")
    ax2.set_title("混淆矩阵（前15类）", fontsize=14, pad=20)
    ax2.tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    plt.savefig("test_evaluation_100.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n✅ 所有评估完成！")
    print("📁 生成的文件：")
    print("  - best_traffic_model.pth（最优模型权重）")
    print("  - loss_curve_100.png（损失曲线）")
    print("  - accuracy_curve_100.png（准确率曲线）")
    print("  - test_evaluation_100.png（PR曲线+混淆矩阵）")


if __name__ == "__main__":
    train_model()