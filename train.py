import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataload import Dataset
from model import ResUNet


torch.manual_seed(42)
torch.cuda.manual_seed(42)

num_classes = 3  # 分类数
batch_size = 4
learning_rate = 1e-4
num_epochs = 50
fold = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = "./datasets/Bladder/raw_data"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

model = ResUNet(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train():
    train_loss = []
    eval_loss = []
    for epoch in range(num_epochs):
        print("Epoch {}".format(epoch + 1))
        print("--------------------------")
        running_loss = 0.0

        for i in range(fold):
            train_dataset = Dataset(root, mode="train", fold=i + 1, transform=transform)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            val_dataset = Dataset(root, mode="val", fold=i + 1, transform=transform)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model.train()
            print("Train folk {}".format(i + 1))
            for (inputs, masks), file_name in train_loader:
                inputs, masks = inputs.to(device), masks.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, masks.argmax(dim=1))  # masks是one-hot编码，需要取argmax

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            train_loss.append(epoch_loss)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

            model.eval()
            print("Eval folk {}".format(i+1))
            val_loss = 0.0
            with torch.no_grad():
                for (inputs, masks), file_name in val_loader:
                    inputs, masks = inputs.to(device), masks.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, masks.argmax(dim=1))
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            eval_loss.append(val_loss)
            print(f"Validation Loss: {val_loss:.4f}")

    # 保存模型权重
    save_path = "./model"
    if not os.path.exists(save_path):
        os.makedirs('model')
    # 保存模型权重
    # torch.save(model.state_dict(), f"ResUNet.pth")
    # 保存整个模型
    torch.save(model, f"./model/ResUNet.pt")

    return train_loss, eval_loss


def draw(train_loss, eval_loss, epoch=50):
    x = []
    for i in range(epoch):
        x.append(i + 1)
    merge_train_loss = []
    merge_eval_loss = []
    for index in range(0, len(train_loss), 5):
        merge_train_loss.append(sum(train_loss[index:index + 5]))
        merge_eval_loss.append(sum(eval_loss[index:index+5]))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(x, merge_train_loss, 'b*--', alpha=0.5, linewidth=1, label='train loss')
    plt.plot(x, merge_eval_loss, 'rs--', alpha=0.5, linewidth=1, label='eval loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    train_loss, eval_loss = train()
    draw(train_loss, eval_loss)
