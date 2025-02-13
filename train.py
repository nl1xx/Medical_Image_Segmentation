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


train_dataset = Dataset(root, mode="train", fold=fold, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = Dataset(root, mode="val", fold=fold, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = ResUNet(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train():
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for (inputs, masks), file_name in train_loader:
            inputs, masks = inputs.to(device), masks.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, masks.argmax(dim=1))  # masks是one-hot编码，需要取argmax

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (inputs, masks), file_name in val_loader:
                inputs, masks = inputs.to(device), masks.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, masks.argmax(dim=1))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

    # 保存模型权重
    # torch.save(model.state_dict(), f"ResUNet_fold{fold}.pth")
    # print("Model saved.")

if __name__ == "__main__":
    train()
