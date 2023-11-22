import torch
from utils import *
from dataset import TrainDataset, TestDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.functional import relu, softmax, max_pool2d

np.random.seed(2024)
torch.manual_seed(2024)
torch.cuda.memory_summary(device=None, abbreviated=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

class linear_model(nn.Module):
    def __init__(self, num_classes, input_shape=24576):
        super(linear_model, self).__init__()
        self.fc1 = nn.Linear(input_shape, num_classes, bias=True)
    def forward(self,x):
        logits = self.fc1(x)
        return logits, softmax(logits,dim=1)

train_dataset = TrainDataset(path='SHTech/train_1000_0.pt')
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TestDataset(path='SHTech/test_100.pt')
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


model = linear_model(num_classes=2, input_shape=1536).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

num_epochs = 20  # Set the number of epochs
for epoch in range(num_epochs):
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs,_ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():  # Disable gradient computation
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, _ = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        print(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(labels)
        print(correct)

accuracy = 100 * correct / total
print(accuracy)







