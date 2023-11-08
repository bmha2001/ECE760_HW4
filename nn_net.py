import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Hyperparameters
input_size = 28 * 28
hidden_size = 128
output_size = 10
learning_rate = 0.01
num_epochs = 10
batch_size = 64

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define the neural network
model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
)

# Initialize weights to zeros
for param in model.parameters():
    param.data.fill_(0)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Lists to store the learning curve data
learning_curve_zeros = []
test_errors_zeros = []

# Training loop
for epoch in range(num_epochs):
    total_loss = 0

    for batch_images, batch_labels in train_loader:
        # Flatten the images
        batch_images = batch_images.view(-1, input_size)

        # Forward pass
        outputs = model(batch_images)

        # Compute loss
        loss = criterion(outputs, batch_labels)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Calculate test accuracy and store it
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images = batch_images.view(-1, input_size)
            outputs = model(batch_images)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    test_accuracy = 100 * correct / total
    test_errors_zeros.append(100 - test_accuracy)  # Calculate test error
    learning_curve_zeros.append(total_loss / len(train_loader))

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {learning_curve_zeros[-1]}, Test Accuracy: {test_accuracy}%')

# Plot the learning curve
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), learning_curve_zeros, label='Training Loss (Zeros)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot the test error curve
plt.subplot(1, 2, 2)
plt.plot(range(num_epochs), test_errors_zeros, label='Test Error (Zeros)')
plt.xlabel('Epoch')
plt.ylabel('Error Rate (%)')
plt.legend()
plt.show()
