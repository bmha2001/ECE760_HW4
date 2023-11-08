import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define hyperparameters
input_size = 28 * 28  # MNIST input size
hidden_size = 128    # Size of the hidden layer
output_size = 10     # Number of classes (0-9)
learning_rate = 0.1
num_epochs = 10

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Define model parameters (weights and biases)
W1 = torch.randn(input_size, hidden_size)
b1 = torch.zeros(hidden_size)
W2 = torch.randn(hidden_size, output_size)
b2 = torch.zeros(output_size)


# Load the MNIST test dataset
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Function to evaluate the model on the test set
def evaluate_model():
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, input_size)
            z1 = torch.matmul(images, W1) + b1
            a1 = torch.sigmoid(z1)
            z2 = torch.matmul(a1, W2) + b2
            a2 = torch.softmax(z2, dim=1)
            _, predicted = torch.max(a2, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Lists to store the learning curve data
learning_curve = []
test_errors = []

# Training loop
for epoch in range(num_epochs):
    total_loss = 0

    for images, labels in train_loader:
        # Flatten the input images
        images = images.view(-1, input_size)

        # Forward pass
        z1 = torch.matmul(images, W1) + b1
        a1 = torch.sigmoid(z1)
        z2 = torch.matmul(a1, W2) + b2
        a2 = torch.softmax(z2, dim=1)

        # Compute loss
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=output_size).float()
        loss = -torch.sum(one_hot_labels * torch.log(a2 + 1e-8))  # Adding a small epsilon to avoid log(0)

        total_loss += loss.item()

        # Backpropagation
        dz2 = a2 - one_hot_labels
        dW2 = torch.matmul(a1.t(), dz2)
        db2 = torch.sum(dz2, dim=0)
        dz1 = torch.matmul(dz2, W2.t()) * a1 * (1 - a1)
        dW1 = torch.matmul(images.t(), dz1)
        db1 = torch.sum(dz1, dim=0)

        # Parameter updates
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    # Print the average loss for this epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader)}')


    # Calculate test accuracy and store it
    test_accuracy = evaluate_model()
    test_errors.append(100 - test_accuracy)  # Calculate test error
    learning_curve.append(total_loss / len(train_loader))

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {learning_curve[-1]}, Test Accuracy: {test_accuracy}%')

# Plot the learning curve
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), learning_curve, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot the test error curve
plt.subplot(1, 2, 2)
plt.plot(range(num_epochs), test_errors, label='Test Error')
plt.xlabel('Epoch')
plt.ylabel('Error Rate (%)')
plt.legend()
plt.show()
