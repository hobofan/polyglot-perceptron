import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F


class Multilayer(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Multilayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 10)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, self.input_size)
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        output = F.log_softmax(output, dim=1)
        return output


transform_mnist = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=(0), std=(1))]
)

batch_size = 20

mnist_data_train = torch.utils.data.random_split(
    torchvision.datasets.MNIST(
        "data", train=True, transform=transform_mnist, download=True
    ),
    [6000, 54000],
)[0]
data_loader_train = torch.utils.data.DataLoader(
    mnist_data_train, batch_size=batch_size, shuffle=True
)

mnist_data_test = torchvision.datasets.MNIST(
    "data", train=False, transform=transform_mnist, download=True
)
data_loader_test = torch.utils.data.DataLoader(
    mnist_data_test, batch_size=batch_size, shuffle=True
)

model = Multilayer(28 * 28, 100)
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 20
for epoch in range(epochs):
    model.train()
    for batch_idx, sample_batched in enumerate(data_loader_train):
        optimizer.zero_grad()
        # TODO: try to destructure sample_batched tuple
        # Forward pass
        y_pred = model(sample_batched[0])
        # Compute Loss
        loss = criterion(y_pred.squeeze(), sample_batched[1])

        print("Trained batch {} of epoch {}".format(batch_idx, epoch))
        # Backward pass
        loss.backward()
        optimizer.step()

    loss_sum = 0
    num_correct = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(data_loader_test):
            y_pred = model(sample_batched[0])
            loss_sum += criterion(y_pred.squeeze(), sample_batched[1])

            y_pred = y_pred.argmax(dim=1, keepdim=True)
            num_correct += y_pred.eq(sample_batched[1].view_as(y_pred)).sum()

    print("Test loss after training epoch", epoch, " - ", loss_sum)
    print("{} out of {}".format(num_correct, len(mnist_data_test)))
