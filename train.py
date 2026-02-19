from torch import nn, no_grad, optim

from dataset_loader import get_dataloader
from model import SmolCNN


def train(model, train_dataloader, loss_fn, optimizer, epochs, val_dataloader):
    for epoch in range(epochs):
        running_loss = 0.0
        print(f"Epoch {epoch + 1}")
        for idx, (images, labels) in enumerate(train_dataloader):
            print(f"\rBatch {idx + 1}/{len(train_dataloader)}", end="")
            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"\nLoss: {running_loss / len(train_dataloader)}")

        with no_grad():
            tp = 0
            for image, label in val_dataloader:
                predictions = model(image)
                pred_class = predictions.argmax(dim=1)
                tp += (pred_class == label).sum().item()
            val_accuracy = tp / len(val_dataloader.dataset)
            print(f"Validation Accuracy: {val_accuracy}\n")


if __name__ == "__main__":
    trainer_loader = get_dataloader("train", batch_size=32)
    val_loader = get_dataloader("val", batch_size=1)
    model = SmolCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, trainer_loader, loss_fn, optimizer, 30, val_loader)
