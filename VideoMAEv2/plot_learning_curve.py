import re

import matplotlib.pyplot as plt

# Path to your log file
log_file = "./work_dir/res.out"

# Pattern for training loss at each epoch (adjust according to your log)
train_loss_pattern = re.compile(r"Epoch: \[(\d+)\].*loss: ([\d\.]+) \(([\d\.]+)\)")

# Pattern for validation loss at each epoch
val_loss_pattern = re.compile(r"\* Acc@1 [\d\.]+ Acc@5 [\d\.]+ loss ([\d\.]+)")

# Pattern for test/top-1
test_acc_pattern = re.compile(
    r"Accuracy of the network on the (\d+) test videos: Top-1: ([\d\.]+)%"
)


train_epochs = []
train_losses = []

val_epochs = []
val_losses = []

epoch = -1

with open(log_file, "r") as f:
    for line_num, line in enumerate(f):
        # Training loss (per epoch, last value only to avoid too much density)
        train_match = train_loss_pattern.search(line)
        if train_match:
            epoch = int(train_match.group(1))
            avg_loss = float(train_match.group(3))
            if len(train_epochs) == 0 or train_epochs[-1] != epoch:
                train_epochs.append(epoch)
                train_losses.append(avg_loss)

        # Validation loss (after epoch)
        val_match = val_loss_pattern.search(line)
        if val_match and epoch != -1:
            val_loss = float(val_match.group(1))
            val_epochs.append(epoch)
            val_losses.append(val_loss)

# Plotting learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_epochs, train_losses, label="Training loss")
plt.plot(val_epochs, val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./work_dir/learning_curve.png")
plt.show()
