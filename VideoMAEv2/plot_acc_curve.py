import re

import matplotlib.pyplot as plt

train_loss_pattern = re.compile(r"Averaged stats: .*loss: [\d\.]+ \(([\d\.]+)\)")

val_acc_loss_pattern = re.compile(r"\* Acc@1 ([\d\.]+) Acc@5 ([\d\.]+) loss ([\d\.]+)")
val_acc_summary_pattern = re.compile(
    r"Accuracy of the network on the [\d]+ val images: ([\d\.]+)%"
)
train_acc1_pattern = re.compile(
    r"Averaged stats: .*acc1: ([\d\.]+) \(([\d\.]+)\)"  # Not seen in your log
)

train_epochs = []
train_losses = []
train_accs = []
val_epochs = []
val_accs = []
val_losses = []

curr_epoch = -1
with open("./work_dir/res.out") as f:
    for line in f:
        # Training loss per epoch
        m = train_loss_pattern.search(line)
        if m:
            curr_epoch += 1
            train_epochs.append(curr_epoch)
            train_losses.append(float(m.group(1)))
        # Training accuracy if in 'Averaged stats:' (not present in your sample log)
        acc_m = train_acc1_pattern.search(line)
        if acc_m:
            train_accs.append(float(acc_m.group(2)))  # average accuracy
        # Val block with * Acc@1
        v = val_acc_loss_pattern.search(line)
        if v and curr_epoch >= 0:
            val_epochs.append(curr_epoch)
            val_accs.append(float(v.group(1)))
            val_losses.append(float(v.group(3)))
        # or using summary after Val "Accuracy of the network on the 271 val images: 96.31%"
        va = val_acc_summary_pattern.search(line)
        if va:
            if val_epochs and val_epochs[-1] == curr_epoch:
                continue  # skip duplicate epoch
            val_epochs.append(curr_epoch)
            val_accs.append(float(va.group(1)))

plt.figure(figsize=(12, 6))
if train_accs:
    plt.plot(train_epochs[: len(train_accs)], train_accs, "o-", label="Train Acc@1")
plt.plot(val_epochs, val_accs, "s-", label="Val Acc@1")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("./work_dir/accuracy_curve.png")
plt.show()
