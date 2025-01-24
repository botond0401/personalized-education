import matplotlib.pyplot as plt


def plot_auc_loss(list_train_loss, list_train_auc, list_val_loss, list_val_auc, num_epochs):
    epochs = range(1, num_epochs + 1)

    # Plot Training and Validation Loss
    plt.figure(figsize=(12, 5))

    # Add a supertitle
    plt.suptitle('With features Ease, Time spent and Hint used', fontsize=16)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, list_train_loss, label='Training Loss', color='b', linestyle='-', marker='o')
    plt.plot(epochs, list_val_loss, label='Validation Loss', color='r', linestyle='-', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Training and Validation AUC
    plt.subplot(1, 2, 2)
    plt.plot(epochs, list_train_auc, label='Training AUC', color='b', linestyle='-', marker='o')
    plt.plot(epochs, list_val_auc, label='Validation AUC', color='r', linestyle='-', marker='o')
    plt.title('Training and Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()

    # Display the plots
    plt.tight_layout()
    plt.show()
    