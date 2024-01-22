

"""Imports:"""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import time


from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay

"""#**Prepare data preprocessing transformations:**"""
if __name__ == '__main__':

    path_train_data = "archive/Car_Brand_Logos/Train"
    path_test_data = "archive/Car_Brand_Logos/Test"


    transform_for_std_mean = transforms.Compose([transforms.Resize((64, 64)),transforms.ToTensor()])

    # Calculate mean and std of the training dataset
    train_dataset = ImageFolder(root=path_train_data, transform=transform_for_std_mean)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    mean = torch.zeros(3)
    std = torch.zeros(3)

    for inputs, _ in train_loader:
        mean += inputs.mean(dim=(0, 2, 3))
        std += inputs.std(dim=(0, 2, 3))

    mean /= len(train_loader)
    std /= len(train_loader)
    print("mean of the dataset is: ", mean)
    print("std of the dataset is: ", std)
    # Define data transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=15, scale=(0.9, 1.1)),  # Shear and zoom
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist())
    ])

    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist())
    ])

    """#Create the dataloaders and set the device (cuda/cpu)"""

    # Load the dataset
    # train_dataset = ImageFolder(root='/content/gdrive/MyDrive/karin/Car_Brand_Logos/Train', transform=transform)
    # test_dataset = ImageFolder(root='/content/gdrive/MyDrive/karin/Car_Brand_Logos/Test', transform=transform)
    train_dataset = ImageFolder(root=path_train_data, transform=transform)
    test_dataset = ImageFolder(root=path_test_data,  transform=transform_test)
    # Split train dataset into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=256,num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=256)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """#Plot some examples of the augmented data"""

    # Load a few images from the dataset
    num_images_to_display = 5
    data_loader = DataLoader(train_dataset, batch_size=num_images_to_display, shuffle=True)
    images, _ = next(iter(data_loader))

    # Denormalize the images
    denormalize = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2))
    ])

    denorm_images = []
    for i in range(num_images_to_display):
        denorm_image = denormalize(images[i])
        denorm_images.append(denorm_image)

    # Display the images
    fig, axes = plt.subplots(1, num_images_to_display, figsize=(15, 3))

    for i in range(num_images_to_display):
        ax = axes[i]
        ax.imshow(np.transpose(denorm_images[i], (1, 2, 0)))
        ax.axis('off')

    plt.show()

    """# Models"""

    class SimpleCNNModel(nn.Module):
        def __init__(self):
            super(SimpleCNNModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32 * 16 * 16, 128)
            self.fc2 = nn.Linear(128, 8)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 32 * 16 * 16)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x


    class SimpleCNNModelWithDropout(nn.Module):
        def __init__(self, dropout_rate=0.25):
            super(SimpleCNNModelWithDropout, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32 * 16 * 16, 128)
            self.fc2 = nn.Linear(128, 8)
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.dropout(x)
            x = self.pool(F.relu(self.conv2(x)))
            x = self.dropout(x)
            x = x.view(-1, 32 * 16 * 16)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x


    class SimpleCNNModelWithBN(nn.Module):
        def __init__(self):
            super(SimpleCNNModelWithBN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn3 = nn.BatchNorm1d(128)

            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32 * 16 * 16, 128)
            self.fc2 = nn.Linear(128, 8)

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = x.view(-1, 32 * 16 * 16)
            x = F.relu(self.bn3(self.fc1(x)))
            x = self.fc2(x)
            return x


    class SimpleCNNModelWithAll(nn.Module):
        def __init__(self, dropout_rate=0.25):
            super(SimpleCNNModelWithAll, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn3 = nn.BatchNorm1d(128)

            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32 * 16 * 16, 128)
            self.fc2 = nn.Linear(128, 8)
            self.dropout = nn.Dropout(dropout_rate)
            # self.l2_strength = l2_strength
            # self.register_buffer('l1_reg', torch.tensor(0.0))

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.dropout(x)
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.dropout(x)
            x = x.view(-1, 32 * 16 * 16)
            x = F.relu(self.bn3(self.fc1(x)))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    """# Train All Models"""

    def train_model(model, train_loader, val_loader, model_name, learning_rate=0.001, epochs=100, regularization=None,
                        regularization_strength=None):
          # Define loss and optimizer
          criterion = nn.CrossEntropyLoss()
          if regularization == None or regularization == "l1":
              optimizer = optim.Adam(model.parameters(), lr=learning_rate)
          elif regularization == "l2":
              optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization_strength)

          if regularization != None:
              print("Training the model: ", model_name, " with ", regularization, " regularization")
          else:
              print("Training the model: ", model_name)

          train_losses = []  # To store training losses for each epoch
          val_accuracies = []  # To store validation accuracies for each epoch
          val_losses = []
          best_val_acc = 0
          best_val_loss = 100
          best_acc_epoch = 0
          best_loss_epoch = 0
          model.to(device)
          # Training loop
          for epoch in range(epochs):
              model.train()
              epoch_train_loss = 0.0  # To accumulate training loss

              for batch_index, (inputs, labels) in enumerate(train_loader):
                  inputs, labels = inputs.to(device), labels.to(device)
                  optimizer.zero_grad()
                  outputs = model(inputs)

                  loss = criterion(outputs, labels)

                  if regularization == "l1":
                      l1_lambda = regularization_strength  # Adjust this value for the strength of L1 regularization
                      l1_regularization = torch.tensor(0.).to(device)
                      for param in model.parameters():
                          l1_regularization += torch.norm(param, p=1)
                      loss += l1_lambda * l1_regularization

                  loss.backward()
                  optimizer.step()

                  # Accumulate training loss for the epoch
                  epoch_train_loss += loss.item()

              # Validation
              model.eval()
              val_loss = 0.0
              correct = 0
              total = 0
              with torch.no_grad():
                  for inputs, labels in val_loader:
                      inputs, labels = inputs.to(device), labels.to(device)
                      outputs = model(inputs)
                      val_loss += criterion(outputs, labels).item()
                      _, predicted = outputs.max(1)
                      total += labels.size(0)
                      correct += predicted.eq(labels).sum().item()

              val_accuracy = 100.0 * correct / total
              average_train_loss = epoch_train_loss / len(train_loader)  # Calculate average training loss

              print(
                  f'Epoch [{epoch + 1}/{epochs}], Train Loss: {average_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

              if val_accuracy > best_val_acc:
                  best_val_acc = val_accuracy
                  # Save model weights
                  torch.save(model.state_dict(), "./" + model_name + "_weights.pth")
                  best_acc_epoch = epoch + 1

              # if val_loss < best_val_loss:
              #     best_val_loss = val_loss
              #     # Save model weights
              #     torch.save(model.state_dict(), "./" + model_name + "_weights.pth")
              #     best_loss_epoch = epoch + 1

              train_losses.append(average_train_loss)
              val_accuracies.append(val_accuracy)
              val_losses.append(val_loss)
          print("best weights model saved on: epoch ", best_acc_epoch)
          print("From the next epoch, overfit start to occur: epoch ", best_loss_epoch)

          return train_losses, val_accuracies, val_losses

    print("device: ", device)
    print(len(train_dataset))
    # Train each model
    model_names = ['cnn_baseline', 'cnn_bn', 'cnn_dropout', 'cnn_all']
    models = [SimpleCNNModel(), SimpleCNNModelWithBN(), SimpleCNNModelWithDropout(), SimpleCNNModelWithAll()]
    epochs = 200

    all_train_losses = []
    all_val_accuracies = []
    all_val_losses = []

    for model, model_name in zip(models, model_names):
        train_losses, val_accuracies, val_losses = train_model(model, train_loader, val_loader, model_name=model_name,
                                                                epochs=epochs)
        all_train_losses.append(train_losses)
        all_val_accuracies.append(val_accuracies)
        all_val_losses.append(val_losses)

    # L2 regularization
    model_name = "cnn_all_l2_regularization"
    model = SimpleCNNModelWithAll()
    model_names.append(model_name)

    train_losses_l2, val_accuracies_l2, val_losses = train_model(model, train_loader, val_loader, model_name=model_name,
                                                                  epochs=epochs, regularization="l2",
                                                                  regularization_strength=0.001)
    all_train_losses.append(train_losses_l2)
    all_val_accuracies.append(val_accuracies_l2)
    all_val_losses.append(val_losses)

    # L1 regularization
    model_name = "cnn_all_l1_regularization"
    model = SimpleCNNModelWithAll()
    model_names.append(model_name)

    train_losses_l1, val_accuracies_l1, val_losses = train_model(model, train_loader, val_loader, model_name=model_name,
                                                                  epochs=epochs, regularization="l1",
                                                                  regularization_strength=0.001)
    all_train_losses.append(train_losses_l1)
    all_val_accuracies.append(val_accuracies_l1)
    all_val_losses.append(val_losses)

    """# Train and Validation loss and accuracy graphs"""

    """# Training and Validations Graphs for the different models"""

    # Plot validation graphs
    plt.figure(figsize=(10, 6))
    for i, model_name in enumerate(model_names):
        if len(all_val_accuracies) > i:
            plt.plot(range(1, epochs + 1), all_val_accuracies[i], label=model_name + "_val")
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("validation_accuracy.png")

    # Plot validation loss graphs
    plt.figure(figsize=(10, 6))
    for i, model_name in enumerate(model_names):
        if len(all_val_accuracies) > i:
            plt.plot(range(1, epochs + 1), all_val_losses[i], label=model_name + "_val_loss")
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("validation_loss.png")


    # Plot training graphs
    plt.figure(figsize=(10, 6))
    for i, model_name in enumerate(model_names):
        if len(all_val_accuracies) > i:
            plt.plot(range(1, epochs + 1), all_train_losses[i], label=model_name + "_train")
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("training_loss.png")


    # Plot training graphs
    plt.figure(figsize=(10, 6))
    for i, model_name in enumerate(model_names):
        if len(all_val_accuracies) > i and "l1" not in model_name:
            plt.plot(range(1, epochs + 1), all_train_losses[i], label=model_name + "_train")
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("training_loss_without_l1.png")

    """# Load the final models"""

    """# Models Evaluation on the test data"""

    def load_model(model, model_name):
        print(model_name)
        model.load_state_dict(torch.load("./" + model_name + "_weights.pth"))
        model.eval()
        return model

    model_names = ['cnn_baseline', 'cnn_bn', 'cnn_dropout', 'cnn_all']
    models = [SimpleCNNModel(), SimpleCNNModelWithBN(), SimpleCNNModelWithDropout(), SimpleCNNModelWithAll()]
    # Load the trained models
    loaded_models = []

    model_name = "cnn_all_l1_regularization"
    model = SimpleCNNModelWithAll()
    model_names.append(model_name)
    models.append(model)
    model_name = "cnn_all_l2_regularization"
    model = SimpleCNNModelWithAll()
    model_names.append(model_name)
    models.append(model)

    for model, model_name in zip(models, model_names):
        model = load_model(model, model_name)
        loaded_models.append(model)

    """# Models evaluation:"""

    best_accuracy = 0
    best_model = ""
    for model, model_name in zip(loaded_models, model_names):
        model.to(device)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_accuracy = 100.0 * correct / total
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = model_name
        print(f'Model {model_name} - Test Accuracy: {test_accuracy:.2f}%')

    print(f'Best Model on test dataset {best_model} - with Test Accuracy: {best_accuracy:.2f}%')

    """# Additional Graph for the best model (Confusion Matrix)"""

    def evaluate_model(model, test_loader):
        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = 100.0 * correct / total
        return accuracy, all_labels, all_predictions

    model_name = "cnn_all_l2_regularization"
    model = SimpleCNNModelWithAll()
    model = load_model(model, model_name)
    # After loading the trained models
    model.to(device)
    # After loading the trained models
    # After loading the trained models
    test_accuracy, true_labels, predicted_labels = evaluate_model(model, test_loader)

    print(f'Model {model_name} - Test Accuracy: {test_accuracy:.2f}%')

    # Calculate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    print(f'Confusion Matrix for {model_name}:\n', cm)



    class_names = test_dataset.classes

    TP = cm.diagonal()
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    # Calculate recall (Sensitivity) for each class
    recall = TP / (TP + FN)

    # Calculate precision for each class
    precision = TP / (TP + FP)

    # Calculate F1-score for each class
    f1 = 2 * (precision * recall) / (precision + recall)
    # Create a DataFrame to organize the metrics
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Recall (Sensitivity)': recall,
        'Precision': precision,
        'F1-Score': f1
    })

    # Save the metrics table as an image (e.g., PNG)
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    plt.axis('off')  # Hide axes
    table = plt.table(cellText=metrics_df.values,
                      colLabels=metrics_df.columns,
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)  # Adjust font size as needed
    plt.title('Metrics Table')
    plt.tight_layout()

    # Save the table as an image (e.g., PNG)
    plt.savefig('metrics_table.png', bbox_inches='tight', dpi=300)
    plt.show()

    # Plot the confusion matrix using ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues,  values_format='d')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=8)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

    """# NOT PRACTICAL JUST FOR DEMONSTRATION
    Grid Search for all models, but due computional limitations it's not practical, so **the code in a comment.**
    """

    """# Grid Search Over Multiple models
    
    The code is Proof of Concept that I could do Grid search over multiple models.
    Due computational limits, the code is in comment.
    In practice I chose SimpleCNNModelWithAll with L2 regularization because is it combinning multiple strategies for that we learned in the course, dealing with overfit.
    """

    # # Define hyperparameter values for grid search
    # dropout_rates = [0.25, 0.5]
    # regularization_strengths = [0.001, 0.01]
    # learning_rates = [0.001, 0.01]
    # batch_sizes = [64, 128]
    # # train_sizes = [0.6, 0.7, 0.8]  # Proportion of the training dataset to use

    # model_names = ['cnn_baseline','cnn_bn','cnn_dropout',  'cnn_all','cnn_all_l2_regularization','cnn_all_l1_regularization']
    # models = [SimpleCNNModel,SimpleCNNModelWithBN, SimpleCNNModelWithDropout, SimpleCNNModelWithAll,SimpleCNNModelWithAll,SimpleCNNModelWithAll]

    # model_names = ['cnn_all_l1_regularization',  'cnn_all','cnn_all_l2_regularization','cnn_all_l1_regularization']
    # models = [SimpleCNNModelWithAll, SimpleCNNModelWithAll]

    # # Dictionary to map model names to their corresponding hyperparameters
    # model_hyperparams = {
    #     "cnn_baseline": {"dropout_rate": None, "regularization_strength": None, "learning_rate": learning_rates, "batch_size": batch_sizes},
    #     "cnn_bn": {"dropout_rate": None, "regularization_strength": None, "learning_rate": learning_rates, "batch_size": batch_sizes},
    #     "cnn_dropout": {"dropout_rate": dropout_rates, "regularization_strength": None, "learning_rate": learning_rates, "batch_size": batch_sizes},
    #     "cnn_all": {"dropout_rate": dropout_rates, "regularization_strength": None, "learning_rate": learning_rates, "batch_size": batch_sizes},
    #     "cnn_all_l2_regularization":{"dropout_rate": dropout_rates, "regularization_strength": regularization_strengths, "learning_rate": learning_rates, "batch_size": batch_sizes},
    #     "cnn_all_l1_regularization":{"dropout_rate": dropout_rates, "regularization_strength": regularization_strengths, "learning_rate": learning_rates, "batch_size": batch_sizes},
    # }

    # # Lists to store best hyperparameters and their corresponding performance metrics
    # best_hyperparams = []
    # best_val_accuracies = []

    # # Nested loop for grid search
    # for model, model_name in zip(models, model_names):
    #     hyperparams = model_hyperparams[model_name]
    #     best_accuracy = 0.0
    #     best_params = {}

    #     for dropout_rate in hyperparams["dropout_rate"] if hyperparams["dropout_rate"] else [None]:
    #         for regularization_strength in hyperparams["regularization_strength"] if hyperparams["regularization_strength"] else [None]:
    #             for learning_rate in hyperparams["learning_rate"]:
    #                 for batch_size in hyperparams["batch_size"]:
    #                           # Create train and validation loaders with the current dataset size
    #                           train_size_idx = int(train_size * len(train_dataset))
    #                           train_loader_subset = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #                           val_loader_subset = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    #                           # Create a new instance of the model with the current hyperparameters
    #                           if dropout_rate:
    #                               model_instance = model(dropout_rate=dropout_rate)
    #                               print(model_instance.dropout)
    #                           else:
    #                               model_instance = model()

    #                           if "l1" in model_name:
    #                               train_losses, val_accuracies, _ = train_model(model_instance, train_loader_subset, val_loader_subset, model_name=model_name, epochs=epochs, learning_rate=learning_rate,regularization="l1", regularization_strength=regularization_strength)
    #                           elif "l2" in model_name: # "l2"
    #                               train_losses, val_accuracies, _ = train_model(model_instance, train_loader_subset, val_loader_subset, model_name=model_name, epochs=epochs, learning_rate=learning_rate,regularization="l2", regularization_strength=regularization_strength)
    #                           else:
    #                               train_losses, val_accuracies, _ = train_model(model_instance, train_loader_subset, val_loader_subset, model_name=model_name, epochs=epochs, learning_rate=learning_rate)

    #                           final_val_accuracy = val_accuracies[-1]

    #                           if final_val_accuracy > best_accuracy:
    #                               best_accuracy = final_val_accuracy
    #                               best_params = {"dropout_rate": dropout_rate, "regularization_strength": regularization_strength, "learning_rate": learning_rate, "batch_size": batch_size, "train_size": train_size, "epochs": epochs}

    #     best_hyperparams.append(best_params)
    #     best_val_accuracies.append(best_accuracy)

    # # Print best hyperparameters and corresponding val accuracies
    # for model_name, best_params, best_accuracy in zip(model_names, best_hyperparams, best_val_accuracies):
    #     print(f"Best hyperparameters for {model_name}: {best_params}")
    #     print(f"Best validation accuracy: {best_accuracy:.2f}%")

    """# Optional Grid Search for the best model.
    It's practical to run it but gonna take sometime, so it's unnecesarry, I saved the best parameters from my previous running. read it on my report.
    
    """

    # """# Grid Search for the best model

    # """
    # dropout_rates = [0.25, 0.5]
    # regularization_strengths = [0.0001, 0.001]
    # learning_rates = [0.001, 0.01]
    # batch_sizes = [128, 256]
    # epochs = 30

    # # Model and model name for grid search
    # model = SimpleCNNModelWithAll
    # model_name = 'cnn_all_l2_regularization_grid_search'
    # # Lists to store best hyperparameters and their corresponding performance metrics
    # best_hyperparams = []
    # best_val_accuracy = 0.0

    # # Lists to store validation accuracies, validation losses, and training losses for each hyperparameter combination
    # all_val_accuracies = []
    # all_val_losses = []
    # all_train_losses = []
    # all_hyperparams = []
    # # Nested loop for grid search
    # for dropout_rate in dropout_rates:
    #     for regularization_strength in regularization_strengths:
    #         for learning_rate in learning_rates:
    #             for batch_size in batch_sizes:
    #                 all_hyperparams.append({"dropout_rate": dropout_rate, "regularization_strength": regularization_strength, "learning_rate": learning_rate, "batch_size": batch_size})
    #                 # Create train and validation loaders
    #                 train_loader_subset = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #                 val_loader_subset = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    #                 # Create a new instance of the model with the current hyperparameters
    #                 model_instance = model(dropout_rate=dropout_rate, l2_strength=regularization_strength)

    #                 # Train the model
    #                 train_losses, val_accuracies, val_losses = train_model(model_instance, train_loader_subset,
    #                                                                         val_loader_subset, model_name=model_name,
    #                                                                         epochs=epochs, learning_rate=learning_rate)
    #                 highest_val_accuracy = max(val_accuracies)

    #                 # Append metrics to lists
    #                 all_val_accuracies.append(val_accuracies)
    #                 all_val_losses.append(val_losses)
    #                 all_train_losses.append(train_losses)

    #                 if highest_val_accuracy > best_val_accuracy:
    #                     best_val_accuracy = highest_val_accuracy
    #                     best_hyperparams = {"dropout_rate": dropout_rate,
    #                                         "regularization_strength": regularization_strength,
    #                                         "learning_rate": learning_rate, "batch_size": batch_size}


    # # Print best hyperparameters and corresponding val accuracy
    # print(f"Best hyperparameters for {model_name}: {best_hyperparams}")
    # print(f"Highest validation accuracy: {best_val_accuracy:.2f}%")

    """# Early Stopping for best model"""

    def train_model_with_early_stopping(model, train_loader, val_loader, model_name, learning_rate=0.001, patience=70,
                                        epochs=100):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        print("Training the model:", model_name)

        train_losses = []
        val_accuracies = []

        best_val_accuracy = 0.0
        early_stopping_counter = 0
        best_epoch = 0
        model.to(device)

        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0.0

            for batch_index, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            model.eval()
            val_accuracy = evaluate_model(model, val_loader)

            average_train_loss = epoch_train_loss / len(train_loader)
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {average_train_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

            train_losses.append(average_train_loss)
            val_accuracies.append(val_accuracy)

            if val_accuracy > best_val_accuracy:
                torch.save(model.state_dict(), "./" + model_name + "_weights.pth")
                best_val_accuracy = val_accuracy
                early_stopping_counter = 0
                best_epoch = epoch
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= patience:
                print("Early stopping triggered on epoch: ", epoch)
                print("Best weights on epoch: ", best_epoch)
                break

        return train_losses, val_accuracies



    def evaluate_model(model, data_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        val_accuracy = 100.0 * correct / total
        return val_accuracy


    model = SimpleCNNModelWithAll()
    patience = 30
    train_losses, val_accuracies = train_model_with_early_stopping(model, train_loader, val_loader,
                                                                    model_name="cnn_early_stopping", epochs=200,patience=patience)

    print("Train Losses:", train_losses)
    print("Validation Accuracies:", val_accuracies)

    model = load_model(model, "cnn_early_stopping")
    model.to(device)

    print("early stopping test accuracy:",evaluate_model(model, test_loader))