import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # TODO: Implement this function (task 3c)
    output = model.forward(X)
    prediction = output == np.amax(output, axis=1)[:, None]
    num_correct = np.count_nonzero(prediction == targets)
    accuracy = num_correct / targets.shape[0]
    return accuracy

class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # TODO: Implement this function (task 3b)
        output = self.model.forward(X_batch)
        grad = self.model.backward(X_batch, output, Y_batch)
        
        self.model.w -= self.learning_rate * grad
        loss = cross_entropy_loss(Y_batch, output)
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(Y_val, logits)

        accuracy_train = calculate_accuracy(
            X_train, Y_train, self.model)
        accuracy_val = calculate_accuracy(
            X_val, Y_val, self.model)
        return loss, accuracy_train, accuracy_val
# %%

if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True
    early_stopping = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.
# %%
    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val, early_stopping
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    # Plot loss
    plot5 = plt.figure(5)
    plt.ylim([0.2, .6])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()
# %% 
    # Plot accuracy
    plot6 = plt.figure(6)
    plt.ylim([0.93, .99])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()
# %%
    # Train a model with L2 regularization (task 4b)

    model1 = SoftmaxModel(l2_reg_lambda=0.0)
    trainer1 = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val, early_stopping
    )
    train_history_reg01, val_history_reg01 = trainer1.train(num_epochs)

    model2 = SoftmaxModel(l2_reg_lambda=1.0)
    trainer2 = SoftmaxTrainer(
        model2, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val, early_stopping
    )
    train_history_reg02, val_history_reg02 = trainer2.train(num_epochs)
    # You can finish the rest of task 4 below this point.


    # Plotting of softmax weights (Task 4b)
    images = []

    images1 = []
    for i in range(10):
        image1 = np.zeros((28 , 28))
        for j in range (28):
            for k in range(28):
                image1[j][k] = model1.w[:,i][k + j*28]
        images1.append(image1)
   
    images2 = []
    for i in range(10):
        image2 = np.zeros((28 , 28))
        for j in range (28):
            for k in range(28):
                image2[j][k] = model2.w[:,i][k + j*28]
        images2.append(image2)


    images1 = np.concatenate(images1, axis = 1)
    images2 = np.concatenate(images2, axis = 1)
    plt.imshow(images1)
    plt.imshow(images2)
    plt.imsave("task4b_softmax_weight_L0.png", images1, cmap="gray")
    plt.imsave("task4b_softmax_weight_L1.png", images2, cmap="gray")
    plt.show()

# %%
     # Task 4c) Train the softmax with several lambdas and plot accuracy in the
     # Same plot
    model3 = SoftmaxModel(l2_reg_lambda=1.0)
    
    trainer3 = SoftmaxTrainer(
        model3, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    
    
    train_history_reg03, val_history_reg03 = trainer3.train(num_epochs)
    plot8 = plt.figure(8)
    # plt.ylim([0.5, 1])
    # l2_lambdas = [1, .1, .01, .001]
    utils.plot_loss(train_history_reg03["accuracy"], "Training Accuracy, lambda = 1")
    utils.plot_loss(val_history_reg03["accuracy"], "Validation Accuracy, lambda = 1")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    # plt.savefig("task4c_l2_reg_accuracy.png")
    plt.legend()
    # plt.show()

   
    model4 = SoftmaxModel(l2_reg_lambda=0.1)
    trainer4 = SoftmaxTrainer(
        model4, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val, early_stopping
    )
   
    train_history_reg04, val_history_reg04 = trainer4.train(num_epochs)
    plot8 = plt.figure(8)
    # plt.ylim([0.5, 1])
    # l2_lambdas = [1, .1, .01, .001]
    utils.plot_loss(train_history_reg04["accuracy"], "Training Accuracy, lambda = 0.1")
    utils.plot_loss(val_history_reg04["accuracy"], "Validation Accuracy, lambda = 0.1")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    # plt.savefig("task4c_l2_reg_accuracy.png")
    plt.legend()
    # plt.show()

    
    model5 = SoftmaxModel(l2_reg_lambda=0.01)
    trainer5 = SoftmaxTrainer(
        model5, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val, early_stopping
    )
    
    train_history_reg05, val_history_reg05 = trainer5.train(num_epochs)
    plot8 = plt.figure(8)
    # plt.ylim([0.5, 1])
    # l2_lambdas = [1, .1, .01, .001]
    utils.plot_loss(train_history_reg05["accuracy"], "Training Accuracy, lambda = 0.01")
    utils.plot_loss(val_history_reg05["accuracy"], "Validation Accuracy, lambda = 0.01")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    # plt.savefig("task4c_l2_reg_accuracy.png")
    plt.legend()
    plt.show()

   
    model6 = SoftmaxModel(l2_reg_lambda=0.001)
    trainer6 = SoftmaxTrainer(
        model6, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val, early_stopping
    )
    train_history_reg06, val_history_reg06 = trainer6.train(num_epochs)

    # Plotting of accuracy for difference values of lambdas (task 4c)
    plot8 = plt.figure(8)
    # plt.ylim([0.5, 1])
    # 
    utils.plot_loss(train_history_reg06["accuracy"], "Training Accuracy, lambda = 0.001")
    utils.plot_loss(val_history_reg06["accuracy"], "Validation Accuracy, lambda = 0.001")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.legend()
    plt.show()
# %%
    # Task 4d - Plotting of the l2 norm for each weight x = lambda, y = norm
    l2_lambdas = [1, .1, .01, .001]
    plot9 = plt.figure(9)
    l2norm3 = (np.linalg.norm(model3.w, ord=None, axis=None, keepdims=False))
    l2norm4 = (np.linalg.norm(model4.w, ord=None, axis=None, keepdims=False))
    l2norm5 = (np.linalg.norm(model5.w, ord=None, axis=None, keepdims=False))
    l2norm6 = (np.linalg.norm(model6.w, ord=None, axis=None, keepdims=False))
    plt.plot(l2_lambdas, [l2norm3,l2norm4, l2norm5, l2norm6])
    # plt.plot(l2norm4)
    # plt.plot(l2norm5)
    # plt.plot(l2norm6)
    plt.xlabel("Lambda value")
    plt.ylabel("L2 Norm")
    plt.savefig("task4d_l2_reg_norms.png")
