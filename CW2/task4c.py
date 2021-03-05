import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import cross_entropy_loss, one_hot_encode, pre_process_images, SoftmaxModel, gradient_approximation_test
from task2 import SoftmaxTrainer, calculate_accuracy

if __name__ == "__main__":
    # Simple test on one-hot encoding
    # Y = np.zeros((1, 1), dtype=int)
    # Y[0, 0] = 3
    # Y = one_hot_encode(Y, 10)
    # assert Y[0, 3] == 1 and Y.sum() == 1, \
    #     f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    # Modify your network here
    neurons_per_layer = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10]
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    momentum_gamma = .9
    learning_rate = 0.02
    batch_size = 32
    num_epochs = 50
    shuffle_dataset = True
    early_stopping = True
    
    # Load dataset
    X_val = pre_process_images(X_val)
    Y_val = one_hot_encode(Y_val, 10)
    
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val, early_stopping,
    )
    train_history, val_history = trainer.train(num_epochs)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)
        
    
    # Plot loss and accuracy (task 4)
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([2.2, 2.55])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.ylim([0.8, 0.85])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task4d_64_10times_10.png")

    #gradient_approximation_test(model, X_train, Y_train)
