function [Theta1 Theta2] = trainNN(X, y, input_layer_size, hidden_layer_size, num_labels)



    %% ================ Part 6: Initializing Pameters ================
    %  In this part of the exercise, you will be starting to implment a two
    %  layer neural network that classifies digits. You will start by
    %  implementing a function to initialize the weights of the neural network
    %  (randInitializeWeights.m)

    fprintf('\nInitializing Neural Network Parameters ...\n')

    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

    % Unroll parameters
    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];



    %% =================== Part 8: Training NN ===================
    %  You have now implemented all the code necessary to train a neural 
    %  network. To train your neural network, we will now use "fmincg", which
    %  is a function which works similarly to "fminunc". Recall that these
    %  advanced optimizers are able to train our cost functions efficiently as
    %  long as we provide them with the gradient computations.
    %
    fprintf('\nTraining Neural Network... \n')

    %  After you have completed the assignment, change the MaxIter to a larger
    %  value to see how more training helps.
    options = optimset('MaxIter', 750);

    %  You should also try different values of lambda
    lambda = 1;

    % Create "short hand" for the cost function to be minimized
    costFunction = @(p) nnCostFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, X, y, lambda);

    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

    % Obtain Theta1 and Theta2 back from nn_params
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));

    fprintf('Training Complete.\n');




    %% ================= Part 10: Implement Predict =================
    %  After training the neural network, we would like to use it to predict
    %  the labels. You will now implement the "predict" function to use the
    %  neural network to predict the labels of the training set. This lets
    %  you compute the training set accuracy.

    pred = predict1(Theta1, Theta2, X);

    fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


