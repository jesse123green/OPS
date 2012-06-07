function [J grad] = nnCostFunctionReg2(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
s2 = (1 + (hidden_layer_size * (input_layer_size + 1)));

Theta2 = reshape(nn_params(s2:s2+(hidden_layer_size + 1) * (hidden_layer_size) - 1), ...
                 (hidden_layer_size), (hidden_layer_size + 1));
s3 = s2 + (hidden_layer_size + 1) * (hidden_layer_size);     

Theta3 = reshape(nn_params(s3:end), ...
    num_labels, (hidden_layer_size + 1));
% Setup some useful variables

m = size(X, 1);




% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%% Part 1

A_1= [ones(m,1) X];   % dont forget to add bias unit column! should be (5000 x 401)
Z_2= A_1*Theta1';   % (5000 x 25)
A_2= [ones(size(Z_2,1),1) Z_2];   % (5000 x 26)
Z_3= A_2*Theta2';    % (5000 x 10)
A_3= [ones(size(Z_3,1),1) Z_3];
Z_4= A_3*Theta3';
A_4= Z_4;    % = H_theta (5000 x 10)

J = sum((A_4 - y).^2)/2/m;

J = J + lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))+ sum(sum(Theta3(:,2:end).^2)));



%% Matrix implementation

% forward propagate


%You need these matrices for both cost function and backprop.

% unroll y
% y =   repmat(y,1,num_labels) == repmat(1:num_labels,size(y,1),1);% calculate y (5000 x 10) (can be done with repmat and ==)

% % cost function
% J=  % (1 x 1) calculate in matrix form then use a double sum

% back propagate
delta_4= A_4 - y;% (5000 x 10)
% calculate delta_2 in 2 parts (because you have to remove the bias column)
delta_3 = delta_4*Theta3;% (5000 x 26)
delta_3 = delta_3(:,2:end);% (5000 x 25)


delta_2 = delta_3*Theta2;% (5000 x 26)
delta_2 = delta_2(:,2:end);% (5000 x 25)

Theta1_grad = delta_2'*A_1/m; % (25 x 401)
Theta2_grad = delta_3'*A_2/m; % (10 x 26)
Theta3_grad = delta_4'*A_3/m; % (10 x 26)

%% Add regularization -------------------------------------------------------------

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + Theta1(:,2:end).*lambda/m;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + Theta2(:,2:end).*lambda/m;
Theta3_grad(:,2:end) = Theta3_grad(:,2:end) + Theta3(:,2:end).*lambda/m;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];




end
