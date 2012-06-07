function p = predictReg2(Theta1, Theta2, Theta3, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);


% You need to return the following variables correctly 


h1 = [ones(m, 1) X] * Theta1';
h2 = [ones(m, 1) h1] * Theta2';

p = [ones(m, 1) h2] * Theta3';


% =========================================================================


end
