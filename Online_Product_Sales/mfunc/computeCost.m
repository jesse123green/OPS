function [J grad] = computeCost(theta, X, y, lambda)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

J = sum((X * theta - y).^2)/2/m + lambda/2/m*sum(theta(2:end).^2);

grad = (X*theta - y)'*X/m + lambda/m*theta';

grad(1) = (X*theta - y)'*X(:,1)/m;




% =========================================================================

end
