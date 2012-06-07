function out = mapFeature5(X,degree)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%

% degree = 3;
out = ones(size(X(:,1)));
for k1 = 1:degree
    for k2 = 0:k1
        for k3 = 0:k2
            for k4 = 0:k3
                for k5 = 0:k4
                   out(:, end+1) = (X(:,1).^(k1-k2)).*(X(:,2).^(k2-k3)).*X(:,3).^(k3-k4).*X(:,4).^(k4-k5).*X(:,5).^(k5);
                end
            end
            
        end
    end
end

end