function out = mapFeature10(X,degree)
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
                    for k6 = 0:k5
                        for k7 = 0:k6
                            for k8 = 0:k7
                                for k9 = 0:k8
                                    for k10 = 0:k9

                                        out(:, end+1) = (X(:,1).^(k1-k2)).*(X(:,2).^(k2-k3)).*X(:,3).^(k3-k4).*X(:,4).^(k4-k5)...
                                            .*X(:,5).^(k5-k6).*X(:,6).^(k6-k7).*X(:,7).^(k7-k8).*X(:,8).^(k8-k9).*X(:,9).^(k9-k10)...
                                            .*X(:,10).^(k10);

                                    end
                                end
                            end
                        end
                    end
                end
            end
            
        end
    end
end

end