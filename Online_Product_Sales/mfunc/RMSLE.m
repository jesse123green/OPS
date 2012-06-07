function [score] = RMSLE(x,y)
x(x<0) = 0;
n = size(x,1)*size(x,2);
x = reshape(x,n,1);
y = reshape(y,n,1);
x = x(~isnan(y));
y = y(~isnan(y));
n = length(y)
score = sqrt(sum(sum((log(x+1) - log(y+1)).^2))/n);
% score = sum((log(x)-log(y)).^2);



end
