function [x1,y1] = removeNaNs(x1,y1)


toKeep = ones(length(y1),1);
for k = 1:size(x1,1)
    if sum(isnan(x1(k,:))) > 0 || isnan(y1(k)) == 1
        toKeep(k) = 0;
    end
end

y1 = y1(logical(toKeep));
x1 = x1(logical(toKeep),:);


end