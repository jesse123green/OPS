function [x1,y1] = removeNaN(x1,y1) % removes rows from x1 and y1 where y1 contains NaN

toKeep = ones(length(y1),1);
for k = 1:size(y1,1)
    if sum(isnan(y1(k,:))) > 0
        toKeep(k) = 0;
    end
end

y1 = y1(logical(toKeep),:);
x1 = x1(logical(toKeep),:);

end