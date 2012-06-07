function y_hat = BLUE(y_1,Y_bag)

% ac = zeros(length(y_1),12);
% for k = 1:length(y_1)
%     ac(k,:) = autocorr(y_1(k,:));
% end
% z = autocorr(mean(y_1,1));
% R1 = toeplitz(z)
% R1 = zeros(12);
% for i = 1:12
%     R1(i,:) = circshift(autocorr(circshift(mean(y_1,1),[i-1 1-i])),[i-1 i-1]);
% end
% mean(y_1,1)
% R1
R1 = zeros(12);
R1_final = zeros(12);
Rxx = zeros(12);
for i = 1:length(y_1)
    for k = 1:12
        for k1 = k:12
            R1(k,k1) = y_1(i,k)*y_1(i,k1);
        end
        R1(k,k:end) = R1(k,k:end)/R1(k,k);
        R1(k+1:end,k) = R1(k,k+1:end)';
    end

    R1_final = R1_final + R1;
end

R1_final = R1_final/length(y_1)

% R1_final = corrcoef(y_1)
% R1=corrcoef(y_1)
% Rxx = corrcoef(Y_bag)
y_hat = zeros(size(Y_bag,1),12);

for i=1:size(Y_bag,1)
    
    for k = 1:12
        for k1 = k:12
            Rxx(k,k1) = Y_bag(i,k)*Y_bag(i,k1);
        end
        Rxx(k,k:end) = Rxx(k,k:end)/Rxx(k,k);
        Rxx(k+1:end,k) = Rxx(k,k+1:end)';
    end
%     Rxx = corrcov(Rxx);
    a = zeros(12);
    for k = 1:12
        a(k,:) = pinv(Rxx)*(R1_final(k,:)');
    end
    
    y_hat(i,:) = (a*Y_bag(i,:)')';
    
end




end