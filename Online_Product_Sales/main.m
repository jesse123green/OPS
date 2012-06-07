%% Kaggle Competition: Online Product Sales
%  Created By: Jesse Green



%% Clear and Close Figures
clear all; close all; clc
% matlabpool open 2
ms.UseParallel = 'always';
options = psoptimset('UseParallel', 'always', 'CompletePoll', 'on', 'Vectorized', 'off');
options = gaoptimset('UseParallel', 'always', 'Vectorized', 'off');

warning off; 
addpath([pwd '/mfunc']); 
warning on;



fprintf('Loading data ...\n');

%% Load Data
fid = fopen('TrainingDataset.csv','r');  % Open text file
InputText=textscan(fid,'%s',558,'delimiter',',');
fclose(fid);
data = csvread('TrainingDataset.csv',1,0);
X = data(:,13:end);
y = data(:, 1:12);
X(:,end+1) = X(:,2) - X(:,7);

Labels = InputText{1}(13:end);

index = 1;
index2 = 1;
for k = 1:length(Labels)
    if strcmp(Labels{k,1}(2),'C') && length(unique(X(:,k))) < 80
        cats(index) = k;
        index = index + 1;
    elseif strcmp(Labels{k,1}(2),'Q')
        quants(index2) = k;
        index2 = index2 + 1;
    end
end



num_labels = 5;



quants = [quants 547];


num_hidden_layers = 1;

qi = quants([1 6 7 9 11 13 28:29]);

[xNN yNN] = removeNaNs(X(:,qi),y(:,4));

ly = log(yNN);
%% kmeans  cluster and neural network
if false
    
    % [idx,ctrs] =
    % kmeans(log(yNN),num_labels,'replicates',10,'start','uniform');
    hidden_layer_size = 3*length(qi);
    idx = clusterdata(ly,'maxclust',num_labels,'linkage','complete');
    ctrs = zeros(num_labels,1);

    for k = 1:num_labels
        ctrs(k) = mean(ly(idx == k));
    end

    cc=hsv(12);
    figure;
    
    hold on;
    for i=1:num_labels
        plot(ly(idx==i,1),ones(sum(idx==i),1),'color',cc(i,:),'MarkerSize',12)
        plot(ly(idx==i,1),ones(sum(idx==i),1),'o','color',cc(i,:),'MarkerSize',5)
    end

    if num_hidden_layers == 2
        [Theta1 Theta2 Theta3] = trainNN2(xNN,idx, length(qi), hidden_layer_size, num_labels);%[Theta1 Theta2] = trainNN(X, y, input_layer_size, hidden_layer_size, num_labels)
        RMSLE(ctrs(predict2(Theta1, Theta2, Theta3, xNN)),yNN)
    else

        [Theta1 Theta2] = trainNN(xNN,idx, length(qi), hidden_layer_size, num_labels);%[Theta1 Theta2] = trainNN(X, y, input_layer_size, hidden_layer_size, num_labels)
        RMSLE(exp(ctrs(predict(Theta1, Theta2, xNN))),yNN)
    end
end

%% Neural Network for Regression
if false
    coeff = 1:10;
    hidden_layer_size = 350;
    lds = [.01*2.^(8:15)];



    i1 = isnan(X(:,8));
    i2 = ~i1;

    num_hidden_layers = 2;

    percTT = .75; % Number of test data

    % Pull test and training data randomly




    for k1 = 1:12
        [x_avail y_avail] = removeNaNs(X(i2,[3:6 8 9 11 13:15 18:22 32]),y(i2,k1));

        index = (rand(size(x_avail,1),1) <= percTT);

        x_r = x_avail(index,:);
        y_r = y_avail(index,:);
        % x_r = [ones(size(x_r,1),1) x_r];

        x_rt = x_avail(~index,:);
        y_rt = y_avail(~index,:);
        % x_rt = [ones(size(x_rt,1),1) x_rt];
        degree = 2;

        x_r = mapFeature16(x_r,degree);
        x_rt = mapFeature16(x_rt,degree);


        % Feature scale training data
        xm = mean(x_r);
        xm(1) = 0;
        xs = std(x_r);
        xs(1) = 1;
        x_r = (x_r - repmat(xm,size(x_r,1),1))./repmat(xs,size(x_r,1),1);
        y_r = log(y_r);
        ym = mean(y_r);
        ys = std(y_r);
        y_r = (y_r - ym)/ys;

        % Feature scale test data
        x_rt = (x_rt - repmat(xm,size(x_rt,1),1))./repmat(xs,size(x_rt,1),1);

        for k=1:length(lds)

            if num_hidden_layers == 2
                [Theta1 Theta2 Theta3] = trainNNReg2(x_r, y_r, size(x_r,2), hidden_layer_size, 1,1,lds(k));
                Scv(k,k1) = RMSLE(exp(predictReg2(Theta1, Theta2, Theta3,x_rt) * ys + ym), y_rt);
                Strain(k,k1) = RMSLE(exp(predictReg2(Theta1, Theta2, Theta3, x_r) * ys + ym),exp(y_r * ys + ym));
            else
                [Theta1 Theta2 pred] = trainNNReg(x_r, y_r, size(x_r,2), hidden_layer_size, 1,1,lds(k));%[Theta1 Theta2] = trainNN(X, y, input_layer_size, hidden_layer_size, num_labels)
                Scv(k,k1) = RMSLE(exp(predictReg(Theta1, Theta2, x_rt) * ys + ym), y_rt);
                Strain(k,k1) = RMSLE(exp(predictReg(Theta1, Theta2, x_r) * ys + ym),exp(y_r * ys + ym));
            end

        %     [predictReg(Theta1, Theta2, x_r) exp(y_r)]
        %     RMSLE(exp(predictReg(Theta1, Theta2, x_r)),exp(y_r))
        end

        figure(k1)
        hold on
        plot(Scv(:,k1))
        plot(Strain(:,k1),'g')
    end


    [smin,smini] = min(Scv);
    smin
    lds(smini)

    return
end
%% correlation cluster
if false
    [x1 y1] = removeNaN(X,y);
    % [y1 ~] = removeNaNs(y,y);

    Y = pdist(y1,'correlation'); 
    length(Y)
    Z = linkage(Y,'complete'); 
    % dendrogram(Z,0)
    T = cluster(Z,'maxclust',9);
    length(T)
end





% for k=quants
%     k
%     close all
%     
%     
%     hold on
%     plot(X(:,k),log(y(:,1)),'rx')
% %     plot(1000:10000,exp(fit(2)+fit(1).*(1000:10000)))
% %     plot(x1,x1*theta)
%     pause
% end

%% Verify Linear Regression
if false
    xtest = 1:100;

    tt = xtest.^2 + rand(1,100)*1000-500;

    xu = [ones(100,1) xtest' xtest'.^2];


    % Initialize fitting parameters
    initial_theta = zeros(size(xu, 2), 1);

    % Set regularization parameter lambda
    lambda = 1;

    % Set Options
    options = optimset('GradObj', 'on', 'MaxIter', 400);

    % Optimize
    [theta, J, exit_flag] = ...
        fminunc(@(t)(computeCost(t, xu, tt', lambda)), initial_theta, options);

    hold on
    plot(xtest,tt)
    plot(xtest,xu*theta,'g')
    hold off
    return
end

%% Linear Regression
if false
    num_trials = 300;
    lds = [.01*2.^(4:18)];
    Strain = zeros(12,length(lds));
    Scv = zeros(12,length(lds));

    % lds = 20:5:.7*size(x_avail,1);
    
    
    for iter = 1:12
        [x_avail y_avail] = removeNaNs(X(:,[3:6 8 9 11 13 14 15]),y(:,iter));
        for k = 1:length(lds)
            for k1 = 1:num_trials
            percTT = .7; % Number of test data

            % Pull test and training data randomly

            index = (rand(size(x_avail,1),1) <= percTT);

            x_r = x_avail(index,:);
            y_r = y_avail(index,:);

            x_rt = x_avail(~index,:);
            y_rt = y_avail(~index,:);

        %     x_r = x_avail(1:lds(k),:);
        %     y_r = y_avail(1:lds(k),:);
        % 
        %     x_rt = x_avail(lds(k)+1:end,:);
        %     y_rt = y_avail(lds(k)+1:end,:);

    %         x_r = x_avail(1:round(percTT*size(x_avail,1)),:);
    %         y_r = y_avail(1:round(percTT*size(x_avail,1)),:);
    % 
    %         x_rt = x_avail(round(percTT*size(x_avail,1)+1):end,:);
    %         y_rt = y_avail(round(percTT*size(x_avail,1)+1):end,:);

            % Polynomial feature mapping
            degree = 2;
            % x_r = mapFeature3(x_r(:,1),x_r(:,2),x_r(:,3));
            % x_rt = mapFeature3(x_rt(:,1),x_rt(:,2),x_rt(:,3));
            x_r = mapFeature10(x_r,degree);
            x_rt = mapFeature10(x_rt,degree);

            % Feature scale training data
            xm = mean(x_r);
            xm(1) = 0;
            xs = std(x_r);
            xs(1) = 1;
            x_r = (x_r - repmat(xm,size(x_r,1),1))./repmat(xs,size(x_r,1),1);
            y_r = log(y_r);
            ym = mean(y_r);
            ys = std(y_r);
            y_r = (y_r - ym)/ys;

            % Feature scale test data
            x_rt = (x_rt - repmat(xm,size(x_rt,1),1))./repmat(xs,size(x_rt,1),1);
            % y_rt = (log(y_rt) - ym)./ys;



            % Initialize fitting parameters
            initial_theta = zeros(size(x_r, 2), 1);

            % Set regularization parameter lambda

            lambda = lds(k);

            % Set Options
            options = optimset('GradObj', 'on', 'MaxIter', 400);

            % Optimize
            [theta, Jtrain(k,k1), exit_flag] = ...
                fminunc(@(t)(computeCost(t, x_r, y_r, lambda)), initial_theta, options);


            Jcv(k,k1) = computeCost(theta,x_rt,(log(y_rt) - ym)/ys,lambda);
            % hold on
            % plot(X(:,4),log(y(:,1)),'rx')
            % plot(X(:,4),max(log(2000),x_r*theta))
            % exp([x_r*theta y_r]*ys+ym)


            Strain(iter,k) = Strain(iter,k) + RMSLE(exp(x_r*theta * ys + ym), exp(y_r * ys + ym));
            % 
            Scv(iter,k) = Scv(iter,k) + RMSLE(exp(x_rt*theta .*ys + ym), y_rt);
        %     RMSLE(min(5e5,exp(x_rt*theta .*ys + ym)), y_rt)
            
            end
        end
        Strain(iter,:) = Strain(iter,:)/num_trials;
        Scv(iter,:) = Scv(iter,:)/num_trials;
        
        figure(iter)
        hold on
        plot(Scv(iter,:))
        plot(Strain(iter,:),'g')
    
    end
    
    Strain = Strain/num_trials;
    Scv = Scv/num_trials;
    
    [smin,smini] = min(Scv,[],2);
    smin
    lds(smini)

%     hold on
%     plot(mean(Jcv,2))
%     plot(mean(Jtrain,2),'g')





end
% x1(:,2) = log(x1(:,2))/fit(1);



% fprintf('clustering....');
% size(x1)
% size(T)
% CorrCluster = TreeBagger(10,x1,T,'Method','category','OOBPred','on','categorical', cats);
% plot(oobError(CorrCluster))
% xlabel('number of grown trees')
% ylabel('out-of-bag classification error')
% pause
%% Treebagger on groups
% B = TreeBagger(50,xNN,ly,'method','regression','OOBPred','on');
% pred = predict(B,xNN);
% % pred = cellfun(@str2num,pred);
% RMSLE(exp(pred),yNN)

i1 = isnan(X(:,8));
i2 = ~i1;



B = cell(12,1);
B2 = cell(12,1);

dataToPredict = csvread('TestDataset.csv',1,0);
dP = dataToPredict(:,2:end);
dP(:,end+1) = dP(:,2) - dP(:,7);

i1o = isnan(dP(:,8));
i2o = ~i1o;

finalPrediction = zeros(size(dP,1),12);

pred = cell(12,1);

%% Min leaf
if false
    leaf = [1 5 10 20 50 100];
    col = 'rgbcmy';
    figure(1);
    for i=1:length(leaf)
        b = TreeBagger(100,xa(i1,:),log(ya(i1,2)),'method','r','oobpred','on',...
       'cat',cats,'minleaf',leaf(i));
        plot(oobError(b),col(i));
        hold on;
    end
    xlabel('Number of Grown Trees');
    ylabel('Mean Squared Error');
    legend({'1' '5' '10' '20' '50' '100'},'Location','NorthEast');
    hold off;

    pause
end
%% Eliminate unnecessary data for group 1
if false
    cat = cell(9,1);
    cat{1} = cats;
    idxvar = cell(9,1);
    idxvar{1} = 1:547;
    for k = 1:8
        B{k} = TreeBagger(250,xa(i2,idxvar{k}),log(ya(i2,4)),'method','r','categorical',cat{k},'oobvarimp','on','minleaf',1);
%         figure(2);
%         plot(oobError(B{k}));
%         xlabel('Number of Grown Trees');
%         ylabel('Out-of-Bag Mean Squared Error');
%         figure(1)
%         bar(B{k}.OOBPermutedVarDeltaError)
        temp = oobError(B{k});
        temp(end)

        idxvar{k+1} = find(B{k}.OOBPermutedVarDeltaError>0);
        idxvar{k+1} = idxvar{k}(idxvar{k+1});

        cat{k+1} = find(ismember(idxvar{k+1},cat{1}) == 1);
%         pause;
    end

    pause
end
%% Eliminate data outliers
if false
    cat = cell(9,1);
    cat{1} = cats;
    idxvar = cell(9,1);
    idxvar{1} = 1:547;
    for k = 1:8
        
        B{k} = TreeBagger(250,xa(i2,idxvar{k}),log(ya(i2,4)),'method','r','categorical',cat{k},'oobvarimp','on','minleaf',1);
        B{k} = fillProximities(B{k});
        figure(7);
        hist(b5v.OutlierMeasure);
        xlabel('Outlier Measure');
        ylabel('Number of Observations');
        [~,temp] = sort(B{1}.OutlierMeasure);
        temp(end)

        idxvar{k+1} = find(B{k}.OOBPermutedVarDeltaError>0);
        idxvar{k+1} = idxvar{k}(idxvar{k+1});

        cat{k+1} = find(ismember(idxvar{k+1},cat{1}) == 1);
%         pause;
    end

    pause
end
%% 

iterations = 8;
i2_score = zeros(12,iterations);
% group i2
for k = 3:12
    for iter = 1:iterations


        [xa ya] = removeNaN(X(i2,:),y(i2,k));
        if k > 12
            B2{k} = TreeBagger(250,xa,log(ya),'OOBPred','on');
            pred{k} = log(cellfun(@str2num,predict(B2{k},xa(i2,qi))));
            finalPrediction(i2o,k) = log(cellfun(@str2num,predict(B2{k},dP(i2o,qi))));
        else
            B2{k} = TreeBagger(150,xa,log(ya),'method','regression','OOBPred','on','categorical',cats,'minleaf',1);
            pred{k} = predict(B2{k},xa);
            finalPrediction(i2o,k) = predict(B2{k},dP(i2o,:));
        end

        temp = sqrt(oobError(B2{k}));
        i2_score(k,iter) = temp(end);
%         fprintf('Trial %i - Size: %i - Score: %f\n', k ,size(ya,1)
%         ,temp(end));
    end
    k
    i2_score(k,:)
    mean(i2_score(k,:),2)
end


% 
% 
% 
% 
% 
% fprintf('\n');
%group i1
i1_score = zeros(12,iterations);
for k = 1:12
    for iter = 1:iterations
    [xa ya] = removeNaN(X(i1,:),y(i1,k));

    
    
    B{k} = TreeBagger(150,xa,log(ya),'method','regression','OOBPred','on','categorical',cats,'minleaf',1);
    pred{k} = predict(B{k},xa);
    finalPrediction(i1o,k) = predict(B{k},dP(i1o,:));
    
    temp = sqrt(oobError(B{k}));
    i1_score(k,iter) = temp(end);
%     fprintf('Trial %i - Size: %i - Score: %f\n', k ,size(ya,1)
%     ,temp(end));
    
    end
    k
    i1_score(k,:)
    mean(i1_score(k,:),2)
end


return
% for num = 80:20:200
%     clear cats
%     index = 1;
%     index2 = 1;
%     for k = 1:length(Labels)
%         if strcmp(Labels{k,1}(2),'C') && length(unique(X(:,k))) < num
%             cats(index) = k;
%             index = index + 1;
%         elseif strcmp(Labels{k,1}(2),'Q')
%             quants(index2) = k;
%             index2 = index2 + 1;
%         end
%     end
% 
% 
%     %both groups
%     for k = 1:1
%         fprintf('%i ', k);
%         B{k} = TreeBagger(50,xa,log(ya(:,k)),'method','regression','OOBPred','on','categorical',cats,'minleaf',1);
%         pred(:,k) = predict(B{k},xa);
%         finalPrediction(:,k) = predict(B{k},dP);
% 
%     end
% 
%     temp = oobError(B{k});
%     fprintf('%i: %f \n',num,temp(end))
% end



RMSLE(exp(pred),ya)


finalPrediction = [(1:519)' exp(finalPrediction)];
head = {'id','Outcome_M1','Outcome_M2','Outcome_M3','Outcome_M4','Outcome_M5','Outcome_M6','Outcome_M7','Outcome_M8','Outcome_M9','Outcome_M10','Outcome_M11','Outcome_M12'};
fid = fopen('output.txt','w');
for row = 1:size(head,1)
    fprintf(fid, repmat('%s,',1,size(head,2)-1), head{row,1:end-1});
    fprintf(fid, '%s\n', head{row,end});
end
fclose(fid);

dlmwrite('output.txt',finalPrediction,'-append');

pause

% pred = cellfun(@str2num,pred);

%% Tree bagger and BLUE
if false
    fprintf('Planting Forest ...\n');

    N = 10;
    pred = cell(N,1);


    for j = 1:N
        fprintf('Planting Trees in Region %i: \n',j);
        pred{j} = zeros(size(y1,1),size(y1,2));
        B = cell(12,1);
        for k = 1:12
            fprintf('%i ', k);
        %     B{k} =
        %     TreeBagger(50,X,y(:,k),'method','regression','OOBPred','on','categorical', cats);
            B{k} = TreeBagger(50,x1,y1(:,k),'method','regression','OOBPred','on');
            pred{j}(:,k) = predict(B{k},x1);
        %     Y_bag(:,k) = predict(B{k},X);
        end
    end

    y_hat1 = zeros(length(y1(T==4,:)),12);

    p1 = zeros(size(y_hat1));

    for j = 1:N
        y_hat1 = y_hat1 + BLUE(y1(T==4,:),pred{j}(T==4,:));
        p1 = p1 + pred{j}(T==4,:);
    end

    y_hat1 = y_hat1/N;

    p1 = p1/N;

    y_hat2 = BLUE(mean(y1(T==4,:)),p1);


    fprintf('RMSLE bag =  %f \n', RMSLE(p1,y1(T==4,:)))

    fprintf('RMSLE BLUE 1 bag =  %f \n', RMSLE(y_hat1,y1(T==4,:)))
    fprintf('RMSLE BLUE 2 bag =  %f \n', RMSLE(y_hat2,y1(T==4,:)))
end


%% Neural Network to groups







