clc;clear;close all;
%-----------------------------------------------取CSI数据，动作组一（10个动作为一组，坐下站起交替）
fileID1 = fopen('Activities_Data\r0');
rawData1 = fread(fileID1, 'uint32', 'ieee-be'); % 'ieee-be'表示大端字节序
csi1 = typecast(swapbytes(uint32(rawData1)), 'single');% 二进制32位反转并转浮点数
fclose(fileID1);

csi1= reshape(csi1, 64, [])'; % 将数据重构为64列的CSI数据（对应64个子载波）
csi1(:,all(csi1==0,1))=[]; % 去除csi中数值为0的列

[~, score, ~, ~, ~] = pca(csi1);% 对CSI数据进行PCA主成分分析
reducedData1 = score(:, 1);% 将数据降到1维

waveletName1 = 'db4'; % 使用Daubechies小波基（用于DWT）
[cA, cD] = dwt(reducedData1, waveletName1);% 对数据执行一维DWT离散小波变换
csi_denoised1 = idwt(zeros(size(cA)), cD, waveletName1);% 重构信号，去除低频信号，保留高频信号（动作）

figure;
plot(csi_denoised1);
title('Denoised CSI Data using PCA and DWT');% 绘制PCA和DWT后的信号


% 设置LOF算法，并找出异常点
n = length(csi_denoised1);
k = 20; % 定义k值，表示每个点的k个邻居
distances1 = pdist2(csi_denoised1(:), csi_denoised1(:));  % 欧氏距离矩阵
    % 找到每个点的 k 个最近邻居
[~, indices] = sort(distances1, 2);  % 对每行距离排序，返回每个点的邻居
k_neighbors1 = indices(:, 2:k+1);    % 第 2 列到 k+1 列是最近的 k 个邻居（忽略自己）
reach_dist1 = zeros(n, 1);% 初始化可达距离
    % 计算每个点的局部可达性距离（reachability distance）
for i = 1:n
    neighbors_idx = k_neighbors1(i, :);% 获取当前点的 k 个邻居的索引
    dist_to_neighbors4 = distances1(i, neighbors_idx);% 计算点 i 到其 k 个邻居的距离
    % 对邻居，计算可达距离(取 max(自身到邻居的距离, 邻居到达距离))
    reach_dist_neighbors = max(dist_to_neighbors4, distances1(neighbors_idx, i)');
    % 计算平均的可达距离，并赋值给reach_dist(i)
    reach_dist1(i) = mean(reach_dist_neighbors); 
end
    % 正常化可达距离
reach_dist1 = reach_dist1 / mean(reach_dist1);


lof_threshold1 = 10.0;  % LOF 阈值，选择阈值来确定异常点
anomaly_indices1 = find(reach_dist1 > lof_threshold1); % LOF值大于设定阈值的点，作为异常点

figure;
plot(csi_denoised1);
hold on;
plot(anomaly_indices1, csi_denoised1(anomaly_indices1), 'r*');
hold off;
title('Anomaly Detection using LOF');% 显示异常点（用*标出）


% 划分含动作的CSI区间
wave_intervals1 = zeros(10, 2);  % 用来存储动作的起止点
jiange=500; % 用于找下一个动作起点的间隔值
    %存储动作区间
for i = 1:10
    if i == 1
        wave_intervals1(i, 1) = anomaly_indices1(1);  % 第一个动作的起始点
    else
        % 找到下一个动作的起始点
        next_start = find(anomaly_indices1 > wave_intervals1(i-1, 2), 1,'first');
        % 检查是否找到有效的起点，如果没有找到，跳过或者处理
        if ~isempty(next_start)
            wave_intervals1(i, 1) = anomaly_indices1(next_start);
        else
            warning(['No valid start point found for wave ', num2str(i)]);
            break; 
        end
    end
        % 动作结束点是该动作信号的最后一个异常点
    next_end = find(wave_intervals1(i,1)+jiange >anomaly_indices1, 1,'last');
        % 检查是否找到有效的终点
    if ~isempty(next_end)
        wave_intervals1(i, 2) = anomaly_indices1(next_end);
    else
        warning(['No valid end point found for wave ', num2str(i)]);
        break;
    end
end
    % 限制并统一动作区间范围
for i=1:size(wave_intervals1)
    interval= wave_intervals1(i,2)-wave_intervals1(i,1);
    if interval<=100
        wave_intervals1(i,1)=round(wave_intervals1(i,1)-(100-interval)/2);
        wave_intervals1(i,2)=round(wave_intervals1(i,2)+(100-interval)/2);
    elseif interval>100
        wave_intervals1(i,1)=round(wave_intervals1(i,1)+(100-interval)/2);
        wave_intervals1(i,2)=round(wave_intervals1(i,2)-(100-interval)/2);
    end
end

    % 显示这些动作区间
figure;
plot(csi_denoised1);
hold on;
for i = 1:size(wave_intervals1)
    xline(wave_intervals1(i, 1), 'r--', 'Start');
    xline(wave_intervals1(i, 2), 'b--', 'End');
end
hold off;
title('Detected Wave Intervals using LOF');

%取出对应CSI矩阵，用于活动决策
csi_wave_data1=cell(10,1);  % 初始化csi矩阵
for i=1:size(wave_intervals1)
    csi_wave_data1{i}=csi1(wave_intervals1(i,1):wave_intervals1(i,2),:);
end
%-----------------------------------------------取CSI数据，动作组二（重复上述操作）
fileID2 = fopen('Activities_Data\r01');
rawData2 = fread(fileID2, 'uint32', 'ieee-be'); 
csi2 = typecast(swapbytes(uint32(rawData2)), 'single');
fclose(fileID2);

csi2= reshape(csi2, 64, [])';
csi2(:,all(csi2==0,1))=[];

[~, score, ~, ~, ~] = pca(csi2);
reducedData2 = score(:, 1);

waveletName2 = 'db4';
[cA, cD] = dwt(reducedData2, waveletName2);
csi_denoised2 = idwt(zeros(size(cA)), cD, waveletName2);

figure;
plot(csi_denoised2);
title('Denoised CSI Data using PCA and DWT');

n = length(csi_denoised2);
k = 20;
distances2 = pdist2(csi_denoised2(:), csi_denoised2(:));  
[~, indices] = sort(distances2, 2);  
k_neighbors2 = indices(:, 2:k+1);   
reach_dist2 = zeros(n, 1);
for i = 1:n
    neighbors_idx = k_neighbors2(i, :);
    dist_to_neighbors4 = distances2(i, neighbors_idx);
    reach_dist_neighbors = max(dist_to_neighbors4, distances2(neighbors_idx, i)');
    reach_dist2(i) = mean(reach_dist_neighbors);  
end
reach_dist2 = reach_dist2 / mean(reach_dist2);

lof_threshold2 = 10.0;  
anomaly_indices2 = find(reach_dist2 > lof_threshold2);

figure;
plot(csi_denoised2);
hold on;
plot(anomaly_indices2, csi_denoised2(anomaly_indices2), 'r*');
hold off;
title('Anomaly Detection using LOF');

wave_intervals2 = zeros(10, 2);
jiange=500;   
for i = 1:10
    if i == 1
        wave_intervals2(i, 1) = anomaly_indices2(1);  
    else
        next_start = find(anomaly_indices2 > wave_intervals2(i-1, 2), 1,'first');
        if ~isempty(next_start)
            wave_intervals2(i, 1) = anomaly_indices2(next_start);
        else
            warning(['No valid start point found for wave ', num2str(i)]);
            break; 
        end
    end
    next_end = find(wave_intervals2(i,1)+jiange >anomaly_indices2, 1,'last');
    if ~isempty(next_end)
        wave_intervals2(i, 2) = anomaly_indices2(next_end);
    else
        warning(['No valid end point found for wave ', num2str(i)]);
        break;  
    end
end

for i=1:size(wave_intervals2)
    interval= wave_intervals2(i,2)-wave_intervals2(i,1);
    if interval<=100
        wave_intervals2(i,1)=round(wave_intervals2(i,1)-(100-interval)/2);
        wave_intervals2(i,2)=round(wave_intervals2(i,2)+(100-interval)/2);
    elseif interval>100
        wave_intervals2(i,1)=round(wave_intervals2(i,1)+(100-interval)/2);
        wave_intervals2(i,2)=round(wave_intervals2(i,2)-(100-interval)/2);
    end
end

figure;
plot(csi_denoised2);
hold on;
for i = 1:size(wave_intervals2)
    xline(wave_intervals2(i, 1), 'r--', 'Start');
    xline(wave_intervals2(i, 2), 'b--', 'End');
end
hold off;
title('Detected Wave Intervals using LOF');

csi_wave_data2=cell(10,1);
for i=1:size(wave_intervals2)
    csi_wave_data2{i}=csi2(wave_intervals2(i,1):wave_intervals2(i,2),:);
end
%-----------------------------------------------取CSI数据，动作组三（重复上述操作）
fileID3 = fopen('Activities_Data\r02');
rawData3 = fread(fileID3, 'uint32', 'ieee-be'); 
csi3 = typecast(swapbytes(uint32(rawData3)), 'single');
fclose(fileID3);

csi3= reshape(csi3, 64, [])';
csi3(:,all(csi3==0,1))=[];

[~, score, ~, ~, ~] = pca(csi3);
reducedData3 = score(:, 1);

waveletName3 = 'db4';
[cA, cD] = dwt(reducedData3, waveletName3);
csi_denoised3 = idwt(zeros(size(cA)), cD, waveletName3);

figure;
plot(csi_denoised3);
title('Denoised CSI Data using PCA and DWT');

n = length(csi_denoised3);
k = 20;
distances3 = pdist2(csi_denoised3(:), csi_denoised3(:));
[~, indices] = sort(distances3, 2); 
k_neighbors3 = indices(:, 2:k+1);  
reach_dist3 = zeros(n, 1);

for i = 1:n
    neighbors_idx = k_neighbors3(i, :);
    dist_to_neighbors4 = distances3(i, neighbors_idx);
    reach_dist_neighbors = max(dist_to_neighbors4, distances3(neighbors_idx, i)');
    reach_dist3(i) = mean(reach_dist_neighbors); 
end
reach_dist3 = reach_dist3 / mean(reach_dist3);

lof_threshold3 = 10.0;
anomaly_indices3 = find(reach_dist3 > lof_threshold3);

figure;
plot(csi_denoised3);
hold on;
plot(anomaly_indices3, csi_denoised3(anomaly_indices3), 'r*');
hold off;
title('Anomaly Detection using LOF');

wave_intervals3 = zeros(10, 2);
jiange=500;  
for i = 1:10
    if i == 1
        wave_intervals3(i, 1) = anomaly_indices3(1);  
    else
        next_start = find(anomaly_indices3 > wave_intervals3(i-1, 2), 1,'first');
        if ~isempty(next_start)
            wave_intervals3(i, 1) = anomaly_indices3(next_start);
        else
            warning(['No valid start point found for wave ', num2str(i)]);
            break;  
        end
    end
    next_end = find(wave_intervals3(i,1)+jiange >anomaly_indices3, 1,'last');
    if ~isempty(next_end)
        wave_intervals3(i, 2) = anomaly_indices3(next_end);
    else
        warning(['No valid end point found for wave ', num2str(i)]);
        break; 
    end
end

for i=1:size(wave_intervals3)
    interval= wave_intervals3(i,2)-wave_intervals3(i,1);
    if interval<=100
        wave_intervals3(i,1)=round(wave_intervals3(i,1)-(100-interval)/2);
        wave_intervals3(i,2)=round(wave_intervals3(i,2)+(100-interval)/2);
    elseif interval>100
        wave_intervals3(i,1)=round(wave_intervals3(i,1)+(100-interval)/2);
        wave_intervals3(i,2)=round(wave_intervals3(i,2)-(100-interval)/2);
    end
end

figure;
plot(csi_denoised3);
hold on;
for i = 1:size(wave_intervals3)
    xline(wave_intervals3(i, 1), 'r--', 'Start');
    xline(wave_intervals3(i, 2), 'b--', 'End');
end
hold off;
title('Detected Wave Intervals using LOF');

csi_wave_data3=cell(10,1); 
for i=1:size(wave_intervals3)
    csi_wave_data3{i}=csi3(wave_intervals3(i,1):wave_intervals3(i,2),:);
end
%-----------------------------------------------取CSI数据，动作组四（重复上述操作）
fileID4 = fopen('Activities_Data\r03');
rawData4 = fread(fileID4, 'uint32', 'ieee-be'); 
csi4 = typecast(swapbytes(uint32(rawData4)), 'single');
fclose(fileID4);

csi4= reshape(csi4, 64, [])';
csi4(:,all(csi4==0,1))=[];

[coeff, score, latent, tsquared, explained] = pca(csi4);
reducedData4 = score(:, 1);

waveletName4 = 'db4'; 
[cA, cD] = dwt(reducedData4, waveletName4);
csi_denoised4 = idwt(zeros(size(cA)), cD, waveletName4);

figure;
plot(csi_denoised4);
title('Denoised CSI Data using PCA and DWT');

n = length(csi_denoised4);
k = 20;
distances4 = pdist2(csi_denoised4(:), csi_denoised4(:));
[~, indices] = sort(distances4, 2);  
k_neighbors4 = indices(:, 2:k+1);  
reach_dist4 = zeros(n, 1);

for i = 1:n
    neighbors_idx = k_neighbors4(i, :);
    dist_to_neighbors4 = distances4(i, neighbors_idx);  
    reach_dist_neighbors = max(dist_to_neighbors4, distances4(neighbors_idx, i)');
    reach_dist4(i) = mean(reach_dist_neighbors);  
end
reach_dist4 = reach_dist4 / mean(reach_dist4);

lof_threshold4 = 10.0;  
anomaly_indices4 = find(reach_dist4 > lof_threshold4);

figure;
plot(csi_denoised4);
hold on;
plot(anomaly_indices4, csi_denoised4(anomaly_indices4), 'r*');
hold off;
title('Anomaly Detection using LOF');

wave_intervals4 = zeros(10, 2); 
jiange=500;  
for i = 1:10
    if i == 1
        wave_intervals4(i, 1) = anomaly_indices4(1); 
    else
        next_start = find(anomaly_indices4 > wave_intervals4(i-1, 2), 1,'first');
        if ~isempty(next_start)
            wave_intervals4(i, 1) = anomaly_indices4(next_start);
        else
            warning(['No valid start point found for wave ', num2str(i)]);
            break;  
        end
    end
    next_end = find(wave_intervals4(i,1)+jiange >anomaly_indices4, 1,'last');
    if ~isempty(next_end)
        wave_intervals4(i, 2) = anomaly_indices4(next_end);
    else
        warning(['No valid end point found for wave ', num2str(i)]);
        break;  
    end
end

for i=1:size(wave_intervals4)
    interval= wave_intervals4(i,2)-wave_intervals4(i,1);
    if interval<=100
        wave_intervals4(i,1)=round(wave_intervals4(i,1)-(100-interval)/2);
        wave_intervals4(i,2)=round(wave_intervals4(i,2)+(100-interval)/2);
    elseif interval>100
        wave_intervals4(i,1)=round(wave_intervals4(i,1)+(100-interval)/2);
        wave_intervals4(i,2)=round(wave_intervals4(i,2)-(100-interval)/2);
    end
end

figure;
plot(csi_denoised4);
hold on;
for i = 1:size(wave_intervals4)
    xline(wave_intervals4(i, 1), 'r--', 'Start');
    xline(wave_intervals4(i, 2), 'b--', 'End');
end
hold off;
title('Detected Activities Intervals using LOF');

csi_wave_data4=cell(10,1); 
for i=1:size(wave_intervals4)
    csi_wave_data4{i}=csi4(wave_intervals4(i,1):wave_intervals4(i,2),:);
end
%------------------------------------------------单类支持向量机（One-Class SVM）算法
% 初始化空矩阵用于存储数据 X 和标签 Y
X = [];
Y = [];

% 动作标签：1 表示坐下，-1 表示站起
sit_label = 1;
stand_label = -1;

% 整合四个 cell 中的数据到特征矩阵 X 和标签向量 Y
for i = 1:4
    % 获取当前 cell 组
    current_cell = eval(['csi_wave_data', num2str(i)]);
    % 遍历当前 cell 中的每个动作矩阵
    for j = 1:10
        % 将矩阵展平为一维向量，并添加到 X 中
        X = [X; current_cell{j}(:)'];
        % 根据交替模式设置标签
        if mod(j, 2) == 1
            Y = [Y; sit_label];  % 奇数动作为坐下
        else
            Y = [Y; stand_label]; % 偶数动作为站起
        end
    end
end

% 将数据集划分为 75% 的训练集和 25% 的测试集
cv = cvpartition(size(X, 1), 'HoldOut', 0.25);
XTrain = X(training(cv), :);
YTrain = Y(training(cv), :);
XTest = X(test(cv), :);
YTest = Y(test(cv), :);

% 训练 SVM 分类器，使用线性核函数
svmModel = fitcsvm(XTrain, YTrain, 'KernelFunction', 'linear', 'Standardize', true);

% 使用训练好的模型对测试集进行预测
YPred = predict(svmModel, XTest);

% 计算并显示测试集的准确率
accuracy = sum(YPred == YTest) / numel(YTest);
disp(['测试集准确率: ', num2str(accuracy * 100), '%']);
