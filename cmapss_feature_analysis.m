%% CMAPSS FEATURE RELEVANCE ANALYSIS & VISUALIZATION
% Comprehensive feature importance ranking with multiple visualization methods:
% 1. Feature Importance Ranking (Bar plots, Heatmaps)
% 2. t-SNE dimensionality reduction visualization
% 3. Correlation analysis
% 4. Mutual Information
% 5. Feature contribution to RUL prediction

clear all; close all; clc;
rng(42);

fprintf('================================================================\n');
fprintf('   CMAPSS FEATURE RELEVANCE ANALYSIS & VISUALIZATION\n');
fprintf('================================================================\n\n');

%% ===========================
%% STEP 1: LOAD DATA
%% ===========================
fprintf('[STEP 1/7] Loading data...\n');

base_path = 'C:\Users\DELL\Downloads\CMAPSSData\';
dataset_name = 'FD001';

train_raw = readmatrix([base_path, 'train_', dataset_name, '.txt']);
test_raw = readmatrix([base_path, 'test_', dataset_name, '.txt']);
true_rul = readmatrix([base_path, 'RUL_', dataset_name, '.txt']);

unit_col = 1;
time_col = 2;
sensor_cols = 6:26;

% Remove dead sensors
train_sensors = train_raw(:, sensor_cols);
sensor_variance = var(train_sensors, 0, 1);
dead_sensors = find(sensor_variance < 1e-8);
sensor_cols(dead_sensors) = [];

sensor_names = {'T2','T24','T30','T50','P2','P15','P30','Nf','Nc','epr',...
                'Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd',...
                'PCNfR_dmd','W31','W32'};
sensor_names(dead_sensors) = [];

fprintf('  Active sensors: %d\n', length(sensor_cols));

%% ===========================
%% STEP 2: FEATURE ENGINEERING
%% ===========================
fprintf('\n[STEP 2/7] Engineering features...\n');

function [features, feature_names] = engineer_features_with_names(data, sensor_cols, sensor_names, unit_col, time_col)
    units = data(:, unit_col);
    unique_units = unique(units);
    sensors = data(:, sensor_cols);
    time_cycles = data(:, time_col);
    
    num_sensors = length(sensor_cols);
    num_rows = size(data, 1);
    
    % Initialize
    features_out = sensors;
    names_out = sensor_names;
    
    % Temporal features
    temporal = zeros(num_rows, 3);
    for u = 1:length(unique_units)
        idx = find(units == unique_units(u));
        cycles = time_cycles(idx);
        max_cycle = max(cycles);
        
        temporal(idx, 1) = cycles / max_cycle;
        temporal(idx, 2) = 1 ./ (cycles + 1);
        temporal(idx, 3) = sqrt(cycles) / sqrt(max_cycle);
    end
    features_out = [features_out, temporal];
    names_out = [names_out, {'Norm_Time', 'Urgency', 'NonLinear_Time'}];
    
    % Rolling statistics
    window = 5;
    rolling_mean = zeros(num_rows, num_sensors);
    rolling_std = zeros(num_rows, num_sensors);
    
    for u = 1:length(unique_units)
        idx = find(units == unique_units(u));
        block = sensors(idx, :);
        
        rolling_mean(idx, :) = movmean(block, window, 1, 'Endpoints', 'shrink');
        rolling_std(idx, :) = movstd(block, window, 0, 1, 'Endpoints', 'shrink');
    end
    features_out = [features_out, rolling_mean, rolling_std];
    
    for i = 1:num_sensors
        names_out = [names_out, {[sensor_names{i} '_RollMean']}, {[sensor_names{i} '_RollStd']}];
    end
    
    % EWMA
    alpha = 0.25;
    ewma = zeros(num_rows, num_sensors);
    
    for u = 1:length(unique_units)
        idx = find(units == unique_units(u));
        for s = 1:num_sensors
            signal = sensors(idx, s);
            ewma(idx, s) = filter(alpha, [1 -(1-alpha)], signal);
        end
    end
    features_out = [features_out, ewma];
    
    for i = 1:num_sensors
        names_out = [names_out, {[sensor_names{i} '_EWMA']}];
    end
    
    features = features_out;
    feature_names = names_out;
end

[train_features, feature_names] = engineer_features_with_names(train_raw, sensor_cols, ...
    sensor_names, unit_col, time_col);

num_features = size(train_features, 2);
fprintf('  Total features: %d\n', num_features);

%% ===========================
%% STEP 3: CALCULATE RUL LABELS
%% ===========================
fprintf('\n[STEP 3/7] Calculating RUL labels...\n');

train_engines = unique(train_raw(:, unit_col));
rul_clip = 125;

train_rul = zeros(size(train_raw, 1), 1);
for i = 1:length(train_engines)
    mask = train_raw(:, unit_col) == train_engines(i);
    cycles = train_raw(mask, time_col);
    max_cycle = max(cycles);
    rul = min(max_cycle - cycles, rul_clip);
    train_rul(mask) = rul;
end

fprintf('  RUL labels calculated for %d samples\n', length(train_rul));

%% ===========================
%% STEP 4: NORMALIZE FEATURES
%% ===========================
fprintf('\n[STEP 4/7] Normalizing features...\n');

feature_medians = median(train_features, 1);
feature_mads = mad(train_features, 1, 1);
feature_mads(feature_mads < 1e-6) = 1;

train_features_norm = (train_features - feature_medians) ./ feature_mads;
train_features_norm = max(min(train_features_norm, 3), -3);

%% ===========================
%% STEP 5: FEATURE IMPORTANCE ANALYSIS
%% ===========================
fprintf('\n[STEP 5/7] Computing feature importance metrics...\n');

% Sample for efficiency
sample_size = min(10000, size(train_features_norm, 1));
sample_idx = randperm(size(train_features_norm, 1), sample_size);
features_sample = train_features_norm(sample_idx, :);
rul_sample = train_rul(sample_idx);

% 1. Pearson Correlation
fprintf('  Computing Pearson correlation...\n');
correlation_scores = zeros(num_features, 1);
for i = 1:num_features
    correlation_scores(i) = abs(corr(features_sample(:, i), rul_sample));
end

% 2. Mutual Information (simplified)
fprintf('  Computing mutual information...\n');
mi_scores = zeros(num_features, 1);
num_bins = 10;

for i = 1:num_features
    try
        % Discretize
        feature_bins = discretize(features_sample(:, i), num_bins);
        rul_bins = discretize(rul_sample, num_bins);
        
        % Remove NaN
        valid = ~isnan(feature_bins) & ~isnan(rul_bins);
        feature_bins = feature_bins(valid);
        rul_bins = rul_bins(valid);
        
        if length(feature_bins) < 10
            mi_scores(i) = 0;
            continue;
        end
        
        % Calculate MI
        [counts, ~] = hist3([feature_bins, rul_bins], {1:num_bins, 1:num_bins});
        joint_prob = counts / sum(counts(:));
        
        feature_prob = sum(joint_prob, 2);
        rul_prob = sum(joint_prob, 1)';
        
        mi = 0;
        for j = 1:num_bins
            for k = 1:num_bins
                if joint_prob(j,k) > 0 && feature_prob(j) > 0 && rul_prob(k) > 0
                    mi = mi + joint_prob(j,k) * log2(joint_prob(j,k) / (feature_prob(j) * rul_prob(k)));
                end
            end
        end
        mi_scores(i) = max(mi, 0);
    catch
        mi_scores(i) = 0;
    end
end

% 3. Variance-based importance
variance_scores = var(features_sample, 0, 1)';
variance_scores = variance_scores / sum(variance_scores);

% 4. Combined importance score
combined_scores = (correlation_scores / max(correlation_scores) + ...
                   mi_scores / max(mi_scores + 1e-6) + ...
                   variance_scores / max(variance_scores)) / 3;

% Rank features
[sorted_combined, idx_combined] = sort(combined_scores, 'descend');
[sorted_corr, idx_corr] = sort(correlation_scores, 'descend');
[sorted_mi, idx_mi] = sort(mi_scores, 'descend');

fprintf('  Feature importance computed!\n');

%% ===========================
%% STEP 6: t-SNE VISUALIZATION
%% ===========================
fprintf('\n[STEP 6/7] Computing t-SNE for dimensionality reduction...\n');

% Sample even more for t-SNE (computational efficiency)
tsne_sample_size = min(3000, size(train_features_norm, 1));
tsne_idx = randperm(size(train_features_norm, 1), tsne_sample_size);
features_tsne = train_features_norm(tsne_idx, :);
rul_tsne = train_rul(tsne_idx);

% Select top features for t-SNE
top_k_tsne = min(30, num_features);
features_tsne_selected = features_tsne(:, idx_combined(1:top_k_tsne));

fprintf('  Running t-SNE on %d samples with %d features...\n', tsne_sample_size, top_k_tsne);
fprintf('  (This may take 1-2 minutes...)\n');

tsne_result = tsne(features_tsne_selected, 'NumDimensions', 2, 'Perplexity', 30, 'Verbose', 0);

fprintf('  t-SNE completed!\n');

%% ===========================
%% STEP 7: COMPREHENSIVE VISUALIZATIONS
%% ===========================
fprintf('\n[STEP 7/7] Generating visualizations...\n');

%% FIGURE 1: FEATURE IMPORTANCE RANKINGS
figure('Name', 'Feature Importance Rankings', 'Position', [50 50 1800 1000]);

% 1. Top features by combined score
subplot(2,3,1);
top_n = min(20, num_features);
barh(sorted_combined(1:top_n), 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k', 'LineWidth', 1.5);
set(gca, 'YTick', 1:top_n, 'YTickLabel', feature_names(idx_combined(1:top_n)), 'YDir', 'reverse');
xlabel('Combined Importance Score', 'FontWeight', 'bold', 'FontSize', 11);
title('Top 20 Features (Combined Score)', 'FontWeight', 'bold', 'FontSize', 13);
grid on;

% 2. Correlation ranking
subplot(2,3,2);
barh(sorted_corr(1:top_n), 'FaceColor', [0.8 0.4 0.3], 'EdgeColor', 'k', 'LineWidth', 1.5);
set(gca, 'YTick', 1:top_n, 'YTickLabel', feature_names(idx_corr(1:top_n)), 'YDir', 'reverse');
xlabel('|Correlation| with RUL', 'FontWeight', 'bold', 'FontSize', 11);
title('Top 20 by Correlation', 'FontWeight', 'bold', 'FontSize', 13);
grid on;

% 3. Mutual Information ranking
subplot(2,3,3);
barh(sorted_mi(1:top_n), 'FaceColor', [0.4 0.7 0.4], 'EdgeColor', 'k', 'LineWidth', 1.5);
set(gca, 'YTick', 1:top_n, 'YTickLabel', feature_names(idx_mi(1:top_n)), 'YDir', 'reverse');
xlabel('Mutual Information', 'FontWeight', 'bold', 'FontSize', 11);
title('Top 20 by Mutual Information', 'FontWeight', 'bold', 'FontSize', 13);
grid on;

% 4. Correlation vs MI scatter
subplot(2,3,4);
scatter(correlation_scores, mi_scores, 80, combined_scores, 'filled', 'MarkerEdgeColor', 'k');
xlabel('Correlation', 'FontWeight', 'bold', 'FontSize', 11);
ylabel('Mutual Information', 'FontWeight', 'bold', 'FontSize', 11);
title('Feature Importance Landscape', 'FontWeight', 'bold', 'FontSize', 13);
colorbar;
colormap('jet');
grid on;

% 5. Cumulative importance
subplot(2,3,5);
cumulative_importance = cumsum(sorted_combined) / sum(sorted_combined);
plot(1:num_features, cumulative_importance, 'b-', 'LineWidth', 2.5);
hold on;
yline(0.8, 'r--', 'LineWidth', 2, 'Label', '80% threshold');
yline(0.9, 'g--', 'LineWidth', 2, 'Label', '90% threshold');
xlabel('Number of Features', 'FontWeight', 'bold', 'FontSize', 11);
ylabel('Cumulative Importance', 'FontWeight', 'bold', 'FontSize', 11);
title('Cumulative Feature Importance', 'FontWeight', 'bold', 'FontSize', 13);
grid on;
legend('Location', 'southeast');

% Find 80% and 90% thresholds
idx_80 = find(cumulative_importance >= 0.8, 1);
idx_90 = find(cumulative_importance >= 0.9, 1);

% 6. Feature importance summary
subplot(2,3,6);
text(0.1, 0.95, 'FEATURE ANALYSIS SUMMARY', 'FontSize', 12, 'FontWeight', 'bold');
text(0.1, 0.85, sprintf('Total features: %d', num_features), 'FontSize', 10);
text(0.1, 0.78, sprintf('80%% importance: %d features', idx_80), 'FontSize', 10);
text(0.1, 0.71, sprintf('90%% importance: %d features', idx_90), 'FontSize', 10);
text(0.1, 0.62, 'TOP 5 FEATURES:', 'FontSize', 11, 'FontWeight', 'bold');
for i = 1:5
    text(0.1, 0.55-0.07*(i-1), sprintf('%d. %s (%.3f)', i, feature_names{idx_combined(i)}, sorted_combined(i)), 'FontSize', 9);
end
text(0.1, 0.15, 'RECOMMENDATIONS:', 'FontSize', 10, 'FontWeight', 'bold');
text(0.1, 0.08, sprintf('• Use top %d features (80%% coverage)', idx_80), 'FontSize', 9);
text(0.1, 0.01, sprintf('• Consider removing bottom %d features', num_features - idx_90), 'FontSize', 9);
axis off;

%% FIGURE 2: t-SNE VISUALIZATION
figure('Name', 't-SNE Feature Space Visualization', 'Position', [100 100 1600 900]);

% 1. t-SNE colored by RUL
subplot(2,3,1);
scatter(tsne_result(:,1), tsne_result(:,2), 50, rul_tsne, 'filled', 'MarkerEdgeColor', 'k', 'MarkerEdgeAlpha', 0.3);
xlabel('t-SNE Dimension 1', 'FontWeight', 'bold', 'FontSize', 11);
ylabel('t-SNE Dimension 2', 'FontWeight', 'bold', 'FontSize', 11);
title('t-SNE: Colored by RUL', 'FontWeight', 'bold', 'FontSize', 13);
colorbar;
colormap('jet');
grid on;

% 2. t-SNE colored by health state
subplot(2,3,2);
health_states = zeros(size(rul_tsne));
health_states(rul_tsne < 31) = 4;  % Critical
health_states(rul_tsne >= 31 & rul_tsne < 62) = 3;  % Advanced
health_states(rul_tsne >= 62 & rul_tsne < 94) = 2;  % Early
health_states(rul_tsne >= 94) = 1;  % Normal

scatter(tsne_result(:,1), tsne_result(:,2), 50, health_states, 'filled', 'MarkerEdgeColor', 'k');
xlabel('t-SNE Dimension 1', 'FontWeight', 'bold', 'FontSize', 11);
ylabel('t-SNE Dimension 2', 'FontWeight', 'bold', 'FontSize', 11);
title('t-SNE: Colored by Health State', 'FontWeight', 'bold', 'FontSize', 13);
colormap(gca, [0 0.7 0; 0.9 0.9 0; 1 0.5 0; 0.8 0 0]);
cb = colorbar;
set(cb, 'Ticks', [1.4 2.2 3 3.8], 'TickLabels', {'Normal', 'Early', 'Advanced', 'Critical'});
grid on;

% 3. t-SNE with contour density
subplot(2,3,3);
scatter(tsne_result(:,1), tsne_result(:,2), 30, rul_tsne, 'filled', 'MarkerEdgeAlpha', 0.4);
hold on;
% Add density contours
[f, xi] = ksdensity(tsne_result);
% Create 2D density (simplified)
contour(xi(:,1), xi(:,2), reshape(f, sqrt(length(f)), sqrt(length(f))), 5, 'LineWidth', 1.5);
xlabel('t-SNE Dimension 1', 'FontWeight', 'bold', 'FontSize', 11);
ylabel('t-SNE Dimension 2', 'FontWeight', 'bold', 'FontSize', 11);
title('t-SNE with Density Contours', 'FontWeight', 'bold', 'FontSize', 13);
colorbar;
grid on;

% 4. RUL distribution in t-SNE space
subplot(2,3,4);
histogram2(tsne_result(:,1), tsne_result(:,2), 30, 'DisplayStyle', 'tile', 'ShowEmptyBins', 'on');
xlabel('t-SNE Dimension 1', 'FontWeight', 'bold', 'FontSize', 11);
ylabel('t-SNE Dimension 2', 'FontWeight', 'bold', 'FontSize', 11);
title('Sample Density in t-SNE Space', 'FontWeight', 'bold', 'FontSize', 13);
colorbar;
view(2);
grid on;

% 5. Feature contribution to t-SNE
subplot(2,3,5);
% Calculate feature contribution (correlation with t-SNE dimensions)
feature_contrib = zeros(top_k_tsne, 1);
for i = 1:top_k_tsne
    contrib1 = abs(corr(features_tsne_selected(:,i), tsne_result(:,1)));
    contrib2 = abs(corr(features_tsne_selected(:,i), tsne_result(:,2)));
    feature_contrib(i) = sqrt(contrib1^2 + contrib2^2);
end

[sorted_contrib, idx_contrib] = sort(feature_contrib, 'descend');
top_tsne = min(15, top_k_tsne);

barh(sorted_contrib(1:top_tsne), 'FaceColor', [0.6 0.4 0.8], 'EdgeColor', 'k', 'LineWidth', 1.5);
set(gca, 'YTick', 1:top_tsne, 'YTickLabel', feature_names(idx_combined(idx_contrib(1:top_tsne))), 'YDir', 'reverse');
xlabel('Contribution to t-SNE', 'FontWeight', 'bold', 'FontSize', 11);
title('Features Driving t-SNE Separation', 'FontWeight', 'bold', 'FontSize', 13);
grid on;

% 6. t-SNE interpretation
subplot(2,3,6);
text(0.1, 0.95, 't-SNE INTERPRETATION', 'FontSize', 12, 'FontWeight', 'bold');
text(0.1, 0.85, sprintf('Samples visualized: %d', tsne_sample_size), 'FontSize', 10);
text(0.1, 0.78, sprintf('Features used: %d (top)', top_k_tsne), 'FontSize', 10);
text(0.1, 0.71, sprintf('Perplexity: 30'), 'FontSize', 10);

% Calculate cluster separation
normal_idx = health_states == 1;
critical_idx = health_states == 4;
if sum(normal_idx) > 0 && sum(critical_idx) > 0
    normal_center = mean(tsne_result(normal_idx, :));
    critical_center = mean(tsne_result(critical_idx, :));
    separation = norm(normal_center - critical_center);
    text(0.1, 0.62, sprintf('Normal-Critical separation: %.2f', separation), 'FontSize', 10);
end

text(0.1, 0.52, 'KEY INSIGHTS:', 'FontSize', 11, 'FontWeight', 'bold');
text(0.1, 0.45, '• Clustering indicates degradation', 'FontSize', 9);
text(0.1, 0.38, '  patterns in feature space', 'FontSize', 9);
text(0.1, 0.31, '• Colors show RUL progression', 'FontSize', 9);
text(0.1, 0.24, '• Overlap suggests similar states', 'FontSize', 9);
text(0.1, 0.17, '• Clear separation = good features', 'FontSize', 9);
text(0.1, 0.08, 'USE: Verify feature quality before', 'FontSize', 9, 'FontWeight', 'bold');
text(0.1, 0.01, 'training LSTM model', 'FontSize', 9, 'FontWeight', 'bold');
axis off;

%% FIGURE 3: CORRELATION HEATMAP
figure('Name', 'Top Features Correlation Matrix', 'Position', [150 150 1200 900]);

top_heatmap = min(25, num_features);
top_features = idx_combined(1:top_heatmap);
corr_matrix_top = corrcoef(features_sample(:, top_features));

imagesc(corr_matrix_top);
colorbar;
colormap('jet');
caxis([-1 1]);

set(gca, 'XTick', 1:top_heatmap, 'XTickLabel', feature_names(top_features), ...
    'XTickLabelRotation', 90, 'FontSize', 8);
set(gca, 'YTick', 1:top_heatmap, 'YTickLabel', feature_names(top_features), 'FontSize', 8);

xlabel('Features', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('Features', 'FontWeight', 'bold', 'FontSize', 12);
title(sprintf('Correlation Matrix: Top %d Features', top_heatmap), 'FontWeight', 'bold', 'FontSize', 14);

% Add correlation values
for i = 1:top_heatmap
    for j = 1:top_heatmap
        if abs(corr_matrix_top(i,j)) > 0.7 && i ~= j
            text(j, i, sprintf('%.2f', corr_matrix_top(i,j)), ...
                'HorizontalAlignment', 'center', 'Color', 'white', ...
                'FontWeight', 'bold', 'FontSize', 7);
        end
    end
end

%% PRINT SUMMARY
fprintf('\n================================================================\n');
fprintf('                  FEATURE ANALYSIS COMPLETE\n');
fprintf('================================================================\n\n');

fprintf('TOP 10 FEATURES (Combined Score):\n');
fprintf('%-5s %-30s %-10s %-10s %-10s\n', 'Rank', 'Feature', 'Combined', 'Corr', 'MI');
fprintf('%s\n', repmat('-', 1, 75));
for i = 1:10
    feat_idx = idx_combined(i);
    fprintf('%-5d %-30s %-10.4f %-10.4f %-10.4f\n', i, feature_names{feat_idx}, ...
        combined_scores(feat_idx), correlation_scores(feat_idx), mi_scores(feat_idx));
end

fprintf('\n\nFEATURE SELECTION RECOMMENDATIONS:\n');
fprintf('  • %d features capture 80%% of importance\n', idx_80);
fprintf('  • %d features capture 90%% of importance\n', idx_90);
fprintf('  • Consider using top %d-%d features for training\n', idx_80, idx_90);

fprintf('\n\nt-SNE VISUALIZATION:\n');
fprintf('  • %d samples plotted in 2D\n', tsne_sample_size);
fprintf('  • Check for clear health state separation\n');
fprintf('  • Clusters indicate feature quality\n');

fprintf('\n================================================================\n');
fprintf('All visualizations generated! Review the figures.\n');
fprintf('================================================================\n\n');S