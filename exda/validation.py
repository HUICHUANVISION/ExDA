
"""验证模式模块 - 专门验证RQ1和RQ2"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import shutil

from data_loader import load_data_from_csv, create_small_dataset
from exda_model import ExDA
from sampling_methods import apply_sampling_methods
from metrics import evaluate_model_comprehensive, calculate_improvements
from config import (SAMPLING_METHODS, EXDA_PARAMS, VALIDATION_METRICS, 
                   VALIDATION_DEFECT_RATES, VALIDATION_MODE, 
                   VALIDATION_DATASET_SPECIFIC_PARAMS, REDUCTION_RATIO)

from sklearn.utils import shuffle

def get_dataset_specific_params(dataset_name):
    """
    获取特定数据集的参数（包括reduction_ratio和ExDA参数）
    """
    if dataset_name in VALIDATION_DATASET_SPECIFIC_PARAMS:
        # 使用数据集特定参数
        specific_params = VALIDATION_DATASET_SPECIFIC_PARAMS[dataset_name].copy()
        print(f"使用数据集 '{dataset_name}' 的特定参数")
        print(f"  - reduction_ratio: {specific_params.get('reduction_ratio', 'default')}")
        print(f"  - augmentation_percentage: {specific_params.get('augmentation_percentage', 'default')}")
        print(f"  - target_ratio: {specific_params.get('target_ratio', 'default')}")
    else:
        # 使用默认参数
        specific_params = VALIDATION_DATASET_SPECIFIC_PARAMS['default'].copy()
        print(f"数据集 '{dataset_name}' 使用默认参数")
    
    # 确保包含所有必要的参数
    default_params = VALIDATION_DATASET_SPECIFIC_PARAMS['default']
    for key in default_params:
        if key not in specific_params:
            specific_params[key] = default_params[key]
    
    return specific_params

def validate_rq1_small_datasets(folder_path, output_dir, target_column='class', dataset_name=None):
    """
    RQ1验证: 在4个人造小数据集上验证ExDA相对于基线方法的优势
    """
    print("=== RQ1验证: 小数据集上的整体优势 ===")
    
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if len(csv_files) < 2:
        print("数据集数量不足")
        return None, None
    
    # 获取数据集特定参数
    if dataset_name:
        dataset_params = get_dataset_specific_params(dataset_name)
    else:
        dataset_params = VALIDATION_DATASET_SPECIFIC_PARAMS['default'].copy()
    
    # 分离参数
    reduction_ratio = dataset_params.pop('reduction_ratio', REDUCTION_RATIO)
    exda_params = dataset_params  # 剩余参数用于ExDA
    
    results_rq1 = []
    
    for train_file in tqdm(csv_files, desc="RQ1 - 训练数据集"):
        try:
            # 加载完整数据
            X_full, y_full, _ = load_data_from_csv(
                os.path.join(folder_path, train_file), target_column
            )
            
            # 确保y是数值类型
            y_full = y_full.astype(int)
            
            # 创建小数据集（使用特定数据集的reduction_ratio）
            X_small, y_small = create_small_dataset(X_full, y_full, reduction_ratio=reduction_ratio)
            
            # 训练ExDA（使用特定数据集参数）
            exda = ExDA(**exda_params)
            exda.fit(X_small, y_small)
            X_aug, y_aug = exda.augment(X_small, y_small)
            
            # 应用基线采样方法
            sampling_results = apply_sampling_methods(X_small, y_small, target_ratio=exda_params.get('target_ratio', 0.3))
            
            # 测试所有其他文件
            test_files = [f for f in csv_files if f != train_file]
            for test_file in test_files:
                try:
                    X_test, y_test, _ = load_data_from_csv(
                        os.path.join(folder_path, test_file), target_column
                    )
                    
                    # 确保y是数值类型
                    y_test = y_test.astype(int)
                    
                    # 评估所有方法
                    methods_metrics = {}
                    
                    # 1. 小数据集（无采样）
                    small_model = LogisticRegression(random_state=42, max_iter=1000)
                    small_model.fit(X_small, y_small)
                    y_pred_small = small_model.predict(X_test)
                    y_prob_small = small_model.predict_proba(X_test)[:, 1]
                    methods_metrics['small'] = evaluate_model_comprehensive(X_small, y_small, X_test, y_test)
                    
                    # 2. ExDA增强
                    aug_model = LogisticRegression(random_state=42, max_iter=1000)
                    aug_model.fit(X_aug, y_aug)
                    methods_metrics['exda'] = evaluate_model_comprehensive(X_aug, y_aug, X_test, y_test)
                    
                    # 3. 基线采样方法
                    for method in SAMPLING_METHODS:
                        if method in sampling_results:
                            model = LogisticRegression(random_state=42, max_iter=1000)
                            X_method = sampling_results[method]['X']
                            y_method = sampling_results[method]['y']
                            model.fit(X_method, y_method)
                            methods_metrics[method] = evaluate_model_comprehensive(X_method, y_method, X_test, y_test)
                    
                    # 记录RQ1结果
                    for metric in VALIDATION_METRICS:
                        result = {
                            'train_file': train_file,
                            'test_file': test_file,
                            'metric': metric,
                            'defect_rate': np.sum(y_small) / len(y_small),
                            'exda_performance': methods_metrics['exda'][metric],
                            'small_performance': methods_metrics['small'][metric],
                            'dataset_name': dataset_name if dataset_name else 'unknown',
                            'reduction_ratio': reduction_ratio  # 记录使用的缩减比例
                        }
                        
                        # 记录使用的参数
                        result['reduction_ratio_used'] = reduction_ratio
                        for param_name, param_value in exda_params.items():
                            result[f'exda_param_{param_name}'] = param_value
                        
                        # 记录基线方法性能
                        for method in SAMPLING_METHODS:
                            if method in methods_metrics:
                                result[f'{method}_performance'] = methods_metrics[method][metric]
                        
                        results_rq1.append(result)
                        
                except Exception as e:
                    print(f"测试文件 {test_file} 处理失败: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"训练文件 {train_file} 处理失败: {str(e)}")
            continue
    
    # 保存RQ1结果
    if results_rq1:
        rq1_df = pd.DataFrame(results_rq1)
        rq1_file = os.path.join(output_dir, 'rq1_small_datasets_comparison.csv')
        rq1_df.to_csv(rq1_file, index=False)
        
        # 生成RQ1汇总统计
        rq1_summary = generate_rq1_summary(rq1_df)
        rq1_summary_file = os.path.join(output_dir, 'rq1_summary_statistics.csv')
        rq1_summary.to_csv(rq1_summary_file, index=False)
        
        print(f"RQ1验证完成，结果保存至: {rq1_file}")
        return rq1_df, rq1_summary
    
    print("RQ1验证没有生成任何结果")
    return None, None

def validate_rq2_defect_rates(folder_path, output_dir, target_column='class', dataset_name=None):
    """
    RQ2验证: 在不同缺陷率情况下验证F1和Recall提升
    """
    print("=== RQ2验证: 不同缺陷率下的性能提升 ===")
    
    # 获取数据集特定参数
    if dataset_name:
        dataset_params = get_dataset_specific_params(dataset_name)
    else:
        dataset_params = VALIDATION_DATASET_SPECIFIC_PARAMS['default'].copy()
    
    # 分离参数
    reduction_ratio = dataset_params.pop('reduction_ratio', REDUCTION_RATIO)
    exda_params = dataset_params  # 剩余参数用于ExDA
    
    # 首先创建不同缺陷率的合成数据集
    base_dataset = None
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if csv_files:
        base_dataset = os.path.join(folder_path, csv_files[0])
    
    if not base_dataset:
        print("找不到基础数据集")
        return None, None
    
    # 创建不同缺陷率的数据集
    synthetic_dir = os.path.join(output_dir, 'synthetic_defect_rates')
    datasets_info = create_synthetic_datasets_with_defect_rates(
        base_dataset, synthetic_dir, VALIDATION_DEFECT_RATES, target_column
    )
    
    if not datasets_info:
        print("创建合成数据集失败")
        return None, None
    
    results_rq2 = []
    
    # 对每个缺陷率数据集进行实验
    for dataset_info in tqdm(datasets_info, desc="RQ2 - 不同缺陷率"):
        defect_rate = dataset_info['defect_rate']
        dataset_path = dataset_info['path']
        
        try:
            # 加载数据
            X, y, _ = load_data_from_csv(dataset_path, target_column)
            
            # 确保y是数值类型
            y = y.astype(int)
            
            # 创建小数据集（使用特定数据集的reduction_ratio）
            X_small, y_small = create_small_dataset(X, y, reduction_ratio=reduction_ratio)
            
            # 训练ExDA（使用特定数据集参数）
            exda = ExDA(**exda_params)
            exda.fit(X_small, y_small)
            X_aug, y_aug = exda.augment(X_small, y_small)
            
            # 应用基线采样方法
            sampling_results = apply_sampling_methods(X_small, y_small, target_ratio=exda_params.get('target_ratio', 0.3))
            
            # 使用其他数据集进行测试
            test_datasets = [info for info in datasets_info if info['defect_rate'] != defect_rate]
            
            for test_info in test_datasets[:2]:  # 限制测试数据集数量
                try:
                    X_test, y_test, _ = load_data_from_csv(test_info['path'], target_column)
                    
                    # 确保y是数值类型
                    y_test = y_test.astype(int)
                    
                    # 评估方法
                    methods_metrics = {}
                    
                    # 小数据集
                    small_model = LogisticRegression(random_state=42, max_iter=1000)
                    small_model.fit(X_small, y_small)
                    methods_metrics['small'] = evaluate_model_comprehensive(X_small, y_small, X_test, y_test)
                    
                    # ExDA
                    aug_model = LogisticRegression(random_state=42, max_iter=1000)
                    aug_model.fit(X_aug, y_aug)
                    methods_metrics['exda'] = evaluate_model_comprehensive(X_aug, y_aug, X_test, y_test)
                    
                    # 基线方法
                    for method in ['SMOTE', 'ADASYN', 'RUS', 'ROS']:
                        if method in sampling_results:
                            model = LogisticRegression(random_state=42, max_iter=1000)
                            X_method = sampling_results[method]['X']
                            y_method = sampling_results[method]['y']
                            model.fit(X_method, y_method)
                            methods_metrics[method] = evaluate_model_comprehensive(X_method, y_method, X_test, y_test)
                    
                    # 记录RQ2结果
                    for method in ['exda', 'SMOTE', 'ADASYN', 'RUS', 'ROS']:
                        if method in methods_metrics:
                            improvements = calculate_improvements(methods_metrics[method], methods_metrics['small'])
                            
                            result = {
                                'train_defect_rate': defect_rate,
                                'test_defect_rate': test_info['defect_rate'],
                                'method': method,
                                'f1_improvement': improvements.get('F1', 0),
                                'recall_improvement': improvements.get('Recall', 0),
                                'f1_absolute': methods_metrics[method]['F1'],
                                'recall_absolute': methods_metrics[method]['Recall'],
                                'small_f1': methods_metrics['small']['F1'],
                                'small_recall': methods_metrics['small']['Recall'],
                                'dataset_name': dataset_name if dataset_name else 'unknown',
                                'reduction_ratio': reduction_ratio  # 记录使用的缩减比例
                            }
                            
                            # 记录参数（仅对ExDA方法）
                            if method == 'exda':
                                result['reduction_ratio_used'] = reduction_ratio
                                for param_name, param_value in exda_params.items():
                                    result[f'exda_param_{param_name}'] = param_value
                            
                            results_rq2.append(result)
                            
                except Exception as e:
                    print(f"测试数据集 {test_info['path']} 处理失败: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"训练数据集 {dataset_path} 处理失败: {str(e)}")
            continue
    
    # 保存RQ2结果
    if results_rq2:
        rq2_df = pd.DataFrame(results_rq2)
        rq2_file = os.path.join(output_dir, 'rq2_defect_rates_comparison.csv')
        rq2_df.to_csv(rq2_file, index=False)
        
        # 生成RQ2汇总统计
        rq2_summary = generate_rq2_summary(rq2_df)
        rq2_summary_file = os.path.join(output_dir, 'rq2_summary_statistics.csv')
        rq2_summary.to_csv(rq2_summary_file, index=False)
        
        print(f"RQ2验证完成，结果保存至: {rq2_file}")
        return rq2_df, rq2_summary
    
    print("RQ2验证没有生成任何结果")
    return None, None

def create_synthetic_datasets_with_defect_rates(base_dataset_path, output_dir, defect_rates, target_column='class'):
    """
    RQ2: 创建不同缺陷率的合成数据集（修复版本）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        X, y, feature_names = load_data_from_csv(base_dataset_path, target_column)
        
        # 确保y是数值类型
        y = y.astype(int)
        
        datasets_info = []
        
        # 计算原始数据集的缺陷率
        original_defect_rate = np.sum(y) / len(y)
        print(f"原始数据集缺陷率: {original_defect_rate:.3f}")
        
        for defect_rate in tqdm(defect_rates, desc="创建不同缺陷率数据集"):
            # 调整缺陷率 - 使用更保守的方法
            positive_indices = np.where(y == 1)[0]
            negative_indices = np.where(y == 0)[0]
            
            # 确定最大可能的样本数量
            max_positive = min(len(positive_indices), int(len(y) * defect_rate))
            max_negative = min(len(negative_indices), int(len(y) * (1 - defect_rate)))
            
            # 调整目标数量以避免超出可用样本
            n_positive_target = max_positive
            n_negative_target = max_negative
            
            # 如果调整后样本太少，按比例缩放
            total_target = n_positive_target + n_negative_target
            if total_target < len(y) * 0.5:  # 如果目标样本太少
                scale_factor = (len(y) * 0.8) / total_target  # 放大到原始数据的80%
                n_positive_target = min(int(n_positive_target * scale_factor), len(positive_indices))
                n_negative_target = min(int(n_negative_target * scale_factor), len(negative_indices))
            
            print(f"目标缺陷率: {defect_rate:.3f}, 实际使用正样本: {n_positive_target}, 负样本: {n_negative_target}")
            
            # 选择正样本（允许重复如果不够）
            if len(positive_indices) >= n_positive_target:
                selected_positive = np.random.choice(positive_indices, n_positive_target, replace=False)
            else:
                selected_positive = np.random.choice(positive_indices, n_positive_target, replace=True)
            
            # 选择负样本（允许重复如果不够）
            if len(negative_indices) >= n_negative_target:
                selected_negative = np.random.choice(negative_indices, n_negative_target, replace=False)
            else:
                selected_negative = np.random.choice(negative_indices, n_negative_target, replace=True)
            
            selected_indices = np.concatenate([selected_positive, selected_negative])
            np.random.shuffle(selected_indices)
            
            X_synthetic = X[selected_indices]
            y_synthetic = y[selected_indices]
            
            # 计算实际缺陷率
            actual_defect_rate = np.sum(y_synthetic) / len(y_synthetic)
            print(f"  实际缺陷率: {actual_defect_rate:.3f}, 总样本数: {len(y_synthetic)}")
            
            # 保存数据集
            dataset_name = f"defect_rate_{defect_rate:.2f}.csv"
            dataset_path = os.path.join(output_dir, dataset_name)
            
            df_synthetic = pd.DataFrame(X_synthetic, columns=feature_names)
            df_synthetic['class'] = y_synthetic
            df_synthetic.to_csv(dataset_path, index=False)
            
            datasets_info.append({
                'name': dataset_name,
                'path': dataset_path,
                'defect_rate': defect_rate,
                'actual_defect_rate': actual_defect_rate,
                'samples': len(y_synthetic),
                'positive_count': np.sum(y_synthetic),
                'negative_count': len(y_synthetic) - np.sum(y_synthetic)
            })
        
        return datasets_info
        
    except Exception as e:
        print(f"创建合成数据集失败: {str(e)}")
        return []

def generate_rq1_summary(rq1_df):
    """生成RQ1汇总统计 - 修复版本"""
    summary_data = []
    
    for metric in VALIDATION_METRICS:
        metric_data = rq1_df[rq1_df['metric'] == metric]
        
        if len(metric_data) == 0:
            continue
            
        # ExDA性能统计
        exda_perf = metric_data['exda_performance']
        small_perf = metric_data['small_performance']
        
        # 创建基础汇总记录
        summary_record = {
            'metric': metric,
            'exda_mean': exda_perf.mean(),
            'exda_std': exda_perf.std(),
            'small_mean': small_perf.mean(),
            'small_std': small_perf.std(),
            'num_experiments': len(metric_data),
            'exda_vs_small_improvement': (exda_perf.mean() - small_perf.mean()) / small_perf.mean() * 100,
            'exda_win_rate_vs_small': (exda_perf > small_perf).mean() * 100
        }
        
        # 与每个基线方法的比较
        for method in SAMPLING_METHODS:
            method_col = f'{method}_performance'
            if method_col in metric_data.columns:
                method_perf = metric_data[method_col]
                
                # 计算改进百分比和胜率
                improvement = (exda_perf.mean() - method_perf.mean()) / method_perf.mean() * 100
                win_rate = (exda_perf > method_perf).mean() * 100
                
                # 添加到汇总记录
                summary_record[f'{method}_mean'] = method_perf.mean()
                summary_record[f'{method}_std'] = method_perf.std()
                summary_record[f'exda_vs_{method}_improvement'] = improvement
                summary_record[f'exda_vs_{method}_win_rate'] = win_rate
        
        summary_data.append(summary_record)
    
    # 创建DataFrame并返回
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def generate_rq2_summary(rq2_df):
    """生成RQ2汇总统计"""
    summary_data = []
    
    # 按缺陷率分组
    for defect_rate in VALIDATION_DEFECT_RATES:
        rate_data = rq2_df[rq2_df['train_defect_rate'] == defect_rate]
        
        if len(rate_data) == 0:
            continue
            
        for method in ['exda', 'SMOTE', 'ADASYN', 'RUS', 'ROS']:
            method_data = rate_data[rate_data['method'] == method]
            
            if len(method_data) == 0:
                continue
                
            summary_data.append({
                'defect_rate': defect_rate,
                'method': method,
                'avg_f1_improvement': method_data['f1_improvement'].mean(),
                'avg_recall_improvement': method_data['recall_improvement'].mean(),
                'f1_improvement_std': method_data['f1_improvement'].std(),
                'recall_improvement_std': method_data['recall_improvement'].std(),
                'avg_f1_absolute': method_data['f1_absolute'].mean(),
                'avg_recall_absolute': method_data['recall_absolute'].mean(),
                'num_tests': len(method_data)
            })
    
    return pd.DataFrame(summary_data)

def run_statistical_validation(parent_folder_path, num_runs=1):
    """
    运行统计性验证：重复实验30次，分别保存每次的RQ1汇总统计结果
    """
    if not VALIDATION_MODE:
        print("验证模式未启用")
        return
    
    # 创建主验证目录
    validation_dir = os.path.join(parent_folder_path, "statistical_validation")
    os.makedirs(validation_dir, exist_ok=True)
    
    # 获取所有数据集文件夹
    dataset_folders = [f for f in os.listdir(parent_folder_path) 
                      if os.path.isdir(os.path.join(parent_folder_path, f))]
    
    all_runs_summary = []
    
    # 运行30次实验
    for run_num in tqdm(range(1, num_runs + 1), desc="统计性验证运行"):
        print(f"\n=== 第 {run_num} 次运行 ===")
        
        # 为每次运行创建单独的目录
        run_dir = os.path.join(validation_dir, f"run_{run_num:02d}")
        os.makedirs(run_dir, exist_ok=True)
        
        run_summary = {
            'run_number': run_num,
            'run_directory': run_dir
        }
        
        # 对每个数据集运行验证
        for dataset_folder in dataset_folders:
            folder_path = os.path.join(parent_folder_path, dataset_folder)
            
            # 检查是否有CSV文件
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            if len(csv_files) < 2:
                print(f"数据集 {dataset_folder} 中CSV文件不足，跳过")
                continue
            
            dataset_run_dir = os.path.join(run_dir, dataset_folder)
            os.makedirs(dataset_run_dir, exist_ok=True)
            
            print(f"正在验证数据集: {dataset_folder}")
            
            # 只运行RQ1验证（根据要求）
            rq1_df, rq1_summary = validate_rq1_small_datasets(
                folder_path, dataset_run_dir, dataset_name=dataset_folder
            )
            
            if rq1_summary is not None:
                # 在汇总统计中添加运行信息
                rq1_summary['run_number'] = run_num
                rq1_summary['dataset'] = dataset_folder
                
                # 保存到运行目录
                rq1_summary_file = os.path.join(dataset_run_dir, 'rq1_summary_statistics.csv')
                rq1_summary.to_csv(rq1_summary_file, index=False)
                
                # 添加到总汇总
                all_runs_summary.append(rq1_summary)
                
                # 清理其他文件，只保留rq1_summary_statistics.csv
                cleanup_run_directory(dataset_run_dir)
                
                # 记录运行信息
                run_summary[f'{dataset_folder}_completed'] = True
                run_summary[f'{dataset_folder}_summary_file'] = rq1_summary_file
            else:
                print(f"数据集 {dataset_folder} 的RQ1验证没有结果")
                run_summary[f'{dataset_folder}_completed'] = False
        
        # 保存本次运行的汇总信息
        run_summary_file = os.path.join(run_dir, 'run_summary.json')
        pd.Series(run_summary).to_json(run_summary_file)
    
    # 合并所有运行的RQ1汇总统计
    if all_runs_summary:
        # 合并所有数据集的汇总统计
        combined_summary = pd.concat(all_runs_summary, ignore_index=True)
        combined_summary_file = os.path.join(validation_dir, 'all_runs_combined_summary.csv')
        combined_summary.to_csv(combined_summary_file, index=False)
        
        # 生成统计性分析报告
        statistical_report = generate_statistical_report(combined_summary)
        report_file = os.path.join(validation_dir, 'statistical_analysis_report.csv')
        statistical_report.to_csv(report_file, index=False)
        
        print(f"\n统计性验证完成！")
        print(f"总共运行了 {num_runs} 次实验")
        print(f"合并汇总文件: {combined_summary_file}")
        print(f"统计分析报告: {report_file}")
    else:
        print("统计性验证没有生成任何结果")
    
    return combined_summary

def cleanup_run_directory(run_dir):
    """
    清理运行目录，只保留rq1_summary_statistics.csv文件
    """
    try:
        for filename in os.listdir(run_dir):
            if filename != 'rq1_summary_statistics.csv':
                file_path = os.path.join(run_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        print(f"已清理目录 {run_dir}，只保留RQ1汇总统计文件")
    except Exception as e:
        print(f"清理目录 {run_dir} 时出错: {str(e)}")

def generate_statistical_report(combined_summary):
    """
    生成统计性分析报告
    """
    report_data = []
    
    # 按数据集和指标分组
    for dataset in combined_summary['dataset'].unique():
        dataset_data = combined_summary[combined_summary['dataset'] == dataset]
        
        for metric in VALIDATION_METRICS:
            metric_data = dataset_data[dataset_data['metric'] == metric]
            
            if len(metric_data) == 0:
                continue
            
            # 计算统计量
            exda_means = metric_data['exda_mean']
            small_means = metric_data['small_mean']
            
            report_record = {
                'dataset': dataset,
                'metric': metric,
                'num_runs': len(metric_data),
                'exda_mean_over_runs': exda_means.mean(),
                'exda_std_over_runs': exda_means.std(),
                'exda_min_over_runs': exda_means.min(),
                'exda_max_over_runs': exda_means.max(),
                'small_mean_over_runs': small_means.mean(),
                'small_std_over_runs': small_means.std(),
                'mean_improvement_percentage': metric_data['exda_vs_small_improvement'].mean(),
                'improvement_std': metric_data['exda_vs_small_improvement'].std(),
                'mean_win_rate': metric_data['exda_win_rate_vs_small'].mean()
            }
            
            # 计算t检验统计量（简化版）
            improvement_significant = (metric_data['exda_vs_small_improvement'] > 0).mean() * 100
            report_record['improvement_significant_rate'] = improvement_significant
            
            report_data.append(report_record)
    
    return pd.DataFrame(report_data)

def run_validation_mode(parent_folder_path):
    """运行完整的验证模式"""
    if not VALIDATION_MODE:
        print("验证模式未启用")
        return
    
    # 询问用户是否运行统计性验证
    response = 'y'
    if response.lower() in ['y', 'yes']:
        return run_statistical_validation(parent_folder_path, num_runs=30)
    
    # 原有的单次验证模式
    validation_dir = os.path.join(parent_folder_path, "validation_results")
    os.makedirs(validation_dir, exist_ok=True)
    
    # 获取所有数据集文件夹
    dataset_folders = [f for f in os.listdir(parent_folder_path) 
                      if os.path.isdir(os.path.join(parent_folder_path, f))]
    
    all_rq1_results = []
    all_rq2_results = []
    
    for dataset_folder in tqdm(dataset_folders, desc="验证数据集"):
        folder_path = os.path.join(parent_folder_path, dataset_folder)
        
        # 检查是否有CSV文件
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        if len(csv_files) < 2:
            print(f"数据集 {dataset_folder} 中CSV文件不足，跳过")
            continue
        
        dataset_validation_dir = os.path.join(validation_dir, dataset_folder)
        os.makedirs(dataset_validation_dir, exist_ok=True)
        
        print(f"\n正在验证数据集: {dataset_folder}")
        
        # RQ1验证（传递数据集名称）
        rq1_df, rq1_summary = validate_rq1_small_datasets(
            folder_path, dataset_validation_dir, dataset_name=dataset_folder
        )
        
        if rq1_df is not None:
            rq1_df['dataset'] = dataset_folder
            all_rq1_results.append(rq1_df)
        else:
            print(f"数据集 {dataset_folder} 的RQ1验证没有结果")
        
        # RQ2验证（传递数据集名称）
        rq2_df, rq2_summary = validate_rq2_defect_rates(
            folder_path, dataset_validation_dir, dataset_name=dataset_folder
        )
        
        if rq2_df is not None:
            rq2_df['dataset'] = dataset_folder
            all_rq2_results.append(rq2_df)
        else:
            print(f"数据集 {dataset_folder} 的RQ2验证没有结果")
    
    # 保存汇总结果
    if all_rq1_results:
        final_rq1 = pd.concat(all_rq1_results, ignore_index=True)
        final_rq1.to_csv(os.path.join(validation_dir, 'all_rq1_results.csv'), index=False)
        print(f"保存了 {len(all_rq1_results)} 个数据集的RQ1结果")
    else:
        print("没有可保存的RQ1结果")
    
    if all_rq2_results:
        final_rq2 = pd.concat(all_rq2_results, ignore_index=True)
        final_rq2.to_csv(os.path.join(validation_dir, 'all_rq2_results.csv'), index=False)
        print(f"保存了 {len(all_rq2_results)} 个数据集的RQ2结果")
    else:
        print("没有可保存的RQ2结果")
    
    print("\n=== 验证模式完成 ===")
    print(f"结果保存在: {validation_dir}")