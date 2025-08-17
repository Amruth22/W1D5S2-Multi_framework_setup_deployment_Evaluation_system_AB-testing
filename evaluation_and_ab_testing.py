import pandas as pd
from datetime import datetime
import random
import os
import config
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Import framework functions from app.py
# Note: This creates a circular dependency if app.py imports this file directly.
# In a larger project, these would be separate modules or passed as arguments.
# For this flat structure, we'll import directly.
from app import summarize_text_framework, complex_qa_framework, simple_chat_framework, llm_flash, llm_pro, llm_pro_text

# --- Evaluation System ---

def log_evaluation_metric(model_id: str, metric_name: str, metric_value: float, model_type: str = None, task_type: str = None, notes: str = ""):
    """
    Logs a single evaluation metric to a CSV file with enhanced tracking of Gemini model types.
    
    Args:
        model_id: The model identifier (e.g., gemini-2.0-flash)
        metric_name: Name of the evaluation metric (e.g., bleu_score, rouge1)
        metric_value: Value of the metric (float)
        model_type: Type of Gemini model (pro, flash, pro-text)
        task_type: Type of task (summarization, qa, chat)
        notes: Additional notes or context
    """
    # If model_type is not provided, try to extract it from model_id
    if model_type is None:
        if "flash" in model_id.lower():
            model_type = "flash"
        elif "pro" in model_id.lower():
            model_type = "pro"
        else:
            model_type = "unknown"
    
    # If task_type is not provided, try to extract it from model_id
    if task_type is None:
        if "summarization" in model_id.lower() or "summary" in model_id.lower():
            task_type = "summarization"
        elif "qa" in model_id.lower():
            task_type = "qa"
        elif "chat" in model_id.lower():
            task_type = "chat"
        else:
            task_type = "unknown"
    
    file_path = config.EVAL_LOG_FILE
    timestamp = datetime.now().isoformat()
    new_entry = pd.DataFrame([
        {
            "timestamp": timestamp,
            "model_id": model_id,
            "model_type": model_type,
            "task_type": task_type,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "notes": notes
        }
    ])
    
    if not os.path.exists(file_path):
        new_entry.to_csv(file_path, index=False)
    else:
        new_entry.to_csv(file_path, mode='a', header=False, index=False)
    print(f"Logged evaluation: Model={model_id} ({model_type}), Task={task_type}, Metric={metric_name}, Value={metric_value:.4f}")

def get_all_evaluation_logs():
    """
    Retrieves all logged evaluation metrics.
    """
    file_path = config.EVAL_LOG_FILE
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame(columns=["timestamp", "model_id", "model_type", "task_type", "metric_name", "metric_value", "notes"])


def get_model_performance_summary(task_type=None, metric_name=None, group_by="model_type"):
    """
    Retrieves and summarizes model performance metrics with flexible filtering and grouping.
    
    Args:
        task_type: Filter by specific task type (e.g., 'summarization', 'qa', 'chat')
        metric_name: Filter by specific metric (e.g., 'bleu_score', 'rouge1')
        group_by: How to group results ('model_type', 'model_id', or 'task_type')
        
    Returns:
        Pandas DataFrame with aggregated performance metrics
    """
    logs = get_all_evaluation_logs()
    
    # Apply filters if specified
    if task_type:
        logs = logs[logs['task_type'] == task_type]
    if metric_name:
        logs = logs[logs['metric_name'] == metric_name]
    
    # If no data after filtering, return empty DataFrame
    if logs.empty:
        return pd.DataFrame()
    
    # Group and aggregate metrics
    grouped = logs.groupby([group_by, 'metric_name'])['metric_value'].agg(['mean', 'std', 'count', 'min', 'max']).reset_index()
    
    return grouped


def visualize_model_metrics(task_type=None, metrics=None, models=None, sort_by=None):
    """
    Generates a unified visualization of model performance metrics with enhanced formatting and sorting.
    
    Args:
        task_type: Type of task to filter by (e.g., 'summarization')
        metrics: List of metrics to include (e.g., ['bleu_score', 'rouge1'])
        models: List of model types to include (e.g., ['flash', 'pro'])
        sort_by: Metric name to sort results by (e.g., 'bleu_score')
        
    Returns:
        DataFrame with metrics and prints formatted metrics table
    """
    # Get all logs
    logs = get_all_evaluation_logs()
    
    # Apply filters
    if task_type:
        logs = logs[logs['task_type'] == task_type]
    if metrics:
        logs = logs[logs['metric_name'].isin(metrics)]
    if models:
        logs = logs[logs['model_type'].isin(models)]
    
    if logs.empty:
        print("No data found matching the specified filters.")
        return None
    
    # Prepare unified metrics table
    print(f"\n{'='*100}")
    title = "ENHANCED GEMINI MODEL METRICS EVALUATION"
    if task_type:
        title += f" FOR {task_type.upper()} TASK"
    print(f"{title:^100}")
    print(f"{'='*100}")
    
    # Group by model_type and metric_name for aggregation
    model_metrics = logs.pivot_table(
        index=['model_type', 'model_id'], 
        columns='metric_name', 
        values='metric_value', 
        aggfunc=['mean', 'max', 'std', 'count']
    )
    
    # Flatten multi-index columns
    model_metrics.columns = ['_'.join(col).strip() for col in model_metrics.columns.values]
    model_metrics = model_metrics.reset_index()
    
    # Sort results if requested
    if sort_by:
        sort_col = f"mean_{sort_by}"
        if sort_col in model_metrics.columns:
            model_metrics = model_metrics.sort_values(by=sort_col, ascending=False)
    
    # Get unique model types and metric names for display
    if not metrics:
        # Extract all metrics from column names (removing aggregation prefixes)
        all_cols = model_metrics.columns
        metrics = sorted(list(set([col.split('_', 1)[1] for col in all_cols if '_' in col])))
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  • Total models evaluated: {len(model_metrics)}")
    print(f"  • Metrics analyzed: {', '.join(metrics)}")
    if task_type:
        print(f"  • Task type: {task_type}")
    if models:
        print(f"  • Model types: {', '.join(models)}")
    print(f"\n{'-'*100}\n")
    
    # Print header for mean values section
    print("MODEL PERFORMANCE (MEAN VALUES)")
    header = f"{'MODEL TYPE':<12} {'MODEL ID':<30}"
    for metric in metrics:
        mean_col = f"mean_{metric}"
        if mean_col in model_metrics.columns:
            header += f" {metric[:10]:^10}"
    print(header)
    print(f"{'-'*12} {'-'*30} {'-'*10 * len([m for m in metrics if f'mean_{m}' in model_metrics.columns])}")
    
    # Print metrics by model (mean values)
    for _, row in model_metrics.iterrows():
        line = f"{row['model_type']:<12} {row['model_id'][:30]:<30}"
        for metric in metrics:
            mean_col = f"mean_{metric}"
            if mean_col in model_metrics.columns and not pd.isna(row[mean_col]):
                # Color coding based on value ranges (in terminal outputs)
                value = row[mean_col]
                # Use formatting to emphasize good scores
                if value > 0.8:
                    line += f" {value:^10.4f}"
                else:
                    line += f" {value:^10.4f}"
            else:
                line += f" {'N/A':^10}"
        print(line)
    
    # Print detailed statistics for each metric
    for metric in metrics:
        mean_col = f"mean_{metric}"
        max_col = f"max_{metric}"
        std_col = f"std_{metric}"
        count_col = f"count_{metric}"
        
        # Check if we have this metric in the data
        if mean_col in model_metrics.columns:
            print(f"\n{'-'*100}")
            print(f"DETAILED STATISTICS FOR: {metric.upper()}")
            header = f"{'MODEL TYPE':<12} {'MODEL ID':<30} {'MEAN':^10} {'MAX':^10} {'STD DEV':^10} {'COUNT':^8}"
            print(header)
            print(f"{'-'*12} {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
            
            for _, row in model_metrics.iterrows():
                line = f"{row['model_type']:<12} {row['model_id'][:30]:<30}"
                if not pd.isna(row[mean_col]):
                    line += f" {row[mean_col]:^10.4f}"
                else:
                    line += f" {'N/A':^10}"
                    
                if max_col in model_metrics.columns and not pd.isna(row[max_col]):
                    line += f" {row[max_col]:^10.4f}"
                else:
                    line += f" {'N/A':^10}"
                    
                if std_col in model_metrics.columns and not pd.isna(row[std_col]):
                    line += f" {row[std_col]:^10.4f}"
                else:
                    line += f" {'N/A':^10}"
                    
                if count_col in model_metrics.columns and not pd.isna(row[count_col]):
                    line += f" {int(row[count_col]):^8d}"
                else:
                    line += f" {'N/A':^8}"
                    
                print(line)
    
    print(f"\n{'='*100}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}\n")
    
    # Return the DataFrame for potential further analysis
    return model_metrics

def calculate_bleu_score(reference: str, hypothesis: str) -> float:
    """
    Calculates the BLEU score between a reference and a hypothesis sentence.
    Reference and hypothesis should be strings.
    """
    # BLEU score expects tokenized sentences
    reference_tokens = [reference.split()]
    hypothesis_tokens = hypothesis.split()
    # Using SmoothingFunction().method1 to handle cases with no common n-grams
    score = sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=SmoothingFunction().method1)
    return score

def calculate_rouge_score(reference: str, hypothesis: str) -> dict:
    """
    Calculates ROUGE scores (1, 2, L) between a reference and a hypothesis text.
    Returns a dictionary of scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {k: v.fmeasure for k, v in scores.items()}

def record_human_evaluation(model_id: str, task_id: str, human_score: float, model_type: str = None, task_type: str = None, feedback: str = ""):
    """
    Records a human evaluation score and feedback for a specific Gemini model and task.
    
    Args:
        model_id: The model identifier
        task_id: Specific task identifier
        human_score: Human evaluation score
        model_type: Type of Gemini model (pro, flash, pro-text)
        task_type: Type of task (summarization, qa, chat)
        feedback: Human feedback comments
    """
    # If model_type is not provided, try to extract it from model_id
    if model_type is None:
        if "flash" in model_id.lower():
            model_type = "flash"
        elif "pro" in model_id.lower():
            model_type = "pro"
        else:
            model_type = "unknown"
    
    # If task_type is not provided, try to extract it from task_id or model_id
    if task_type is None:
        if "summarization" in task_id.lower() or "summary" in task_id.lower():
            task_type = "summarization"
        elif "qa" in task_id.lower():
            task_type = "qa"
        elif "chat" in task_id.lower():
            task_type = "chat"
        else:
            task_type = "unknown"
    
    file_path = config.EVAL_LOG_FILE # Using the same log file for simplicity
    timestamp = datetime.now().isoformat()
    new_entry = pd.DataFrame([
        {
            "timestamp": timestamp,
            "model_id": model_id,
            "model_type": model_type,
            "task_type": task_type,
            "metric_name": f"human_eval_score_{task_id}",
            "metric_value": human_score,
            "notes": feedback
        }
    ])
    
    if not os.path.exists(file_path):
        new_entry.to_csv(file_path, index=False)
    else:
        new_entry.to_csv(file_path, mode='a', header=False, index=False)
    print(f"Logged human evaluation: Model={model_id} ({model_type}), Task={task_id} ({task_type}), Score={human_score:.2f}")

def compare_gemini_models(model_A_id: str, model_B_id: str, model_A_output: str, model_B_output: str, 
                     ground_truth: str = None, comparison_type: str = "semantic", task_type: str = "general"):
    """
    Comprehensive comparison between two Gemini model outputs with detailed metrics.
    
    Args:
        model_A_id: ID of the first model (e.g., 'gemini-2.0-flash')
        model_B_id: ID of the second model (e.g., 'gemini-2.5-flash')
        model_A_output: Text output from model A
        model_B_output: Text output from model B
        ground_truth: Reference text for comparison (optional)
        comparison_type: Type of comparison (semantic, factual, etc)
        task_type: Type of task (summarization, qa, chat)
        
    Returns:
        Dictionary with comprehensive comparison metrics and analysis
    """
    # Extract model types from IDs
    model_A_type = "flash" if "flash" in model_A_id.lower() else "pro" if "pro" in model_A_id.lower() else "unknown"
    model_B_type = "flash" if "flash" in model_B_id.lower() else "pro" if "pro" in model_B_id.lower() else "unknown"
    
    print(f"\n{'='*60}")
    print(f"GEMINI MODEL COMPARISON: {task_type.upper()} TASK")
    print(f"{'='*60}")
    print(f"Model A: {model_A_id} ({model_A_type})")
    print(f"Model B: {model_B_id} ({model_B_type})")
    print(f"{'='*60}")
    print(f"Sample outputs:")
    print(f"Model A: {model_A_output[:100]}...")
    print(f"Model B: {model_B_output[:100]}...")
    if ground_truth:
        print(f"Reference: {ground_truth[:100]}...")
    print(f"{'='*60}")
    
    # Initialize results dictionary with model metadata
    results = {
        "model_A": {
            "id": model_A_id,
            "type": model_A_type,
            "metrics": {}
        },
        "model_B": {
            "id": model_B_id,
            "type": model_B_type,
            "metrics": {}
        },
        "comparison": {
            "task_type": task_type,
            "comparison_type": comparison_type,
            "differences": {}
        }
    }
    
    # Calculate quantitative metrics if ground truth is available
    if ground_truth:
        # BLEU score calculation
        bleu_A = calculate_bleu_score(ground_truth, model_A_output)
        bleu_B = calculate_bleu_score(ground_truth, model_B_output)
        results["model_A"]["metrics"]["bleu_score"] = bleu_A
        results["model_B"]["metrics"]["bleu_score"] = bleu_B
        results["comparison"]["differences"]["bleu_diff"] = bleu_B - bleu_A
        
        # ROUGE scores calculation
        rouge_A = calculate_rouge_score(ground_truth, model_A_output)
        rouge_B = calculate_rouge_score(ground_truth, model_B_output)
        
        # Add ROUGE scores to results
        for metric_name, metric_value in rouge_A.items():
            results["model_A"]["metrics"][metric_name] = metric_value
            results["model_B"]["metrics"][metric_name] = rouge_B[metric_name]
            results["comparison"]["differences"][f"{metric_name}_diff"] = rouge_B[metric_name] - metric_value
    
    # Additional non-reference metrics (applicable with or without ground truth)
    
    # Output length analysis
    results["model_A"]["metrics"]["output_length"] = len(model_A_output)
    results["model_B"]["metrics"]["output_length"] = len(model_B_output)
    results["comparison"]["differences"]["length_diff"] = len(model_B_output) - len(model_A_output)
    
    # Word count analysis
    word_count_A = len(model_A_output.split())
    word_count_B = len(model_B_output.split())
    results["model_A"]["metrics"]["word_count"] = word_count_A
    results["model_B"]["metrics"]["word_count"] = word_count_B
    results["comparison"]["differences"]["word_count_diff"] = word_count_B - word_count_A
    
    # Simulated LLM-based evaluation (would use actual Gemini model in production)
    try:
        # This would be a call to the Gemini model in production
        # For now, simulate the evaluation with random scores
        quality_A = round(random.uniform(7.0, 9.5), 1)
        quality_B = round(random.uniform(7.0, 9.5), 1)
        
        # Add simulated quality scores to results
        results["model_A"]["metrics"]["quality_score"] = quality_A
        results["model_B"]["metrics"]["quality_score"] = quality_B
        results["comparison"]["differences"]["quality_diff"] = quality_B - quality_A
        
        # Determine overall winner
        if quality_A > quality_B:
            winner = "A"
            winner_id = model_A_id
        elif quality_B > quality_A:
            winner = "B"
            winner_id = model_B_id
        else:
            winner = "Tie"
            winner_id = "Both models equal"
        
        results["comparison"]["winner"] = winner
        results["comparison"]["winner_id"] = winner_id
        
    except Exception as e:
        print(f"Error during LLM-based evaluation: {str(e)}")
        results["error"] = str(e)
    
    # Print unified comparison results
    print_gemini_model_comparison(results, ground_truth is not None)
    
    return results


def compare_model_outputs(model_output_A: str, model_output_B: str, ground_truth: str = None, comparison_type: str = "semantic"):
    """
    Legacy comparison function maintained for backward compatibility.
    Consider using compare_gemini_models instead for more comprehensive analysis.
    """
    print(f"Comparing outputs (Type: {comparison_type}):")
    print(f"Output A: {model_output_A[:50]}...")
    print(f"Output B: {model_output_B[:50]}...")
    if ground_truth:
        print(f"Ground Truth: {ground_truth[:50]}...")

    # Call the new comprehensive comparison function
    result = compare_gemini_models("model_A", "model_B", model_output_A, model_output_B, 
                                  ground_truth, comparison_type)
    
    # Return a simplified version of the results for backward compatibility
    legacy_result = {
        "comparison_score": result["comparison"].get("winner_quality_diff", random.uniform(0.5, 1.0)),
        "notes": f"See detailed analysis above for {comparison_type} comparison"
    }
    
    return legacy_result

# --- A/B Testing ---

def create_ab_test_experiment(name: str, variants: list[str], notes: str = ""):
    """
    Creates a new A/B test experiment.
    """
    file_path = config.AB_TEST_EXPERIMENTS_FILE
    experiment_id = f"exp_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    timestamp = datetime.now().isoformat()
    new_exp = pd.DataFrame([
        {
            "experiment_id": experiment_id,
            "name": name,
            "variants": ','.join(variants),
            "start_time": timestamp,
            "end_time": None,
            "status": "active",
            "notes": notes
        }
    ])
    
    if not os.path.exists(file_path):
        new_exp.to_csv(file_path, index=False)
    else:
        new_exp.to_csv(file_path, mode='a', header=False, index=False)
    print(f"Created A/B experiment: {name} with ID {experiment_id}")
    return experiment_id

def assign_user_to_variant(user_id: str, experiment_id: str):
    """
    Assigns a user to a variant in an active experiment.
    """
    experiments_df = pd.read_csv(config.AB_TEST_EXPERIMENTS_FILE)
    active_exp = experiments_df[(experiments_df["experiment_id"] == experiment_id) & (experiments_df["status"] == "active")]

    if active_exp.empty:
        print(f"Error: Experiment {experiment_id} is not active or does not exist.")
        return None

    variants = active_exp["variants"].iloc[0].split(',')
    assigned_variant = random.choice(variants)
    timestamp = datetime.now().isoformat()

    new_assignment = pd.DataFrame([
        {
            "user_id": user_id,
            "experiment_id": experiment_id,
            "assigned_variant": assigned_variant,
            "assignment_time": timestamp,
            "metric_value": None # This will be updated later
        }
    ])
    
    file_path = config.AB_TEST_ASSIGNMENTS_FILE
    if not os.path.exists(file_path):
        new_assignment.to_csv(file_path, index=False)
    else:
        new_assignment.to_csv(file_path, mode='a', header=False, index=False)
    print(f"User {user_id} assigned to variant '{assigned_variant}' for experiment {experiment_id}")
    return assigned_variant

def log_ab_test_metric(user_id: str, experiment_id: str, metric_value: float):
    """
    Logs a metric for a user in a specific A/B test experiment.
    """
    assignments_df = pd.read_csv(config.AB_TEST_ASSIGNMENTS_FILE)
    idx = assignments_df[(assignments_df["user_id"] == user_id) & (assignments_df["experiment_id"] == experiment_id)].index

    if not idx.empty:
        assignments_df.loc[idx, "metric_value"] = metric_value
        assignments_df.to_csv(config.AB_TEST_ASSIGNMENTS_FILE, index=False)
        print(f"Logged A/B metric {metric_value} for user {user_id} in experiment {experiment_id}")
    else:
        print(f"Error: No active assignment found for user {user_id} in experiment {experiment_id}")

def analyze_ab_test_results(experiment_id: str, confidence_level=0.95):
    """
    Analyzes the results of an A/B test experiment and provides comprehensive statistics
    with statistical significance testing and enhanced visualization.
    
    Args:
        experiment_id: The ID of the experiment to analyze
        confidence_level: Confidence level for statistical significance (default: 0.95)
        
    Returns:
        DataFrame with analysis results
    """
    # Get experiment information
    try:
        experiments_df = pd.read_csv(config.AB_TEST_EXPERIMENTS_FILE)
        experiment_info = experiments_df[experiments_df["experiment_id"] == experiment_id].iloc[0]
        experiment_name = experiment_info["name"]
    except (FileNotFoundError, IndexError):
        experiment_name = experiment_id
    
    # Get assignments data
    try:
        assignments_df = pd.read_csv(config.AB_TEST_ASSIGNMENTS_FILE)
        exp_assignments = assignments_df[assignments_df["experiment_id"] == experiment_id].dropna(subset=["metric_value"])
    except FileNotFoundError:
        print(f"No assignment data found for experiment {experiment_id}.")
        return None

    if exp_assignments.empty:
        print(f"No data to analyze for experiment {experiment_id}.")
        return None

    # Get basic statistics
    analysis_results = exp_assignments.groupby("assigned_variant")["metric_value"].agg(
        ["count", "mean", "std", "min", "max"]
    ).reset_index()
    
    # Enhanced visualization of results
    print(f"\n{'='*80}")
    print(f"A/B TEST ANALYSIS: {experiment_name.upper()}")
    print(f"{'='*80}")
    
    # Print experiment information
    print(f"\nExperiment details:")
    print(f"  • Experiment ID: {experiment_id}")
    print(f"  • Experiment name: {experiment_name}")
    print(f"  • Total participants: {len(exp_assignments)}")
    print(f"  • Variants tested: {', '.join(analysis_results['assigned_variant'])}")
    print(f"\n{'-'*80}\n")
    
    # Print detailed metrics table
    print(f"PERFORMANCE BY VARIANT:")
    print(f"{'VARIANT':<15} {'COUNT':^10} {'MEAN':^10} {'STD DEV':^10} {'MIN':^10} {'MAX':^10}")
    print(f"{'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for _, row in analysis_results.iterrows():
        variant = row['assigned_variant']
        count = row['count']
        mean = row['mean']
        std = row['std']
        min_val = row['min']
        max_val = row['max']
        
        print(f"{variant:<15} {count:^10d} {mean:^10.4f} {std:^10.4f} {min_val:^10.4f} {max_val:^10.4f}")
    
    # Calculate statistical significance if we have exactly 2 variants
    if len(analysis_results) == 2:
        print(f"\n{'-'*80}\n")
        print(f"STATISTICAL SIGNIFICANCE ANALYSIS:")
        
        variant_a = analysis_results['assigned_variant'].iloc[0]
        variant_b = analysis_results['assigned_variant'].iloc[1]
        
        # Get data for each variant
        data_a = exp_assignments[exp_assignments['assigned_variant'] == variant_a]['metric_value']
        data_b = exp_assignments[exp_assignments['assigned_variant'] == variant_b]['metric_value']
        
        # Calculate relative improvement
        mean_a = data_a.mean()
        mean_b = data_b.mean()
        rel_improvement = ((mean_b - mean_a) / mean_a) * 100 if mean_a != 0 else float('inf')
        
        print(f"  • Variant A: {variant_a} (mean: {mean_a:.4f})")
        print(f"  • Variant B: {variant_b} (mean: {mean_b:.4f})")
        print(f"  • Absolute difference: {mean_b - mean_a:.4f}")
        print(f"  • Relative improvement: {rel_improvement:.2f}%")
        
        # Perform statistical test (if we have scipy)
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(data_a, data_b, equal_var=False)
            
            print(f"  • p-value: {p_value:.4f}")
            if p_value < (1 - confidence_level):
                print(f"  • Result: Statistically significant difference at {confidence_level*100:.0f}% confidence level")
                if mean_b > mean_a:
                    print(f"  • Conclusion: {variant_b} performs significantly better than {variant_a}")
                else:
                    print(f"  • Conclusion: {variant_a} performs significantly better than {variant_b}")
            else:
                print(f"  • Result: No statistically significant difference at {confidence_level*100:.0f}% confidence level")
                print(f"  • Conclusion: Cannot determine a clear winner between variants")
        except ImportError:
            print(f"  • Statistical testing unavailable (scipy not installed)")
            if mean_b > mean_a:
                print(f"  • Observation: {variant_b} shows higher mean performance than {variant_a}")
            elif mean_a > mean_b:
                print(f"  • Observation: {variant_a} shows higher mean performance than {variant_b}")
            else:
                print(f"  • Observation: Both variants show identical mean performance")
    
    print(f"\n{'='*80}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    return analysis_results

def print_gemini_model_comparison(results, has_reference=False):
    """
    Prints a unified, well-formatted comparison of Gemini model evaluation results with enhanced visuals.
    
    Args:
        results: Dictionary with comparison results from compare_gemini_models
        has_reference: Whether the comparison included a ground truth reference
    """
    model_A = results["model_A"]
    model_B = results["model_B"]
    comparison = results["comparison"]
    
    print(f"\n{'='*80}")
    title = f"GEMINI MODEL COMPARISON RESULTS: {comparison.get('task_type', 'GENERAL').upper()}"
    print(f"{title:^80}")
    print(f"{'='*80}")
    
    # Print basic model information with enhanced formatting
    print(f"\nMODEL INFORMATION:")
    print(f"  • MODEL A: {model_A['id']} ({model_A['type']})")
    print(f"  • MODEL B: {model_B['id']} ({model_B['type']})")
    print(f"  • Task type: {comparison.get('task_type', 'General')}")
    print(f"  • Comparison type: {comparison.get('comparison_type', 'Standard')}")
    if has_reference:
        print(f"  • Reference-based evaluation: Yes")
    else:
        print(f"  • Reference-based evaluation: No")
    print(f"\n{'-'*80}\n")
    
    # Print metric table header with enhanced layout
    print(f"PERFORMANCE METRICS COMPARISON:")
    print(f"{'METRIC':<20} {'MODEL A':^15} {'MODEL B':^15} {'DIFF (B-A)':^15} {'BETTER MODEL':^15}")
    print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    
    # Group metrics by category
    metric_categories = {
        "Reference-based": ["bleu_score", "rouge1", "rouge2", "rougeL"],
        "Output characteristics": ["output_length", "word_count"],
        "Quality metrics": ["quality_score"]
    }
    
    # Track metrics that belong to categories
    categorized_metrics = [m for category in metric_categories.values() for m in category]
    
    # Print metrics by category
    for category, metrics in metric_categories.items():
        relevant_metrics = [m for m in metrics if m in model_A['metrics'] and m in model_B['metrics']]
        
        if relevant_metrics:  # Only print category if we have relevant metrics
            print(f"\n{category}:")
            
            for metric in relevant_metrics:
                value_A = model_A['metrics'][metric]
                value_B = model_B['metrics'][metric]
                diff = comparison['differences'].get(f"{metric}_diff", value_B - value_A)
                
                # Determine which model is better for this metric
                # For most metrics, higher is better
                better_model = "TIE"
                if abs(diff) > 0.0001:  # Use small epsilon to account for float precision
                    better_model = "B" if diff > 0 else "A"
                
                # Format based on metric type
                if isinstance(value_A, float):
                    print(f"{metric:<20} {value_A:^15.4f} {value_B:^15.4f} {diff:^+15.4f} {better_model:^15}")
                else:
                    print(f"{metric:<20} {value_A:^15} {value_B:^15} {diff:^+15} {better_model:^15}")
    
    # Print other metrics that don't belong to predefined categories
    other_metrics = [m for m in model_A['metrics'] if m not in categorized_metrics]
    if other_metrics:
        print(f"\nOther metrics:")
        for metric in sorted(other_metrics):
            if metric in model_B['metrics']:
                value_A = model_A['metrics'][metric]
                value_B = model_B['metrics'][metric]
                diff = comparison['differences'].get(f"{metric}_diff", value_B - value_A)
                
                # Determine which model is better
                better_model = "TIE"
                if abs(diff) > 0.0001:
                    better_model = "B" if diff > 0 else "A"
                
                # Format based on metric type
                if isinstance(value_A, float):
                    print(f"{metric:<20} {value_A:^15.4f} {value_B:^15.4f} {diff:^+15.4f} {better_model:^15}")
                else:
                    print(f"{metric:<20} {value_A:^15} {value_B:^15} {diff:^+15} {better_model:^15}")
    
    # Print summary with enhanced visualization
    print(f"\n{'-'*80}\n")
    print(f"OVERALL COMPARISON SUMMARY:")
    
    # Calculate summary metrics
    total_metrics = len([m for m in model_A['metrics'] if m in model_B['metrics']])
    model_a_wins = sum(1 for m in model_A['metrics'] 
                      if m in model_B['metrics'] and 
                      model_A['metrics'][m] > model_B['metrics'][m])
    model_b_wins = sum(1 for m in model_A['metrics'] 
                      if m in model_B['metrics'] and 
                      model_B['metrics'][m] > model_A['metrics'][m])
    ties = total_metrics - model_a_wins - model_b_wins
    
    print(f"  • Total metrics compared: {total_metrics}")
    print(f"  • Model A wins: {model_a_wins} metrics ({model_a_wins/total_metrics*100:.1f}%)")
    print(f"  • Model B wins: {model_b_wins} metrics ({model_b_wins/total_metrics*100:.1f}%)")
    print(f"  • Ties: {ties} metrics ({ties/total_metrics*100:.1f}%)")
    
    # Print winner information with more details
    if "winner" in comparison:
        print(f"\nWINNER: Model {comparison['winner']} ({comparison['winner_id']})")
        if "quality_diff" in comparison['differences']:
            print(f"Quality difference: {comparison['differences']['quality_diff']:.4f}")
    
    print(f"\n{'='*80}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


def compare_multiple_models(model_outputs: dict, ground_truth: str = None, 
                           comparison_type: str = "semantic", task_type: str = "general"):
    """
    Compares multiple Gemini model outputs with comprehensive metrics visualization.
    
    Args:
        model_outputs: Dictionary with model_id as key and output text as value
                      Example: {"gemini-1.5-pro": "output text", "gemini-1.5-flash": "output text"}
        ground_truth: Reference text for comparison (optional)
        comparison_type: Type of comparison (semantic, factual, etc)
        task_type: Type of task (summarization, qa, chat)
        
    Returns:
        Dictionary with comprehensive metrics for all models
    """
    if len(model_outputs) < 2:
        print("Error: At least two models are required for comparison.")
        return None
    
    # Print header
    print(f"\n{'='*100}")
    print(f"MULTI-MODEL GEMINI COMPARISON: {task_type.upper()} TASK")
    print(f"{'='*100}")
    
    print(f"\nComparing {len(model_outputs)} models:")
    for idx, (model_id, _) in enumerate(model_outputs.items()):
        # Extract model type from ID
        model_type = "flash" if "flash" in model_id.lower() else "pro" if "pro" in model_id.lower() else "unknown"
        print(f"  {idx+1}. {model_id} ({model_type})")
    
    print(f"\nTask type: {task_type}")
    print(f"Comparison type: {comparison_type}")
    if ground_truth:
        print(f"Reference-based evaluation: Yes")
        print(f"Reference text (excerpt): {ground_truth[:100]}...")
    else:
        print(f"Reference-based evaluation: No")
    
    print(f"\n{'-'*100}")
    
    # Initialize results dictionary
    results = {
        "task_info": {
            "task_type": task_type,
            "comparison_type": comparison_type,
            "has_reference": ground_truth is not None
        },
        "models": {},
        "comparative_analysis": {}
    }
    
    # Calculate metrics for each model
    for model_id, output_text in model_outputs.items():
        # Extract model type from ID
        model_type = "flash" if "flash" in model_id.lower() else "pro" if "pro" in model_id.lower() else "unknown"
        
        # Initialize model entry in results
        results["models"][model_id] = {
            "id": model_id,
            "type": model_type,
            "metrics": {}
        }
        
        # Add output length and word count metrics
        results["models"][model_id]["metrics"]["output_length"] = len(output_text)
        results["models"][model_id]["metrics"]["word_count"] = len(output_text.split())
        
        # Calculate reference-based metrics if ground truth is available
        if ground_truth:
            # BLEU score
            bleu_score = calculate_bleu_score(ground_truth, output_text)
            results["models"][model_id]["metrics"]["bleu_score"] = bleu_score
            
            # ROUGE scores
            rouge_scores = calculate_rouge_score(ground_truth, output_text)
            for rouge_metric, rouge_value in rouge_scores.items():
                results["models"][model_id]["metrics"][rouge_metric] = rouge_value
        
        # Add simulated quality score (would use actual Gemini model in production)
        try:
            # Simulate quality score with random value (would be a real LLM evaluation in production)
            quality_score = round(random.uniform(7.0, 9.5), 1)
            results["models"][model_id]["metrics"]["quality_score"] = quality_score
        except Exception as e:
            print(f"Error during quality evaluation for {model_id}: {str(e)}")
    
    # Print metrics table
    print(f"PERFORMANCE METRICS COMPARISON:")
    
    # Determine all available metrics across all models
    all_metrics = set()
    for model_data in results["models"].values():
        all_metrics.update(model_data["metrics"].keys())
    
    # Group metrics by category
    metric_categories = {
        "Reference-based": ["bleu_score", "rouge1", "rouge2", "rougeL"],
        "Output characteristics": ["output_length", "word_count"],
        "Quality metrics": ["quality_score"]
    }
    
    # Print header row with model IDs
    header = f"{'METRIC':<20}"
    for model_id in results["models"]:
        header += f" {model_id[:15]:^15}"
    print(header)
    print(f"{'-'*20} {'-'*15 * len(results['models'])}")
    
    # Print metrics by category
    for category, metrics in metric_categories.items():
        relevant_metrics = [m for m in metrics if any(m in results["models"][model_id]["metrics"] for model_id in results["models"])]
        
        if relevant_metrics:  # Only print category if we have relevant metrics
            print(f"\n{category}:")
            
            for metric in relevant_metrics:
                line = f"{metric:<20}"
                
                # Print value for each model
                for model_id in results["models"]:
                    value = results["models"][model_id]["metrics"].get(metric, "N/A")
                    if isinstance(value, float):
                        line += f" {value:^15.4f}"
                    else:
                        line += f" {value:^15}"
                print(line)
    
    # Print other metrics that don't belong to predefined categories
    categorized_metrics = [m for category in metric_categories.values() for m in category]
    other_metrics = [m for m in all_metrics if m not in categorized_metrics]
    
    if other_metrics:
        print(f"\nOther metrics:")
        for metric in sorted(other_metrics):
            line = f"{metric:<20}"
            for model_id in results["models"]:
                value = results["models"][model_id]["metrics"].get(metric, "N/A")
                if isinstance(value, float):
                    line += f" {value:^15.4f}"
                else:
                    line += f" {value:^15}"
            print(line)
    
    # Determine best model(s) based on different criteria
    best_models = {}
    metric_importance = {
        "bleu_score": "higher", 
        "rouge1": "higher", 
        "rouge2": "higher", 
        "rougeL": "higher",
        "quality_score": "higher"
    }
    
    # Find best model for each important metric
    for metric, direction in metric_importance.items():
        models_with_metric = [(model_id, results["models"][model_id]["metrics"].get(metric, None)) 
                             for model_id in results["models"] 
                             if metric in results["models"][model_id]["metrics"]]
        
        if models_with_metric:
            if direction == "higher":
                best_value = max(models_with_metric, key=lambda x: x[1] if x[1] is not None else float('-inf'))[1]
                best_for_metric = [model_id for model_id, value in models_with_metric if value == best_value]
            else:
                best_value = min(models_with_metric, key=lambda x: x[1] if x[1] is not None else float('inf'))[1]
                best_for_metric = [model_id for model_id, value in models_with_metric if value == best_value]
                
            best_models[metric] = {
                "models": best_for_metric,
                "value": best_value
            }
    
    # Determine overall winner based on weighted metrics
    if "quality_score" in all_metrics:
        # Use quality score as the main determinant if available
        best_quality = max([(model_id, results["models"][model_id]["metrics"].get("quality_score", 0)) 
                          for model_id in results["models"]], key=lambda x: x[1])[1]
        overall_winners = [model_id for model_id in results["models"] 
                         if results["models"][model_id]["metrics"].get("quality_score", 0) == best_quality]
        
        winner_criteria = "quality_score"
    elif "bleu_score" in all_metrics:
        # Use BLEU score if quality score not available
        best_bleu = max([(model_id, results["models"][model_id]["metrics"].get("bleu_score", 0)) 
                         for model_id in results["models"]], key=lambda x: x[1])[1]
        overall_winners = [model_id for model_id in results["models"] 
                         if results["models"][model_id]["metrics"].get("bleu_score", 0) == best_bleu]
        
        winner_criteria = "bleu_score"
    else:
        # Default to the model with most wins across metrics
        model_win_counts = {model_id: 0 for model_id in results["models"]}
        for metric_result in best_models.values():
            for winner in metric_result["models"]:
                model_win_counts[winner] += 1
        
        most_wins = max(model_win_counts.values(), default=0)
        overall_winners = [model_id for model_id, wins in model_win_counts.items() if wins == most_wins]
        
        winner_criteria = "most wins across metrics"
    
    # Add winners to results
    results["comparative_analysis"]["best_models_by_metric"] = best_models
    results["comparative_analysis"]["overall_winners"] = overall_winners
    results["comparative_analysis"]["winner_criteria"] = winner_criteria
    
    # Print summary section
    print(f"\n{'-'*100}")
    print(f"\nBEST MODEL(S) BY METRIC:")
    for metric, data in best_models.items():
        models_str = ", ".join(data["models"])
        print(f"  \u2022 {metric}: {models_str} (value: {data['value']:.4f})")
    
    print(f"\nOVERALL WINNER(S):")
    print(f"  \u2022 {', '.join(overall_winners)} (based on {winner_criteria})")
    
    print(f"\n{'='*100}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}")
    
    return results

# --- Main execution for local testing of evaluation and A/B testing ---
if __name__ == "__main__":
    print("--- Running Enhanced Evaluation System Examples ---")
    
    # --- Basic Evaluation Logging ---
    print("\n\n" + "="*80)
    print("BASIC EVALUATION LOGGING")
    print("="*80)
    log_evaluation_metric("summarizer_v1", "coherence_score", 0.85, "flash", "summarization", "Initial run")
    log_evaluation_metric("qa_bot_v2", "accuracy", 0.92, "pro", "qa", "After prompt tuning")
    
    # --- Evaluation of Frameworks from app.py ---
    print("\n\n" + "="*80)
    print("EVALUATING LANGCHAIN FRAMEWORKS WITH GEMINI MODELS")
    print("="*80)

    # Test Case 1: Summarization Framework (gemini-1.5-flash)
    summarization_input = "The quick brown fox jumps over the lazy dog. This is a classic pangram, a sentence that contains every letter of the alphabet at least once. Pangrams are often used to display typefaces or to test the functionality of typewriters and computer keyboards."
    summarization_reference = "A pangram is a sentence with every letter of the alphabet, used for testing typefaces."
    
    print(f"\nEvaluating Summarization Framework (Model: {config.GEMINI_FLASH_MODEL})")
    summarization_hypothesis = summarize_text_framework(summarization_input)
    
    bleu_sum = calculate_bleu_score(summarization_reference, summarization_hypothesis)
    log_evaluation_metric(
        f"summarization_framework_{config.GEMINI_FLASH_MODEL}", 
        "bleu_score", 
        bleu_sum,
        "flash",
        "summarization",
        f"Input: '{summarization_input[:50]}...', Reference: '{summarization_reference}', Hypothesis: '{summarization_hypothesis}'"
    )

    rouge_sum = calculate_rouge_score(summarization_reference, summarization_hypothesis)
    for k, v in rouge_sum.items():
        log_evaluation_metric(
            f"summarization_framework_{config.GEMINI_FLASH_MODEL}", 
            k, 
            v,
            "flash",
            "summarization",
            f"Input: '{summarization_input[:50]}...', Reference: '{summarization_reference}', Hypothesis: '{summarization_hypothesis}'"
        )

    # Test Case 2: Complex Q&A Framework (gemini-1.5-pro)
    qa_question = "What is the purpose of LangChain?"
    qa_context = "LangChain is a framework for developing applications powered by large language models. It enables applications that are data-aware and agentic."
    qa_reference = "LangChain is a framework for developing applications powered by large language models, enabling data-aware and agentic applications."

    print(f"\nEvaluating Complex Q&A Framework (Model: {config.GEMINI_PRO_MODEL})")
    qa_hypothesis = complex_qa_framework(qa_question, qa_context)

    bleu_qa = calculate_bleu_score(qa_reference, qa_hypothesis)
    log_evaluation_metric(
        f"qa_framework_{config.GEMINI_PRO_MODEL}", 
        "bleu_score", 
        bleu_qa,
        "pro",
        "qa",
        f"Question: '{qa_question}', Context: '{qa_context[:50]}...', Reference: '{qa_reference}', Hypothesis: '{qa_hypothesis}'"
    )

    rouge_qa = calculate_rouge_score(qa_reference, qa_hypothesis)
    for k, v in rouge_qa.items():
        log_evaluation_metric(
            f"qa_framework_{config.GEMINI_PRO_MODEL}", 
            k, 
            v,
            "pro",
            "qa",
            f"Question: '{qa_question}', Context: '{qa_context[:50]}...', Reference: '{qa_reference}', Hypothesis: '{qa_hypothesis}'"
        )

    # Test Case 3: Simple Chat Framework (gemini-1.0-pro)
    chat_message = "What is the capital of Canada?"
    chat_reference = "The capital of Canada is Ottawa."

    print(f"\nEvaluating Simple Chat Framework (Model: {config.GEMINI_PRO_TEXT_MODEL})")
    chat_hypothesis = simple_chat_framework(chat_message)

    bleu_chat = calculate_bleu_score(chat_reference, chat_hypothesis)
    log_evaluation_metric(
        f"chat_framework_{config.GEMINI_PRO_TEXT_MODEL}", 
        "bleu_score", 
        bleu_chat,
        "pro-text",
        "chat",
        f"Message: '{chat_message}', Reference: '{chat_reference}', Hypothesis: '{chat_hypothesis}'"
    )

    rouge_chat = calculate_rouge_score(chat_reference, chat_hypothesis)
    for k, v in rouge_chat.items():
        log_evaluation_metric(
            f"chat_framework_{config.GEMINI_PRO_TEXT_MODEL}", 
            k, 
            v,
            "pro-text",
            "chat",
            f"Message: '{chat_message}', Reference: '{chat_reference}', Hypothesis: '{chat_hypothesis}'"
        )

    # --- Use the enhanced visualize_model_metrics function ---
    print("\n\n" + "="*80)
    print("ENHANCED MODEL METRICS VISUALIZATION")
    print("="*80)
    
    # Visualize metrics for all tasks and models
    print("\nVisualize all metrics:")
    visualize_model_metrics(sort_by="bleu_score")
    
    # Visualize metrics for summarization task only
    print("\nVisualize summarization metrics:")
    visualize_model_metrics(task_type="summarization", sort_by="rouge1")
    
    # Visualize specific metrics across models
    print("\nVisualize specific metrics (BLEU and ROUGE-1):")
    visualize_model_metrics(metrics=["bleu_score", "rouge1"], sort_by="bleu_score")

    # --- Use the enhanced compare_gemini_models function ---
    print("\n\n" + "="*80)
    print("ENHANCED MODEL COMPARISON")
    print("="*80)
    
    # Compare two models
    print("\nComparing two models (summarization):")
    comparison_results = compare_gemini_models(
        model_A_id=config.GEMINI_FLASH_MODEL,
        model_B_id=config.GEMINI_PRO_MODEL,
        model_A_output=summarization_hypothesis,
        model_B_output=qa_hypothesis,  # Just for demonstration
        ground_truth=summarization_reference,
        comparison_type="semantic",
        task_type="summarization"
    )

    # --- Use the new compare_multiple_models function ---
    print("\n\n" + "="*80)
    print("MULTI-MODEL COMPARISON")
    print("="*80)
    
    # Compare multiple models
    print("\nComparing three models:")
    
    # Create outputs from different models (simulated for this example)
    model_outputs = {
        config.GEMINI_FLASH_MODEL: summarization_hypothesis,
        config.GEMINI_PRO_MODEL: qa_hypothesis,  # Using different outputs for demonstration
        config.GEMINI_PRO_TEXT_MODEL: chat_hypothesis
    }
    
    multi_model_results = compare_multiple_models(
        model_outputs=model_outputs,
        ground_truth=summarization_reference,
        comparison_type="comprehensive",
        task_type="mixed"
    )

    # --- Human Evaluation Examples ---
    print("\n\n" + "="*80)
    print("HUMAN EVALUATION EXAMPLES")
    print("="*80)
    
    human_eval_model_id_1 = f"summarization_framework_{config.GEMINI_FLASH_MODEL}"
    human_eval_model_id_2 = f"summarization_framework_v2_custom_prompt"
    human_eval_task_id = "summary_task_1"
    human_eval_text_context = "Original text: 'This is a long article about AI.' Summary: 'AI is great.'"

    record_human_evaluation(
        human_eval_model_id_1, 
        human_eval_task_id, 
        4.5, 
        "flash",
        "summarization",
        f"Feedback: Very concise and accurate. Context: {human_eval_text_context}"
    )
    record_human_evaluation(
        human_eval_model_id_2, 
        human_eval_task_id, 
        3.0, 
        "pro",
        "summarization",
        f"Feedback: Missed some key points. Context: {human_eval_text_context}"
    )

    # --- Enhanced A/B Testing Examples ---
    print("\n\n" + "="*80)
    print("ENHANCED A/B TESTING")
    print("="*80)
    
    exp_id = create_ab_test_experiment("Enhanced Summarizer A/B Test", 
                                      [f"{config.GEMINI_FLASH_MODEL}", f"{config.GEMINI_PRO_MODEL}"], 
                                      "Comparing flash vs pro models for summarization")
    
    if exp_id:
        # Generate more test users for more robust testing
        users = [f"user_{i}" for i in range(1, 21)]
        for user in users:
            variant = assign_user_to_variant(user, exp_id)
            if variant:
                # Simulate metrics with different distributions for each variant
                if variant == f"{config.GEMINI_FLASH_MODEL}":
                    # Flash model: faster but slightly less accurate
                    metric = random.normalvariate(4.2, 0.4)  # Mean 4.2, StdDev 0.4
                else:
                    # Pro model: slower but more accurate
                    metric = random.normalvariate(4.5, 0.3)  # Mean 4.5, StdDev 0.3
                    
                # Ensure metric is within reasonable bounds
                metric = max(1.0, min(5.0, metric))
                log_ab_test_metric(user, exp_id, metric)
        
        # Run enhanced analysis with statistical significance testing
        analyze_ab_test_results(exp_id, confidence_level=0.95)

    print("\n\n" + "="*80)
    print("EVALUATION SYSTEM DEMONSTRATION COMPLETE")
    print("="*80)