"""
Data validation module for MLOps project.

Provides comprehensive data validation including schema checks, data quality metrics,
leakage detection, and statistical analysis.
"""

from typing import Any

import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

from ..utils.io import save_html, save_json
from ..utils.logging import get_logger
from ..utils.metrics import calculate_class_imbalance

logger = get_logger(__name__)


class DataValidator:
    """
    Comprehensive data validator for tabular datasets.
    
    Performs schema validation, data quality checks, leakage detection,
    and statistical analysis with MLflow logging.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the data validator.
        
        Args:
            config: Configuration dictionary containing validation parameters
        """
        self.config = config
        self.validation_results = {}
        self.issues = []

    def validate_dataset(
        self,
        df: pd.DataFrame,
        target_column: str,
        dataset_name: str = "dataset"
    ) -> dict[str, Any]:
        """
        Perform comprehensive dataset validation.
        
        Args:
            df: Input DataFrame to validate
            target_column: Name of the target column
            dataset_name: Name identifier for the dataset
            
        Returns:
            Dictionary containing validation results and issues
        """
        logger.info(f"Starting validation for dataset: {dataset_name}")

        # Reset results
        self.validation_results = {}
        self.issues = []

        # Basic dataset info
        self._validate_basic_info(df, dataset_name)

        # Schema validation
        self._validate_schema(df, dataset_name)

        # Data quality checks
        self._validate_data_quality(df, dataset_name)

        # Target analysis
        self._validate_target(df, target_column, dataset_name)

        # Leakage detection
        self._detect_leakage(df, target_column, dataset_name)

        # Statistical analysis
        self._analyze_statistics(df, dataset_name)

        # Log results to MLflow
        self._log_validation_results(dataset_name)

        # Generate validation report
        self._generate_validation_report(dataset_name)

        return {
            "validation_results": self.validation_results,
            "issues": self.issues,
            "is_valid": len(self.issues) == 0
        }

    def _validate_basic_info(self, df: pd.DataFrame, dataset_name: str) -> None:
        """Validate basic dataset information."""
        logger.info("Validating basic dataset information")

        self.validation_results[f"{dataset_name}_basic_info"] = {
            "shape": df.shape,
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "has_duplicates": df.duplicated().any(),
            "duplicate_count": df.duplicated().sum()
        }

        if df.duplicated().any():
            self.issues.append(f"Dataset contains {df.duplicated().sum()} duplicate rows")

    def _validate_schema(self, df: pd.DataFrame, dataset_name: str) -> None:
        """Validate dataset schema and data types."""
        logger.info("Validating dataset schema")

        schema_info = {}

        for col in df.columns:
            col_info = {
                "dtype": str(df[col].dtype),
                "null_count": df[col].isnull().sum(),
                "null_percentage": (df[col].isnull().sum() / len(df)) * 100,
                "unique_count": df[col].nunique(),
                "unique_percentage": (df[col].nunique() / len(df)) * 100
            }

            # Check for suspicious patterns
            if col_info["null_percentage"] > 50:
                self.issues.append(f"Column '{col}' has {col_info['null_percentage']:.1f}% null values")

            if col_info["unique_percentage"] > 95:
                self.issues.append(f"Column '{col}' has {col_info['unique_percentage']:.1f}% unique values (potential ID column)")

            schema_info[col] = col_info

        self.validation_results[f"{dataset_name}_schema"] = schema_info

    def _validate_data_quality(self, df: pd.DataFrame, dataset_name: str) -> None:
        """Validate data quality metrics."""
        logger.info("Validating data quality")

        quality_metrics = {}

        # Check for constant columns
        constant_columns = []
        for col in df.columns:
            if df[col].nunique() == 1:
                constant_columns.append(col)
                self.issues.append(f"Column '{col}' is constant (single unique value)")

        quality_metrics["constant_columns"] = constant_columns

        # Check for high cardinality categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        high_cardinality = []

        for col in categorical_columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.5:
                high_cardinality.append({
                    "column": col,
                    "unique_ratio": unique_ratio,
                    "unique_count": df[col].nunique()
                })
                self.issues.append(f"Column '{col}' has high cardinality ({unique_ratio:.1%})")

        quality_metrics["high_cardinality_categorical"] = high_cardinality

        # Check for numeric columns with suspicious patterns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_issues = []

        for col in numeric_columns:
            col_stats = df[col].describe()

            # Check for extreme values
            if col_stats['std'] == 0:
                numeric_issues.append({
                    "column": col,
                    "issue": "zero_std",
                    "description": "Column has zero standard deviation"
                })
                self.issues.append(f"Column '{col}' has zero standard deviation")

            # Check for suspicious ranges
            if col_stats['min'] == col_stats['max']:
                numeric_issues.append({
                    "column": col,
                    "issue": "constant_numeric",
                    "description": "Numeric column has single value"
                })
                self.issues.append(f"Column '{col}' is constant numeric")

        quality_metrics["numeric_issues"] = numeric_issues

        self.validation_results[f"{dataset_name}_quality"] = quality_metrics

    def _validate_target(self, df: pd.DataFrame, target_column: str, dataset_name: str) -> None:
        """Validate target column characteristics."""
        logger.info(f"Validating target column: {target_column}")

        if target_column not in df.columns:
            self.issues.append(f"Target column '{target_column}' not found in dataset")
            return

        target_info = {
            "dtype": str(df[target_column].dtype),
            "null_count": df[target_column].isnull().sum(),
            "null_percentage": (df[target_column].isnull().sum() / len(df)) * 100
        }

        # Check for nulls in target
        if target_info["null_percentage"] > 0:
            self.issues.append(f"Target column '{target_column}' contains {target_info['null_percentage']:.1f}% null values")

        # Determine task type
        if df[target_column].dtype in ['object', 'category']:
            task_type = "classification"
            target_info["task_type"] = task_type
            target_info["unique_values"] = df[target_column].nunique()
            target_info["class_distribution"] = df[target_column].value_counts().to_dict()

            # Check class imbalance
            imbalance_info = calculate_class_imbalance(df[target_column])
            target_info["class_imbalance_ratio"] = imbalance_info

            if imbalance_info["imbalance_ratio"] < 0.1:
                self.issues.append(f"Target column '{target_column}' shows severe class imbalance (ratio: {imbalance_info['imbalance_ratio']:.3f})")
            elif imbalance_info["imbalance_ratio"] < 0.2:
                self.issues.append(f"Target column '{target_column}' shows moderate class imbalance (ratio: {imbalance_info['imbalance_ratio']:.3f})")

        else:
            task_type = "regression"
            target_info["task_type"] = task_type
            target_info["statistics"] = df[target_column].describe().to_dict()

            # Check for extreme values in regression
            target_std = df[target_column].std()
            target_mean = df[target_column].mean()

            if target_std == 0:
                self.issues.append(f"Target column '{target_column}' has zero variance")
            elif abs(target_mean) > 10 * target_std:
                self.issues.append(f"Target column '{target_column}' has extreme mean relative to std")

        self.validation_results[f"{dataset_name}_target"] = target_info

    def _detect_leakage(self, df: pd.DataFrame, target_column: str, dataset_name: str) -> None:
        """Detect potential data leakage."""
        logger.info("Detecting potential data leakage")

        leakage_indicators = []

        # Check for target-like columns (high correlation with target)
        if df[target_column].dtype in ['object', 'category']:
            # For classification, check if any column has similar distribution
            target_dist = df[target_column].value_counts(normalize=True)

            for col in df.columns:
                if col == target_column:
                    continue

                if df[col].dtype in ['object', 'category']:
                    col_dist = df[col].value_counts(normalize=True)

                    # Check distribution similarity
                    if len(col_dist) == len(target_dist):
                        # Simple similarity check
                        if abs(col_dist.iloc[0] - target_dist.iloc[0]) < 0.1:
                            leakage_indicators.append({
                                "column": col,
                                "type": "distribution_similarity",
                                "description": "Column distribution similar to target"
                            })
                            self.issues.append(f"Potential leakage: Column '{col}' has similar distribution to target")
        else:
            # For regression, check correlations
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            correlations = df[numeric_cols].corr()[target_column].abs()

            high_corr_cols = correlations[correlations > 0.95].index.tolist()
            high_corr_cols = [col for col in high_corr_cols if col != target_column]

            for col in high_corr_cols:
                leakage_indicators.append({
                    "column": col,
                    "type": "high_correlation",
                    "correlation": correlations[col],
                    "description": f"Very high correlation ({correlations[col]:.3f}) with target"
                })
                self.issues.append(f"Potential leakage: Column '{col}' has {correlations[col]:.3f} correlation with target")

        # Check for time-based leakage
        time_columns = [col for col in df.columns if any(time_word in col.lower() for time_word in
                       ['date', 'time', 'timestamp', 'year', 'month', 'day', 'hour'])]

        for col in time_columns:
            if df[col].dtype in ['object', 'category']:
                try:
                    # Try to parse as datetime
                    pd.to_datetime(df[col], errors='raise')
                    leakage_indicators.append({
                        "column": col,
                        "type": "temporal",
                        "description": "Temporal column that might cause leakage"
                    })
                    self.issues.append(f"Temporal column '{col}' might cause time-based leakage")
                except:
                    pass

        self.validation_results[f"{dataset_name}_leakage"] = {
            "indicators": leakage_indicators,
            "high_correlation_threshold": 0.95,
            "distribution_similarity_threshold": 0.1
        }

    def _analyze_statistics(self, df: pd.DataFrame, dataset_name: str) -> None:
        """Perform statistical analysis of the dataset."""
        logger.info("Performing statistical analysis")

        stats = {
            "summary_statistics": {},
            "correlation_matrix": None,
            "missing_data_pattern": None
        }

        # Summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats["summary_statistics"] = df[numeric_cols].describe().to_dict()

        # Correlation matrix for numeric columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            stats["correlation_matrix"] = corr_matrix.to_dict()

            # Check for high correlations between features
            high_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:
                        high_correlations.append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": corr_value
                        })
                        self.issues.append(f"High correlation ({corr_value:.3f}) between '{col1}' and '{col2}'")

            stats["high_correlations"] = high_correlations

        # Missing data pattern analysis
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df)) * 100

        if missing_data.sum() > 0:
            stats["missing_data_pattern"] = {
                "total_missing": missing_data.sum(),
                "missing_percentage": (missing_data.sum() / (len(df) * len(df.columns))) * 100,
                "columns_with_missing": missing_data[missing_data > 0].to_dict(),
                "missing_percentages": missing_percentage[missing_percentage > 0].to_dict()
            }

        self.validation_results[f"{dataset_name}_statistics"] = stats

    def _log_validation_results(self, dataset_name: str) -> None:
        """Log validation results to MLflow."""
        logger.info("Logging validation results to MLflow")

        try:
            # Log validation metrics
            mlflow.log_metrics({
                f"{dataset_name}_total_issues": len(self.issues),
                f"{dataset_name}_validation_score": max(0, 100 - len(self.issues) * 10)
            })

            # Log validation parameters
            mlflow.log_params({
                f"{dataset_name}_validation_timestamp": pd.Timestamp.now().isoformat(),
                f"{dataset_name}_validation_config": str(self.config)
            })

            # Log validation results as artifact
            results_file = f"validation_results_{dataset_name}.json"
            save_json(self.validation_results, results_file)
            mlflow.log_artifact(results_file)

            # Log issues as artifact
            if self.issues:
                issues_file = f"validation_issues_{dataset_name}.txt"
                with open(issues_file, 'w') as f:
                    for issue in self.issues:
                        f.write(f"- {issue}\n")
                mlflow.log_artifact(issues_file)

            logger.info("Successfully logged validation results to MLflow")

        except Exception as e:
            logger.warning(f"Failed to log validation results to MLflow: {e}")

    def _generate_validation_report(self, dataset_name: str) -> None:
        """Generate comprehensive validation report."""
        logger.info("Generating validation report")

        try:
            # Create HTML report with shorter filename
            report_html = self._create_html_report(dataset_name)
            report_file = f"val_report_{dataset_name[:10]}.html"  # Limit filename length
            save_html(report_file, report_html)

            # Log to MLflow
            mlflow.log_artifact(report_file)

            # Create summary report
            summary = {
                "dataset_name": dataset_name,
                "validation_timestamp": pd.Timestamp.now().isoformat(),
                "total_issues": len(self.issues),
                "validation_score": max(0, 100 - len(self.issues) * 10),
                "critical_issues": [issue for issue in self.issues if "leakage" in issue.lower() or "null" in issue.lower()],
                "warnings": [issue for issue in self.issues if "leakage" not in issue.lower() and "null" not in issue.lower()],
                "recommendations": self._generate_recommendations()
            }

            summary_file = f"validation_summary_{dataset_name}.yaml"
            with open(summary_file, 'w') as f:
                yaml.dump(summary, f, default_flow_style=False)

            mlflow.log_artifact(summary_file)

            logger.info("Successfully generated validation report")

        except Exception as e:
            logger.warning(f"Failed to generate validation report: {e}")

    def _create_html_report(self, dataset_name: str) -> str:
        """Create HTML validation report."""
        # Create subplots for different visualizations
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Data Quality Overview",
                "Missing Data Pattern",
                "Target Distribution",
                "Feature Correlations",
                "Validation Issues",
                "Recommendations"
            ),
            specs=[
                [{"type": "indicator"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "heatmap"}],
                [{"type": "bar"}, {"type": "table"}]
            ]
        )

        # Data Quality Overview
        quality_score = max(0, 100 - len(self.issues) * 10)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=quality_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Data Quality Score"},
                delta={'reference': 100},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )

        # Missing Data Pattern
        if "schema" in self.validation_results.get(f"{dataset_name}_schema", {}):
            schema = self.validation_results[f"{dataset_name}_schema"]
            columns = list(schema.keys())
            null_percentages = [schema[col]["null_percentage"] for col in columns]

            fig.add_trace(
                go.Bar(
                    x=columns,
                    y=null_percentages,
                    name="Null Percentage",
                    marker_color='red'
                ),
                row=1, col=2
            )

        # Target Distribution
        if "target" in self.validation_results.get(f"{dataset_name}_target", {}):
            target_info = self.validation_results[f"{dataset_name}_target"]
            if target_info.get("task_type") == "classification":
                class_dist = target_info.get("class_distribution", {})
                fig.add_trace(
                    go.Histogram(
                        x=list(class_dist.keys()),
                        y=list(class_dist.values()),
                        name="Class Distribution"
                    ),
                    row=2, col=1
                )

        # Feature Correlations (placeholder)
        fig.add_trace(
            go.Heatmap(
                z=[[1, 0.5], [0.5, 1]],
                x=["Feature 1", "Feature 2"],
                y=["Feature 1", "Feature 2"],
                colorscale="RdBu"
            ),
            row=2, col=2
        )

        # Validation Issues
        issue_counts = {}
        for issue in self.issues:
            issue_type = issue.split(":")[0] if ":" in issue else "Other"
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        if issue_counts:
            fig.add_trace(
                go.Bar(
                    x=list(issue_counts.keys()),
                    y=list(issue_counts.values()),
                    name="Issue Counts",
                    marker_color='orange'
                ),
                row=3, col=1
            )

        # Update layout
        fig.update_layout(
            height=1200,
            title_text=f"Data Validation Report - {dataset_name}",
            showlegend=False
        )

        return fig.to_html(include_plotlyjs='cdn')

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on validation issues."""
        recommendations = []

        if not self.issues:
            recommendations.append("Dataset passed all validation checks. No immediate action required.")
            return recommendations

        # Data quality recommendations
        null_issues = [issue for issue in self.issues if "null" in issue.lower()]
        if null_issues:
            recommendations.append("Address missing data: Consider imputation strategies or data collection improvements.")

        leakage_issues = [issue for issue in self.issues if "leakage" in issue.lower()]
        if leakage_issues:
            recommendations.append("Investigate potential data leakage: Review feature engineering and data collection processes.")

        imbalance_issues = [issue for issue in self.issues if "imbalance" in issue.lower()]
        if imbalance_issues:
            recommendations.append("Handle class imbalance: Consider resampling, class weights, or specialized algorithms.")

        correlation_issues = [issue for issue in self.issues if "correlation" in issue.lower()]
        if correlation_issues:
            recommendations.append("Address high correlations: Consider feature selection or dimensionality reduction.")

        # General recommendations
        if len(self.issues) > 5:
            recommendations.append("Multiple validation issues detected. Consider comprehensive data quality review.")

        recommendations.append("Monitor data quality metrics during model training and inference.")
        recommendations.append("Document any data preprocessing steps applied to address validation issues.")

        return recommendations

    def get_validation_summary(self) -> dict[str, Any]:
        """Get a summary of validation results."""
        return {
            "total_issues": len(self.issues),
            "validation_score": max(0, 100 - len(self.issues) * 10),
            "critical_issues": [issue for issue in self.issues if "leakage" in issue.lower() or "null" in issue.lower()],
            "warnings": [issue for issue in self.issues if "leakage" not in issue.lower() and "null" not in issue.lower()],
            "is_valid": len(self.issues) == 0
        }

    def generate_reports(self, validation_results: dict[str, Any]) -> None:
        """
        Generate and save validation reports.
        
        Args:
            validation_results: Validation results dictionary
        """
        try:
            # Extract dataset name from results
            dataset_name = "dataset"
            for key in validation_results.keys():
                if "_basic_info" in key:
                    dataset_name = key.replace("_basic_info", "")
                    break
            
            # Generate validation report
            self._generate_validation_report(dataset_name)
            
            # Create summary report
            summary = self.get_validation_summary()
            summary["dataset_name"] = dataset_name
            summary["validation_timestamp"] = pd.Timestamp.now().isoformat()
            
            # Save summary as YAML
            summary_file = f"validation_summary_{dataset_name}.yaml"
            with open(summary_file, 'w') as f:
                yaml.dump(summary, f, default_flow_style=False)
            
            logger.info(f"Generated validation reports for {dataset_name}")
            
        except Exception as e:
            logger.warning(f"Failed to generate reports: {e}")
