import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class HDFSIsolationForest:
    """
    Isolation Forest implementation for unsupervised anomaly detection in HDFS system logs.
    """
    
    def __init__(self, contamination=0.1, random_state=42, n_estimators=100):
        """
        Initialize the Isolation Forest detector.
        
        Parameters:
        -----------
        contamination : float
            The expected proportion of outliers in the dataset (default: 0.1)
        random_state : int
            Random seed for reproducibility (default: 42)
        n_estimators : int
            Number of trees in the forest (default: 100)
        """
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None
        
    def fit(self, X):
        """
        Fit the Isolation Forest model on the data.
        
        Parameters:
        -----------
        X : array-like
            Training data
        """
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            max_samples='auto'
        )
        self.model.fit(X_scaled)
        
        return self
    
    def predict(self, X):
        """
        Predict anomalies (1 for normal, -1 for anomaly).
        
        Parameters:
        -----------
        X : array-like
            Data to predict
        
        Returns:
        --------
        predictions : array
            1 for normal, -1 for anomaly
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def anomaly_scores(self, X):
        """
        Get anomaly scores (lower = more anomalous).
        
        Parameters:
        -----------
        X : array-like
            Data to score
        
        Returns:
        --------
        scores : array
            Anomaly scores
        """
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)
    
    def fit_predict(self, X):
        """
        Fit the model and return predictions.
        
        Parameters:
        -----------
        X : array-like
            Training data
        
        Returns:
        --------
        predictions : array
            1 for normal, -1 for anomaly
        """
        self.fit(X)
        return self.predict(X)


def generate_sample_data(n_samples=1000, n_features=5, anomaly_ratio=0.1):
    """
    Generate synthetic HDFS log data for demonstration.
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples
    n_features : int
        Number of features
    anomaly_ratio : float
        Proportion of anomalies
    
    Returns:
    --------
    X, y : tuple
        Features and labels (0=normal, 1=anomaly)
    """
    np.random.seed(42)
    
    # Generate normal data
    n_normal = int(n_samples * (1 - anomaly_ratio))
    n_anomalies = n_samples - n_normal
    
    # Normal data from multivariate normal distribution
    normal_data = np.random.multivariate_normal(
        mean=[10, 50, 5, 100, 20],
        cov=[[2, 0.5, 0.3, 0.8, 0.4],
             [0.5, 10, 0.6, 1.2, 0.7],
             [0.3, 0.6, 1, 0.4, 0.2],
             [0.8, 1.2, 0.4, 20, 1.0],
             [0.4, 0.7, 0.2, 1.0, 5]],
        size=n_normal
    )
    
    # Anomalous data with different distribution
    anomaly_data = np.random.multivariate_normal(
        mean=[25, 150, 15, 300, 60],
        cov=[[5, 2, 1, 3, 2],
             [2, 30, 3, 8, 5],
             [1, 3, 3, 2, 1],
             [3, 8, 2, 50, 10],
             [2, 5, 1, 10, 15]],
        size=n_anomalies
    )
    
    X = np.vstack([normal_data, anomaly_data])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])
    
    # Shuffle the data
    idx = np.random.permutation(n_samples)
    X = X[idx]
    y = y[idx]
    
    # Create feature names typical for HDFS log analysis
    feature_names = [
        'total_event_count',
        'unique_event_types',
        'max_event_count',
        'log_entry_count',
        'error_frequency'
    ]
    
    return pd.DataFrame(X, columns=feature_names), y


def visualize_results(X, y_true, y_pred, anomaly_scores, save_path='figures/'):
    """
    Create comprehensive visualizations of the Isolation Forest results.
    
    Parameters:
    -----------
    X : DataFrame
        Feature data
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    anomaly_scores : array
        Anomaly scores
    save_path : str
        Path to save figures
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # Convert predictions to binary (0=normal, 1=anomaly)
    y_pred_binary = (y_pred == -1).astype(int)
    
    # 1. PCA Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Plot true labels
    scatter = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, 
                                   cmap='RdYlBu', alpha=0.6, s=50)
    axes[0, 0].set_xlabel('First Principal Component', fontsize=12)
    axes[0, 0].set_ylabel('Second Principal Component', fontsize=12)
    axes[0, 0].set_title('True Labels (PCA)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=axes[0, 0], label='True Class')
    
    # Plot predicted labels
    scatter = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_binary, 
                                   cmap='RdYlBu', alpha=0.6, s=50)
    axes[0, 1].set_xlabel('First Principal Component', fontsize=12)
    axes[0, 1].set_ylabel('Second Principal Component', fontsize=12)
    axes[0, 1].set_title('Predicted Labels (PCA)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=axes[0, 1], label='Predicted Class')
    
    # Plot anomaly scores
    scatter = axes[0, 2].scatter(X_pca[:, 0], X_pca[:, 1], c=anomaly_scores, 
                                   cmap='viridis_r', alpha=0.6, s=50)
    axes[0, 2].set_xlabel('First Principal Component', fontsize=12)
    axes[0, 2].set_ylabel('Second Principal Component', fontsize=12)
    axes[0, 2].set_title('Anomaly Scores (PCA)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=axes[0, 2], label='Anomaly Score')
    
    # 2. Feature Distribution Comparison
    feature_idx = 0
    for i in range(3):
        ax = axes[1, i]
        feature = X.columns[feature_idx]
        
        # Normal data
        ax.hist(X[y_true == 0][feature], bins=30, alpha=0.5, 
                label='Normal', color='blue', density=True)
        # Anomalous data
        ax.hist(X[y_true == 1][feature], bins=30, alpha=0.5, 
                label='Anomaly', color='red', density=True)
        
        ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Distribution: {feature.replace("_", " ").title()}', 
                     fontsize=14, fontweight='bold')
        ax.legend()
        
        feature_idx += 1
    
    plt.tight_layout()
    plt.savefig(f'{save_path}isolation_forest_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - Isolation Forest', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_path}isolation_forest_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, -anomaly_scores)  # Negative because lower scores = more anomalous
    auc_score = roc_auc_score(y_true, -anomaly_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Isolation Forest', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}isolation_forest_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Anomaly Score Distribution
    plt.figure(figsize=(10, 6))
    
    plt.hist(anomaly_scores[y_true == 0], bins=50, alpha=0.6, 
             label='Normal', color='blue', density=True)
    plt.hist(anomaly_scores[y_true == 1], bins=50, alpha=0.6, 
             label='Anomaly', color='red', density=True)
    
    plt.axvline(x=np.percentile(anomaly_scores, 100 * 0.1), 
                color='green', linestyle='--', linewidth=2, 
                label=f'Decision Threshold (contamination=0.1)')
    
    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Anomaly Scores', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}isolation_forest_score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Feature Importance (based on anomaly scores correlation)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Correlation with anomaly scores
    correlations = X.corrwith(pd.Series(anomaly_scores))
    correlations = correlations.sort_values(ascending=False)
    
    colors = ['red' if c < 0 else 'blue' for c in correlations]
    axes[0].barh(range(len(correlations)), correlations.values, color=colors, alpha=0.7)
    axes[0].set_yticks(range(len(correlations)))
    axes[0].set_yticklabels([name.replace('_', ' ').title() for name in correlations.index])
    axes[0].set_xlabel('Correlation with Anomaly Score', fontsize=12)
    axes[0].set_title('Feature Correlation with Anomaly Scores', fontsize=14, fontweight='bold')
    axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Box plot of features by class
    X_plot = X.copy()
    X_plot['class'] = ['Normal' if y == 0 else 'Anomaly' for y in y_true]
    X_melted = X_plot.melt(id_vars='class', var_name='Feature', value_name='Value')
    
    sns.boxplot(data=X_melted, x='Feature', y='Value', hue='class', ax=axes[1])
    axes[1].set_xlabel('Feature', fontsize=12)
    axes[1].set_ylabel('Value', fontsize=12)
    axes[1].set_title('Feature Distribution by Class', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(title='Class', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}isolation_forest_feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_performance(y_true, y_pred, anomaly_scores):
    """
    Evaluate and print the performance of the Isolation Forest model.
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    anomaly_scores : array
        Anomaly scores
    
    Returns:
    --------
    metrics : dict
        Dictionary of performance metrics
    """
    y_pred_binary = (y_pred == -1).astype(int)
    
    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc_score = roc_auc_score(y_true, -anomaly_scores)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'auc_score': auc_score,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }
    
    # Print results
    print("="*60)
    print("ISOLATION FOREST - ANOMALY DETECTION RESULTS")
    print("="*60)
    print("\nClassification Report:")
    print("-"*60)
    print(classification_report(y_true, y_pred_binary, 
                                target_names=['Normal', 'Anomaly']))
    
    print("\nConfusion Matrix:")
    print("-"*60)
    print(f"                Predicted")
    print(f"                Normal  Anomaly")
    print(f"Actual Normal    {tn:5d}   {fp:5d}")
    print(f"Actual Anomaly   {fn:5d}   {tp:5d}")
    
    print("\nPerformance Metrics:")
    print("-"*60)
    print(f"Accuracy:      {accuracy:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1-Score:      {f1_score:.4f}")
    print(f"Specificity:   {specificity:.4f}")
    print(f"AUC-ROC:       {auc_score:.4f}")
    print("="*60)
    
    return metrics


def main():
    """
    Main function to run Isolation Forest anomaly detection.
    """
    print("\n" + "="*60)
    print("ISOLATION FOREST FOR HDFS LOG ANOMALY DETECTION")
    print("="*60)
    
    # Generate sample data (replace with actual HDFS log data)
    print("\nGenerating sample HDFS log data...")
    X, y_true = generate_sample_data(n_samples=1000, n_features=5, anomaly_ratio=0.1)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Normal samples: {sum(y_true == 0)}")
    print(f"Anomalous samples: {sum(y_true == 1)}")
    print(f"Anomaly ratio: {sum(y_true == 1) / len(y_true):.2%}")
    
    # Display sample statistics
    print("\nFeature Statistics:")
    print("-"*60)
    print(X.describe())
    
    # Initialize and fit Isolation Forest
    print("\nTraining Isolation Forest model...")
    iso_forest = HDFSIsolationForest(
        contamination=0.1,
        random_state=42,
        n_estimators=100
    )
    
    # Fit and predict
    y_pred = iso_forest.fit_predict(X)
    anomaly_scores = iso_forest.anomaly_scores(X)
    
    # Evaluate performance
    print("\nEvaluating model performance...")
    metrics = evaluate_performance(y_true, y_pred, anomaly_scores)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_results(X, y_true, y_pred, anomaly_scores, save_path='figures/')
    
    print("\nVisualizations saved to 'figures/' directory:")
    print("  - isolation_forest_results.png")
    print("  - isolation_forest_confusion_matrix.png")
    print("  - isolation_forest_roc_curve.png")
    print("  - isolation_forest_score_distribution.png")
    print("  - isolation_forest_feature_analysis.png")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60 + "\n")
    
    return iso_forest, metrics


if __name__ == "__main__":
    model, metrics = main()
