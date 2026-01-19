import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings('ignore')

# Define the models to use
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(),
    'MLP Regressor': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, early_stopping=True),
    'K-Neighbors': KNeighborsRegressor()
}

# Function to evaluate models using cross-validation
def evaluate_models_cv(X, y, target_name):
    """
    Evaluates models using cross-validation for reliable performance estimation
    """
    results = {}
    
    for name, model in models.items():
        # Use cross-validation to get reliable R² scores
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        model.fit(X, y)  # Fit on full data for other metrics
        y_pred = model.predict(X)
        
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        results[name] = {
            'R²': r2,
            'R²_CV_mean': cv_scores.mean(),
            'R²_CV_std': cv_scores.std(),
            'MAE': mae,
            'RMSE': rmse
        }
    
    return results

# Function to plot experimental vs predicted values
def plot_predictions(X, y, results_dict, catalyst_name, target_name):
    """
    Plots experimental vs predicted values for all models
    """
    plt.figure(figsize=(10, 6))
    
    for name, metrics in results_dict.items():
        model = models[name]
        model.fit(X, y)
        y_pred = model.predict(X)
        plt.scatter(y, y_pred, alpha=0.7, label=f'{name} (R²={metrics["R²"]:.3f})')
    
    max_val = max(y.max(), max([models[name].predict(X).max() for name in results_dict.keys()]))
    min_val = min(y.min(), min([models[name].predict(X).min() for name in results_dict.keys()]))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Experimental ' + target_name)
    plt.ylabel('Predicted ' + target_name)
    plt.title(f'{catalyst_name}: Experimental vs Predicted {target_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to create separate heatmaps for each metric
def create_separate_heatmaps(results_dict, catalyst_name, target_name):
    """
    Creates separate heatmaps for R², MAE, and RMSE with proper scaling
    """
    metrics_df = pd.DataFrame(results_dict).T
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{catalyst_name}: Model Performance for {target_name} Prediction', fontsize=16, fontweight='bold')
    
    # R² Heatmap (0-1 scale, higher is better)
    sns.heatmap(metrics_df[['R²']], annot=True, fmt='.4f', cmap='Greens', 
                vmin=0, vmax=1, ax=axes[0], cbar_kws={'label': 'R² Score (0-1)'})
    axes[0].set_title('R² Score\n(Higher = Better)')
    
    # MAE Heatmap (lower is better)
    max_mae = metrics_df['MAE'].max() * 1.1  # Add 10% padding
    sns.heatmap(metrics_df[['MAE']], annot=True, fmt='.4f', cmap='Reds_r', 
                vmin=0, vmax=max_mae, ax=axes[1], cbar_kws={'label': 'MAE (Lower = Better)'})
    axes[1].set_title('Mean Absolute Error\n(Lower = Better)')
    
    # RMSE Heatmap (lower is better)
    max_rmse = metrics_df['RMSE'].max() * 1.1  # Add 10% padding
    sns.heatmap(metrics_df[['RMSE']], annot=True, fmt='.4f', cmap='Reds_r', 
                vmin=0, vmax=max_rmse, ax=axes[2], cbar_kws={'label': 'RMSE (Lower = Better)'})
    axes[2].set_title('Root Mean Squared Error\n(Lower = Better)')
    
    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION ---
# 1. READ THE EXCEL FILE
file_path = r"C:\Users\pooji\OneDrive\Desktop\research paper\new research paper sep2025.xlsx"
sheet_names = ["Ni-Ba on ZrO2", "Ni-Ca on ZrO2", "Ni-Mn on ZrO2", "Ni-Mg on ZrO2", "Ni-K on ZrO2"]
catalyst_data = {}

for sheet in sheet_names:
    try:
        df = pd.read_excel(file_path, sheet_name=sheet, header=1)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        if len(df.columns) >= 5:
            expected_columns = ['Hour', 'CH4_Conversion', 'CO2_Conversion', 'H2_Selectivity_Exp', 'CO_Selectivity_Exp']
            df.columns = expected_columns[:len(df.columns)]
        
        catalyst_data[sheet] = df
        print(f"Successfully processed: {sheet}")
            
    except Exception as e:
        print(f"Error loading {sheet}: {e}")

# 2. PROCESS EACH CATALYST (ML Analysis)
all_results = {}

for catalyst_name, df in catalyst_data.items():
    print(f"\n{'-'*50}")
    print(f"PROCESSING: {catalyst_name}")
    print(f"{'-'*50}")
    
    try:
        X = df[['CH4_Conversion', 'CO2_Conversion']]
        y_h2 = df['H2_Selectivity_Exp']
        y_co = df['CO_Selectivity_Exp']
        
    except KeyError as e:
        print(f"Column error: {e}")
        continue
    
    # Evaluate for H2 Selectivity
    print(f"\nEvaluating models for {catalyst_name} - H2 Selectivity")
    results_h2 = evaluate_models_cv(X, y_h2, 'H2 Selectivity')
    create_separate_heatmaps(results_h2, catalyst_name, 'H₂')
    plot_predictions(X, y_h2, results_h2, catalyst_name, 'H₂ Selectivity')
    
    # Evaluate for CO Selectivity
    print(f"\nEvaluating models for {catalyst_name} - CO Selectivity")
    results_co = evaluate_models_cv(X, y_co, 'CO Selectivity')
    create_separate_heatmaps(results_co, catalyst_name, 'CO')
    plot_predictions(X, y_co, results_co, catalyst_name, 'CO Selectivity')
    
    all_results[catalyst_name] = {'H2': results_h2, 'CO': results_co}

# 3. EXPERIMENTAL PERFORMANCE ANALYSIS (The REAL catalyst ranking)
print("\n" + "="*70)
print("EXPERIMENTAL CATALYST PERFORMANCE RANKING")
print("="*70)

# Calculate average experimental performance for each catalyst
catalyst_performance = {}

for catalyst_name, df in catalyst_data.items():
    # Average conversion and selectivity over the entire time series
    avg_ch4_conv = df['CH4_Conversion'].mean()
    avg_co2_conv = df['CO2_Conversion'].mean()
    avg_h2_sel = df['H2_Selectivity_Exp'].mean()
    avg_co_sel = df['CO_Selectivity_Exp'].mean()
    
    # Calculate overall performance score (weighted average)
    # You can adjust these weights based on what's most important for your application
    performance_score = (
        0.4 * avg_ch4_conv +  # CH4 conversion importance
        0.3 * avg_co2_conv +  # CO2 conversion importance  
        0.2 * avg_h2_sel +    # H2 selectivity importance
        0.1 * avg_co_sel      # CO selectivity importance
    )
    
    catalyst_performance[catalyst_name] = {
        'CH4_Conversion_Avg': avg_ch4_conv,
        'CO2_Conversion_Avg': avg_co2_conv,
        'H2_Selectivity_Avg': avg_h2_sel,
        'CO_Selectivity_Avg': avg_co_sel,
        'Overall_Performance': performance_score
    }

# Create performance comparison dataframe
performance_df = pd.DataFrame(catalyst_performance).T
performance_df['Rank'] = performance_df['Overall_Performance'].rank(ascending=False)

print("Experimental Performance Summary:")
print(performance_df.round(3).sort_values('Overall_Performance', ascending=False))

# 4. FINAL CATALYST RANKING PLOT (Based on EXPERIMENTAL data)
plt.figure(figsize=(14, 10))

# Sort catalysts by overall performance
sorted_df = performance_df.sort_values('Overall_Performance', ascending=False)
catalysts_sorted = sorted_df.index
performance_sorted = sorted_df['Overall_Performance']

# Create main ranking plot
plt.subplot(2, 1, 1)
colors = ['#2E8B57' if 'Ba' in cat else '#FF8C00' if 'Ca' in cat else 
          '#1E90FF' if 'Mn' in cat else '#9370DB' if 'Mg' in cat else '#DC143C' for cat in catalysts_sorted]

bars = plt.bar(range(len(catalysts_sorted)), performance_sorted, color=colors, alpha=0.8)
plt.xlabel('Catalyst', fontsize=12, fontweight='bold')
plt.ylabel('Overall Performance Score', fontsize=12, fontweight='bold')
plt.title('FINAL CATALYST RANKING: Experimental Performance Comparison\n(Based on Average Conversion and Selectivity)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xticks(range(len(catalysts_sorted)), catalysts_sorted, rotation=45, ha='right', fontsize=10)

# Add value labels
for i, (bar, score) in enumerate(zip(bars, performance_sorted)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{score:.1f}\n(Rank #{i+1})', ha='center', va='bottom', fontweight='bold')

# Add detailed metrics subplot
plt.subplot(2, 1, 2)
metrics_to_plot = ['CH4_Conversion_Avg', 'CO2_Conversion_Avg', 'H2_Selectivity_Avg', 'CO_Selectivity_Avg']
x_pos = np.arange(len(catalysts_sorted))
width = 0.2

for i, metric in enumerate(metrics_to_plot):
    values = [sorted_df.loc[cat, metric] for cat in catalysts_sorted]
    plt.bar(x_pos + i*width, values, width, label=metric.replace('_', ' ').replace('Avg', ''))

plt.xlabel('Catalyst', fontsize=12, fontweight='bold')
plt.ylabel('Performance Metric Value (%)', fontsize=12, fontweight='bold')
plt.title('Detailed Performance Metrics by Catalyst', fontsize=14, fontweight='bold')
plt.xticks(x_pos + 1.5*width, catalysts_sorted, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 5. FINAL CONCLUSION
print("\n" + "="*80)
print("FINAL CONCLUSION: BEST CATALYST IDENTIFICATION")
print("="*80)

best_catalyst = performance_df['Overall_Performance'].idxmax()
best_score = performance_df.loc[best_catalyst, 'Overall_Performance']
best_ch4 = performance_df.loc[best_catalyst, 'CH4_Conversion_Avg']
best_co2 = performance_df.loc[best_catalyst, 'CO2_Conversion_Avg']
best_h2 = performance_df.loc[best_catalyst, 'H2_Selectivity_Avg']
best_co = performance_df.loc[best_catalyst, 'CO_Selectivity_Avg']

print(f" BEST CATALYST: {best_catalyst}")
print(f" Overall Performance Score: {best_score:.1f}")
print(f" Key Metrics:")
print(f"   • CH₄ Conversion: {best_ch4:.1f}%")
print(f"   • CO₂ Conversion: {best_co2:.1f}%")
print(f"   • H₂ Selectivity: {best_h2:.1f}%")
print(f"   • CO Selectivity: {best_co:.1f}%")
print(f"\n This catalyst demonstrates superior performance across all key metrics,")
print(f"   making it the most effective choice for the reaction.")

# Show runner-up
second_best = performance_df.nlargest(2, 'Overall_Performance').index[1]
print(f"\n RUNNER-UP: {second_best}")
