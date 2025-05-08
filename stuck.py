import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

# Custom CSS for enhanced styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stHeader {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
        font-size: 28px;
        font-weight: bold;
    }
    .stText {
        color: #34495e;
        font-family: 'Verdana', sans-serif;
        font-size: 14px;
    }
    .stWarning {
        background-color: #fff3f3;
        padding: 10px;
        border-left: 4px solid #e74c3c;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit App Layout
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<h1 class="stHeader">Stuck Pipe Detection App</h1>', unsafe_allow_html=True)
st.markdown('<p class="stText">This app detects the risk of stuck pipe incidents by monitoring drilling parameters in real-time. Upload data files, adjust settings, and visualize results interactively.</p>', unsafe_allow_html=True)

# Sidebar for Controls
st.sidebar.title("Settings")
window_size = st.sidebar.slider("Window Size", min_value=10, max_value=100, value=36, step=5, help="Adjust the size of the data window for real-time monitoring.")
model_selection = st.sidebar.multiselect("Select Models to Train", ["Torque", "Hookload", "SPP"], default=["Torque", "Hookload", "SPP"], help="Choose which predictive models to train.")
st.sidebar.markdown("---")
st.sidebar.markdown('<p class="stText">Adjust settings and upload files to analyze drilling data.</p>', unsafe_allow_html=True)

# Specify Data Needed
with st.expander("📋 Required Data", expanded=True):
    st.markdown("""
    **Files Needed:**
    - `well2.csv`: A CSV file containing drilling data.
    - `well 16A.csv`: A CSV file containing drilling data.
    - `well F-4.xlsx`: An Excel file containing drilling data on the 'Drilling Parameters' sheet.

    **Expected Parameters (Columns):**
    The files should contain the following parameters (case-insensitive, with possible variations):
    - Rate of Penetration (ROP): e.g., 'ROP (ft/hr)', 'ROP Depth/Hour', 'ROP'
    - Rotary Speed (rpm): e.g., 'Rotary Speed (rpm)', 'Top Drive RPM', 'RPm'
    - Weight on Bit (WOB): e.g., 'Weight on bit (k-lbs)', 'Weight on Bit', 'WOB'
    - Torque: e.g., 'Surface Torque (psi)', 'Top Drive Torque (ft-lbs)', 'Torque'
    - Hookload: e.g., 'Hook load (k-lbs)', 'Hookload'
    - Standpipe Pressure (SPP): e.g., 'SPP'
    - Flow Pumps: e.g., 'Flow In (gal/min)', 'Flow In', 'Flow pumps'
    - Block Position: e.g., 'Block Position'

    **Notes:**
    - Missing values should be marked as -999.25 or NaN.
    - Non-numeric values will be coerced to NaN and reported.
    - Ensure data represents drilling phases (WOB > 0).
    """)

# File Upload Section
st.header("📤 Upload Drilling Data Files")
col1, col2, col3 = st.columns(3)
with col1:
    well2_file = st.file_uploader("Upload well2.csv", type=["csv"], key="well2", help="Upload the first CSV file.")
with col2:
    well16a_file = st.file_uploader("Upload well 16A.csv", type=["csv"], key="well16a", help="Upload the second CSV file.")
with col3:
    wellf4_file = st.file_uploader("Upload well F-4.xlsx", type=["xlsx"], key="wellf4", help="Upload the Excel file.")

# Check if all files are uploaded
if well2_file and well16a_file and wellf4_file:
    # Load and clean data
    @st.cache_data
    def load_and_clean_data(file_objects):
        dataframes = []
        for file_obj, name in [(well2_file, "well2.csv"), (well16a_file, "well 16A.csv"), (wellf4_file, "well F-4.xlsx")]:
            try:
                if name.endswith('.csv'):
                    df = pd.read_csv(file_obj)
                else:
                    df = pd.read_excel(file_obj, sheet_name='Drilling Parameters')
                st.write(f"📊 Columns in {name}:")
                st.write(df.columns.tolist())
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    non_numeric = df[col][df[col].isna() & df[col].notna()]
                    if not non_numeric.empty:
                        st.warning(f"⚠️ Non-numeric values found in {name} column '{col}': {non_numeric}")
                dataframes.append(df)
            except Exception as e:
                st.error(f"❌ Error loading {name}: {e}. Please check the file format or content.")
                return None
        
        if not dataframes:
            st.error("❌ No dataframes loaded successfully. Please upload valid files.")
            return None
        
        param_mappings = {
            'ROP': ['ROP (ft/hr)', 'ROP Depth/Hour', 'ROP'],
            'rpm': ['Rotary Speed (rpm)', 'Top Drive RPM', 'RPm'],
            'WOB': ['Weight on bit (k-lbs)', 'Weight on Bit', 'WOB'],
            'Torque': ['Surface Torque (psi)', 'Top Drive Torque (ft-lbs)', 'Torque'],
            'Hookload': ['Hook load (k-lbs)', 'Hookload'],
            'SPP': ['SPP'],
            'Flow pumps': ['Flow In (gal/min)', 'Flow In', 'Flow pumps'],
            'Block Position': ['Block Position']
        }
        
        combined_data = {}
        for param, aliases in param_mappings.items():
            series_list = [df[alias] for df in dataframes for alias in aliases if alias in df.columns]
            if series_list:
                combined_data[param] = pd.concat(series_list, ignore_index=True)
            else:
                st.warning(f"⚠️ No columns found for parameter '{param}' with aliases {aliases}")
        
        if not combined_data:
            st.error("❌ No parameters could be extracted from the datasets.")
            return None
        
        data = pd.DataFrame(combined_data)
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            non_numeric = data[col][pd.to_numeric(data[col], errors='coerce').isna() & ~data[col].isna()]
            if not non_numeric.empty:
                st.warning(f"⚠️ Non-numeric values found in combined data column '{col}': {non_numeric}")
        
        st.header("📊 Data Overview")
        st.write("Data types after concatenation and conversion:")
        st.write(data.dtypes)
        st.write("Sample values (first 5 rows):")
        st.write(data.head())
        
        data = data.replace(-999.25, np.nan).dropna()
        for col in ['ROP', 'rpm', 'WOB', 'Hookload', 'SPP', 'Flow pumps', 'Block Position']:
            if col in data.columns:
                data = data[data[col] >= 0]
        if 'WOB' in data.columns:
            data = data[data['WOB'] > 0]
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        z_scores = np.abs(stats.zscore(data[numeric_cols]))
        data = data[(z_scores < 3).all(axis=1)]
        
        st.header("📈 Data Distributions")
        fig, axes = plt.subplots(2, 4, figsize=(12, 8))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            axes[i].hist(data[col], bins=30, edgecolor='black')
            axes[i].set_title(f'Distribution of {col}')
        plt.tight_layout()
        st.pyplot(fig)
        
        return data

    # Predictive Models Construction
    def train_and_evaluate_model(X, y, target_name):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"🎯 {target_name} Model - R²: {r2:.4f}, MSE: {mse:.4f}")
        
        residuals = y_test - y_pred
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].hist(residuals, bins=30, edgecolor='black')
        axes[0].set_title('Residuals Histogram')
        sm.qqplot(residuals, line='s', ax=axes[1])
        axes[1].set_title('Normal Probability Plot')
        X_test_ols = sm.add_constant(X_test_scaled)
        ols_model = sm.OLS(y_test, X_test_ols).fit()
        influence = OLSInfluence(ols_model)
        cooks_d = influence.cooks_distance[0]
        axes[2].stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
        axes[2].set_title("Cook's Distance")
        plt.tight_layout()
        st.pyplot(fig)
        
        importances = model.feature_importances_
        st.write("📋 Feature Importances:")
        for feature, importance in zip(X.columns, importances):
            st.write(f"  - {feature}: {importance:.4f}")
        return model, scaler, list(X.columns)

    # Real-Time Monitoring Functions
    def preprocess_window(window):
        window = window.dropna()
        numeric_cols = window.select_dtypes(include=[np.number]).columns
        z_scores = np.abs(stats.zscore(window[numeric_cols]))
        window = window[(z_scores < 3).all(axis=1)]
        return window

    def calculate_deviation(actual, predicted):
        if len(actual) == 0 or len(predicted) == 0:
            return 0
        actual_avg = np.mean(actual)
        predicted_avg = np.mean(predicted)
        return abs(actual_avg - predicted_avg) / predicted_avg * 100 if predicted_avg != 0 else 0

    def generate_alert(prev_deviation, current_deviation):
        if prev_deviation == 0:
            return 0
        deviation_relation = current_deviation / prev_deviation
        if deviation_relation > 3:
            return 3
        elif deviation_relation > 1:
            return deviation_relation
        return 0

    def calculate_stuck_probability(alert_levels):
        max_alert_level = 3 * len(alert_levels)
        total_alert = sum(alert_levels)
        probability = (total_alert / max_alert_level) * 100
        return min(probability, 100)

    # Run Analysis
    if st.button("🚀 Run Analysis"):
        with st.spinner("🔄 Processing data and running analysis..."):
            progress_bar = st.progress(0)
            data = load_and_clean_data([(well2_file, "well2.csv"), (well16a_file, "well 16A.csv"), (wellf4_file, "well F-4.xlsx")])
            progress_bar.progress(20)
            if data is None:
                st.stop()

            # Train models
            st.header("📊 Model Training Results")
            models = {}
            scalers = {}
            feature_names = {}
            model_configs = {
                'Torque': ['ROP', 'rpm', 'WOB'],
                'Hookload': ['ROP', 'rpm', 'WOB', 'Torque', 'Flow pumps', 'Block Position'],
                'SPP': ['ROP', 'WOB', 'Flow pumps']
            }
            progress_step = 60 / len(model_selection) if model_selection else 60

            for i, target in enumerate(model_selection):
                if target in data.columns and all(col in data.columns for col in model_configs[target]):
                    X = data[model_configs[target]]
                    y = data[target]
                    model, scaler, features = train_and_evaluate_model(X, y, target)
                    models[target] = model
                    scalers[target] = scaler
                    feature_names[target] = features
                else:
                    st.warning(f"⚠️ Skipping model for {target}: required columns missing.")
                progress_bar.progress(20 + int((i + 1) * progress_step))

            # Real-time monitoring
            st.header("📈 Real-Time Stuck Pipe Probability Monitoring")
            real_time_df = data.copy()
            progress_step = 20 / (len(real_time_df) // window_size + 1) if len(real_time_df) // window_size else 20
            prev_deviations = {key: 0 for key in models.keys()}
            probabilities = []
            window_indices = []

            for idx, start in enumerate(range(0, len(real_time_df), window_size)):
                window = real_time_df.iloc[start:start + window_size]
                window = preprocess_window(window)
                if len(window) == 0:
                    continue
                
                alert_levels = []
                for param in models.keys():
                    input_cols = [col for col in feature_names[param] if col in window.columns]
                    if not input_cols:
                        continue
                    features = scalers[param].transform(window[input_cols])
                    actual = window[param].values
                    predicted = models[param].predict(features)
                    
                    window_size_smooth = min(5, len(actual))
                    actual_smooth = np.convolve(actual, np.ones(window_size_smooth)/window_size_smooth, mode='valid')
                    predicted_smooth = np.convolve(predicted, np.ones(window_size_smooth)/window_size_smooth, mode='valid')
                    
                    deviation = calculate_deviation(actual_smooth, predicted_smooth)
                    alert = generate_alert(prev_deviations[param], deviation)
                    prev_deviations[param] = deviation
                    alert_levels.append(alert)
                
                stuck_probability = calculate_stuck_probability(alert_levels)
                probabilities.append(stuck_probability)
                window_indices.append(idx)
                st.write(f"🕒 Window {idx}: Probability = {stuck_probability:.2f}%")
                progress_bar.progress(20 + int((idx + 1) * progress_step))

            # Plot stuck pipe probability
            st.subheader("📉 Probability Plot")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(window_indices, probabilities, marker='o', linestyle='-', color='#3498db', label='Stuck Pipe Probability')
            for i, prob in enumerate(probabilities):
                ax.text(window_indices[i], prob, f'{prob:.1f}', ha='center', va='bottom', color='#2c3e50')
                if prob > 50:
                    ax.scatter(window_indices[i], prob, color='#e74c3c', label='Alarm' if i == 0 else "")
            ax.axhline(y=50, color='#e74c3c', linestyle='--', label='Alarm Threshold (50%)')
            ax.set_xlabel('Window Number', fontsize=12)
            ax.set_ylabel('Stuck Pipe Probability (%)', fontsize=12)
            ax.set_title('Real-Time Stuck Pipe Probability Monitoring', fontsize=14, pad=15)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
            progress_bar.progress(100)

st.markdown('</div>', unsafe_allow_html=True)
if not (well2_file and well16a_file and wellf4_file):
    st.info("📥 Please upload all three files to proceed.")