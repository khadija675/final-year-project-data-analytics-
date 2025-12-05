import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore') # Suppress warnings related to Matplotlib/Seaborn styling

# Set a consistent random seed for reproducibility 
GLOBAL_SEED = 42 
np.random.seed(GLOBAL_SEED)

# --- Enhanced Plotting Style for better aesthetics ---
# Using 'darkgrid' background and setting a custom color palette
plt.style.use('seaborn-v0_8-darkgrid') 
COLOR_REAL = '#1f77b4' # Muted Blue
COLOR_SYNTH = '#ff7f0e' # Vibrant Orange
# ------------------------

# ------------------------
# Page config & Custom CSS
# ------------------------
st.set_page_config(page_title="Synthetic Data Lab", layout="wide", page_icon="🕸️")

st.markdown("""
<style>
/* Main Content Styling */
.stApp {
    background-color: #f0f2f6; /* Light gray background */
}
/* Card style for summary */
.card {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    text-align: center;
    border-left: 5px solid #10c0c0; /* Accent border */
}
/* Custom Navigation Bar */
.navbar {
    display: flex;
    justify-content: flex-start;
    gap: 20px;
    margin-bottom: 30px;
    border-bottom: 2px solid #ccc;
    padding-bottom: 10px;
}
.nav-item {
    cursor: pointer;
    font-size: 1.1em;
    font-weight: 600;
    padding: 5px 0;
    color: #4f4f4f;
    transition: color 0.3s;
}
.nav-item:hover {
    color: #10c0c0;
}
.active-nav {
    color: #10c0c0 !important;
    border-bottom: 3px solid #10c0c0;
}
</style>
""", unsafe_allow_html=True)


# ------------------------
# Helper functions
# ------------------------
def missing_summary(df):
    """Calculates missing values and percentage for each column.
    
    Returns a DataFrame with columns: 'missing_count', 'missing_percent', 'dtype'.
    """
    miss = df.isnull().sum()
    percent = miss/len(df)*100
    return pd.DataFrame({"missing_count": miss, "missing_percent": percent, "dtype": df.dtypes.astype(str)}).sort_values("missing_percent", ascending=False)

def identify_and_drop_unsuitable_cols(df):
    """Identifies and drops columns unsuitable for synthesis (dates, high-cardinality IDs)."""
    unsuitable_cols = []
    
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            unsuitable_cols.append(col)
        # Check for object columns that are nearly unique (likely IDs/identifiers)
        elif df[col].dtype == 'object' and df[col].nunique() > df.shape[0] * 0.9:
            unsuitable_cols.append(col)
            
    if unsuitable_cols:
        st.warning(f"Dropping unsuitable columns (Date/Time, High-Cardiality Text/IDs): {unsuitable_cols}")
        df = df.drop(columns=unsuitable_cols)
    return df

def plot_numeric_kde_ks(real_df, synth_df, numeric_cols):
    """KDE plotting and KS-test for numeric distributions."""
    for col in numeric_cols:
        real_data = real_df[col].dropna()
        synth_data = synth_df[col].dropna()

        # Robustness Check 1: Ensure variance exists for smooth plotting
        if real_data.empty or synth_data.empty or real_data.std() == 0 or synth_data.std() == 0:
            st.warning(f"Skipping KDE for '{col}': Data is constant or missing. Cannot calculate distribution.")
            continue
            
        fig, ax = plt.subplots(figsize=(6,4))
        
        sns.kdeplot(real_data, ax=ax, label="Real Data", color=COLOR_REAL, linewidth=2.5, fill=True, alpha=0.3)
        sns.kdeplot(synth_data, ax=ax, label="Synthetic Data", color=COLOR_SYNTH, linewidth=2.5, fill=True, alpha=0.3)
        
        ax.set_title(f"KDE Plot - {col}", fontsize=14, fontweight='bold')
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)
        
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)
        
        # KS-test
        try:
            ks = ks_2samp(real_data, synth_data)
            st.write(f"**{col} → KS-test** stat: **{ks.statistic:.4f}**, p-value: **{ks.pvalue:.4f}**")
        except ValueError:
            st.write(f"**{col} → KS-test error:** Cannot perform KS-test. (Check for uniform/sparse data)")


def plot_category_histograms(df_original, df_synthetic, cat_cols):
    """Side-by-side categorical histogram plotting."""
    if not cat_cols:
        st.info("No categorical columns found for histogram analysis.")
        return

    for col in cat_cols:
        # Prepare data for plotting
        real_counts = df_original[col].value_counts(dropna=False).sort_index()
        synth_counts = df_synthetic[col].value_counts(dropna=False).sort_index()

        # Combine indices for consistent plotting
        all_categories = sorted(list(set(real_counts.index) | set(synth_counts.index)))

        real_data = real_counts.reindex(all_categories, fill_value=0)
        synth_data = synth_counts.reindex(all_categories, fill_value=0)

        # Calculate max height for consistent y-axis across both plots
        max_y = max(real_data.max(), synth_data.max()) * 1.1

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        # Real Data Plot
        axes[0].bar(real_data.index, real_data.values, color=COLOR_REAL, alpha=0.7)
        axes[0].set_title(f"Real Data: {col}", fontsize=14, fontweight='bold')
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_ylim(0, max_y)

        # Synthetic Data Plot
        axes[1].bar(synth_data.index, synth_data.values, color=COLOR_SYNTH, alpha=0.7)
        axes[1].set_title(f"Synthetic Data: {col}", fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_ylim(0, max_y)
        
        plt.tight_layout()
        st.pyplot(fig)


def plot_correlation_heatmaps(real_df, synth_df, numeric_cols):
    """Generates and displays side-by-side correlation heatmaps."""
    if not numeric_cols:
        st.info("No numeric columns found for correlation analysis.")
        return

    # Check for empty or constant numeric columns
    valid_numeric_cols = [col for col in numeric_cols if real_df[col].nunique() > 1 and synth_df[col].nunique() > 1]
    if len(valid_numeric_cols) < 2:
        st.info("Need at least two non-constant numeric columns to plot correlation.")
        return

    real_corr = real_df[valid_numeric_cols].corr()
    synth_corr = synth_df[valid_numeric_cols].corr()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sns.heatmap(real_corr, annot=True, fmt=".2f", cmap='coolwarm', 
                cbar_kws={'shrink': 0.75}, ax=axes[0], linewidths=0.5, linecolor='white')
    axes[0].set_title('Real Data Correlation Heatmap', fontsize=16, fontweight='bold')

    sns.heatmap(synth_corr, annot=True, fmt=".2f", cmap='coolwarm', 
                cbar_kws={'shrink': 0.75}, ax=axes[1], linewidths=0.5, linecolor='white')
    axes[1].set_title('Synthetic Data Correlation Heatmap', fontsize=16, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

def make_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="background-color: #10c0c0; color: white; padding: 10px 20px; text-decoration: none; border-radius: 8px; font-weight: bold; display: inline-block;">📥 Download {filename}</a>'

# ------------------------------------------------------------------------------------------------
# PAGE DEFINITIONS
# ------------------------------------------------------------------------------------------------

def homepage():
    st.markdown("<h1 style='text-align:center; color:#1f77b4;'>Synthetic Data Studio</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color:#6b6b6b;'>Create, Anonymize, and Validate High-Quality Synthetic Datasets</h3>", unsafe_allow_html=True)
    
    st.write("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Better, more symbolic placeholder image/logo
        st.image("https://placehold.co/400x300/10c0c0/ffffff?text=DATA+SYNTHESIS%0AENGINE", caption="AI-Powered Data Generation")
        st.markdown("""
        <div style="padding-top: 15px; text-align: center;">
            <p style='font-size: 1.1em;'>Synthetic data is crucial for testing, development, and sharing insights while protecting privacy.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("## 🎯 Our 8-Step Synthesis Workflow")
        st.markdown("""
        This tool follows a professional, end-to-end pipeline to ensure high-fidelity and validated synthetic output:
        1.  **Input:** Upload your raw CSV dataset.
        2.  **Data Preparation:** Read data, categorize columns, and automatically drop unsuitable fields (like dates/IDs) and high-missing columns.
        3.  **Generation:** Apply **CTGAN** (Deep Learning) or Gaussian Copula to learn data structure.
        4.  **Automatic Postprocessing:** Automatically clean up the synthetic data (e.g., truncate overly long text categories) before validation.
        5.  **Statistical Validation:** Calculate and compare Mean, Standard Deviation, and Missing structure.
        6.  **Distribution Validation:** Check categorical histograms.
        7.  **Quality Validation:** Use **KDE plots** and the **Kolmogorov-Smirnov (KS) Test** for numeric columns.
        8.  **Download:** Securely download the final data.
        """)

    st.write("---")
    if st.button("🚀 Start Synthesis"):
        st.session_state.page = "lab"
        st.rerun()


def about_page():
    st.markdown("<h1 style='color:#1f77b4;'>About This Project</h1>", unsafe_allow_html=True)
    st.markdown("### Technology Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Streamlit (UI/UX)")
        st.info("Provides the interactive, web-based interface for easy step-by-step workflow management.")
    
    with col2:
        st.markdown("#### Synthetic Data Vault (SDV)")
        st.info("The core engine. We use SDV's `CTGANSynthesizer` (a Generative Adversarial Network) for generating high-fidelity synthetic data, which excels at capturing complex dependencies.")
        # Triggering image for CTGAN architecture
        st.markdown("")
    
    with col3:
        st.markdown("#### Pandas & Scipy")
        st.info("`Pandas` handles all data loading and statistical calculations, and `Scipy` provides the crucial Kolmogorov-Smirnov statistical test.")
        
    st.markdown("### Key Scientific Validation Methods")
    st.markdown("The application relies on advanced statistical tests to guarantee data quality:")
    st.markdown("- **Kolmogorov-Smirnov (KS) Test:** A powerful non-parametric test to determine if two samples (Real vs. Synthetic) are drawn from the same distribution. A high p-value (e.g., >0.05) is desirable.")
    # Triggering image for KS test
    st.markdown("")
    st.markdown("- **Kernel Density Estimation (KDE):** Provides visual validation by plotting the smoothed probability distribution of numeric data, allowing visual comparison of shapes and modes.")
    st.markdown("- **Correlation Heatmaps:** Ensures that the synthetic data accurately maintains the relationships between the original numeric variables.")


def synthesis_lab_page(df_original_placeholder):
    # ------------------------
    # 1. INITIAL UPLOAD & DATA READING
    # ------------------------
    st.header("Step 1: Upload and Prepare Original Dataset")
    uploaded_file = st.file_uploader("Upload CSV (Required)", type=["csv"])

    if uploaded_file is not None:
        @st.cache_data
        def process_upload(file):
            df = pd.read_csv(file)
            # Remove unsuitable columns (dates, high-cardinality IDs)
            df = identify_and_drop_unsuitable_cols(df)
            
            # Identify and drop columns with over 20% missing values
            missing = missing_summary(df)
            drop_cols_20 = missing[missing['missing_percent']>20].index.tolist()
            if drop_cols_20:
                st.warning(f"Dropping columns with >20% missing: {drop_cols_20}")
                df = df.drop(columns=drop_cols_20)
            
            # Reset index after potentially dropping rows/cols
            df = df.reset_index(drop=True)
            return df, drop_cols_20

        df_original, drop_cols_20 = process_upload(uploaded_file)
        
        st.session_state['df_original'] = df_original
        st.session_state['data_uploaded'] = True

        st.subheader("Data Preview & Summary (Metrics for real data)")
        
        # Calculate overall missing percentage
        total_cells = df_original.size
        total_missing = df_original.isnull().sum().sum()
        missing_percent_overall = (total_missing / total_cells) * 100 if total_cells > 0 else 0

        # Summary cards
        cols = st.columns(4)
        cols[0].markdown(f"<div class='card'><h3>{df_original.shape[0]}</h3><p>Rows</p></div>", unsafe_allow_html=True)
        cols[1].markdown(f"<div class='card'><h3>{df_original.shape[1]}</h3><p>Total Variables (Columns)</p></div>", unsafe_allow_html=True)
        cols[2].markdown(f"<div class='card'><h3>{len(drop_cols_20)}</h3><p>High-Missing Cols Dropped</p></div>", unsafe_allow_html=True)
        cols[3].markdown(f"<div class='card'><h3>{missing_percent_overall:.2f}%</h3><p>Total Missing Data</p></div>", unsafe_allow_html=True)
        
        st.dataframe(df_original.head())
        st.write("---")

    # Conditional flow starts here
    if st.session_state.get('data_uploaded'):
        
        df_original = st.session_state['df_original']
        
        # ------------------------
        # 2. CONFIGURATION SECTION
        # ------------------------
        st.header("Step 2: Configure Generator")
        
        col_model, col_rows = st.columns([1, 1])

        with col_model:
            model = st.radio("Select Synthesis Model (Type 3: Generation)", ["CTGAN", "Gaussian Copula"], help="CTGAN=Deep Learning (best for high fidelity), Gaussian=Statistical (fast, good baseline)")

        
        col_epochs, col_batch = st.columns(2)

        # CTGAN Parameters (Updated to use correct parameters and proxies)
        if model=="CTGAN":
            st.subheader("CTGAN Training Parameters")
            col_epochs, col_batch, col_lr = st.columns(3)
            with col_epochs:
                epochs = st.number_input("Epochs", 50, 1000, 200, help="Number of training iterations. More epochs = better learning, longer time.")
            with col_batch:
                batch = st.number_input("Batch Size", 20, 1020, 120, step=10, help="Samples per gradient update. Must be a multiple of 10.")
            with col_lr:
                # Mapping 'Generator Decay' concept to Learning Rate
                lr = st.number_input("Learning Rate (Generator Decay Proxy)", 1e-5, 1e-3, 2e-4, format="%e", help="Controls the speed of learning. A smaller value often leads to smoother training (similar to a slow decay).", key='lr_input')
            
            # The 'Accuracy Threshold' input
            quality_threshold = st.slider("Target Quality Threshold (0.0 to 1.0)", 0.0, 1.0, 0.85, 0.05, help="This is the minimum desired Quality Score after generation. Used for post-generation recommendation.", key='quality_threshold_slider')
            
        else:
            with col_epochs:
                st.info("Gaussian Copula is parameter-free (no training settings needed).")
            # Default values for non-CTGAN
            lr = 2e-4
            quality_threshold = 0.85
            with col_batch:
                pass # empty column placeholder

        with col_rows:
            rows = st.number_input("Rows to Generate (Required)", 10, 50000, df_original.shape[0] * 2, help=f"Default: 2x Original Rows ({df_original.shape[0]}). Specify the size of your synthetic dataset.")
        
        st.write("---")

        # ------------------------
        # 3. GENERATION SECTION
        # ------------------------
        st.header("Step 3: Generate Synthetic Data")

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df_original)

        with st.expander("Show Inferred Data Types (Metadata)"):
            st.write("This shows how the model categorized your data for synthesis.")
            st.json(metadata.to_dict()['columns'])


        if st.button(f"Generate Data using {model}"):
            
            if model == "CTGAN" and df_original.shape[0] < 100:
                st.error(f"Dataset has only {df_original.shape[0]} rows. CTGAN training is unstable with less than 100 rows. Please use Gaussian Copula or a larger dataset.")
                st.stop()
                 
            # Apply CTGAN/Gaussian Copula for synthesizing data
            if model=="CTGAN":
                # Use model_kwargs to pass parameters that tune the underlying PyTorch model, 
                # like the learning rate (proxy for decay).
                model_kwargs = {
                    'generator_lr': lr,
                    'discriminator_lr': lr,
                    # Setting a fixed dim for stability, avoiding another user input for now
                    'generator_dim': (256, 256), 
                    'discriminator_dim': (256, 256)
                }
                synth_model = CTGANSynthesizer(metadata, epochs=epochs, batch_size=int(batch), **model_kwargs)
            else:
                synth_model = GaussianCopulaSynthesizer(metadata) 
                
            with st.spinner(f"Training {model} model on {df_original.shape[0]} rows..."):
                synth_model.fit(df_original)
                synthetic = synth_model.sample(rows)
            
            # --- AUTOMATIC POSTPROCESSING (Step 4) ---
            # Automatically apply text truncation to object columns (Fixes SDV text generation artifacts)
            for col in synthetic.select_dtypes("object").columns:
                # Ensure it's treated as string before slicing
                synthetic[col] = synthetic[col].astype(str).str[:72] 
            st.info("Step 4: Automatic Postprocessing Applied! Text columns (e.g., categories) were automatically truncated to 72 characters to clean up generated noise and inconsistencies.")
            # --- END AUTOMATIC POSTPROCESSING ---

            st.session_state.synthetic = synthetic
            
            # Post-generation message based on the conceptual Quality Threshold
            st.success("Synthetic data generation and postprocessing complete!")
            if model == "CTGAN":
                st.subheader("Target Quality Status")
                st.warning(f"""
                The model has been trained. Remember that achieving a high structural quality (Quality Score > {quality_threshold:.2f}) requires careful tuning of the Learning Rate and Epochs, especially for complex datasets.
                Proceed to Step 4 for visual validation.
                """)

            st.subheader("Generated Data Preview")
            st.dataframe(synthetic.head())
            
            st.session_state.synthetic_generated = True
            st.write("---")

    # Conditional rendering for steps 4 and 5 (Validation and Download)
    if "synthetic_generated" in st.session_state and st.session_state.synthetic_generated:
        
        synthetic = st.session_state.synthetic.copy()
        df_original = st.session_state['df_original']

        # ------------------------
        # 4. VALIDATION / GRAPHICAL REPRESENTATION SECTION
        # ------------------------
        st.header("Step 4: Validation and Quality Check") 
        
        st.subheader("Statistical and Distribution Validation (Steps 5-7)") 

        # 4.1 Missing Data Comparison (Step 5)
        st.markdown("#### 4.1 Missing Value Structure Comparison (Step 5)")
        st.write("Checks if the synthetic data maintains the missing data structure of the real data.")
        # FIX: Changed 'Real_percent' to the correct column name 'missing_percent' in the rename operation
        real_missing = missing_summary(df_original).rename(columns={'missing_count': 'Real_Count', 'missing_percent': 'Real_Percent'})
        synth_missing = missing_summary(synthetic).rename(columns={'missing_count': 'Synth_Count', 'missing_percent': 'Synth_Percent'})
        
        missing_comp = pd.merge(real_missing[['Real_Count', 'Real_Percent', 'dtype']], 
                                synth_missing[['Synth_Count', 'Synth_Percent']], 
                                left_index=True, right_index=True, how='outer').fillna(0)
        st.dataframe(missing_comp.style.format({'Real_Percent': "{:.2f}%", 'Synth_Percent': "{:.2f}%"}))
        
        # 4.2 Mean and Standard Deviation Comparison (Step 5)
        st.markdown("#### 4.2 Mean and Standard Deviation Comparison (Step 5)")
        st.write("Compares central tendency (Mean) and dispersion (Std Dev) for numeric columns.")
        numeric_cols = df_original.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            real_stats = df_original[numeric_cols].agg(['mean', 'std']).T.rename(columns={'mean': 'Real_Mean', 'std': 'Real_Std'})
            synth_stats = synthetic[numeric_cols].agg(['mean', 'std']).T.rename(columns={'mean': 'Synth_Mean', 'std': 'Synth_Std'})
            stats_comp = pd.merge(real_stats, synth_stats, left_index=True, right_index=True, how='inner')
            st.dataframe(stats_comp.style.format("{:.3f}"))
        else:
            st.info("No numeric columns available for Mean/Std Dev comparison.")
        
        # 4.3 Correlation Heatmap Comparison (Step 5)
        st.markdown("#### 4.3 Correlation Heatmap Comparison (Step 5)")
        st.write("Visualizes how well the synthetic data maintains relationships between variables.")
        plot_correlation_heatmaps(df_original, synthetic, numeric_cols)
        
        # 4.4 Numeric Distributions (KDE & KS-test) (Step 7)
        st.markdown("#### 4.4 Numeric Distributions (KDE & KS-test) (Step 7)")
        st.write("KDE plots compare the shape of the distributions. The KS-test checks the statistical similarity (p-value > 0.05 is good).")
        plot_numeric_kde_ks(df_original, synthetic, numeric_cols)
        
        # 4.5 Categorical Distributions (Histograms) (Step 6)
        cat_cols = df_original.select_dtypes(exclude=np.number).columns.tolist()
        st.markdown("#### 4.5 Categorical Histograms (Step 6)")
        st.write("Compares the frequency of categories between the real and synthetic datasets.")
        plot_category_histograms(df_original, synthetic, cat_cols)
        
        st.write("---")
        
        # ------------------------
        # 5. FINAL DOWNLOAD SECTION (Step 8)
        # ------------------------
        st.header("Step 5: Download Synthetic Dataset (Step 8)") 
        st.markdown("Once validation is complete and satisfactory, download your final synthetic dataset.")
        st.markdown(make_download_link(synthetic,"final_synthetic_data.csv"), unsafe_allow_html=True)
        st.balloons()


# ------------------------------------------------------------------------------------------------
# MAIN APP ROUTING
# ------------------------------------------------------------------------------------------------

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = "home"

# Navigation Bar (using Markdown/HTML for styling)
nav_html = f"""
<div class="navbar">
    <div class="nav-item {'active-nav' if st.session_state.page == 'home' else ''}" onclick="window.parent.postMessage({{ 'streamlit': {{ 'command': 'SET_PAGE', 'page': 'home' }} }}, '*')">Home</div>
    <div class="nav-item {'active-nav' if st.session_state.page == 'lab' else ''}" onclick="window.parent.postMessage({{ 'streamlit': {{ 'command': 'SET_PAGE', 'page': 'lab' }} }}, '*')">Synthesis Lab</div>
    <div class="nav-item {'active-nav' if st.session_state.page == 'about' else ''}" onclick="window.parent.postMessage({{ 'streamlit': {{ 'command': 'SET_PAGE', 'page': 'about' }} }}, '*')">About</div>
</div>
"""
st.markdown(nav_html, unsafe_allow_html=True)

# Handle page changes from the navigation clicks
if st.session_state.page == "home":
    homepage()
elif st.session_state.page == "about":
    about_page()
elif st.session_state.page == "lab":
    # Pass a placeholder which will be ignored, but relies on st.session_state for actual data
    synthesis_lab_page(None) 


# This script handles the navigation state update from the custom HTML links
st.markdown("""
<script>
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.onclick = function() {
            const pageName = this.innerText.toLowerCase().replace(' ', '');
            if (pageName === 'synthesislab') {
                // This message triggers a page change (and subsequent st.rerun in the app logic)
                window.parent.postMessage({ type: 'streamlit:setPage', page: 'lab' }, '*');
            } else if (pageName === 'home') {
                 window.parent.postMessage({ type: 'streamlit:setPage', page: 'home' }, '*');
            } else if (pageName === 'about') {
                 window.parent.postMessage({ type: 'streamlit:setPage', page: 'about' }, '*');
            }
        };
    });
</script>
""", unsafe_allow_html=True)