Synthetic Data Studio

This repository contains the Streamlit application for generating, validating, and comparing synthetic datasets using the Synthetic Data Vault (SDV) library, specifically leveraging the CTGAN and Gaussian Copula models.

🚀 Features

Data Upload: Securely upload and preview original CSV datasets.

Model Selection: Choose between the high-fidelity CTGAN (Deep Learning) or the statistical Gaussian Copula synthesizers.

Statistical Validation: Comparison of mean, standard deviation, and missing data structures between real and synthetic data.

Visual Validation: Use Kernel Density Estimation (KDE) plots and Correlation Heatmaps to assess data quality.

Statistical Quality Check: Utilizes the Kolmogorov-Smirnov (KS) Test for numeric distribution similarity.

Download: Secure download of the final synthetic dataset.

🛠️ Requirements

To run this application locally, you need Python and the following libraries:

pip install streamlit pandas numpy matplotlib seaborn sdv scipy


🏃 Running the Application

Save the app.py file to your local machine.

Navigate to the directory in your terminal.

Run the following command:

streamlit run app.py


The application will automatically open in your web browser.
