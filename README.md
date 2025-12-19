# Time Series Forecasting with Transformers (PatchTST)

## Project Overview
This repository contains a solution for the Time Series Fine-Tuning assignment. The goal was to load a large-scale dataset (>500,000 observations), preprocess it for deep learning, and fine-tune a modern Transformer-based architecture to forecast future electricity demand.

## 1. Dataset Selection & Verification
* **Dataset:** Monash Electricity (Hourly)
* **Source:** Monash Time Series Forecasting Repository / GluonTS
* **Size Verification:**
    * The dataset consists of hourly electricity consumption records for hundreds of clients.
    * **Total Time Steps:** > 500,000 (Verified in the notebook during the loading stage).
    * **Preprocessing:** Data was windowed into "Context" (past 96 hours) and "Prediction" (future 24 hours) segments.

## 2. Model Architecture: PatchTST
We utilized the **PatchTST (Patch Time Series Transformer)** architecture, a state-of-the-art model designed specifically to address common weaknesses of Transformers in forecasting tasks.

### Key Architectural Features:
1.  **Patching:** Instead of feeding individual time steps into the Transformer (like words in a sentence), PatchTST groups time steps into sub-series or "patches" (e.g., grouping 12 hours into one token). This reduces computational complexity and preserves local semantic information.
2.  **Channel Independence:** The model treats each time series (or channel) independently. It shares the same model weights across all series but does not mix their values during the attention mechanism.

## 3. Discussion: Relation to the Article's Critique
*Reference Article Context: This discussion addresses the common critique (e.g., Zeng et al., "Are Transformers Effective for Time Series Forecasting?") that Transformers often fail to capture temporal order due to their permutation-invariant attention mechanisms.*

**The Critique:**
The assignment article likely highlighted that standard Transformers (and even simple Linear models) often outperform complex Transformers because standard attention mechanisms treat time points as isolated tokens, losing the crucial "ordering" information that defines time series data.

**How PatchTST Addresses This:**
Our chosen model, PatchTST, directly refutes the idea that Transformers are unsuitable for time series by introducing **Patching**:
* **Preserving Local Context:** By aggregating adjacent time steps into a single patch *before* the attention layer, the model forces the Transformer to look at local temporal patterns (e.g., a morning spike) rather than isolated points.
* **Long-Term Dependencies:** The Transformer attention mechanism is then applied *between patches*. This allows the model to learn long-term dependencies (e.g., repeating daily patterns) without losing the short-term trend information, effectively solving the "permutation invariance" problem mentioned in the critique.

**Comparison to LSTM/CNN:**
* **vs. LSTM:** LSTMs process data sequentially, which makes them slow on long sequences (like our 96-hour window) and prone to forgetting early information. PatchTST processes patches in parallel, allowing for much faster training and longer history windows.
* **vs. CNN:** While CNNs are good at local patterns, they struggle with global dependencies. PatchTST combines the best of both: patches capture local patterns, and self-attention captures global dependencies.

## 4. Training & Evaluation
* **Framework:** Hugging Face `transformers` + `GluonTS` + `PyTorch`.
* **Hyperparameters:**
    * Context Length: 96 hours
    * Prediction Length: 24 hours
    * Batch Size: 32
    * Optimizer: AdamW
* **Result:**
    * The model successfully converged and was able to predict the mean demand trajectory.
    * **Visual Analysis:** As seen in the generated plots, the Ground Truth (Actual) data is highly volatile with sharp hourly spikes. The PatchTST forecast produces a smoother curve that accurately tracks the central tendency and daily seasonality of the demand, minimizing Mean Squared Error.

## 5. How to Run
1.  Open the `.ipynb` file in Google Colab.
2.  Select `Runtime` > `Run all`.
3.  The notebook will automatically:
    * Install dependencies (`transformers`, `gluonts`, `datasets`).
    * Load and verify the Electricity dataset.
    * Train the PatchTST model.
    * Output a visualization of the forecast vs. actuals.
