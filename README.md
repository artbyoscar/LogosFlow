# Logos Flow: Concept-Driven Reinforcement Learning Agent

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

Logos Flow is a research project exploring **Concept-Driven Reinforcement Learning Agents (CDRLAs)** for language modeling. Unlike traditional LLMs that operate on tokens, Logos Flow reasons and generates text by manipulating **multilingual and multimodal sentence embeddings**, representing underlying concepts. This novel approach aims to improve abstract reasoning, long-range coherence, and efficiency, especially on resource-constrained devices.

**Key Features:**

*   **Concept-Based Reasoning:** Models the underlying reasoning process, not just surface-level language patterns.
*   **Multilingual & Multimodal:** Leverages a shared semantic space, currently based on a modified version of [SONAR](https://github.com/facebookresearch/SONAR) sentence embeddings fine-tuned on the `rotten_tomatoes` dataset, to better capture the nuances of sentiment and improve the performance of our models in this specific domain.
*   **Improved Coherence:** Facilitates the generation of longer, more coherent text through concept-level planning.
*   **Resource Efficiency:** Designed for experimentation and development on machines with limited computational resources (e.g., a personal laptop).
*   **Data Augmentation:** Employs techniques like synonym replacement (using `nlpaug`) to enhance the training dataset.

**Note:** This project is under active development. Expect changes and improvements as we continue our research.

## Project Structure

LogosFlow/
├── backend/                  # Backend code (API, model serving)
│   ├── .venv/              # Virtual environment for backend dependencies
│   ├── app.py              # Main backend application file (e.g., Flask/FastAPI)
│   ├── api/                # API endpoints
│   │   ├── init.py
│   │   └── routes.py
│   ├── models/             # Your core LCM model
│   │   ├── init.py
│   │   ├── model.py        # Core LCM model definition (with Decoder)
│   │   ├── encoder.py      # Sentence encoder module
│   │   ├── decoder.py      # Sentence decoder module
│   │   ├── train.py        # Training script
│   │   └── generate.py     # Text generation script (with beam search fix)
│   ├── utils/
│   │   ├── init.py
│   │   ├── preprocess.py   # Data preprocessing and augmentation script
│   │   └── corpus_manager.py # Manages the corpus data
│   ├── tests/
│   │   ├── init.py
│   │   ├── test_encoder_decoder.py # Tests for encoding/decoding
│   │   ├── test_load.py   # Tests for data loading
│   │   └── test_nltk.py   # Script to test NLTK data and setup
│   ├── requirements.txt    # Backend dependencies
│   └── ...
├── frontend/               # Frontend code (HTML, CSS, JavaScript)
│   ├── public/
│   │   ├── index.html
│   │   └── ...
│   ├── src/                # React, Vue, or other frontend framework
│   │   ├── components/
│   │   ├── App.js
│   │   └── ...
│   ├── package.json        # Frontend dependencies
│   └── ...
├── data/                   # Data directory
│   ├── corpus_data.json    # JSON file containing the base corpus for the model
│   └── embeddings.npy      # Processed embeddings
├── .gitignore              # Files and folders to be ignored by Git
├── README.md               # Project description (this file)
└── download_nltk.py        # Script to download required NLTK resources


## Getting Started

**Prerequisites:**

*   Python 3.9+
*   A machine with limited resources (e.g., a laptop with CPU only is sufficient to start).
*   **NLTK Data:** The `averaged_perceptron_tagger`, `wordnet`, `omw-1.4`, and `punkt` packages are required. (See **NLTK Data Setup** below).

**Installation:**

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/artbyoscar/LogosFlow.git](https://github.com/artbyoscar/LogosFlow.git)
    cd LogosFlow
    ```

2.  **Create and Activate a Virtual Environment:**

    ```bash
    python3 -m venv .venv_new  # Recommended to use .venv_new
    source .venv_new/bin/activate  # On Linux/macOS
    .venv_new\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r backend/requirements.txt
    ```

**NLTK Data Setup:**

*   Run the provided script to download the necessary NLTK data:

    ```bash
    python download_nltk.py
    ```

*   If you encounter issues, you can download them manually within a Python interpreter:

    ```python
    import nltk
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    ```

*   **Important:** You might need to set the `NLTK_DATA` environment variable. See the **Troubleshooting** section for details.

**Data Preparation:**

1.  **Choose a Dataset:**

    *   For initial experiments, we are using the `rotten_tomatoes` dataset. You can adjust the `dataset_name` in `preprocess.py` to use other datasets.

2.  **Preprocess the Data:**

    *   Run the `preprocess.py` script to segment the dataset into sentences, perform data augmentation (synonym replacement), encode them into embeddings, and save them to the `data/` directory:

        ```bash
        python backend/utils/preprocess.py
        ```

**Training:**

1.  **Train the Model:**

    ```bash
    python -m backend.models.train
    ```

    *   This script trains the LCM model using the preprocessed embeddings.
    *   Adjust hyperparameters (e.g., `batch_size`, `learning_rate`, `num_epochs`, `hidden_size`, `num_layers`) in `train.py`.
    *   Model checkpoints will be saved to the `backend/models/models/` directory.

**Generation:**

1.  **Generate Text:**

    ```bash
    python backend/models/generate.py
    ```

    *   This script loads a trained model and generates text based on a given prompt.
    *   Modify the `start_text` and other generation parameters (e.g., `beam_width`, `length`, `repetition_penalty`) in `generate.py`.
    * **`generate.py` now includes the latest fixes for `IndexError` and `RuntimeError` in the `beam_search` function.**

**Testing:**

1.  **Run Unit Tests:**

    ```bash
    python -m unittest discover backend/tests
    ```

2.  **Test NLTK Setup:**
    ```bash
    python backend/tests/test_nltk.py
    ```

## Current Status

*   Successfully set up the development environment and project structure.
*   Using the `rotten_tomatoes` dataset for initial training and evaluation.
*   Implemented encoding and decoding using a modified Sentence-BERT model (384-dimensional embeddings).
*   Data preprocessing pipeline (`preprocess.py`) segments text, performs data augmentation (synonym replacement), encodes sentences, and saves embeddings.
*   Implemented a `SimplePolicyNetwork` (`model.py`) based on a Transformer architecture, including a `Decoder` for improved text generation.
*   Training script (`train.py`) with early stopping, learning rate scheduling, and CPU/memory monitoring.
*   **`generate.py` now includes a fix for the `IndexError` and `RuntimeError` in `beam_search` and other improvements.**
*   Achieved a validation loss of 0.0029 on the `rotten_tomatoes` dataset (current settings).
*   **Next:** We will focus on profiling, hyperparameter tuning, exploring model architectures.

## Experimentation and Development

Here are some suggestions for experiments and further development:

*   **Model Architecture (`model.py`):**
    *   Experiment with different `hidden_size` and `num_layers`.
    *   Try replacing the Transformer encoder with an LSTM or GRU.
    *   Explore adding attention mechanisms.

*   **Training Parameters (`train.py`):**
    *   Tune `initial_learning_rate`, `batch_size`, `ReduceLROnPlateau` parameters.
    *   Fine-tune `num_epochs` and early stopping `patience`.

*   **Generation Parameters (`generate.py`):**
    *   Experiment with different `beam_width`, `length`, and `repetition_penalty` values in the `beam_search` function.

*   **Data Augmentation (`preprocess.py`):**
    *   Experiment with different augmentation techniques and parameters in `nlpaug`.
    *   Consider filtering or post-processing augmented data.

*   **Reward Function (`reward.py` - *Future Development*):**
    *   Implement different reward functions based on perplexity, semantic similarity, or novelty/diversity.

*   **Quantization (*Future Development*):**
    *   Explore quantization to reduce model size and memory usage.

*   **Hierarchical Structure (*Future Development*):**
    *   Experiment with hierarchical planning using topic embeddings or other higher-level representations.

## Troubleshooting

**NLTK `LookupError`:**

*   If you encounter a `LookupError` related to `averaged_perceptron_tagger`, `punkt`, or `wordnet`:
    1.  **Verify NLTK Data:** Ensure the necessary data packages are downloaded in the correct directory (usually `C:\Users\<YourUsername>\nltk_data` on Windows). Check the contents of the `taggers` and `tokenizers` folders within the `nltk_data` directory.
    2.  **Environment Variable:** Set the `NLTK_DATA` environment variable to point to your NLTK data directory:
        *   **Windows:** Search for "environment variables" in the Start menu, click "Edit the system environment variables," then "Environment Variables...". Add a new variable named `NLTK_DATA` with the value `C:\Users\<YourUsername>\nltk_data` (or your custom path).
    3.  **Redownload:** If needed, redownload the NLTK data packages within a Python interpreter, specifying the download directory:

        ```python
        import nltk
        nltk.download('averaged_perceptron_tagger', download_dir='C:\\Users\\<YourUsername>/nltk_data')
        nltk.download('wordnet', download_dir='C:\\Users\\<YourUsername>/nltk_data')
        nltk.download('omw-1.4', download_dir='C:\\Users\\<YourUsername>/nltk_data')
        nltk.download('punkt', download_dir='C:\\Users\\<YourUsername>/nltk_data')
        ```

    4.  **Use `test_nltk.py`:** Run the provided `test_nltk.py` script to isolate and debug NLTK issues.
    5.  **Restart:** After making changes, restart your terminal or computer.

## Contributing

We welcome contributions! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch from `main`.
3.  Make your changes and commit them with clear messages (following [Conventional Commits](https://www.conventionalcommits.org/)).
4.  Write unit tests and ensure they pass.
5.  Submit a pull request targeting the `main` branch.

Please adhere to the following guidelines:

*   **Code Style:** Follow the [black](https://github.com/psf/black) code style.
*   **Testing:** Write comprehensive unit tests.
*   **Documentation:** Keep the documentation up-to-date.
*   **Issue Reporting:** Open an issue for bugs or feature requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. (You'll need to add a `LICENSE` file with the MIT License text).

## Acknowledgements

*   Inspired by "Large Concept Models: Language Modeling in a Sentence Representation Space" \[link to paper, if available] and "OpenMOSS - A Roadmap to o1 from Reinforcement Learning Perspective" \[link to paper, if available].

### Dependencies

*   [Hugging Face Transformers](https://huggingface.co/transformers/)
*   [Sentence Transformers](https://www.sbert.net/)
*   [Datasets](https://huggingface.co/docs/datasets/)
*   [nlpaug](https://github.com/makcedward/nlpaug)
*   [NLTK](https://www.nltk.org/)

## Contact

For questions or feedback, please open an issue on the GitHub repository.