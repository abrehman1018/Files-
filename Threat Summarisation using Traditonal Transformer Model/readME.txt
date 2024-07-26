# Threat Description Summarization with BART

This script summarizes threat descriptions from a dataset using the BART model for conditional generation. The summarized descriptions are saved into a new CSV file.

## File Description

model.py

This script processes the dataset, summarizes the threat descriptions, and saves the results.

- Dependencies: `pandas`, `transformers`, `tqdm`
- Steps:
  1. Load the dataset from `dataset.csv`.
  2. Print column names to identify the correct column containing threat descriptions.
  3. Load the BART model and tokenizer from a local directory (`./bart-large-cnn`).
  4. Define a function `summarize` to generate summaries for given text using the BART model.
  5. Identify and handle NaN values in the threat description column.
  6. Batch process the threat descriptions and generate summaries with a progress bar.
  7. Save the summarized descriptions to `result.csv`.

## Usage

1. Prepare the environment:
   - Ensure `dataset.csv` is in the same directory as the script.
   - Ensure the BART model files are located in `./bart-large-cnn`.

2. Install dependencies:

   pip install pandas transformers tqdm

3. Run the script:
   
   python model.py

## Parameters
max_length: Maximum length of the summary. Default is 130.
min_length: Minimum length of the summary. Default is 30.
length_penalty: Length penalty for the BART model. Default is 2.0.
num_beams: Number of beams for beam search. Default is 4.
column_name: Column name in the dataset containing threat descriptions. Default is 'threat_description'.

