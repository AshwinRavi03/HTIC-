NLP RNN Seq2Seq Transliteration Model
A robust PyTorch pipeline for character-level sequence-to-sequence transliteration from Romanized (Latin) script to Indic native scripts, designed for flexibility and easy training/evaluation across multiple Indian languages. Originally intended for large-scale datasets such as Aksharantar, this project is optimized for Colab and local execution.

Features
Sequence-to-sequence (seq2seq) architecture with GRU-based encoder/decoder and Bahdanau attention.​

Character vocabulary generation and smart data sampling for length diversity.​

Custom text cleaning for romanized and Indic text, ensuring compatibility with noisy real-world data.​

Training pipeline with label smoothing, learning rate warmup, AdamW optimizers, and early stopping.​

Multi-language support: Assamese, Bengali, Bodo, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili, Marathi, Manipuri, Oriya, Punjabi, Sanskrit, Sindhi, Tamil, Telugu, Urdu.​

Checkpointing for model persistence and inference on new words.​

Folder Structure
Organize your dataset as follows:

text
project_root/
│
├── aksharantarsampled/
│   └── [language3]/        # e.g., hindi3, tamil3, bengali3, etc.
│        └── train.csv      # Each file: two columns [source, target]
│
├── nlp_rnn_seq2seq_model.py
└── (other scripts, Colab notebooks, etc.)
Example: aksharantarsampled/hindi3/train.csv with romanized source and native target per line.​

Installation & Setup
Ensure Python 3.8+ and install dependencies:

bash
pip install torch torchvision torchaudio pandas numpy
If running in Colab, mount your Google Drive for dataset access:

python
from google.colab import drive
drive.mount('/content/drive')
Place your train.csv files as described above.

Usage
1. Training a Language Model
python
from nlp_rnn_seq2seq_model import trainlanguagemodel, loadlanguagedata

df = loadlanguagedata("hindi", "/content/drive/MyDrive/aksharantarsampled/hindi3", maxpairs=60000)
# Create vocabularies and train
srcvocab, tgtvocab = CharVocab('hindi_src'), CharVocab('hindi_tgt')
for _, row in df.iterrows():
    srcvocab.addString(row['source'])
    tgtvocab.addString(row['target'])

encoder, decoder = trainlanguagemodel(
    "hindi",
    [(row['source'], row['target']) for _, row in df.iterrows()],
    srcvocab,
    tgtvocab,
    epochs=30
)
2. Inference / Transliteration
python
from nlp_rnn_seq2seq_model import evaluatemodel

result = evaluatemodel(
    encoder,
    decoder,
    "namaste",     # any romanized word
    srcvocab,
    tgtvocab
)
print(f"Native script: {result}")
3. Evaluation
Accuracy on validation/test sets and random samples can be triggered using provided functions and metrics.

Label smoothing, gradient clipping, and NaN detection are provided for robust training.​

Advanced Configuration
Hyperparameters: Adjust HIDDENSIZE, EMBEDDINGDIM, DROPOUT, BATCHSIZE, etc., as needed either in the script or via arguments.​

Model saving/loading logic allows checkpoint re-use for deployment.

Tips for Data Preparation
CSV files must have columns: source (romanized input) and target (native script output).​

Extensive text cleaning (punctuation/digits/normalization) is done automatically.

Stratified sampling by length maintains training diversity for robust sequence learning.​

Citation & Credits
Code and dataset logic inspired by Aksharantar project conventions.

Bahdanau attention, label smoothing, and AdamW optimizations follow best practices in NLP modeling.​
