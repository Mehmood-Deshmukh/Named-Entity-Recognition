# Named Entity Recognition with Fine-Tuned BERT

This repository hosts a Named Entity Recognition (NER) application powered by a fine-tuned BERT model. The model has been trained on the CoNLL-2003 dataset, which is a widely used dataset for NER tasks, containing annotations for entities like persons, locations, organizations.

## Live Demo

Explore the NER application live at: [NER BERT Fine-Tuned App](https://named-entity-recognition-bert-finetuned.streamlit.app/)

## Overview
* This repository includes a complete pipeline for fine-tuning a BERT model for NER tasks.
* After training, the model is deployed using Streamlit for easy and interactive NER inference.
* **Dataset**: The model is trained on the CoNLL-2003 dataset, which provides labeled data for training and evaluation.

## How It Works

1. **Model Initialization**: We use a pre-trained `bert-base-cased` model from Hugging Face's `transformers` library.
2. **Data Processing**: The input text is tokenized, and labels are aligned with the tokenized words.
3. **Fine-Tuning**: The model is fine-tuned using the CoNLL-2003 dataset, optimizing it to predict NER tags.
4. **Inference Pipeline**: A pipeline is set up to handle NER inference on new text inputs.
5. **Streamlit App**: The trained model is deployed in a Streamlit application for interactive use.

## Installation and Setup

To run the NER application locally:

1. **Clone the repository:**

   ```
   git clone https://github.com/Mehmood-Deshmukh/Named-Entity-Recognition.git
   cd ner-bert-finetuned
   ```
2. **Install the dependencies:**

   ```
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app:**
   ```
    streamlit run app.py
   ```
## Technologies Used

* **Python**
* **Transformers (Hugging Face)**
* **Streamlit**
* **Datasets (Hugging Face)**
* **PyTorch**


### Connect with Me

- LinkedIn: [Mehmood Deshmukh](https://www.linkedin.com/in/mehmood-deshmukh-93533a2a7/)
- GitHub: [Mehmood-Deshmukh](https://github.com/Mehmood-Deshmukh)

Feel free to reach out for collaboration, feedback, or just to say hi!
