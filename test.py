import streamlit as st
from transformers import pipeline, AutoModelForTokenClassification, BertTokenizerFast

model_path = "./save/ner_model"
tokenizer_path = "./save/tokenizer"
model_fine_tuned = AutoModelForTokenClassification.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

nlp = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer)

def merge_continuous_words(ner_results):
    merged_results = []
    current_word = ""
    current_entity = None
    for token in ner_results:
        if token['word'].startswith('##'):
            current_word += token['word'][2:]
        else:
            if current_word:
                merged_results.append({'word': current_word, 'entity': current_entity})
            current_word = token['word']
            current_entity = token['entity']
    if current_word:
        merged_results.append({'word': current_word, 'entity': current_entity})
    return merged_results


entity_colors = {
    'B-ORG': 'lightblue',
    'I-ORG': 'lightblue',
    'B-PER': 'lightgreen',
    'I-PER': 'lightgreen',
    'B-LOC': 'lightcoral',
    'I-LOC': 'lightcoral',
    'B-MISC': 'lightgoldenrodyellow',
    'I-MISC': 'lightgoldenrodyellow'
}

readable_entity_names = {
    'B-ORG': 'Organization',
    'I-ORG': 'Organization',
    'B-PER': 'Person',
    'I-PER': 'Person',
    'B-LOC': 'Location',
    'I-LOC': 'Location',
    'B-MISC': 'Miscellaneous',
    'I-MISC': 'Miscellaneous'
}


st.title("Named Entity Recognition (NER)")

input_text = st.text_area("Enter a sentence:", "")

if st.button("Analyze"):
    if input_text:
        ner_results = nlp(input_text)
        merged_results = merge_continuous_words(ner_results)

        st.write("### Recognized Entities")
        st.write("The following entities were recognized in the input text:")

        for item in merged_results:
            word = item['word']
            entity = item['entity']
            color = entity_colors.get(entity, 'white')
            st.markdown(f"<span style='background-color: {color}; padding: 2px 4px; border-radius: 3px;'>{word} ({readable_entity_names[entity]})</span>", unsafe_allow_html=True)

        st.write("### Full Text with Entities Highlighted")
        formatted_text = input_text
        for item in merged_results:
            word = item['word']
            entity = item['entity']
            color = entity_colors.get(entity, 'white')
            formatted_text = formatted_text.replace(word, f"<span style='background-color: {color}; padding: 2px 4px; border-radius: 3px;'>{word}</span>")

        st.markdown(formatted_text, unsafe_allow_html=True)
