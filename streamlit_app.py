import streamlit as st
from transformers import pipeline, AutoModelForTokenClassification, BertTokenizerFast

labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_name):
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    return model, tokenizer

model_name = "Mehmood-Deshmukh/BERT-Finetuned-NER"
model, tokenizer = load_model_and_tokenizer(model_name)

id_to_label = {i: label for i, label in enumerate(labels)}
label_to_id = {label: i for i, label in enumerate(labels)}

model.config.label2id = label_to_id
model.config.id2label = id_to_label

nlp = pipeline("ner", model=model, tokenizer=tokenizer)

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

def get_entity_spans(ner_results):
    entities = []
    current_entity = None
    current_word = ""
    start_idx = 0
    
    for token in ner_results:
        if token['word'].startswith("##"):
            current_word += token['word'][2:]
        else:
            if current_word:
                entities.append((current_word, current_entity, start_idx))
            start_idx = token['start']
            current_word = token['word']
            current_entity = token['entity']
    
    if current_word:
        entities.append((current_word, current_entity, start_idx))
    
    return entities

if st.button("Analyze"):
    if input_text:
        try:
            ner_results = nlp(input_text)
            entity_spans = get_entity_spans(ner_results)
            merged_results = [{'word': word, 'entity': entity} for word, entity, _ in entity_spans]

            st.write("### Recognized Entities")
            st.write("The following entities were recognized in the input text:")

            for item in merged_results:
                word = item['word']
                entity = item['entity']
                color = entity_colors.get(entity, 'white')
                st.markdown(
                    f"<span style='background-color: {color}; padding: 2px 4px; border-radius: 3px;'>{word} ({readable_entity_names[entity]})</span>", 
                    unsafe_allow_html=True
                )

            highlighted_text = []
            last_idx = 0

            for word, entity, start_idx in entity_spans:
                end_idx = start_idx + len(word)
                color = entity_colors.get(entity, 'white')
                
                highlighted_text.append(input_text[last_idx:start_idx])
                
                highlighted_text.append(
                    f"<span style='background-color: {color}; padding: 2px 4px; border-radius: 3px;'>{word}</span>"
                )
                
                last_idx = end_idx

            highlighted_text.append(input_text[last_idx:])
            
            st.write("### Full Text with Entities Highlighted")
            formatted_text = "".join(highlighted_text)
            st.markdown(formatted_text, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
