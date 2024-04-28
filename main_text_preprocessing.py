import argparse
import os
import re

import langdetect
from langdetect.lang_detect_exception import LangDetectException
import pandas as pd
import spacy

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from essentials.config import ABSTRACTS
from essentials.data_functions import read_data


# NLTK Resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
sentiment_bearing_stopwords = [
    'not',
    'no',
    'nor',
    'never',
    'yes',
    'should',
    'could',
    'would'
]
STOP_WORDS_WITHOUT_SENTIMENT = [word for word in stop_words
                                if word not in sentiment_bearing_stopwords]

# spaCY for NER
NLP = spacy.load('en_core_web_sm')


def read_synthetic_data(base_dir=os.getcwd(), model='llama-3-70b'):

    synthetic_data = []

    dir = os.path.join(base_dir, "synthetic_data", model)

    for folder in os.listdir(dir):
        for data in os.listdir(os.path.join(dir, folder)):
            if data.endswith(".jsonl"):
                df = pd.read_json(os.path.join(dir, folder, data), lines=True)
                synthetic_data.append(df)

    df_synthetic = pd.concat(synthetic_data)
    return df_synthetic


def clean_synth_text(text: str) -> str:
    if 'Web of Science' in text:
        return text.split('Web of Science ')[-1]
    if 'abstract' in text:
        return text.split('abstract ')[-1]
    else:
        return text


def remove_urls_and_html_tags(text: str) -> str:
    html_tags_pattern = r'<.*?>'
    text_without_html_tags = re.sub(html_tags_pattern, '', text)
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text_without_html_tags)


def named_entity_regocnition(text: str) -> list[str]:
    doc = NLP(text)
    return ["".join(ent.text) for ent in doc.ents]


def detect_language(text: str) -> str:
    try:
        return langdetect.detect(text)
    except LangDetectException:
        return 'error'


def preprocess_zofa_data(df_zofa: pd.DataFrame) -> pd.DataFrame:
    # Add is_abstract dummy to zofa
    df_zofa['is_abstract'] = 1
    # Rename columns
    df_zofa = df_zofa.rename(columns={'ABSTRACT': 'text', 'SDG': 'label'})
    # Clean OSDG data
    df_zofa['language'] = df_zofa.text.apply(detect_language)
    # Filter non-english texts out
    df_zofa = df_zofa[df_zofa.language == 'en'].copy()

    return df_zofa[['text', 'label', 'is_abstract']]


def preprocess_semantic_scholar_data(df_scholar: pd.DataFrame, null_class=True) -> pd.DataFrame:
    # Add is_abstract dummy to zofa
    df_scholar['is_abstract'] = 1
    # Rename columns
    if null_class:
        df_scholar = df_scholar.rename(columns={'abstract': 'text',
                                                'sdg_label': 'label'})
    else:
        df_scholar = df_scholar.rename(columns={'abstract': 'text',
                                                'sdg_verdict': 'label'})
    # Drop if no abstract
    df_scholar = df_scholar.dropna(subset='text')
    # Clean OSDG data
    df_scholar['language'] = df_scholar.text.apply(detect_language)
    # Filter non-english texts out
    df_scholar = df_scholar[df_scholar.language == 'en'].copy()

    return df_scholar[['text', 'label', 'is_abstract']]


def preprocess_osdg_data(df_osdg: pd.DataFrame, threshold=0.5) -> pd.DataFrame:
    # Filter out entries with low agreement score
    df_osdg[df_osdg.agreement >= threshold]
    # Clean OSDG data
    df_osdg['language'] = df_osdg.text.apply(detect_language)
    # Filter non-english texts out
    df_osdg = df_osdg[df_osdg.language == 'en'].copy()
    # Naive search for abstracts in the OSDG data
    wanted_words = ['abstract', 'this paper', 'this study', 'this article']
    df_osdg['is_abstract'] = [1 if any(word in text.lower() for word in wanted_words) else 0 for text in df_osdg.text]
    # Rename columns
    df_osdg = df_osdg.rename(columns={'sdg': 'label'})
    return df_osdg[['text', 'label', 'is_abstract']]


def preprocess_synth_data(df_synth: pd.DataFrame) -> pd.DataFrame:
    df_synth['is_abstract'] = 2
    df_synth['text'] = df_synth.text.apply(clean_synth_text)
    df_synth = df_synth.rename(columns={'sdg_id': 'label'})
    return df_synth[['text', 'label', 'is_abstract']]


def process_text(text: str) -> str:

    # Lowercasing
    text = text.lower()

    # Removal of urls and html tags
    text = remove_urls_and_html_tags(text)

    # Removal of Numeric values
    text = re.sub(r'\d+', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove punctuation and non-alphabetic characters
    tokens = [token for token in tokens if token.isalpha()]

    # Selective removal of stopwords
    tokens = [token for token in tokens if token not in STOP_WORDS_WITHOUT_SENTIMENT]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(tokens)


def save_data(df: pd.DataFrame, args: argparse.Namespace) -> None:
    null_included = '_with_null' if args.include_null else ''
    synth_included = '_with_synth' if args.include_synth else ''
    weak_included = '_with_weakly_labeled' if args.include_weakly_labeled_data else ''
    file_path = f"{args.output_dir}/cleaned_data{null_included}{weak_included}{synth_included}.csv"
    df.to_csv(file_path, index=False)
    print(f"---Saved data to {file_path}---")


def main(args):

    # Load zofa
    df_zofa = read_data(ABSTRACTS)

    # Load osdg data
    df_osdg = read_data('osdg_data/osdg-community-data-v2024-01-01.csv', format='csv', delimiter='\t')

    # preprocess data
    print("---Preprocessing zofa and osdg data---")
    df_zofa_prep = preprocess_zofa_data(df_zofa)
    df_osdg_prep = preprocess_osdg_data(df_osdg)

    # Combine OSDG and ZOFA
    df_prep = pd.concat([df_zofa_prep, df_osdg_prep])

    if args.include_null:
        print("---Adding Null data for garbage class---")
        # Load garbage data
        df_null = read_data('garbage_class/null_labels_clean.csv', format='csv')
        df_null_prep = preprocess_semantic_scholar_data(df_null)
        df_prep = pd.concat([df_prep, df_null_prep])
    else:  # Remove 0 since None class is not part of OSDG
        print("---Removing Null class from data---")
        df_prep = df_prep[df_prep.label != 0].copy()

    if args.include_weakly_labeled_data:
        print("---Adding weakly labeled data to underrepresented classes---")
        # Load garbage data
        df_weak = read_data('garbage_class/labeled_null_class.csv', format='csv')
        df_weak_prep = preprocess_semantic_scholar_data(df_weak, null_class=False)
        df_prep = pd.concat([df_prep, df_weak_prep])

    if args.include_synth:
        print("---Adding synthetic data---")
        # Load synth data
        df_synth = read_synthetic_data()
        df_synth_prep = preprocess_synth_data(df_synth)
        # Create ZOFA + OSDG + SYNTH DataFrame
        df_prep = pd.concat([df_prep, df_synth_prep])

    # Apply custom pre-processing
    print("---Applying text cleaning---")
    df_prep['text_clean'] = df_prep.text.apply(process_text)
    df_prep = df_prep[['text_clean', 'label', 'is_abstract']]

    save_data(df_prep, args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('')

    parser.add_argument(
        '--output_dir',
        default='.',
        type=str,
        help='output directory for the cleaned data'
    )
    parser.add_argument(
        '--include_null',
        default=False,
        type=bool,
        help='if True, NULL class is included in the data'
    )
    parser.add_argument(
        '--include_synth',
        default=False,
        type=bool,
        help='if True, synthetic data is added to the data'
    )
    parser.add_argument(
        '--include_weakly_labeled_data',
        default=False,
        type=bool,
        help='if True, weakly labeled abstracts are added to the data'
    )

    args = parser.parse_args()

    main(args)
