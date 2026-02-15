import csv
import sys
import os
import json
import subprocess
import pandas as pd
import spacy
import spacy_transformers

from string import punctuation
from wordfreq import zipf_frequency

from config.paths_config import *
from config.models_list import SPACY_MODELS
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)


class DataProcessor:
    
    def __init__(self, 
                models_list : dict, 
                data_dir : str,
                ):
        
        self.spacy_models = models_list
        self.data_dir = data_dir
        self.nlp = None

        os.makedirs(self.data_dir, exist_ok = True)

        logger.info("Data Processor Initialized.")

    
    def create_language_dirs(self):
        try:

            for language in self.spacy_models.keys():
                try:
                    os.mkdir(f"{self.data_dir}/{language}")
                    logger.info(f"Directory {language} created.")
                except:
                    logger.info(f"Directory {language} already exists.")
        
        except Exception as e:
            logger.error(f"Error while creating language dirs - {e}")
            raise CustomException("Failed to craete language dirs : ", e)
        
    
    def load_and_clean_word_list(self, language : str) -> pd.DataFrame:   
        try: 

            with open(f"{RAW_WORD_LIST_DIR}/{language}/{language}.txt", "r", encoding = "utf-8") as f:
                word_list = f.read().split(",")

            word_df = pd.DataFrame({
                "word" : word_list
            })

            word_df['word'] = word_df["word"].str.strip(punctuation)

            return word_df 

        except Exception as e:
            logger.error(f"Error while loading and cleaning word list - {e}")
            raise CustomException("Failed to load and clean word list : ", e)    
        
    
    def add_lemma(self, df : pd.DataFrame,
                        batch_size : int = 1000) -> pd.DataFrame:
        try:

            docs = self.nlp.pipe(df["word"].to_list(), batch_size = batch_size)
            lemmas = [doc[0].lemma_ for doc in docs]
            df["lemma"] = pd.DataFrame(lemmas, index = df.index)

            return df

        except Exception as e:
            logger.error(f"Error while adding lemmatizer - {e}")
            raise CustomException("Failed to add lemmatizer : ", e)
        
    
    def add_word_frequencies(self, df : pd.DataFrame,
                         language : str)-> pd.DataFrame :
        try:
            
            language_group = self.spacy_models[language].split("_")[0]
            df["zipf_freq_lemma"] = [zipf_frequency(w, language_group) for w in df["lemma"]]

            return df
        
        except Exception as e:
            logger.error(f"Error while adding word frequencies - {e}")
            raise CustomException("Failed to add word frequencies : ", e)
        
    
    def clean_up_and_export(self, df : pd.DataFrame, language : str) -> None:
        try:

            df = (
                df.loc[df.groupby("lemma", sort = False)["zipf_freq_lemma"].idxmax()]
                .reset_index(drop = True)
            )

            df = df[(df["zipf_freq_lemma"] > 0)]

            df.loc[:, "word_difficulty"] = pd.cut(
                df["zipf_freq_lemma"],
                bins = [-float("inf"), 2.0, 4.0, float("inf")],
                labels = ["advanced", "intermediate", "beginner"],
                include_lowest = True,
                right = True
            )

            df = df.drop(columns = ["word", "zipf_freq_lemma"])
            df = df.rename(columns = {
                "lemma" : "word"
            })

            df.to_json(f"{RAW_WORD_LIST_DIR}/{language}/word-list-cleaned.json", orient = "index")
    
        except Exception as e:
            logger.error(f"Error while cleaning up and exporting data - {e}")
            raise CustomException("Failed to clean and export data : ", e)
    

    def create_clean_word_list(self, language : str) -> None:
        try:

            self.nlp = spacy.load(self.spacy_models[language], disable = ["parser", "ner", "textcat"])

            logger.info("Load in dataset")
            lang_df = self.load_and_clean_word_list(language)

            logger.info("Lemmatise Words")
            lang_df = self.add_lemma(lang_df, self.nlp)

            logger.info("Add the word frequencies")
            lang_df = self.add_word_frequencies(lang_df, language)

            logger.info("Do the final clean ups and export the file")
            self.clean_up_and_export(lang_df, language)

            return None
    
        except Exception as e:
            logger.error(f"Error while creating clean word list - {e}")
            raise CustomException("Failed to clean word list : ", e)
    

    def process_all_languages(self):
        try:

            for language in self.spacy_models.keys():
                logger.info(f"Processing language: {language}")
                self.create_clean_word_list(language)
        
        except Exception as e:
            logger.error(f"Error while processing all languages - {e}")
            raise CustomException("Failed to process languages : ", e)
        

