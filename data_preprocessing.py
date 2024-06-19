import pandas as pd
import emoji
import re

class TwitterDataProcessing():
    """ Class to preprocess the twitter data file for further topic modeling """

    def __init__(self, filepath) -> None:
        self.filepath = filepath
        self.df = None
        self.stopwords = None


    def read_csv_to_df(self):
        """Reads a csv file to a pandas Dataframe
        """

        df = pd.read_csv(self.filepath)
        self.df = df
    
    def remove_emojis(self, text):
        """Removes emojis from a string

        Args:
            text (str): text contaning emojis

        Returns:
            str: text without emojis
        """

        for char in text:
            if char in emoji.UNICODE_EMOJI['en'].keys():
                text = text.replace(char, '')
        return text

    def remove_additinal_emojis(self, text):
        """Remove additional emojis that haven't  been removed with the remove_emoji() function.

        Args:
            text (str): text contaning emojis

        Returns:
            str: text without emojis
        """

        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    
    def remove_urls(self, text):
        """Removes URLs in the given text.

        Args:
            text (str): text contaning URLs

        Returns:
            str: text without URLs
        """

        text = re.sub(r'http\S+', '', text)
        return text
    
    def remove_handles(self, text):
        """Removes twitter handles using regex patter: starting with @ and ending at a whitespace.

        Args:
            text (str): text contaning twitter handles

        Returns:
            str: text without twitter handles
        """

        pattern = r'(@\w+)(\s|$)'
        return re.sub(pattern, '', text)
    
    def remove_other_char(self, text):
        """Remove characters that have been seen in the data: 'RT', '_', '-', '&gt', '...'

        Args:
            text (str): text contaning mentioned chars

        Returns:
            str: text without mentioned chars
        """

        text_new = text.replace('RT', '')
        text_new = text_new.replace('-', '')
        text_new = text_new.replace('_', ' ')
        text_new = text_new.replace('...', '')
        text_new = text_new.replace('&gt', '')
        return text_new
    
    def remove_repeated_chars(self, text):
        """Removes repeated characters (if they appear 3 or more times). For example something like 'noooo' would be changed to 'no'

        Args:
            text (str): text contaning repeated chars

        Returns:
            str: text without repeated chars
        """

        reversed_text = text[::-1]
        pattern = r'(\w)\1{2,}'
        cleaned_text = re.sub(pattern, r'\1', reversed_text)
        final_text = cleaned_text[::-1]
        return final_text

    def remove_stopwords(self, text):
        """Removed stopwords from the text. The list of stopwords has been taken from: https://github.com/NNLP-IL/Stop-Words-Hebrew

         Args:
            text (str): text contaning stopwords

        Returns:
            str: text without stopwords
        """

        tokens = text.split()
        filtered_tokens = [token for token in tokens if token not in self.stopwords]
        return ' '.join(filtered_tokens)
    
    def clean_text(self):
        """Performing all steps to process the twitter data.
        """

        self.df['final'] = self.df['text'].apply(self.remove_handles)
        self.df['final'] = self.df['final'].apply(self.remove_urls)
        self.df['final'] = self.df['final'].apply(self.remove_other_char)
        self.df['final'] = self.df['final'].apply(self.remove_emojis)
        self.df['final'] = self.df['final'].apply(self.remove_additinal_emojis)
        self.df['final'] = self.df['final'].apply(self.remove_repeated_chars)
        self.df['final'] = self.df['final'].apply(self.remove_stopwords)

    def process(self):
        """Processes the raw csv data file to a cleaned pandas Dataframe

        Returns:
            pd.DataFrame: datarame containing the tweets per row
        """

        self.read_csv_to_df()
        with open('data/stopswords_list_extend.txt', 'r') as file:
            self.stopwords = file.read().splitlines()
        self.clean_text()
        df_cleaned = self.df[self.df['final'].astype(bool)]
        self.df = df_cleaned
        return self.df
