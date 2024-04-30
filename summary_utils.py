import pathlib
import nltk
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Download NLTK tokenizer data
nltk.download('punkt')


class T5Summ:
    """
    Summarization Class using T5 Models
    """
    def __init__(self, model_name: str, model_path: str ='', cuda=True) -> None:
        self.device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        if model_path:
            model_path = pathlib.Path(model_path)
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))

    def summarize(self, input_text: str) -> str:
        """
        Summarize the input text.
        """
        # Tokenize the input text
        inputs = self.tokenizer.encode("summarize: " + input_text, max_length=2048, truncation=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        # Generate the summary
        summary_ids = self.model.generate(inputs, num_beams=4, min_length=5, max_length=60, early_stopping=True)
        # Decode the generated summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print(summary)
        return summary

class LexRankSumm:
    """
    Summarization Class using LexRank Method.
    """
    def __init__(self) -> None:
        self.summarizer = LexRankSummarizer()

    def summarize(self, input_text: str, num_sentences: int = 2) -> str:
        """
        Summarize the input text.
        """
        parser = PlaintextParser.from_string(input_text, Tokenizer("english"))
        summary = self.summarizer(parser.document, num_sentences)
        plot = " ".join(str(sentence) for sentence in summary)
        print(plot)
        return plot