"""Dataset class implemented as a package that can be imported"""
from datasets import load_dataset


class Process_Dataset:
    def __init__(self, dataset_name, num_samples=None):
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.input_texts, self.target_texts = self.preprocess_data()

    def preprocess_data(self):
        if self.dataset_name == "cnn_dailymail":
            dataset = load_dataset("cnn_dailymail", "3.0.0")
            if self.num_samples is None:
                self.num_samples = len(dataset["test"]["article"])

            input_texts = dataset["test"]["article"][: self.num_samples]
            target_texts = dataset["test"]["highlights"][: self.num_samples]

        elif self.dataset_name == "bookcorpus":
            dataset = load_dataset("bookcorpus", split="train")
            if self.num_samples is None:
                self.num_samples = len(dataset)

            input_texts = dataset["text"][: self.num_samples]
            # For BookCorpus, input and target are the same
            target_texts = input_texts
        elif self.dataset_name == "xsum":
            dataset = load_dataset("xsum", split="test")
            if self.num_samples is None:
                self.num_samples = len(dataset["document"])
            input_texts = dataset["document"][: self.num_samples]
            target_texts = dataset["summary"][: self.num_samples]
        else:
            raise ValueError("Unsupported dataset type.")
        print("Data Preprocessing Complete")
        return input_texts, target_texts
