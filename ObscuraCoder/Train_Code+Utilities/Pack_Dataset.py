from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset

class ConstantLengthDataset(IterableDataset):
    def __init__(
        self, tokenizer, dataset, infinite=False, seq_length=1024, num_of_sequences=1024, chars_per_token=3.6
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.bos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.input_characters = seq_length * chars_per_token * num_of_sequences
        self.epoch = 0
        self.infinite = infinite

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.input_characters:
                    break
                try:
                    buffer.append(next(iterator)["content"])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        self.epoch += 1
                        print(f"Dataset epoch: {self.epoch}")
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    yield {"input_ids": input_ids}
    
    def __call__(self):
        return self.__iter__()

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


tok = AutoTokenizer.from_pretrained("obf-tokenizer")
ds = load_dataset("continued_pretrain", streaming=True, split="train")
constant_length_dataset = ConstantLengthDataset(
	 tokenizer=tok, 
	 dataset=ds, 
	 seq_length=2048, 
	 chars_per_token=3.4
)
hf_dataset = Dataset.from_generator(constant_length_dataset)
