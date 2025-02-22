import copy
import logging
from dataclasses import (
    dataclass,
    field
)
from typing import (
    Dict,
    Optional,
    Sequence
)
from evaluate import load
from datasets import load_dataset
from peft import (
    LoraConfig,
    prepare_model_for_int8_training,
    get_peft_model
)
import numpy as np
import os
import torch
import transformers
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    BitsAndBytesConfig
)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<|pad|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
DEFAULT_BOS_TOKEN = "<|endoftext|>"
DEFAULT_UNK_TOKEN = "<unk>"

MODEL_TO_LORA_MODULES_MAP = {
        "OBF_255M": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        "DOBF_255M": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        "BASE_255M": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        "BASE_CP_OBF_255M": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        "OBF_491M": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        "DOBF_491M": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        "BASE_491M": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        "BASE_CP_OBF_491M": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        "OBF_1229M": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        "DOBF_1229M": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        "BASE_1229M": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        "BASE_CP_OBF_1229M": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        "OBF_2794M": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        "DOBF_2794M": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        "BASE_2794M": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        "BASE_CP_OBF_2794M": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        "deepseek-ai/deepseek-coder-1.3b-base": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        "bigcode/starcoder2-3b": ["q_proj", "k_proj", "v_proj", "o_proj", "c_proj", "c_fc"],
        "bigcode/starcoderbase-1b": ["c_attn", "c_proj", "q_attn", "c_fc"],
        "bigcode/starcoderbase-3b": ["c_attn", "c_proj", "q_attn", "c_fc"],
        "microsoft/phi-2": ["q_proj", "k_proj", "v_proj", "fc1", "fc2"],
        "stabilityai/stable-code-3b": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "google/codegemma-2b": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}


@dataclass
class LoggingArguments:
    project_name: str = field(
        default=None, metadata={"help": "The project name under which the experiment will be logged."}
    )
    wandb_token: str = field(
        default=None, metadata={"help": "API token for WandB hub."}
    )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="DEFAULT_MODEL")
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "The maximum sequence length to be processed during instruction tuning."
        }
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters"
                "when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    llm_int8_threshold: float = field(
        default=6.0, metadata={"help": "The thresholf for a parameter to be designated a quantization outlier."}
    )
    lora_alpha: int = field(
        default=16, metadata={"help": "The interpolation importance factor for the LoRA adapter."}
    )
    lora_r: int = field(
        default=8, metadata={"help": "The LoRA adapter rank."}
    )
    lora_dropout: float = field(
        default=0.1, metadata={"help": "Dropout value for LoRA layers."}
    )


@dataclass
class DataArguments:
    hf_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    language: str = field(default=None, metadata={"help": "Language config of the dataset."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    data_processing_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of workers to use when pre-processing and mapping the dataset."
            )
        }
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self,
            hf_data_path: str,
            config: str,
            tokenizer: transformers.PreTrainedTokenizer,
            token: str,
            split="train",
            limit=None,
            workers=None
        ):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        dataset = load_dataset(
            hf_data_path,
            config,
            split=split,
            token=token
        )
        if limit is not None:
            dataset = dataset.select(range(min(len(dataset), limit)))

        def _preprocess(example):
            """Preprocess the data by tokenizing."""
            source = f"### Diff:\n{example['diff']}\n\n### Message:\n"
            target = f"{example['message']}{tokenizer.eos_token}"
            reformed_example = source + target
            reformed_example_tokenized = tokenizer(
                    reformed_example,
                    return_tensors="pt",
                    padding="longest",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
            )
            source_tokenized = tokenizer(
                    source,
                    return_tensors="pt",
                    padding="longest",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
            )
            source_len = source_tokenized["input_ids"].ne(tokenizer.pad_token_id).sum().item()
            input_ids = reformed_example_tokenized["input_ids"][0]
            labels = copy.deepcopy(input_ids)
            labels[:source_len] = IGNORE_INDEX
            example["input_ids"] = input_ids
            example["labels"] = labels
            return example


        logging.warning("Formatting and Tokenizing inputs... This may take some time...")
        self.dataset = dataset.map(
            _preprocess,
            batched=False,
            num_proc=workers
        ).remove_columns(["diff", "message"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=torch.LongTensor(self.dataset[i]["input_ids"]),
            labels=torch.LongTensor(self.dataset[i]["labels"])
        )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
        model_args
    ) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        hf_data_path=data_args.hf_data_path,
        config=data_args.language,
        token=model_args.token,
        split="train",
        limit=data_args.max_train_samples,
        workers=data_args.data_processing_workers
    )
    val_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        hf_data_path=data_args.hf_data_path,
        config=data_args.language,
        token=model_args.token,
        split="validation",
        limit=data_args.max_eval_samples,
        workers=data_args.data_processing_workers
    )
    predict_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        hf_data_path=data_args.hf_data_path,
        config=data_args.language,
        token=model_args.token,
        split="test",
        limit=data_args.max_predict_samples,
        workers=data_args.data_processing_workers
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator), predict_dataset


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoggingArguments))
    model_args, data_args, training_args, log_args = parser.parse_args_into_dataclasses()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        truncation_side="right"
    )
    os.environ["WANDB_PROJECT"] = log_args.project_name
    os.environ["WANDB_API_KEY"] = log_args.wandb_token

    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=model_args.llm_int8_threshold
    )
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model_base = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        quantization_config=quant_config,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    tokenizer.add_special_tokens(special_tokens_dict=special_tokens_dict)
    embedding_size = model_base.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        input_embeddings = model_base.get_input_embeddings().weight.data
        output_embeddings = model_base.get_output_embeddings().weight.data
        model_base.resize_token_embeddings(len(tokenizer))
        input_embeddings_avg = input_embeddings[:embedding_size].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:embedding_size].mean(dim=0, keepdim=True)
        input_embeddings[embedding_size:] = input_embeddings_avg
        output_embeddings[embedding_size:] = output_embeddings_avg
    elif len(tokenizer) < embedding_size:
        model_base.resize_token_embeddings(len(tokenizer))

    model_base = prepare_model_for_int8_training(model_base)
    adapter_config = LoraConfig(
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        r=model_args.lora_r,
        inference_mode=False,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=MODEL_TO_LORA_MODULES_MAP[model_args.model_name_or_path]
    )
    model = get_peft_model(model_base, adapter_config)

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    rouge_metric = load("rouge")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        labels = labels[:, 1:]
        predictions = predictions[:, :-1]
        # Decode the predictions and labels
        predictions = np.where(labels != IGNORE_INDEX, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        # Replace IGNORE_INDEX in the labels as we did before
        labels = np.where(labels != IGNORE_INDEX, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {key: value * 100 for key, value in result.items()}
        return {
            f"rouge1": result["rouge1"],
            f"rouge2": result["rouge2"],
            f"rougeL": result["rougeL"],
            f"rougeLsum": result["rougeLsum"]
        }

    data_module, predict_dataset = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        args=training_args,
        **data_module
    )
    if training_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(output_dir=training_args.output_dir)

    if training_args.do_predict:
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        predictions = predict_results.predictions
        predictions = np.where(predictions != IGNORE_INDEX, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        predictions = [pred.strip() for pred in predictions]
        output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
        with open(output_prediction_file, "w") as writer:
            writer.write("\n\n=====DELIMITER====\n\n".join(predictions))


if __name__ == "__main__":
    train()
