from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)


# Load dataset
def load_data(dataset_name="squad", split="train[:100]"):
    """Load and split dataset."""
    try:
        dataset = load_dataset(dataset_name)
        print("Dataset loaded successfully.")
        return DatasetDict({
            "train": dataset["train"].select(range(80)),
            "valid": dataset["train"].select(range(80, 100)),
        })

    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


# Load tokenizer and model
#def load_tokenizer_and_model(model_checkpoint="bert-base-cased"):
def load_tokenizer_and_model(model_checkpoint="distilbert-base-uncased-distilled-squad"):
    """Load tokenizer and model."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
        return tokenizer, model
    except Exception as e:
        print(f"Error loading tokenizer and model: {e}")
        raise


# Preprocess training examples
def preprocess_training_examples(examples, tokenizer, max_length=384, doc_stride=128):
    """Tokenize and prepare training examples."""
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            token_end_index = len(input_ids) - 1

            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            start_positions.append(token_start_index - 1)

            while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            end_positions.append(token_end_index + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples


# Train the model
def train_model(train_dataset, model, tokenizer, output_dir="fine_tuned_qa_model"):
    """Train and save the model."""
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        push_to_hub=False,
        fp16=False,  # Disable fp16 if GPU is unavailable
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset["train"],
        eval_dataset=train_dataset["valid"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(output_dir)

    return trainer


# Test the model
def test_model(model_path, tokenizer, question, context):
    """Test the fine-tuned model."""
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)

    #inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=384, padding="max_length")
    inputs = tokenizer(
    question,
    context,
    truncation="only_second",
    max_length=384,
    stride=128,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding="max_length",
    return_tensors="pt"
    )   
    input_ids = inputs["input_ids"].tolist()[0]
    offsets = inputs.pop("offset_mapping")  # Remove it before passing to model
    inputs_for_model = {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
    outputs = model(**inputs_for_model)
    answer_start_index = outputs.start_logits.argmax().item()
    answer_end_index = outputs.end_logits.argmax().item()

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start_index:answer_end_index + 1])
    )
    return answer


# Main function
def main():
    # Load data
    raw_datasets = load_data()
    print("Sample Dataset")
    print(raw_datasets["train"][0])

    # Load tokenizer and model
    tokenizer, model = load_tokenizer_and_model()

    # Preprocess dataset
    train_dataset = raw_datasets.map(
        lambda x: preprocess_training_examples(x, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    # Train model
    save_path = "fine_tuned_qa_model"
    train_model(train_dataset, model, tokenizer, output_dir=save_path)

    # Test the fine-tuned model
    question = "What is the capital of France?"
    context = "France is a country in Europe. Its capital is Paris."
    answer = test_model(save_path, tokenizer, question, context)

    print("Question:", question)
    print("Answer:", answer)


if __name__ == "__main__":
    main()
