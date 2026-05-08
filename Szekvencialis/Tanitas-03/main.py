from wandb import init
from functions import login, setup_model
from functions import setup_dataset_stage_1, setup_dataset_stage_2
from functions import setup_trainer, setup_training_args, inference

def main() -> None:

    # Setup login credentials.
    login()

    # Initialize WanDB project for stage 1.
    init(project="fem_ai_project", name="hole_in_square_vertex")

    # Batch size for the training.
    batch_size = 2

    # Setup training args for stage 1.
    args = setup_training_args("/app/fem_ai_model_vertex", batch_size)

    # Setup the choosen model and tokenizer.
    model, tokenizer = setup_model(
        pretrained_model_name="google/gemma-3-270m-it",
        model_kwargs={
            "attn_implementation": "eager"  # Manual attention implementation.
        },
        tokenizer_kwargs={
            "model_max_length": 32000,
            "padding_side": "right",
            "truncation_side" : "right"
        }
    )
    
    # Setup the dataset split.
    dataset_split = setup_dataset_stage_1(
        "/app/fem_ai_dataset", tokenizer, 
        train_size=8500, valid_size=1450, shuffle=False, batch_size=batch_size
    )

    # Setup the trainer class.
    trainer = setup_trainer(model, args, tokenizer, dataset_split)

    # Start the training process.
    trainer.train()

    # Inference.
    inference("/app/tests_vertex", model, tokenizer, dataset_split["test"])

    # Initialize WanDB project for stage 2.
    init(project="fem_ai_project", name="hole_in_square_mesh", reinit='create_new')

    # Setup training args for stage 2.
    args = setup_training_args("/app/fem_ai_model_mesh", batch_size)

    # Setup the dataset split.
    dataset_split = setup_dataset_stage_2(
        "/app/fem_ai_dataset", tokenizer,
        train_size=8500, valid_size=1450, shuffle=False, batch_size=batch_size
    )

    # Setup the trainer class.
    trainer = setup_trainer(model, args, tokenizer, dataset_split)

    # Start the training process.
    trainer.train()

    # Inference.
    inference("/app/tests_mesh", model, tokenizer, dataset_split["test"])


if __name__ == "__main__":
    main()
