import os
import wandb
import torch
import numpy as np
import huggingface_hub
import transformers
import datasets
from io import TextIOBase, StringIO
from csv import QUOTE_NONE
from transformers import TextGenerationPipeline
from pandas import DataFrame, Series
from typing import Callable, Iterable, Optional, Tuple, List, Literal, Dict, Union, Any
from torch import Tensor, where, argmax
from itertools import repeat
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, PretrainedBartModel
from transformers.trainer_utils import EvalPrediction
from evaluate import load
from transformers.trainer_utils import SchedulerType, IntervalStrategy, HubStrategy, FSDPOption
from transformers.training_args import OptimizerNames


def login() -> None:

    # Authentication with API key, and user token.
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    huggingface_token = os.environ.get("HF_TOKEN")

    # Log into `huggingface.co` and `wandb.ai`.
    wandb.login(key=wandb_api_key, verify=True)
    huggingface_hub.login(huggingface_token)

def inference(
    test_directory: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    test_dataset: Dataset,
):

    # Determine device (GPU if available, otherwise CPU).
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Generator with a batch size of 1 element.
    generator = TextGenerationPipeline(model, tokenizer, device=device)

    # prompts = [str(item) for item in test_dataset["prompt"]]
    # prompts = [ str(test_dataset["prompt"]) ]

    os.makedirs(test_directory, exist_ok=True)

    for index, prompt in enumerate(test_dataset["prompt"], start=1):
   
        if index == 30:
            break

        # Generate the mesh for the test prompts.
        result = generator(
            str(prompt), 
            return_full_text=False,
            max_new_tokens=4096
        )

        with open(f"{test_directory}/result_{index}.txt", "w") as file:
            file.write(result[0]["generated_text"])


def setup_training_args(output_dir: str, batch_size: int) -> TrainingArguments:
    # Initial conservative estimates based on 4k-8k tokens and Gemma 2B on A100 40GB
    # You will likely have to tune per_device_train_batch_size and gradient_accumulation_steps
    # after initial runs to find the largest batch size that fits without OOM.
    
    args = TrainingArguments(
        output_dir,
        # push_to_hub=False,              # Save the model to `huggingface.co`.
        # load_best_model_at_end=True,     # Load the best model after training.
        metric_for_best_model="eval_loss",  # Which metric defines the best model.
        greater_is_better=False,        # Smaller losses define better models.
        remove_unused_columns=False,    # Don't remove colums from the batches.
        run_name="100_train_epochs",
        # use_cpu=False,                  # Use the available CUDE device.
        torch_empty_cache_steps=1,
        save_safetensors=False,
        save_only_model=True      
        # fp16=False,                     # No fp16 (mixed) precision training.
        # bf16=True,                      # Use bf16 (mixed) precision training.
        # fp16_full_eval=False,           # No fp16 (mixed) precision evaluation.
        # bf16_full_eval=False,            # No bf16 (mixed) precision evaluation.
        # fsdp=True
        # fsdp_config="fsdp_config.json"
        # deepspeed="ds_config.json"
    )

    args.set_optimizer(
        name=OptimizerNames.ADAMW_TORCH,
        learning_rate=2e-5,
        weight_decay=0.01,
    )

    args.set_lr_scheduler(
        name=SchedulerType.COSINE,
        warmup_ratio=0.03
    )

    args.set_evaluate(
        strategy=IntervalStrategy.EPOCH,   # Evalute the model every epoch.
        accumulation_steps=None,
        delay=0.0                          # Don't delay the first evaluation.
    )

    args.set_training(
        num_epochs=100,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True
    )

    args.set_logging(
        strategy=IntervalStrategy.EPOCH,    # Log the loss every epoch.
        report_to="wandb"
    )

    args.set_save(
        strategy=IntervalStrategy.EPOCH,    # Save the model every epoch.
        total_limit=1                       # Save the best (and last) model. 
    )

    args.set_dataloader(
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
        # auto_find_batch_size=False       # Half batch sizes, if OOM occures.
    )

    # args.set_push_to_hub(
    #     model_id="zegedi/gemma-3-270m-fem",
    #     strategy=HubStrategy.ALL_CHECKPOINTS,
    #     private_repo=True,
    #     always_push=False
    # )

    args.load_best_model_at_end = True
    args.deepspeed="ds_config.json"

    return args


def compute_metrics(batch: EvalPrediction):
    return np.mean(np.exp(batch.losses))


def setup_trainer(
    model,
    args,
    tokenizer,
    dataset: DatasetDict
) -> Trainer:

    # Use the inputs as labels shifted to the right by one element.
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    return Trainer(
        model, 
        args=args, 
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"], 
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        compute_metrics=None
    )


def setup_model(
    pretrained_model_name: str,
    model_kwargs: Optional[Dict[str, Any]] = None,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    if model_kwargs is None:
        model_kwargs = dict()

    if tokenizer_kwargs is None:
        tokenizer_kwargs = dict()

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name, **model_kwargs
    )
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        pretrained_model_name, **tokenizer_kwargs
    )

    if tokenizer.chat_template is None:
        # tokenizer.add_tokens(["<|user|>", "<|assistant|>", "<|system|>"])
        tokenizer.chat_template = (
            "{%- for message in messages %}"
                "{{- '<|' + message['role'] + '|>\n' }}"
                "{% if (message['role'] == 'assistant') %}"
                    "{% generation %}"
                    "{{- message['content'] + eos_token }}"
                    "{% endgeneration %}"
                "{% else %}"
                    "{{- message['content'] + eos_token }}"
                "{% endif %}"
            "{%- endfor %}"
            "{%- if add_generation_prompt %}"
                "{{- '<|assistant|>\n' }}"
            "{%- endif %}"
        )

    # Use the end-of-sequence token as the padding token.
    # tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def setup_dataset_stage_1(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    train_size: Union[float, int] = 0.8,
    valid_size: Union[float, int] = 0.5,
    batch_size: int = 1,
    shuffle: bool = True,
) -> DatasetDict:
    
    dataset = load_dataset(dataset_path)
    
    dataset_split = split_dataset(dataset, train_size, valid_size, shuffle)
    
    dataset_split["train"].set_transform(
        lambda batch: train_process_stage1(batch, tokenizer)
    )
    dataset_split["valid"] = dataset_split["valid"].map(
        lambda batch: eval_process_stage1(batch, tokenizer), 
        batched=True, batch_size=batch_size,
        remove_columns = [
            'identifier',
            'geometry.point', 
            'geometry.curve', 
            'geometry.surface', 
            'mesh.node',
            'mesh.element'
        ]
    )
    dataset_split["test"] = dataset_split["test"].map(
        lambda batch: process_test_dataset_stage1(batch, tokenizer),
        batched=True, batch_size=batch_size,
        remove_columns = [
            'identifier',
            'geometry.point', 
            'geometry.curve', 
            'geometry.surface', 
            'mesh.node',
            'mesh.element'
        ]
    )

    print("First stage - Vertex generation - Dataset split information.")
    print("Train num rows:", dataset_split["train"].num_rows)
    print("Valid num rows:", dataset_split["valid"].num_rows)
    print("Test num rows:", dataset_split["test"].num_rows)

    return dataset_split


def setup_dataset_stage_2(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    train_size: Union[float, int] = 0.8,
    valid_size: Union[float, int] = 0.5,
    batch_size: int = 1,
    shuffle: bool = True,
) -> DatasetDict:
    
    dataset = load_dataset(dataset_path)
    
    dataset_split = split_dataset(dataset, train_size, valid_size, shuffle)
    
    dataset_split["train"].set_transform(
        lambda batch: train_process_stage2(batch, tokenizer)
    )
    dataset_split["valid"] = dataset_split["valid"].map(
        lambda batch: eval_process_stage2(batch, tokenizer), 
        batched=True, batch_size=batch_size,
        remove_columns = [
            'identifier',
            'geometry.point', 
            'geometry.curve', 
            'geometry.surface', 
            'mesh.node',
            'mesh.element'
        ]
    )
    dataset_split["test"] = dataset_split["test"].map(
        lambda batch: process_test_dataset_stage2(batch, tokenizer),
        batched=True, batch_size=batch_size,
        remove_columns = [
            'identifier',
            'geometry.point', 
            'geometry.curve', 
            'geometry.surface', 
            'mesh.node',
            'mesh.element'
        ]
    )
    
    print("Second stage - Mesh generation - Dataset split information.")
    print("Train num rows:", dataset_split["train"].num_rows)
    print("Valid num rows:", dataset_split["valid"].num_rows)
    print("Test num rows:", dataset_split["test"].num_rows)

    return dataset_split


def load_dataset(dataset_path: str) -> Dataset:
    
    dataset: Dataset = load_from_disk(dataset_path, keep_in_memory=True)

    # Subset of the dataset, for testing purposes.
    # dataset = Dataset.from_dict(dataset[0:16])

    # columns_to_remove = 'geometry.property'
    columns_to_format = [
        'geometry.point', 
        'geometry.curve', 
        'geometry.surface', 
        'mesh.node',
        'mesh.element'
    ]

    dataset = dataset.flatten(max_depth=2)
    # dataset = dataset.remove_columns(columns_to_remove)
    
    # print("Dataset columns:", dataset.column_names)

    # dataset.set_format("numpy", columns_to_format, output_all_columns=False)

    return dataset


def split_dataset(
    dataset: Dataset,
    train_size: Union[float, int],
    valid_size: Union[float, int],
    shuffle: bool = True
) -> DatasetDict:
    dataset_train_test = dataset.train_test_split(
        None, train_size, shuffle
    )
    dataset_valid_test = dataset_train_test["test"].train_test_split(
        None, valid_size, shuffle
    )
    return DatasetDict({
        "train": dataset_train_test["train"],
        "valid": dataset_valid_test["train"],
        "test" : dataset_valid_test["test"]
    })


def default_transform(*args, **kwargs) -> None:
    return


def process_test_dataset_stage2(
    batch: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase
) -> Dict[str, List]:

    batch_chats: List[List[Dict[str, str]]] = []

    for points, curves, surfaces, nodes, elements in get_dataframes(batch):

        # Sort the mesh data.
        # sort_mesh_data(nodes, elements)

        # Create the text input for the batch.
        batch_chats.append(
            get_initial_prompt_obj_mesh(points, curves, surfaces)
        )
    
    # Tokenize the conversations and return the tokenized batch.
    batch["prompt"] = tokenizer.apply_chat_template(
        batch_chats,
        tokenize=False,
        add_generation_prompt=True
    )

    return batch


def process_test_dataset_stage1(
    batch: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase
) -> Dict[str, List]:

    batch_chats: List[List[Dict[str, str]]] = []

    for points, curves, surfaces, nodes, elements in get_dataframes(batch):

        # Sort the mesh data.
        # sort_mesh_data(nodes, elements)

        # Create the text input for the batch.
        batch_chats.append(
            get_initial_prompt_obj_vertex(points, curves, surfaces)
        )
    
    # Tokenize the conversations and return the tokenized batch.
    batch["prompt"] = tokenizer.apply_chat_template(
        batch_chats,
        tokenize=False,
        add_generation_prompt=True
    )

    return batch


def sort_mesh_data(nodes: DataFrame, elements: DataFrame) -> None:

    nodes.sort_values(by=['X', 'Y', 'Z'], inplace=True)

    tag_to_new_index_map = { 
        old: new for new, old in enumerate(nodes['Tag'], start=1)
    }

    elements['Nodes'] = elements['Nodes'].apply(
        lambda node_list: [tag_to_new_index_map[tag] for tag in node_list]
    )

    elements.sort_values(by='Nodes', inplace=True)


def process_dataset(
    batch: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase,
    prompt: Callable,
    transform: Callable[..., None] = default_transform
) -> Dict[str, List]:

    batch_chats: List[List[Dict[str, str]]] = []

    for points, curves, surfaces, nodes, elements in get_dataframes(batch):

        # Data augmentation on the coordinates.
        transform(points, nodes)
        
        # Create the text input for the batch.
        batch_chats.append(
            prompt(points, curves, surfaces, nodes, elements)
        )
    
    # Tokenize the conversations and return the tokenized batch.
    batch_tokenized: Dict[str, Tensor] = tokenizer.apply_chat_template(
        batch_chats, 
        tokenize=True,
        padding="longest",
        truncation=True,
        return_dict=True,
        return_tensors="pt"
    )

    return batch_tokenized


def eval_process_stage1(
    batch: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase
) -> Dict[str, List]:
    return process_dataset(
        batch, 
        tokenizer,
        lambda *args : get_full_prompt_obj_vertex(*args)
    )


def eval_process_stage2(
    batch: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase
) -> Dict[str, List]:
    return process_dataset(
        batch, 
        tokenizer,
        lambda *args : get_full_prompt_obj_mesh(*args)
    )


def train_process_stage1(
    batch: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase
) -> Dict[str, List]:
    return process_dataset(
        batch, 
        tokenizer,
        lambda *args : get_full_prompt_obj_vertex(*args),
        uniform_translation
    )


def train_process_stage2(
    batch: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase
) -> Dict[str, List]:
    return process_dataset(
        batch, 
        tokenizer,
        lambda *args : get_full_prompt_obj_mesh(*args),
        uniform_translation
    )


def train_process(
    batch: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase
) -> Dict[str, List]:
    return process_dataset(
        batch, 
        tokenizer,
        get_full_prompt_obj_mesh,
        uniform_translation
    )


def get_user_prompt_obj_mesh(
    target: TextIOBase,
    point: DataFrame, 
    curve: DataFrame, 
    surface: DataFrame
) -> None:
    def iterable_dataframes() -> Iterable[DataFrame]:
        yield point[["Entity", "Tag", "X", "Y"]]
        yield curve[["Entity", "Tag", "Start", "End", "Center"]]
        yield surface[["Entity", "Tag", "Tags"]]

    target.write(
        "Here is the CSV description of a two-dimensional geometry:\n"
    )

    for dataframe in iterable_dataframes():
        export_csv(dataframe, target)

    target.write(
        "Create a quadrilateral mesh for this geometry. "
        "Directly generate the verticies and elements in OBJ format.\n"
    )


def get_user_prompt_obj_vertex(
    target: TextIOBase,
    point: DataFrame, 
    curve: DataFrame, 
    surface: DataFrame
) -> None:
    def iterable_dataframes() -> Iterable[DataFrame]:
        yield point[["Entity", "Tag", "X", "Y"]]
        yield curve[["Entity", "Tag", "Start", "End", "Center"]]
        yield surface[["Entity", "Tag", "Tags"]]

    target.write(
        "Here is the CSV description of a two-dimensional geometry:\n"
    )

    for dataframe in iterable_dataframes():
        export_csv(dataframe, target)

    target.write(
        "Create quadrilateral element verticies for this geometry. "
        "Directly generate the verticies in OBJ format.\n"
    )


def get_assistant_prompt(
    target: TextIOBase,
    nodes: DataFrame, 
    elements: DataFrame
) -> None:
    export_obj_vertexes(nodes, target)
    export_obj_faces(elements, target)



def get_initial_prompt_obj_vertex(
    point: DataFrame, 
    curve: DataFrame, 
    surface: DataFrame,
) -> List[Dict[str, str]]:
    user_content = StringIO()

    get_user_prompt_obj_vertex(user_content, point, curve, surface)

    # return [{ "role": "user", "content": user_content.getvalue() }]

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": 
"""Role play.

You are an expert in computer-aided design.
Your task is to analyze the CSV description of 2D geometries and produce a
quadrilateral mesh in OBJ format.

Input description.

The input geometry is defined using a CSV format, with semicolons as separators.
The first line of every CSV table contains the header.

There are three different tables, which hierarchically define the points, curves, and surfaces of the geometry.
Points are defined by their X and Y coordinates. Curves are defined by their boundary points. Finally, surfaces are defined by their boundary curves.
The data might include additional information like identifiers (Tag) and names (Type).

Output description.

The output mesh is defined using the OBJ format.
The first section contains the vertex definitions in the form `v x y z`.
The second section includes the face definitions in the form `f v1 v2 v3 v4`."""},]
        },
        { 
            "role": "user", 
            "content": [{ "type": "text", "text": user_content.getvalue() },]
        }
    ]


def get_initial_prompt_obj_mesh(
    point: DataFrame, 
    curve: DataFrame, 
    surface: DataFrame,
) -> List[Dict[str, str]]:
    user_content = StringIO()

    get_user_prompt_obj_mesh(user_content, point, curve, surface)

    # return [{ "role": "user", "content": user_content.getvalue() }]

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": 
"""Role play.

You are an expert in computer-aided design.
Your task is to analyze the CSV description of 2D geometries and produce a
quadrilateral mesh in OBJ format.

Input description.

The input geometry is defined using a CSV format, with semicolons as separators.
The first line of every CSV table contains the header.

There are three different tables, which hierarchically define the points, curves, and surfaces of the geometry.
Points are defined by their X and Y coordinates. Curves are defined by their boundary points. Finally, surfaces are defined by their boundary curves.
The data might include additional information like identifiers (Tag) and names (Type).

Output description.

The output mesh is defined using the OBJ format.
The first section contains the vertex definitions in the form `v x y z`.
The second section includes the face definitions in the form `f v1 v2 v3 v4`."""},]
        },
        { 
            "role": "user", 
            "content": [{ "type": "text", "text": user_content.getvalue() },]
        }
    ]
        


def get_full_prompt_obj_vertex(
    point: DataFrame, 
    curve: DataFrame, 
    surface: DataFrame,
    nodes: DataFrame, 
    elements: DataFrame = None
) -> List[Dict[str, str]]:

    assistant_content = StringIO()

    export_obj_vertexes(nodes, assistant_content)

    prompt = get_initial_prompt_obj_vertex(point, curve, surface)

    prompt.append({
        "role": "assistant",
        "content": [{ "type": "text", "text": assistant_content.getvalue() },]
    })

    return prompt



def get_full_prompt_obj_mesh(
    point: DataFrame, 
    curve: DataFrame, 
    surface: DataFrame,
    nodes: DataFrame, 
    elements: DataFrame
) -> List[Dict[str, str]]:

    assistant_content = StringIO()

    get_assistant_prompt(assistant_content, nodes, elements)

    prompt = get_initial_prompt_obj_mesh(point, curve, surface)

    prompt.append({
        "role": "assistant",
        "content": [{ "type": "text", "text": assistant_content.getvalue() },]
    })

    return prompt


def get_dataframes(batch: Dict[str, List]) -> Iterable[Tuple[DataFrame, ...]]:

    tables_per_example: Iterable[Tuple[List, ...]] = zip(
        batch['geometry.point'],
        batch['geometry.curve'],
        batch['geometry.surface'],
        batch['mesh.node'],
        batch['mesh.element']
    )

    convert_to_dataframe = lambda table: DataFrame(table)

    return (
        tuple(map(convert_to_dataframe, tables)) 
        for tables in tables_per_example
    )


def uniform_translation(
    *dataframes: Tuple[DataFrame, ...],
    low: float = -1,
    high: float = 1,
    columns: List[str] = ['X', 'Y'],
    **kwargs
) -> None:
    translations = np.random.uniform(low, high, size=len(columns))

    for dataframe in dataframes:
        dataframe[columns] += translations


def export_csv(
    object: DataFrame,
    target: Optional[TextIOBase] = None,
    mode: Literal["w", "x", "a"] = "a"
) -> Optional[str]:
    return object.to_csv(
        target, sep=";", na_rep="", lineterminator="\n",
        float_format=lambda number: str(round(number, ndigits=3)), 
        header=True, index=False, mode=mode
    )


def export_obj_vertexes(
    object: DataFrame, 
    target: Optional[TextIOBase] = None,
    mode: Literal["w", "x", "a"] = "a"
) -> Optional[str]:

    # Convert the list of tags into a space separated string.
    vertexes = object[["X", "Y"]].apply(np.round, decimals=3)
    
    # Add custom indexes for the OBJ vertex definition.
    vertexes.index = repeat("v", len(vertexes))

    return vertexes.to_csv(
        target, sep=" ", lineterminator="\n", 
        header=False, index=True, mode=mode
    )


def export_obj_faces(
    object: DataFrame,
    target: Optional[TextIOBase] = None,
    mode: Literal["w", "x", "a"] = "a"
) -> Optional[str]:
    
    # Convert the list of tags into a space separated string.
    faces = object["Nodes"].apply(Series)

    # Add custom indexes for the OBJ face definition.
    faces.index = repeat("f", len(faces))

    return faces.to_csv(
        target, sep=" ", na_rep="", lineterminator="\n", 
        header=False, index=True, mode=mode
    )