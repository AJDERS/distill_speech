"""Config class that carries all the hyperparameters needed for training"""

from pydantic import BaseModel
from typing import Optional, Tuple


class Config(BaseModel):
    """Config class that carries all the hyperparameters needed for training.
    Args:
        pretrained_model_id (str):
            The model id of the pretrained model to finetune.
        distilled_model_id (str):
            The model id of the distilled model.
        hub_token (str):
            Token for use of HF repo
        dataset_id (str):
            The id of the dataset to finetune on.
        dataset_subset (str or None, optional):
            The subset of the dataset to finetune on. If None then no subset
            will be used. Defaults to 'da'.
        sampling_rate (int, optional):
            The sample rate of the audio files. Defaults to 16_000.
        train_name (str, optional):
            The name of the train dataset. Defaults to 'train'.
        validation_name (str or None, optional):
            The name of the validation dataset. If None then a validation set
            will be created from the train dataset. Defaults to 'validation'.
        test_name (str or None, optional):
            The name of the test dataset. If None then the validation set will
            be used as a test set and a new validation set will be created from
            the train dataset. If a validation set is not available either,
            then both a validation and test set will be created from the train
            dataset. Defaults to 'test'.
        audio_column_name (str, optional):
            The name of the column in the dataset containing the audio data.
            Defaults to 'audio'
        max_duration_in_seconds (int, optional):
            The maximum length of audio clips in seconds. Defaults to 25.
        min_duration_in_seconds (int, optional):
            The minimum length of audio clips in seconds. Defaults to 0.
        preprocessing_num_workers (int, optional):
            Number of workers used in multiprocessing of preprocessing.
            Defaults to number of cpus.
        (teacher/student)_activation_dropout (float, optional):
            The dropout rate for the activation layer. Defaults to 0.1.
        (teacher/student)_attention_dropout (float, optional):
            The dropout rate for the attention layer. Defaults to 0.1.
        (teacher/student)_hidden_dropout (float, optional):
            The dropout rate for the hidden layer. Defaults to 0.1.
        (teacher/student)_feat_proj_dropout (float, optional):
            The dropout rate for the feature projection layer. Defaults to 0.1.
        (teacher/student)_final_dropout (float, optional):
            The dropout rate for the final layer. Defaults to 0.1.
        (teacher/student)_mask_time_prob (float, optional):
            The probability of masking the time dimension. Defaults to 0.075.
        (teacher/student)_mask_feature_prob (float, optional):
            The probability of masking the feature dimension. Defaults to
            0.075.
        (teacher/student)_mask_feature_length (int, optional):
            The length of the masking of the feature dimension. Defaults to
            10.
        (teacher/student)_layerdrop (float, optional):
            The dropout rate for the layers. Defaults to 0.1.
        (teacher/student)_ctc_loss_reduction (str, optional):
            The reduction to use for the CTC loss. Defaults to 'sum'.
        distill_factor (int):
            By how many factors we reduce height parameters from teacher to obtain
            student parameters.
        num_hidden_layers (int):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (int):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_conv_pos_embedding_groups (int):
            Number of groups of 1D convolutional positional embeddings layer.
        tdnn_dim (tuple):
            A tuple of integers defining the number of output channels of each 1D
            convolutional layer in the TDNN module of the XVector model. The length
            of tdnn_dim defines the number of TDNN layers.
        tdnn_kernel (tuple):
            A tuple of integers defining the kernel size of each 1D convolutional
            layer in the TDNN module of the XVector model. The length of tdnn_kernel
            has to match the length of tdnn_dim.
        tdnn_dilation (tuple):
            A tuple of integers defining the dilation factor of each 1D convolutional
            layer in TDNN module of the XVector model. The length of tdnn_dilation has
            to match the length of tdnn_dim.
        seed (int):
            Seed for random generation.
        output_dir (str):
            Specifies local folder in which to save model.
        batch_size (int, optional):
            The batch size for training. Defaults to 4.
        gradient_accumulation_steps (int, optional):
            The number of steps to accumulate gradients for. Defaults to 8.
        gradient_checkpointing (bool):
            Wether or not to use gradient checkpointing. Useful for few GPUs.
        epochs (int, optional):
            The number of epochs to train for. Defaults to 500.
        learning_rate (float, optional):
            The learning rate for the optimizer. Defaults to 4e-5.
        lr_scheduler_type (str):
            Which schedule the learning rate decline should follow.
        adam_beta1 (float):
            beta_1 parameter for Adam optimizer
        adam_beta2 (float):
            beta_2 parameter for Adam optimizer
        adam_epsilon (float):
            epsilon parameter for Adam optimizer
        softmax_temperature (float):
            Smoothing factor of softmax calculation in loss function.
        warmup_steps (int, optional):
            The number of warmup steps for the learning rate scheduler.
            Defaults to 500.
        logging_steps (int, optional):
            The number of warmup steps for the learning rate scheduler.
            Defaults to 500.
        saving_steps (int, optional):
            The number of warmup steps for the learning rate scheduler.
            Defaults to 500.
        eval_steps (int, optional):
            The number of warmup steps for the learning rate scheduler.
            Defaults to 500.
        early_stopping (bool, optional):
            Whether to use early stopping. Defaults to True.
        early_stopping_patience (int, optional):
            The patience for early stopping. Only relevant if `early_stopping`
            is True. Defaults to 5.
        push_to_hub (bool, optional):
            Whether to push the model to the hub. Defaults to True.
    """

    # Model IDs
    pretrained_teacher_model_id: str
    distilled_model_id: str

    # HF
    hub_token: str = "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"

    # Dataset hyperparameters
    dataset_id: str
    dataset_subset: Optional[str] = None
    sampling_rate: int = 16_000
    train_name: str = "train"
    validation_name: Optional[str] = "validation"
    test_name: Optional[str] = "test"
    audio_column_name: Optional[str] = "audio"
    max_duration_in_seconds: Optional[int] = 25
    min_duration_in_seconds: Optional[int] = 0
    preprocessing_num_workers: Optional[int] = -1

    # Teacher Model hyperparameters
    teacher_activation_dropout: float = 0.1
    teacher_attention_dropout: float = 0.1
    teacher_hidden_dropout: float = 0.1
    teacher_feat_proj_dropout: float = 0.1
    teacher_final_dropout: float = 0.1
    teacher_mask_time_prob: float = 0.075
    teacher_mask_feature_prob: float = 0.075
    teacher_mask_feature_length: int = 10
    teacher_layerdrop: float = 0.1
    teacher_ctc_loss_reduction: str = "sum"

    # Student Model hyperparameters
    distill_factor: int = 3  # tested, -1, 1, 2, 3
    num_hidden_layers: int = 12  # Only necessary if `distill_factor` > 0
    num_attention_heads: int = (
        8  # Only necessary if `distill_factor` > 0, must divide `embed_dim`
    )
    num_conv_pos_embedding_groups: int = 8  # Only necessary if `distill_factor` > 0
    tdnn_dim: Tuple[int] = (512, 512, 1500)  # Only necessary if `distill_factor` > 0
    tdnn_kernel: Tuple[int] = (5, 3, 3)  # Only necessary if `distill_factor` > 0
    tdnn_dilation: Tuple[int] = (1, 2, 3)  # Only necessary if `distill_factor` > 0
    student_activation_dropout: float = 0.1
    student_attention_dropout: float = 0.1
    student_hidden_dropout: float = 0.1
    student_feat_proj_dropout: float = 0.1
    student_final_dropout: float = 0.1
    student_mask_time_prob: float = 0.075
    student_mask_feature_prob: float = 0.075
    student_mask_feature_length: int = 10
    student_layerdrop: float = 0.1
    student_ctc_loss_reduction: str = "sum"

    # Training hyperparameters
    seed: int = 703
    output_dir: Optional[str] = "models"
    batch_size: int = 1
    gradient_accumulation_steps: int = 128
    gradient_checkpointing: bool = True
    epochs: int = 500
    learning_rate: float = 4e-5
    lr_scheduler_type: str = "linear"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    softmax_temperature: float = 2.0
    warmup_steps: int = 500
    logging_steps: int = 5000
    eval_steps: int = 5000
    early_stopping: bool = True
    early_stopping_patience: int = 5
    push_to_hub: bool = True
    pad_to_multiple_of: int = 32
