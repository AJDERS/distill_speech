"""Student teacher model distilling script for Danish acoustic models"""
import os
import math
import torch
import logging
from tqdm import tqdm
from config import Config
from data import AudioDataset
from math import log, ceil, floor
from typing import Optional, Union
from prettytable import PrettyTable
import torch.nn.functional as functional
from huggingface_hub.repository import Repository
from transformers.utils import get_full_repo_name
from torch.utils.data.dataloader import DataLoader
from data_collator import DataCollatorForWav2Vec2Pretraining
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2ForPreTraining,
    AdamW,
    get_scheduler,
    set_seed,
)

logging.basicConfig(level=logging.INFO)


def train(config: Optional[Union[dict, Config]] = None, **kwargs):
    """Distilling a smaller model from a pretrained audio model on a dataset.
    Args:
        config (Config, dict or None):
            Config object or dict containing the parameters for the finetuning.
            If None then a Config object is created from the default
            parameters. Defaults to None.
        **kwargs:
            Keyword arguments to be passed to the config
    """
    #### Check if GPU is available
    logging.info(f"GPU availability: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #### Load config

    # If no config is provided, create a config object from the default
    # parameters
    if config is None:
        config = Config(**kwargs)

    # If a dict is provided, create a config object from the dict
    elif isinstance(config, dict):
        config = Config(**{**config, **kwargs})

    # If a Config object is provided, update the config object with the
    # provided keyword arguments
    elif isinstance(config, Config) and len(kwargs) > 0:
        config = Config(**{**config.__dict__, **kwargs})

    #### Set training seed.
    if config.seed is not None:
        set_seed(config.seed)

    #### Create HF epository
    if config.output_dir is not None:
        save_dir = os.path.join(
            config.output_dir,
            f"{config.pretrained_teacher_model_id}_{config.dataset_id}",
        )
    if config.push_to_hub:
        if config.distilled_model_id is None:
            repo_name = get_full_repo_name(save_dir, token=config.hub_token)
        else:
            repo_name = config.distilled_model_id
        repo = Repository(save_dir, clone_from=repo_name)
    elif config.output_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    #### Load dataset
    dataset = AudioDataset(
        pretrained_teacher_model_id=config.pretrained_teacher_model_id,
        dataset_id=config.dataset_id,
        dataset_subset=config.dataset_subset,
        sampling_rate=config.sampling_rate,
        train_name=config.train_name,
        validation_name=config.validation_name,
        test_name=config.test_name,
        audio_column_name=config.audio_column_name,
        max_duration_in_seconds=config.max_duration_in_seconds,
        min_duration_in_seconds=config.min_duration_in_seconds,
        preprocessing_num_workers=config.preprocessing_num_workers,
    )
    vectorized_dataset = dataset.process()

    #### Load models, and update configs

    # Teacher model config, used to update student configs.
    teacher_model_config = Wav2Vec2Config.from_pretrained(
        config.pretrained_teacher_model_id,
    )
    # Student model config.
    student_model_config = Wav2Vec2Config.from_pretrained(
        config.pretrained_teacher_model_id
    )

    # Set student parameters not related to model distilling.
    student_model_config.activation_dropout = config.student_activation_dropout
    student_model_config.attention_dropout = config.student_attention_dropout
    student_model_config.hidden_dropout = config.student_hidden_dropout
    student_model_config.feat_proj_dropout = config.student_feat_proj_dropout
    student_model_config.final_dropout = config.student_final_dropout
    student_model_config.mask_time_prob = config.student_mask_time_prob
    student_model_config.mask_feature_prob = config.student_mask_feature_prob
    student_model_config.mask_feature_length = config.student_mask_feature_length
    student_model_config.layerdrop = config.student_layerdrop
    student_model_config.ctc_loss_reduction = config.student_ctc_loss_reduction

    # Set parameters related to model distilling.
    # Check if `distill_factor` is positive, if so divide all model "height" parameters by `distill_factor`
    # if not use parameters from config.
    if config.distill_factor > 0:
        # Hidden layers
        student_model_config.num_hidden_layers = round(
            teacher_model_config.num_hidden_layers / config.distill_factor
        )

        # Attention heads
        student_model_config.num_attention_heads = round(
            teacher_model_config.num_attention_heads / config.distill_factor
        )

        # Convolutional positional embedding groups
        student_model_config.num_conv_pos_embedding_groups = round(
            teacher_model_config.num_conv_pos_embedding_groups / config.distill_factor
        )

        # If number of convolutional positional embeddings groups, or number of attention heads,
        # does not divide hidden size, find closest power of 2, which divides.
        # NOTE: this assumes hidden_size is a power of 2.
        if (
            student_model_config.hidden_size
            % student_model_config.num_conv_pos_embedding_groups
            != 0
        ):
            n_groups = student_model_config.num_conv_pos_embedding_groups
            possible_results = floor(log(n_groups, 2)), ceil(log(n_groups, 2))
            student_model_config.num_conv_pos_embedding_groups = min(
                possible_results, key=lambda z: abs(n_groups - 2**z)
            )
        if (
            student_model_config.hidden_size % student_model_config.num_attention_heads
            != 0
        ):
            n_heads = student_model_config.num_attention_heads
            possible_results = floor(log(n_heads, 2)), ceil(log(n_heads, 2))
            student_model_config.num_attention_heads = min(
                possible_results, key=lambda z: abs(n_heads - 2**z)
            )

        # Convolutional layers in Time Delay Neural Networks
        tdnn_len = round(len(teacher_model_config.tdnn_dim) / config.distill_factor)
        # Convolutional layer dimensions
        # Take last entries from teacher
        student_model_config.tdnn_dim = teacher_model_config.tdnn_dim[(tdnn_len + 1) :]
        # Convolutional layer kernel sizes
        # Take first entries from teacher
        student_model_config.tdnn_kernel = teacher_model_config.tdnn_kernel[:tdnn_len]
        # Convolutional layer time dilation
        # Take first entries from teacher
        student_model_config.tdnn_dilation = teacher_model_config.tdnn_dilation[
            :tdnn_len
        ]
    else:
        student_model_config.num_hidden_layers = config.num_hidden_layers
        student_model_config.num_attention_heads = config.num_attention_heads
        student_model_config.num_conv_pos_embedding_groups = (
            config.num_conv_pos_embedding_groups
        )
        student_model_config.tdnn_dim = config.tdnn_dim
        student_model_config.tdnn_kernel = config.tdnn_kernel
        student_model_config.tdnn_dilation = config.tdnn_dilation

    # Check configs are valid
    assert all(
        [
            len(student_model_config.tdnn_dim) == len(x)
            for x in [
                student_model_config.tdnn_dilation,
                student_model_config.tdnn_kernel,
            ]
        ]
    ), (
        f"tdnn_dim: {len(student_model_config.tdnn_dim)}, "
        + f"tdnn_dilation: {len(student_model_config.tdnn_dilation)}, "
        + f"tdnn_kernel: {len(student_model_config.tdnn_kernel)}, "
    )

    # Initialize models. Teacher model is initialized from a pretrained model.
    teacher_model = Wav2Vec2ForPreTraining.from_pretrained(
        config.pretrained_teacher_model_id
    ).to(device)
    teacher_model.config.activation_dropout = config.teacher_activation_dropout
    teacher_model.config.attention_dropout = config.teacher_attention_dropout
    teacher_model.config.hidden_dropout = config.teacher_hidden_dropout
    teacher_model.config.feat_proj_dropout = config.teacher_feat_proj_dropout
    teacher_model.config.final_dropout = config.teacher_final_dropout
    teacher_model.config.mask_time_prob = config.teacher_mask_time_prob
    teacher_model.config.mask_feature_prob = config.teacher_mask_feature_prob
    teacher_model.config.mask_feature_length = config.teacher_mask_feature_length
    teacher_model.config.layerdrop = config.teacher_layerdrop
    teacher_model.config.ctc_loss_reduction = config.teacher_ctc_loss_reduction

    # Student model is initialized with a random model, using the parameters defined above.
    student_model = Wav2Vec2ForPreTraining(student_model_config).to(device)
    t_total_params, t_table = count_parameters(teacher_model)
    s_total_params, s_table = count_parameters(student_model)
    logging.info(f"  Teacher summary: \n {t_table}")
    logging.info(f"  Student summary: \n {s_table}")
    logging.info(
        f"  Parameters ratio (student/teacher): {round((s_total_params/t_total_params), 2)}."
    )

    # Activate gradient checkpointing if enabled
    if config.gradient_checkpointing:
        student_model.gradient_checkpointing_enable()

    #### Define data collator, loaders, optimizer, loss function and scheduler
    data_collator = DataCollatorForWav2Vec2Pretraining(
        model=teacher_model, feature_extractor=dataset.feature_extractor, device=device
    )

    train_dataloader = DataLoader(
        vectorized_dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=config.batch_size,
    )
    eval_dataloader = DataLoader(
        vectorized_dataset["validation"],
        collate_fn=data_collator,
        batch_size=config.batch_size,
    )

    # Optimizer
    optimizer = AdamW(
        list(student_model.parameters()),
        lr=config.learning_rate,
        betas=[config.adam_beta1, config.adam_beta2],
        eps=config.adam_epsilon,
    )

    # Loss function
    KD_loss = torch.nn.KLDivLoss(reduction="batchmean")

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    max_train_steps = config.epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=max_train_steps,
    )

    #### Train

    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {len(vectorized_dataset['train'])}")
    logging.info(f"  Num Epochs = {config.epochs}")
    logging.info(f"  Instantaneous batch size per device = {config.batch_size}")
    logging.info(
        f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}"
    )
    logging.info(f"  Total optimization steps = {max_train_steps}")

    completed_steps = 0
    starting_epoch = 0

    def evaluate(batch):
        # Validate
        student_model.eval()
        # init logs
        val_logs = {
            "val_loss": 0,
            "val_contrastive_loss_diff": 0,
            "val_diversity_loss_diff": 0,
            "val_num_losses": 0,
        }
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                batch.pop("sub_attention_mask", None)
                teacher_outputs = teacher_model(**batch)
                student_outputs = student_model(**batch)
            teacher_projected_states = teacher_outputs.projected_states
            student_projected_states = student_outputs.projected_states
            loss = KD_loss(
                input=functional.log_softmax(
                    student_projected_states / config.softmax_temperature,
                    dim=-1,
                ),
                target=functional.softmax(
                    teacher_projected_states / config.softmax_temperature,
                    dim=-1,
                ),
            )
        val_logs["val_loss"] += loss
        val_logs["val_contrastive_loss_diff"] += (
            teacher_outputs.contrastive_loss - student_outputs.contrastive_loss
        )
        val_logs["val_diversity_loss_diff"] += (
            teacher_outputs.diversity_loss - student_outputs.diversity_loss
        )
        val_logs["val_num_losses"] += batch["mask_time_indices"].sum()
        val_logs = {k: v / val_logs["val_num_losses"] for k, v in val_logs.items()}
        log_str = ""
        for k, v in val_logs.items():
            log_str += "| {}: {:.3e}".format(k, v.item())
        logging.info(log_str)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps))
    completed_steps = 0
    starting_epoch = 0
    for epoch in range(starting_epoch, config.epochs):
        student_model.train()
        teacher_model.eval()
        for step, batch in enumerate(train_dataloader):
            # compute num of losses
            num_losses = batch["mask_time_indices"].sum()
            sub_attention_mask = batch.pop("sub_attention_mask", None)
            sub_attention_mask = (
                sub_attention_mask
                if sub_attention_mask is not None
                else torch.ones_like(batch["mask_time_indices"])
            )
            percent_masked = num_losses / sub_attention_mask.sum()

            #### Forward
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch)
            student_outputs = student_model(**batch)

            teacher_projected_states = teacher_outputs.projected_states
            student_projected_states = student_outputs.projected_states

            # TODO Think about wether to use `projected_states`, `projected_quantized_states`, or
            # `hidden_states`

            #### Calculate (Kullback-Leibler divergence) loss
            loss = KD_loss(
                input=functional.log_softmax(
                    student_projected_states / config.softmax_temperature, dim=-1
                ),
                target=functional.softmax(
                    teacher_projected_states / config.softmax_temperature, dim=-1
                ),
            )
            # divide loss by gradient accumulation steps since gradients
            # are accumulated for multiple backward passes in PyTorch
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % config.gradient_accumulation_steps == 0 or step == len(
                train_dataloader
            ) - 1:

                # update parameters
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                progress_bar.update(1)
                completed_steps += 1

            #### Log all results
            if (step + 1) % (
                config.gradient_accumulation_steps * config.logging_steps
            ) == 0:
                loss.detach()
                train_logs = {
                    "loss": (loss * config.gradient_accumulation_steps) / num_losses,
                    "constrast_loss_diff": (
                        teacher_outputs.contrastive_loss
                        - student_outputs.contrastive_loss
                    )
                    / num_losses,
                    "div_loss": (
                        teacher_outputs.contrastive_loss
                        - student_outputs.contrastive_loss
                    )
                    / num_losses,
                    "%_mask_idx": percent_masked,
                    "ppl": (
                        teacher_outputs.codevector_perplexity
                        - student_outputs.codevector_perplexity
                    ),
                    "lr": torch.tensor(optimizer.param_groups[0]["lr"]),
                }
                log_str = ""
                for k, v in train_logs.items():
                    log_str += "| {}: {:.3e}".format(k, v.item())
                logging.info(log_str)

            # TODO Early stopping

            #### Save model
            if (step + 1) % (
                config.gradient_accumulation_steps * config.saving_steps
            ) == 0:
                if (
                    config.push_to_hub and epoch < config.epochs - 1
                ) or save_dir is not None:
                    student_model.save_pretrained(save_dir)

                if config.push_to_hub and epoch < config.epochs - 1:
                    repo.push_to_hub(
                        commit_message=f"Training in progress step {completed_steps}",
                        blocking=False,
                        auto_lfs_prune=True,
                    )
                logging.info("Saved model...")

            #### Eval model
            if (step + 1) % (
                config.gradient_accumulation_steps * config.eval_steps
            ) == 0:
                evaluate(batch)

        # Always evaluate at end of epoch
        evaluate(batch)
        if config.output_dir is not None:
            student_model.save_pretrained(save_dir)

            if config.push_to_hub:
                repo.push_to_hub(
                    commit_message=f"End of epoch {epoch}",
                    auto_lfs_prune=True,
                )
            logging.info(f"Saved model, at end of epoch {epoch}...")


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    return total_params, table


if __name__ == "__main__":

    xlsr_300m_config = Config(
        dataset_id="google/fleurs",
        dataset_subset="da_dk",
        pretrained_teacher_model_id="facebook/wav2vec2-xls-r-300m",
        distilled_model_id="ajders/distilled_wav2vec2_xls_r_300m",
    )

    train(xlsr_300m_config)
