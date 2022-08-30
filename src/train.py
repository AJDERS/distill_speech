"""Student teacher model distilling script for Danish acoustic models"""
import os
import math
import torch
import hydra
import logging
from numpy import Inf
from tqdm import tqdm
from torch.optim import AdamW
from data import AudioDataset
from accelerate import Accelerator
from omegaconf import DictConfig
from math import log, ceil, floor
from prettytable import PrettyTable
import torch.nn.functional as functional
from huggingface_hub.repository import Repository
from transformers.utils import get_full_repo_name
from torch.utils.data.dataloader import DataLoader
from data_collator import DataCollatorForWav2Vec2Pretraining
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2ForPreTraining,
    get_scheduler,
    set_seed,
)

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def train(config: DictConfig, **kwargs):
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
    if torch.cuda.is_available():
        logging.info(f"Using {torch.cuda.device_count()} GPU(s)")

    # Initialize the accelerator.
    accelerator = Accelerator()

    #### Set training seed.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #### Create HF epository
    if accelerator.is_main_process:
        if config.training.output_dir is not None:
            save_dir = os.path.join(
                config.training.output_dir,
                f"{config.models.pretrained_teacher_model_id}_{config.data.dataset_id}",
            )
        if config.training.push_to_hub:
            if config.models.distilled_model_id is None:
                if config.models.hub_token != "None":
                    repo_name = get_full_repo_name(save_dir, token=config.models.hub_token)
            else:
                repo_name = config.models.distilled_model_id
            repo = Repository(save_dir, clone_from=repo_name)
        elif config.training.output_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    #### Load dataset
    dataset = AudioDataset(
        accelerator=accelerator,
        pretrained_teacher_model_id=config.models.pretrained_teacher_model_id,
        dataset_id=config.data.dataset_id,
        dataset_subset=config.data.dataset_subset,
        sampling_rate=config.data.sampling_rate,
        train_name=config.data.train_name,
        validation_name=config.data.validation_name,
        test_name=config.data.test_name,
        audio_column_name=config.data.audio_column_name,
        max_duration_in_seconds=config.data.max_duration_in_seconds,
        min_duration_in_seconds=config.data.min_duration_in_seconds,
        preprocessing_num_workers=config.data.preprocessing_num_workers,
        num_train=None if config.data.num_train == -1 else config.data.num_train,
        num_test=None if config.data.num_test == -1 else config.data.num_test,
        num_val=None if config.data.num_val == -1 else config.data.num_val,
        use_cached=config.data.use_cached,
    )
    vectorized_dataset = dataset.process()

    #### Load models, and update configs

    # Teacher model config, used to update student configs.
    teacher_model_config = Wav2Vec2Config.from_pretrained(
        config.models.pretrained_teacher_model_id,
    )
    # Student model config.
    student_model_config = Wav2Vec2Config.from_pretrained(
        config.models.pretrained_teacher_model_id
    )

    # Set student parameters not related to model distilling.
    student_model_config.activation_dropout = config.student.activation_dropout
    student_model_config.attention_dropout = config.student.attention_dropout
    student_model_config.hidden_dropout = config.student.hidden_dropout
    student_model_config.feat_proj_dropout = config.student.feat_proj_dropout
    student_model_config.final_dropout = config.student.final_dropout
    student_model_config.mask_time_prob = config.student.mask_time_prob
    student_model_config.mask_feature_prob = config.student.mask_feature_prob
    student_model_config.mask_feature_length = config.student.mask_feature_length
    student_model_config.layerdrop = config.student.layerdrop
    student_model_config.ctc_loss_reduction = config.student.ctc_loss_reduction

    # Set parameters related to model distilling.
    # Check if `distill_factor` is positive, if so divide all model "height" parameters by `distill_factor`
    # if not use parameters from config.
    if config.student.distill_factor > 0:
        # Hidden layers
        student_model_config.num_hidden_layers = round(
            teacher_model_config.num_hidden_layers / config.student.distill_factor
        )

        # Attention heads
        student_model_config.num_attention_heads = round(
            teacher_model_config.num_attention_heads / config.student.distill_factor
        )

        # Convolutional positional embedding groups
        student_model_config.num_conv_pos_embedding_groups = round(
            teacher_model_config.num_conv_pos_embedding_groups
            / config.student.distill_factor
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

    else:
        student_model_config.num_hidden_layers = config.student.num_hidden_layers
        student_model_config.num_attention_heads = config.student.num_attention_heads
        student_model_config.num_conv_pos_embedding_groups = (
            config.student.num_conv_pos_embedding_groups
        )

    # Initialize models. Teacher model is initialized from a pretrained model.
    teacher_model = Wav2Vec2ForPreTraining.from_pretrained(
        config.models.pretrained_teacher_model_id
    ).to(accelerator.device)
    teacher_model.config.activation_dropout = config.teacher.activation_dropout
    teacher_model.config.attention_dropout = config.teacher.attention_dropout
    teacher_model.config.hidden_dropout = config.teacher.hidden_dropout
    teacher_model.config.feat_proj_dropout = config.teacher.feat_proj_dropout
    teacher_model.config.final_dropout = config.teacher.final_dropout
    teacher_model.config.mask_time_prob = config.teacher.mask_time_prob
    teacher_model.config.mask_feature_prob = config.teacher.mask_feature_prob
    teacher_model.config.mask_feature_length = config.teacher.mask_feature_length
    teacher_model.config.layerdrop = config.teacher.layerdrop
    teacher_model.config.ctc_loss_reduction = config.teacher.ctc_loss_reduction

    # Student model is initialized with a random model, using the parameters defined above.
    student_model = Wav2Vec2ForPreTraining(student_model_config)
    t_total_params, t_table = count_parameters(teacher_model)
    s_total_params, s_table = count_parameters(student_model)
    logging.info(f"  Teacher summary: \n {t_table}")
    logging.info(f"  Student summary: \n {s_table}")
    logging.info(
        f"  Parameters ratio (student/teacher): {round((s_total_params/t_total_params), 2)}."
    )

    # Activate gradient checkpointing if enabled
    if config.training.gradient_checkpointing:
        student_model.gradient_checkpointing_enable()

    #### Define data collator, loaders, optimizer, loss function and scheduler
    data_collator = DataCollatorForWav2Vec2Pretraining(
        model=teacher_model, feature_extractor=dataset.feature_extractor, device=accelerator.device
    )

    train_dataloader = DataLoader(
        vectorized_dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=config.training.batch_size,
    )
    eval_dataloader = DataLoader(
        vectorized_dataset["validation"],
        collate_fn=data_collator,
        batch_size=config.training.batch_size,
    )

    # Optimizer
    optimizer = AdamW(
        list(student_model.parameters()),
        lr=config.training.learning_rate,
        betas=[config.training.adam_beta1, config.training.adam_beta2],
        eps=config.training.adam_epsilon,
    )

    # Prepare everything with our `accelerator`.
    student_model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        student_model, optimizer, train_dataloader, eval_dataloader
    )

    # Loss function
    KD_loss = torch.nn.KLDivLoss(reduction="batchmean")

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.training.gradient_accumulation_steps
    )
    max_train_steps = config.training.epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=config.training.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=max_train_steps,
    )

    #### Train
    total_batch_size = config.training.batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps


    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {len(vectorized_dataset['train'])}")
    logging.info(f"  Num Epochs = {config.training.epochs}")
    logging.info(
        f"  Instantaneous batch size per device = {config.training.batch_size}"
    )
    logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logging.info(
        f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}"
    )
    logging.info(f"  Total optimization steps = {max_train_steps}")

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    patience = 1
    current_best_val_logs = {
        "val_loss": Inf,
        "val_contrastive_loss_diff": Inf,
        "val_diversity_loss_diff": Inf,
        "val_num_losses": Inf,
    }

    ###### Evaluate ######
    ######################
    ######################
    ######################

    def evaluate(batch, current_best_val_logs, patience):
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
                student_outputs = student_model(**batch)
            teacher_projected_states = batch.projected_states
            student_projected_states = student_outputs.projected_states
            loss = KD_loss(
                input=functional.log_softmax(
                    student_projected_states / config.training.softmax_temperature,
                    dim=-1,
                ),
                target=functional.softmax(
                    teacher_projected_states / config.training.softmax_temperature,
                    dim=-1,
                ),
            )
            val_logs["val_loss"] += loss
            val_logs["val_contrastive_loss_diff"] += (
                teacher_contrastive_loss - student_outputs.contrastive_loss
            )
            val_logs["val_diversity_loss_diff"] += (
                teacher_diversity_loss - student_outputs.diversity_loss
            )
            val_logs["val_num_losses"] += batch["mask_time_indices"].sum()

        # sum over devices in multi-processing
        if accelerator.num_processes > 1:
            val_logs = {k: accelerator.gather(v).sum() for k, v in val_logs.items()}

        # Make validation logs, and write to log.
        val_logs = {k: v / val_logs["val_num_losses"] for k, v in val_logs.items()}

        log_str = ""
        for k, v in val_logs.items():
            log_str += "| {}: {:.3e}".format(k, v.item())
        logging.info(log_str)

        if accelerator.is_local_main_process:
            progress_bar.write(log_str)

        # Early stopping
        early_stop = False
        if config.training.early_stopping:
            if val_logs["val_loss"] > current_best_val_logs["val_loss"]:
                patience += 1
                if patience >= config.training.early_stopping_patience:
                    early_stop = True
                    return current_best_val_logs, early_stop, patience
            else:
                patience = 0
        return val_logs, early_stop, patience
    
    ######################
    ######################
    ######################

    #### Training loop
    for epoch in range(starting_epoch, config.training.epochs):
        student_model.train()
        
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
            batch.pop("projected_quantized_states", None)
            batch.pop("hidden_states", None)
            teacher_projected_states = batch.pop("projected_states", None)
            teacher_contrastive_loss = batch.pop("contrastive_loss", None)
            teacher_codevector_perplexity = batch.pop("codevector_perplexity", None)
            teacher_diversity_loss = batch.pop("diversity_loss", None)

            
            student_outputs = student_model(**batch)
            student_projected_states = student_outputs.projected_states

            # TODO Think about wether to use `projected_states`, `projected_quantized_states`, or
            # `hidden_states`

            #### Calculate (Kullback-Leibler divergence) loss
            loss = KD_loss(
                input=functional.log_softmax(
                    student_projected_states / config.training.softmax_temperature,
                    dim=-1,
                ),
                target=functional.softmax(
                    teacher_projected_states / config.training.softmax_temperature,
                    dim=-1,
                ),
            )
            # divide loss by gradient accumulation steps since gradients
            # are accumulated for multiple backward passes in PyTorch
            loss = loss / config.training.gradient_accumulation_steps
            accelerator.backward(loss)

            # make sure that `num_losses` is summed for distributed training
            # and average gradients over losses of all devices
            if accelerator.state.num_processes > 1:
                num_losses = accelerator.gather(num_losses).sum()
                gradient_multiplier = accelerator.state.num_processes / num_losses
                multiply_grads(student_model.module.parameters(), gradient_multiplier)
            else:
                multiply_grads(student_model.parameters(), 1 / num_losses)

            # update parameters
            if (
                step + 1
            ) % config.training.gradient_accumulation_steps == 0 or step == len(
                train_dataloader
            ) - 1:

                # compute grad norm for monitoring
                scale = (
                    accelerator.scaler._scale.item()
                    if hasattr(accelerator, "scaler") and accelerator.scaler is not None
                    else 1
                )
                if accelerator.state.num_processes > 1:
                    grad_norm = get_grad_norm(student_model.module.parameters(), scale)
                else:
                    grad_norm = get_grad_norm(student_model.parameters(), scale)

                if not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()
                elif accelerator.is_local_main_process:
                    progress_bar.write(
                        f"Gradients have overflown - skipping update step... Updating gradient scale to {scale}..."
                    )

                progress_bar.update(1)
                completed_steps += 1
                optimizer.step()
                optimizer.zero_grad()

            #### Log all results
            if (step + 1) % (
                config.training.gradient_accumulation_steps
                * config.training.logging_steps
            ) == 0:
                loss.detach()

                if accelerator.state.num_processes > 1:
                    loss = accelerator.gather(loss).sum()
                    teacher_contrastive_loss = accelerator.gather(teacher_contrastive_loss).sum()
                    student_outputs.diversity_loss = accelerator.gather(student_outputs.diversity_loss).sum()
                    percent_masked = accelerator.gather(percent_masked).sum()


                train_logs = {
                    "loss": (loss * config.training.gradient_accumulation_steps)
                    / num_losses,
                    "constrast_loss_diff": (
                        teacher_contrastive_loss
                        - student_outputs.contrastive_loss
                    )
                    / num_losses,
                    "div_loss": (
                        teacher_contrastive_loss
                        - student_outputs.contrastive_loss
                    )
                    / num_losses,
                    "%_mask_idx": percent_masked,
                    "ppl": (
                        teacher_codevector_perplexity
                        - student_outputs.codevector_perplexity
                    ),
                    "lr": torch.tensor(optimizer.param_groups[0]["lr"]),
                    "grad_norm": torch.tensor(grad_norm),
                }
                log_str = ""
                for k, v in train_logs.items():
                    log_str += "| {}: {:.3e}".format(k, v.item())
                logging.info(log_str)

                if accelerator.is_local_main_process:
                    progress_bar.write(log_str)

            #### Eval model
            if (step + 1) % (
                config.training.gradient_accumulation_steps * config.training.eval_steps
            ) == 0:
                current_best_val_logs, early_stop, patience = evaluate(
                    batch, current_best_val_logs, patience
                )
                accelerator.wait_for_everyone()
                unwrapped_student_model = accelerator.unwrap_model(student_model)
                # Saving new best model
                if patience == 0 and not early_stop:
                    if (
                        config.training.push_to_hub
                        and epoch < config.training.epochs - 1
                    ) or save_dir is not None:
                        unwrapped_student_model.save_pretrained(
                            save_dir,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save
                        )
                        unwrapped_student_model.save_pretrained(save_dir)

                    if (
                        config.training.push_to_hub
                        and epoch < config.training.epochs - 1
                    ) and accelerator.is_main_process:
                        repo.push_to_hub(
                            commit_message=f"Training in progress step {completed_steps}. New best model.",
                            blocking=False,
                            auto_lfs_prune=True,
                        )
                    logging.info("Saved new best model...")
                if early_stop:
                    logging.info(
                        "Stopping training, early stopping patience ran out..."
                    )
                    break

        # Always evaluate at end of epoch
        current_best_val_logs, early_stop, patience = evaluate(
            batch, current_best_val_logs, patience
        )
        if config.training.output_dir is not None:
            student_model.save_pretrained(save_dir)

            if config.training.push_to_hub:
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

def multiply_grads(params, c):
    """Multiplies grads by a constant *c*."""
    for p in params:
        if p.grad is not None:
            if torch.is_tensor(c):
                c = c.to(p.grad.device)
            p.grad.data.mul_(c)

def get_grad_norm(params, scale=1):
    """Compute grad norm given a gradient scale."""
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = (p.grad.detach().data / scale).norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm

if __name__ == "__main__":
    train()
