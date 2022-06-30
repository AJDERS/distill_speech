import torch
from dataclasses import dataclass
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices,
    _sample_negative_indices,
)
from transformers import Wav2Vec2ForPreTraining, Wav2Vec2FeatureExtractor
from typing import Dict, List, Optional, Union


@dataclass
class DataCollatorForWav2Vec2Pretraining:
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices
    for self-supervised pretraining.
    Args:
        model (Wav2Vec2ForPreTraining):
            The Wav2Vec2 model used for pretraining.
        feature_extractor (Wav2Vec2FeatureExtractor):
            The processor used for proccessing the data.
        padding (PaddingStrategy, optional, defaults to True):
            Padding strategy
        max_length (int, optional):
            Maximum length of the input_values of the returned list and optionally padding length.
        pad_to_multiple_of (int, optional):
            If set will pad the sequence to a multiple of the provided value.
    """

    model: Wav2Vec2ForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    device: Optional[str] = "cpu"

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # reformat list to dict and set to pytorch format
        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch = batch.to(self.device)
        batch_size = batch["input_values"].shape[0]

        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(
            batch["input_values"].shape[-1]
        )
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length)

        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        features_shape = (batch_size, mask_indices_seq_length)

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.model.config.mask_time_prob,
            self.model.config.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )

        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        batch["mask_time_indices"] = torch.tensor(
            mask_time_indices, dtype=torch.long, device=self.device
        )
        batch["sampled_negative_indices"] = torch.tensor(
            sampled_negative_indices, dtype=torch.long, device=self.device
        )

        return batch
