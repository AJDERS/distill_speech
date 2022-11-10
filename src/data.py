import os
import pandas as pd
import logging
from typing import Optional, Tuple
from datasets import (
    load_dataset as hugging_load_dataset,
    Dataset,
    DatasetDict,
    IterableDatasetDict
)
from pathlib import Path
from datasets.features import Audio
from transformers import Wav2Vec2FeatureExtractor
from accelerate import Accelerator

logging.basicConfig(level=logging.INFO)

class AudioDataset:
    """A dataset containing audio data.
    Args:
        accelerator (Accelerator):
            The accelerator to use.
        dataset_id (str, optional):
            The HF dataset id. Defaults to
            'mozilla-foundation/common_voice_8_0'.
        dataset_subset (str, optional):
            The HF dataset subset. Defaults to 'da'.
        sampling_rate (int, optional):
            The sampling rate of the audio data. Defaults to 16_000.
        train_name (str, optional):
            The name of the train split. Defaults to 'train'.
        validation_name (str or None, optional):
            The name of the validation split. If None then the validation set
            is created from the train split. Defaults to 'validation'.
        test_name (str or None, optional):
            The name of the test split. If None then the test set is created
            from the validation or train split. Defaults to 'test'.
        audio_column_name (str, optional):
            Name of the column in the data containing the audio data. Defaults to 'audio'
        max_duration_in_seconds (int):
            Maximum duration of audio clips in seconds. Defaults to 25.
        min_duration_in_seconds (int)
            Minimum duration of audio clips in seconds. Defaults to 0.
        preprocessing_num_workers (int)
            Number of workers used when preprocessing. Defaults to -1, indicating number of cpu cores.
        num_train (int):
            The number of entries to use for training.
        num_test (int):
            The number of entries to use for test.
        num_val (int):
            The number of entries to use for validation.
        use_cached (bool):
            Wether to use cached data or not.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        load_local: bool,
        metadata_path: Path,
        pretrained_teacher_model_id: str,
        dataset_id: str = "google/fleurs",
        dataset_subset: Optional[str] = "da_dk",
        sampling_rate: int = 16_000,
        train_name: str = "train",
        validation_name: Optional[str] = "validation",
        test_name: Optional[str] = "test",
        audio_column_name: Optional[str] = "audio",
        max_duration_in_seconds: int = 25,
        min_duration_in_seconds: int = 0,
        preprocessing_num_workers: int = -1,
        num_train: int = None,
        num_test: int = None,
        num_val: int = None,
        use_cached: bool = False,
    ):
        self.accelerator = accelerator
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            pretrained_teacher_model_id
        )
        self.sampling_rate = sampling_rate
        self.train_name = train_name
        self.validation_name = validation_name
        self.test_name = test_name
        self.num_train = num_train
        self.num_test = num_test
        self.num_val = num_val
        self.audio_column_name = audio_column_name
        self.max_duration_in_seconds = max_duration_in_seconds
        self.min_duration_in_seconds = min_duration_in_seconds
        self.preprocessing_num_workers = preprocessing_num_workers
        self.use_cached = use_cached

        # Load the dataset
        if not load_local:
            self._load_dataset()
        else:
            self._load_local_dataset(metadata_path=metadata_path)

    def _load_dataset_split(
        self,
        dataset_id: str,
        name: Optional[str] = None,
        split: str = "train",
        use_auth_token: bool = True,
    ) -> Dataset:
        """Load a dataset split.
        Args:
            dataset_id (str):
                The HF dataset id.
            name (str or None, optional):
                The name of the dataset split. If None then the dataset split
                is created from the train split. Defaults to None.
            split (str, optional):
                The HF dataset split. Defaults to 'train'.
            use_auth_token (bool, optional):
                Whether to use the auth token. Defaults to True.
        Returns:
            Dataset:
                The loaded dataset split.
        """
        try:
            return hugging_load_dataset(
                path=dataset_id, name=name, split=split, use_auth_token=use_auth_token
            )
        except ValueError:
            return DatasetDict.load_from_disk(dataset_id)[split]

    def _load_local_dataset(
        self, metadata_path: Path, stream: bool = True
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Loads a dataset.
        Returns:
            tuple:
                A triple (train, val, test), containing the three splits of the
                dataset.
        """
        # Create paths
        train_data_paths_parquet = metadata_path / Path("train.parquet")
        test_data_paths_parquet = metadata_path / Path("train_backup.parquet")
        val_data_paths_parquet = metadata_path / Path("valid.parquet")

        # Load DFs
        if self.accelerator.is_main_process:
            logging.info(f"Loading metadata parquets.")
        train_metadata = pd.read_parquet(train_data_paths_parquet)
        test_metadata = pd.read_parquet(test_data_paths_parquet)
        val_metadata = pd.read_parquet(val_data_paths_parquet)

        # Get paths from DFs
        train_paths = (
            train_metadata["/work/data/p1-r24syv-segmented/"]
            .apply(lambda x: "/home/ucloud/data/" + x.split(".wav")[0] + ".wav")
            .values.tolist()
        )
        test_paths = (
            test_metadata["/work/data/p1-r24syv-segmented/"]
            .apply(lambda x: "/home/ucloud/data/" + x.split(".wav")[0] + ".wav")
            .values.tolist()
        )
        val_paths = (
            val_metadata["/work/data/p1-r24syv-segmented/"]
            .apply(lambda x: "/home/ucloud/data/" + x.split(".wav")[0] + ".wav")
            .values.tolist()
        )

        # Check if files exists in paths
        train_paths = [fname for fname in train_paths if os.path.isfile(fname)]
        test_paths = [fname for fname in test_paths if os.path.isfile(fname)]
        val_paths = [fname for fname in val_paths if os.path.isfile(fname)]

        self.len_train_data = len(train_paths)

        # Load splits
        if self.accelerator.is_main_process:
            logging.info(f"Creating training dataset from paths.")
        train_dataset = hugging_load_dataset(
            "audiofolder",
            data_files=train_paths,
            streaming=stream,
            drop_metadata=True,
            drop_labels=True,
        )
        if self.accelerator.is_main_process:
            logging.info(f"Creating test dataset from paths.")
        test_dataset = hugging_load_dataset(
            "audiofolder",
            data_files=test_paths,
            streaming=stream,
            drop_metadata=True,
            drop_labels=True,
        )
        if self.accelerator.is_main_process:
            logging.info(f"Creating validation dataset from paths.")
        val_dataset = hugging_load_dataset(
            "audiofolder",
            data_files=val_paths,
            streaming=stream,
            drop_metadata=True,
            drop_labels=True,
        )
        self.raw_datasets = IterableDatasetDict()
        self.raw_datasets["train"] = train_dataset["train"]
        self.raw_datasets["test"] = test_dataset["train"]
        self.raw_datasets["validation"] = val_dataset["train"]

    def _load_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Loads a dataset.
        Returns:
            tuple:
                A triple (train, val, test), containing the three splits of the
                dataset.
        """
        # Load train dataset
        train = self._load_dataset_split(
            dataset_id=self.dataset_id, name=self.dataset_subset, split=self.train_name
        )

        # Load validation and test datasets. If both `validation_name` and
        # `test_name` are not None then these are simply loaded. If only
        # `test_name` is not None then a validation set is created from the
        # train dataset.
        if self.test_name is not None:
            test = self._load_dataset_split(
                dataset_id=self.dataset_id,
                name=self.dataset_subset,
                split=self.test_name,
            )
            if self.validation_name is not None:
                val = self._load_dataset_split(
                    dataset_id=self.dataset_id,
                    name=self.dataset_subset,
                    split=self.validation_name,
                )
            else:
                split_dict = train.train_test_split(test_size=0.1, seed=703)
                train = split_dict["train"]
                val = split_dict["test"]

        # If only `validation_name` is not None then the validation set is used
        # as a test set and a new validation set is created from the train
        # dataset.
        elif self.validation_name is not None:
            test = self._load_dataset_split(
                dataset_id=self.dataset_id,
                name=self.dataset_subset,
                split=self.validation_name,
            )
            split_dict = train.train_test_split(test_size=0.1, seed=703)
            train = split_dict["train"]
            val = split_dict["test"]

        # If both `validation_name` and `test_name` are None then validation
        # and test sets are created from the train dataset.
        else:
            # Split train dataset into train and a combined validation and test
            # set
            split_dict = train.train_test_split(test_size=0.2, seed=703)
            train = split_dict["train"]
            val_test = split_dict["test"]

            # Create validation set from the combined validation and test set
            split_dict = val_test.train_test_split(test_size=0.5, seed=703)
            val = split_dict["train"]
            test = split_dict["test"]

        self.raw_datasets = DatasetDict()
        if self.num_train is not None:
            self.raw_datasets["train"] = Dataset.from_dict(train[: self.num_train])
        else:
            self.raw_datasets["train"] = train

        if self.num_test is not None:
            self.raw_datasets["test"] = Dataset.from_dict(test[: self.num_test])
        else:
            self.raw_datasets["test"] = test

        if self.num_val is not None:
            self.raw_datasets["validation"] = Dataset.from_dict(train[: self.num_val])
        else:
            self.raw_datasets["validation"] = val

    def process(self) -> Dataset:
        """Preprocesses audio data. Casts audio column as `Audio`-tensor and down samples to
        correct sampling rate. Remove to short audio-files and cut too long.

        Returns:
            Dataset:
                The preprocessed dataset splits.
        """

        def prepare_batch(batch):
            sample = batch[self.audio_column_name]

            inputs = self.feature_extractor(
                sample["array"],
                sampling_rate=sample["sampling_rate"],
                max_length=max_length,
                truncation=True,
            )
            batch["input_values"] = inputs.input_values[0]
            batch["input_length"] = len(inputs.input_values[0])

            return batch

        self.raw_datasets = self.raw_datasets.cast_column(
            self.audio_column_name,
            Audio(sampling_rate=self.feature_extractor.sampling_rate),
        )
        # set max & min audio length in number of samples
        max_length = int(
            self.max_duration_in_seconds * self.feature_extractor.sampling_rate
        )
        min_length = int(
            self.min_duration_in_seconds * self.feature_extractor.sampling_rate
        )

        if self.preprocessing_num_workers < 0:
            import multiprocessing

            self.preprocessing_num_workers = multiprocessing.cpu_count()
        with self.accelerator.main_process_first():
            vectorized_datasets = self.raw_datasets.map(prepare_batch)

            if min_length > 0:
                vectorized_datasets = vectorized_datasets.filter(
                    lambda x: x > min_length,
                    num_proc=self.preprocessing_num_workers,
                    input_columns=["input_length"],
                )

            vectorized_datasets = vectorized_datasets.remove_columns("input_length")
        return vectorized_datasets
