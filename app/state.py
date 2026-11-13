"""
FootGuard AI - Streamlit Cached Service Registry
Caches stateless/read-only service instances and inference engines via @st.cache_resource.
Mutable training state is never cached.
"""

import sys
from pathlib import Path
import streamlit as st

ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from services.config_service import ConfigService
from services.dataset_service import DatasetService
from services.validation_service import ValidationService
from services.model_service import ModelService
from services.inference_service import InferenceService
from services.evaluation_service import EvaluationService
from services.experiment_service import ExperimentService
from services.training_service import TrainingService
from services.synthetic_generator_service import SyntheticGeneratorService
from services.synthetic_training_service import SyntheticTrainingService
from services.ingestion_service import DatasetIngestionService


@st.cache_resource
def get_config_service() -> ConfigService:
    return ConfigService()


@st.cache_resource
def get_dataset_service() -> DatasetService:
    return DatasetService(get_config_service())


@st.cache_resource
def get_validation_service() -> ValidationService:
    return ValidationService(get_dataset_service())


@st.cache_resource
def get_model_service() -> ModelService:
    return ModelService(get_config_service())


@st.cache_resource
def get_inference_service() -> InferenceService:
    return InferenceService(get_config_service())


@st.cache_resource
def get_evaluation_service() -> EvaluationService:
    return EvaluationService()


@st.cache_resource
def get_experiment_service() -> ExperimentService:
    return ExperimentService(get_config_service())


def create_training_service() -> TrainingService:
    """Instantiates a fresh TrainingService per run (never cached)."""
    return TrainingService(
        config_service=get_config_service(),
        dataset_service=get_dataset_service(),
        validation_service=get_validation_service(),
        model_service=get_model_service(),
        experiment_service=get_experiment_service(),
    )


def create_synthetic_generator_service() -> SyntheticGeneratorService:
    return SyntheticGeneratorService(get_config_service())


def create_synthetic_training_service() -> SyntheticTrainingService:
    return SyntheticTrainingService(
        config_service=get_config_service(),
        generator_service=create_synthetic_generator_service(),
        experiment_service=get_experiment_service(),
    )


def create_ingestion_service() -> DatasetIngestionService:
    return DatasetIngestionService(get_config_service())

