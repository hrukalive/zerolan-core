"""
More details about the model:
    https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2
"""
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from zerolan.data.pipeline.defense import DefenseQuery, DefenseConversation, DefenseResult, DefensePrediction
from zerolan.data.pipeline.llm import LLMQuery

from common.abs_model import AbstractModel
from common.decorator import log_model_loading
from defense.deberta.config import DebertaPromptDefenseModelConfig


class DebertaPromptDefenseModel(AbstractModel):
    """
    Input-Level Defense Model
    """

    def __init__(self, config: DebertaPromptDefenseModelConfig):
        super().__init__()
        self.model_id = "protectai/deberta-v3-base-prompt-injection-v2"
        self._model_path = config.model_path
        self._device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
        self._max_length = config.max_length

        self._tokenizer: any = None
        self._model: any = None

    @log_model_loading("protectai/deberta-v3-base-prompt-injection-v2")
    def load_model(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_path).eval()

    def predict(self, query: DefenseQuery) -> DefensePrediction:
        classifier = pipeline(
            "text-classification",
            model=self._model,
            tokenizer=self._tokenizer,
            max_length=self._max_length,
            truncation=True,
            device=self._device
        )

        # Output format:
        # [{'label': 'INJECTION', 'score': 0.99998}]
        # [{'label': 'SAFE', 'score': 0.93331}]

        output: List = classifier(query.text)
        # logger.debug(f'Defense Model Output: {output}')

        return self.to_pipeline_format(output[0], query)

    def stream_predict(self, llm_query: LLMQuery):
        raise NotImplementedError()

    @staticmethod
    def to_pipeline_format(output: dict, query: DefenseQuery):
        label: str = output['label']
        if label.lower() == 'safe':
            defense_result = DefenseResult.safe
        elif label.lower() == 'injection':
            defense_result = DefenseResult.injection
        else:
            raise ValueError(f"Unknown label {label}.")
        score = float(output['score'])

        dc = DefenseConversation(role=query.role, content=query.text, defense_result=defense_result, confidence=score)
        query.history.append(dc)

        return DefensePrediction(defense_result=defense_result, confidence=score, history=query.history)
