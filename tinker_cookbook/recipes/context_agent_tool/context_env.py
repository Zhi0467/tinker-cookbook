
import logging
import os
import random
import string
import re
from functools import partial, reduce
from pathlib import Path
from typing import Literal, Sequence, TypedDict, cast

import chz
import pandas as pd
import tinker
from huggingface_hub import hf_hub_download
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Action,
    EnvGroupBuilder,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.recipes.context_agent_tool.tools import ContextToolClient

logger = logging.getLogger(__name__)

def normalize_answer(s: str) -> str:
    """Normalize answer by lowercasing, removing punctuation, articles, and fixing whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    # Apply transformations in order using reduce
    transformations = [lower, remove_punc, remove_articles, white_space_fix]
    return reduce(lambda text, func: func(text), transformations, s)

# This is the new System Prompt for your agent.
# It uses the <function_call> syntax, which the Qwen3Renderer parses (from renderers.py)
CONTEXT_AGENT_SYSTEM_PROMPT = """
You are an expert agent that solves tasks.
You have the ability to modify your own context (conversation history) to save space or correct mistakes.

Tool calling. Execute the tool by wrapping calls in <function_call>...</function_call>

Here is the tool you are given:
```
{
    "name": "delete",
    "title": "Delete Context String",
    "description": "Deletes the first exact match of a specific string from the conversation history. Use this to remove redundant, incorrect, or unnecessary information.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "string_to_delete": {
                "type": "string",
                "description": "The exact string to be deleted from the context. The match is case-sensitive and includes all whitespace.",
            }
        },
        "required": ["string_to_delete"],
    },
}
```

Here are your instructions:
1.  Think step by step about the problem.
2.  If your context becomes cluttered with useless information, you can use the `delete` tool to remove it.
3.  When you have the final answer, you MUST output it in the format:
    `Answer: [your answer]`
"""

class ContextEnv(ProblemEnv):
    """
    This class is the "executor" for the context-managing agent.
    
    It manages the conversation history, parses the model's 'action',
    and executes the 'delete' tool by directly modifying the history.
    """

    def __init__(
        self,
        problem: str,
        answer: list[str], # The ground-truth answer for grading
        tool_client: ContextToolClient,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        max_trajectory_tokens: int = 32 * 1024,
        max_num_calls: int = 10, # Max tool calls before quitting
    ):
        super().__init__(renderer, convo_prefix)
        self.problem: str = problem
        self.answer: list[str] = answer
        self.tool_client: ContextToolClient = tool_client
        self.max_trajectory_tokens: int = max_trajectory_tokens
        
        # This is the agent's "memory" or "state"
        self.past_messages: list[renderers.Message] = convo_prefix.copy() if convo_prefix else []
        
        self.current_num_calls: int = 0
        self.max_num_calls: int = max_num_calls
        
    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Called once at the start of an episode."""
        convo = self.convo_prefix + [
            {"role": "user", "content": self.get_question()},
        ]
        self.past_messages = convo.copy()
        return self.renderer.build_generation_prompt(convo), self.stop_condition

    def get_question(self) -> str:
        return self.problem

    def _extract_answer(self, sample_str: str) -> str | None:
        """Extracts the answer from the model's final response."""
        if "Answer:" not in sample_str:
            return None
        message_pars = sample_str.split("Answer:")
        if len(message_pars) != 2:
            return None
        return message_pars[1].strip()

    def check_format(self, sample_str: str) -> bool:
        return self._extract_answer(sample_str) is not None

    def check_answer(self, sample_str: str) -> bool:
        """Grades the model's extracted answer against the ground truth."""
        model_answer = self._extract_answer(sample_str)
        if model_answer is None or len(self.answer) == 0:
            return False

        normalized_model_answer = normalize_answer(model_answer)
        for gold_answer in self.answer:
            if normalized_model_answer == normalize_answer(gold_answer):
                return True
        return False

    def get_reference_answer(self) -> str:
        """Return the reference answer for logging purposes."""
        return " OR ".join(self.answer) if self.answer else "N/A"

    async def step(self, action: Action) -> StepResult:
        """
        tool execution logic implemented here,
        since the execution modifies the context directly,
        it can only be implemented here to modify the states
        It runs once *after* the model generates a complete 'action'.
        """
        message, parse_success = self.renderer.parse_response(action)

        self.past_messages.append(message)

        is_final_answer = self.check_format(message["content"])
        
        # --- CASE 1: MODEL HAS FINISHED ---
        # reward is format violation punishment + binary answer reward
        if is_final_answer:
            correct_answer = float(self.check_answer(message["content"]))
            correct_format = float(parse_success) and float(self.check_format(message["content"]))
            total_reward = self.format_coef * (correct_format - 1) + correct_answer
            return StepResult(
                reward=total_reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "correct": correct_answer,
                    "format": correct_format,
                },
            )

        # --- CASE 2: MODEL CALLED A TOOL ---
        if "tool_calls" in message:
            if self.current_num_calls >= self.max_num_calls:
                # Force-stop if the agent is in a tool-use loop
                return StepResult(
                    reward=0.0, # Failed the task
                    episode_done=True,
                    next_observation=tinker.ModelInput.empty(),
                    next_stop_condition=self.stop_condition,
                )

            tool_call = message["tool_calls"][0]
            
            # --- execution logic of context manage tools---
            if tool_call["name"] == "delete":
                self.current_num_calls += 1
                string_to_delete = tool_call["args"].get("string_to_delete")

                # This just validates args and returns a Message object
                tool_return_messages = await self.tool_client.invoke(tool_call)

                found_and_deleted = False
                if string_to_delete:
                    # Iterate backward through history (skip the current action)
                    for msg in reversed(self.past_messages[:-1]):
                        if string_to_delete in msg["content"]:
                            # Perform the replacement (only once)
                            msg["content"] = msg["content"].replace(string_to_delete, "", 1)
                            found_and_deleted = True
                            logger.info(f"ContextEnv: Deleted '{string_to_delete}'")
                            break
                
                # Update the tool message with the execution result
                if found_and_deleted:
                    tool_return_messages[0]["content"] += " (Success: String found and deleted.)"
                else:
                    tool_return_messages[0]["content"] += " (Warning: String not found in history.)"

                # Add the tool's result to history
                self.past_messages.extend(tool_return_messages)

            else:
                # Handle unknown tool call
                self.past_messages.append(
                    {"role": "tool", "content": f"Error: Unknown tool {tool_call['name']}"}
                )

            # --- Prepare for the next model turn ---
            next_observation = self.renderer.build_generation_prompt(self.past_messages)
            
            if next_observation.length > self.max_trajectory_tokens:
                # History is too long, end episode
                return StepResult(
                    reward=0.0, episode_done=True, 
                    next_observation=tinker.ModelInput.empty(), 
                    next_stop_condition=self.stop_condition
                )
                
            return StepResult(
                reward=0.0, # No reward for intermediate steps
                episode_done=False, # Continue the episode
                next_observation=next_observation,
                next_stop_condition=self.stop_condition,
            )

        # --- CASE 3: INTERMEDIATE THOUGHT ---
        # No tool call, no final answer. The model is "thinking".
        # Its thought is already in 'past_messages'.
        # We just continue the loop.
        return StepResult(
            reward=0.0,
            episode_done=False,
            next_observation=self.renderer.build_generation_prompt(self.past_messages),
            next_stop_condition=self.stop_condition,
        )

    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        """Returns the system prompt for the agent."""
        return [
            {
                "role": "system",
                "content": CONTEXT_AGENT_SYSTEM_PROMPT,
            },
        ]

class SearchR1Datum(TypedDict):
    question: str
    answer: list[str]
    data_source: str


def process_single_row(row_series: pd.Series) -> SearchR1Datum:
    """
    Process a single row of data for SearchR1-like format.

    Args:
        row: DataFrame row containing the original data
        current_split_name: Name of the current split (train/test)
        row_index: Index of the row in the DataFrame

    Returns:
        pd.Series: Processed row data in the required format
    """
    import numpy as np

    row = row_series.to_dict()
    question: str = row.get("question", "")

    # Extract ground truth from reward_model or fallback to golden_answers
    reward_model_data = row.get("reward_model")
    if isinstance(reward_model_data, dict) and "ground_truth" in reward_model_data:
        ground_truth = reward_model_data.get("ground_truth")
    else:
        ground_truth = row.get("golden_answers", [])

    # NOTE(tianyi)
    # I hate datasets with mixed types but it is what it is.
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth["target"]
    if isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.tolist()

    assert isinstance(ground_truth, list)
    for item in ground_truth:
        assert isinstance(item, str)
    ground_truth = cast(list[str], ground_truth)
    return {
        "question": question,
        "answer": ground_truth,
        "data_source": row["data_source"],
    }


def download_search_r1_dataset(split: Literal["train", "test"]) -> list[SearchR1Datum]:
    hf_repo_id: str = "PeterJinGo/nq_hotpotqa_train"
    parquet_filename: str = f"{split}.parquet"
    # TODO(tianyi): make download dir configurable for release
    user = os.getenv("USER", "unknown")
    assert user is not None
    tmp_download_dir = Path("/tmp") / user / "data" / hf_repo_id / split
    tmp_download_dir.mkdir(parents=True, exist_ok=True)

    hf_repo_id: str = "PeterJinGo/nq_hotpotqa_train"
    parquet_filename: str = f"{split}.parquet"

    local_parquet_filepath = hf_hub_download(
        repo_id=hf_repo_id,
        filename=parquet_filename,
        repo_type="dataset",
        local_dir=tmp_download_dir,
        local_dir_use_symlinks=False,
    )

    df_raw = pd.read_parquet(local_parquet_filepath)

    return df_raw.apply(process_single_row, axis=1).tolist()


class SearchR1Dataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        # tool args
        context_tool_client: ContextToolClient,
        # optional args
        convo_prefix: list[renderers.Message] | None = None,
        seed: int = 0,
        split: Literal["train", "test"] = "train",
        subset_size: int | None = None,
        max_trajectory_tokens: int = 32 * 1024,
    ):
        self.batch_size: int = batch_size
        self.group_size: int = group_size
        self.max_trajectory_tokens: int = max_trajectory_tokens
        self.renderer: renderers.Renderer = renderer
        self.convo_prefix: list[renderers.Message] | None = convo_prefix
        self.context_tool_client: ContextToolClient = context_tool_client
        self.seed: int = seed
        self.split: Literal["train", "test"] = split
        self.ds: list[SearchR1Datum] = download_search_r1_dataset(split)
        # shuffle with seed
        rng = random.Random(self.seed)
        rng.shuffle(self.ds)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return [
            self._make_env_group_builder(row, self.group_size)
            for row in self.ds[index * self.batch_size : (index + 1) * self.batch_size]
        ]

    def __len__(self) -> int:
        return len(self.ds) // self.batch_size

    def _make_env_group_builder(self, row: SearchR1Datum, group_size: int) -> ProblemGroupBuilder:
        return ProblemGroupBuilder(
            env_thunk=partial(
                ContextEnv,
                row["question"],
                row["answer"],
                self.context_tool_client,
                self.renderer,
                convo_prefix=self.convo_prefix,
                max_trajectory_tokens=self.max_trajectory_tokens,
            ),
            num_envs=group_size,
        )


@chz.chz
class SearchR1DatasetBuilder(RLDatasetBuilder):
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0
    max_eval_size: int = 1024
    max_trajectory_tokens: int = 32 * 1024

    async def __call__(self) -> tuple[SearchR1Dataset, None]:
        if self.convo_prefix == "standard":
            convo_prefix = ContextEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        tool_client = ContextToolClient()

        train_dataset = SearchR1Dataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            context_tool_client=tool_client,
            convo_prefix=convo_prefix,
            split="train",
            seed=self.seed,
            max_trajectory_tokens=self.max_trajectory_tokens,
        )
        return (train_dataset, None)
