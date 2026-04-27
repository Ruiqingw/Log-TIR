from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agent_rollout import _append_feedback, _feedback_message
from dataset_adapter import (
    TextToSQLExample,
    load_text_to_sql_examples,
    resolve_dataset_root,
    resolve_db_path,
    resolve_split_file,
)
from infer_eval import (
    _format_chat_prompts,
    _schema_texts_for_spider,
    build_prompt,
    generate_responses_transformers,
    schema_text_for_example,
)
from openrlhf_reward import score_response
from sandbox import execute_sql
from sft_data import parse_tagged_response


class ResponseGenerator:
    def __init__(
        self,
        *,
        backend: str,
        model_name_or_path: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        device_map: str,
        tensor_parallel_size: int,
        dtype: str,
        gpu_memory_utilization: float,
        max_model_len: int | None,
    ) -> None:
        self.backend = backend
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device_map = device_map
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self._llm: Any | None = None

    def _vllm(self) -> Any:
        if self._llm is None:
            from transformers import PreTrainedTokenizerBase
            from vllm import LLM

            if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
                PreTrainedTokenizerBase.all_special_tokens_extended = property(
                    lambda self: self.all_special_tokens
                )

            llm_kwargs: dict[str, Any] = {
                "model": self.model_name_or_path,
                "tensor_parallel_size": self.tensor_parallel_size,
                "dtype": self.dtype,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "trust_remote_code": True,
            }
            if self.max_model_len is not None:
                llm_kwargs["max_model_len"] = self.max_model_len
            self._llm = LLM(**llm_kwargs)
        return self._llm

    def generate(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        if self.backend == "transformers":
            return generate_responses_transformers(
                model_name_or_path=self.model_name_or_path,
                prompts=prompts,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                device_map=self.device_map,
            )
        if self.backend == "vllm":
            from vllm import SamplingParams

            llm = self._vllm()
            tokenizer = llm.get_tokenizer()
            formatted_prompts = _format_chat_prompts(tokenizer, prompts)
            sampling_params = SamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            outputs = llm.generate(formatted_prompts, sampling_params)
            return [
                output.outputs[0].text.strip() if output.outputs else ""
                for output in outputs
            ]
        raise ValueError(f"Unsupported inference backend: {self.backend}")


def _label_for_example(dataset: str, root: Path, example: TextToSQLExample) -> dict[str, Any]:
    return {
        "db_id": example.db_id,
        "db_path": str(resolve_db_path(dataset, root, example.db_id)),
        "gold_sql": example.gold_sql,
    }


def _action_sql(response: str, score: dict[str, Any]) -> str:
    if score.get("action_sql"):
        return str(score["action_sql"])
    try:
        return parse_tagged_response(response)["action"]
    except ValueError:
        return ""


def _feedback_for_response(
    label: dict[str, Any],
    response: str,
    timeout_s: float,
) -> str:
    try:
        action_sql = parse_tagged_response(response)["action"]
        execution = execute_sql(label["db_path"], action_sql, timeout_s=timeout_s)
        return _feedback_message(
            execution.get("error", ""),
            execution.get("rows", []),
        )
    except Exception as exc:
        return _feedback_message(f"{type(exc).__name__}: {exc}", [])


def _turn_record(
    *,
    turn_index: int,
    response: str,
    score: dict[str, Any],
) -> dict[str, Any]:
    error_category = str(score.get("error_category") or "")
    return {
        "turn_index": turn_index,
        "response": response,
        "action_sql": _action_sql(response, score),
        "format_match": bool(score.get("format_match", False)),
        "no_error": bool(score.get("no_error", False)),
        "exec_match": bool(score.get("exec_match", False)),
        "reward": float(score.get("reward", 0.0)),
        "error_category": error_category,
        "raw_execution_error": str(score.get("error") or ""),
        "empty_result": bool(score.get("empty_result", False)),
        "wrong_result": bool(score.get("wrong_result", False)),
    }


def _first_turn_category(turn: dict[str, Any]) -> str:
    if turn["exec_match"]:
        return "success"
    category = str(turn.get("error_category") or "")
    if category:
        return category
    if not turn.get("format_match", False):
        return "format"
    if not turn.get("no_error", False):
        return "execution"
    return "wrong_result"


def _empty_summary(dataset: str, split: str, max_turns: int) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "split": split,
        "max_turns": max_turns,
        "total": 0,
        "matched": 0,
        "accuracy": 0.0,
        "turn1_matched": 0,
        "turn1_accuracy": 0.0,
        "turn1_exec_match": 0.0,
        "final_matched": 0,
        "final_accuracy": 0.0,
        "final_exec_match_with_2_turns": 0.0,
        "rescued_by_turn2": 0,
        "turn2_rescue_rate_all": 0.0,
        "turn2_rescue_rate_among_turn1_failures": 0.0,
        "timeout_first_turn": 0,
        "timeout_rescued_by_turn2": 0,
        "timeout_rescue": 0,
        "non_timeout_rescued_by_turn2": 0,
        "syntax_rescued_by_turn2": 0,
        "syntax_error_rescue": 0,
        "execution_error_rescued_by_turn2": 0,
        "wrong_result_rescued_by_turn2": 0,
        "wrong_result_rescue": 0,
        "turn1_timeout_count": 0,
        "turn1_invalid_format_count": 0,
        "turn1_syntax_error_count": 0,
        "turn1_schema_hallucination_count": 0,
        "turn1_wrong_result_count": 0,
        "final_accuracy_including_timeouts": 0.0,
        "final_accuracy_excluding_first_turn_timeouts": 0.0,
        "first_turn_error_counts": {},
        "turn2_rescue_by_first_turn_error": {},
    }


def summarize_trajectories(
    trajectories: list[dict[str, Any]],
    *,
    dataset: str,
    split: str,
    max_turns: int,
) -> dict[str, Any]:
    if not trajectories:
        return _empty_summary(dataset, split, max_turns)

    total = len(trajectories)
    turn1_matched = sum(int(row["turns"][0]["exec_match"]) for row in trajectories)
    final_matched = sum(int(row["final_exec_match"]) for row in trajectories)
    first_turn_errors: Counter[str] = Counter()
    rescue_by_error: Counter[str] = Counter()

    for row in trajectories:
        category = _first_turn_category(row["turns"][0])
        if category != "success":
            first_turn_errors[category] += 1
        if row.get("rescued_by_turn2", False):
            rescue_by_error[category] += 1

    turn1_failures = total - turn1_matched
    timeout_first_turn = first_turn_errors["timeout"]
    non_timeout_total = total - timeout_first_turn
    non_timeout_final_matched = sum(
        int(row["final_exec_match"])
        for row in trajectories
        if _first_turn_category(row["turns"][0]) != "timeout"
    )

    rescued_by_turn2 = sum(int(row.get("rescued_by_turn2", False)) for row in trajectories)
    summary = {
        "dataset": dataset,
        "split": split,
        "max_turns": max_turns,
        "total": total,
        "matched": final_matched,
        "accuracy": final_matched / total,
        "turn1_matched": turn1_matched,
        "turn1_accuracy": turn1_matched / total,
        "turn1_exec_match": turn1_matched / total,
        "final_matched": final_matched,
        "final_accuracy": final_matched / total,
        "final_exec_match_with_2_turns": final_matched / total,
        "rescued_by_turn2": rescued_by_turn2,
        "turn2_rescue_rate_all": rescued_by_turn2 / total,
        "turn2_rescue_rate_among_turn1_failures": (
            rescued_by_turn2 / turn1_failures if turn1_failures else 0.0
        ),
        "timeout_first_turn": timeout_first_turn,
        "timeout_rescued_by_turn2": rescue_by_error["timeout"],
        "timeout_rescue": rescue_by_error["timeout"],
        "non_timeout_rescued_by_turn2": sum(
            count for category, count in rescue_by_error.items() if category != "timeout"
        ),
        "syntax_rescued_by_turn2": rescue_by_error["syntax"],
        "syntax_error_rescue": rescue_by_error["syntax"],
        "execution_error_rescued_by_turn2": rescue_by_error["execution"],
        "wrong_result_rescued_by_turn2": rescue_by_error["wrong_result"],
        "wrong_result_rescue": rescue_by_error["wrong_result"],
        "turn1_timeout_count": timeout_first_turn,
        "turn1_invalid_format_count": first_turn_errors["format"],
        "turn1_syntax_error_count": first_turn_errors["syntax"],
        "turn1_schema_hallucination_count": first_turn_errors["schema_hallucination"],
        "turn1_wrong_result_count": first_turn_errors["wrong_result"],
        "turn1_empty_result_count": first_turn_errors["empty_result"],
        "turn1_execution_error_count": first_turn_errors["execution"],
        "turn1_unsafe_sql_count": first_turn_errors["unsafe_sql"],
        "final_accuracy_including_timeouts": final_matched / total,
        "final_accuracy_excluding_first_turn_timeouts": (
            non_timeout_final_matched / non_timeout_total if non_timeout_total else 0.0
        ),
        "first_turn_error_counts": dict(sorted(first_turn_errors.items())),
        "turn2_rescue_by_first_turn_error": dict(sorted(rescue_by_error.items())),
    }
    return summary


def run_multi_turn_eval(
    *,
    dataset: str,
    root: Path,
    split: str,
    examples: list[TextToSQLExample],
    prompts: list[str],
    generator: ResponseGenerator,
    max_turns: int,
    timeout_s: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = [_label_for_example(dataset, root, example) for example in examples]
    turn1_responses = generator.generate(prompts)

    trajectories: list[dict[str, Any]] = []
    turn2_prompts: list[str] = []
    turn2_indices: list[int] = []

    for idx, (example, prompt, label, response) in enumerate(
        zip(examples, prompts, labels, turn1_responses)
    ):
        score = score_response(response, label, timeout_s=timeout_s)
        turn = _turn_record(turn_index=1, response=response, score=score)
        row = {
            "index": example.index,
            "db_id": example.db_id,
            "question": example.question,
            "gold_sql": example.gold_sql,
            "turns": [turn],
            "turn1_exec_match": turn["exec_match"],
            "final_exec_match": turn["exec_match"],
            "rescued_by_turn2": False,
        }
        trajectories.append(row)
        if max_turns > 1 and not turn["exec_match"]:
            feedback = _feedback_for_response(label, response, timeout_s)
            turn2_prompts.append(_append_feedback(prompt, response, feedback))
            turn2_indices.append(idx)

    if max_turns > 1 and turn2_prompts:
        turn2_responses = generator.generate(turn2_prompts)
        for idx, response in zip(turn2_indices, turn2_responses):
            label = labels[idx]
            score = score_response(response, label, timeout_s=timeout_s)
            turn = _turn_record(turn_index=2, response=response, score=score)
            row = trajectories[idx]
            row["turns"].append(turn)
            row["final_exec_match"] = turn["exec_match"]
            row["rescued_by_turn2"] = bool(
                (not row["turn1_exec_match"]) and turn["exec_match"]
            )

    summary = summarize_trajectories(
        trajectories,
        dataset=dataset,
        split=split,
        max_turns=max_turns,
    )
    return summary, trajectories


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-turn Text-to-SQL inference and execution-match evaluation."
    )
    parser.add_argument("--dataset", choices=["spider", "bird"], default="spider")
    parser.add_argument("--data-root", type=Path, default=Path("data/spider"))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--dev-file", type=Path, default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectories-out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    args = parser.parse_args()

    if args.max_turns not in (1, 2):
        raise ValueError("This evaluator currently supports --max-turns 1 or 2")

    root = resolve_dataset_root(args.dataset, args.data_root)
    split_file = resolve_split_file(args.dataset, root, args.split, args.dev_file)
    examples = load_text_to_sql_examples(args.dataset, split_file)
    if args.limit is not None:
        examples = examples[: args.limit]

    spider_schemas = _schema_texts_for_spider(root) if args.dataset == "spider" else {}
    prompts = [
        build_prompt(
            example,
            schema_text_for_example(args.dataset, root, example, spider_schemas),
        )
        for example in examples
    ]
    generator = ResponseGenerator(
        backend=args.backend,
        model_name_or_path=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device_map=args.device_map,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    summary, trajectories = run_multi_turn_eval(
        dataset=args.dataset,
        root=root,
        split=args.split,
        examples=examples,
        prompts=prompts,
        generator=generator,
        max_turns=args.max_turns,
        timeout_s=args.timeout,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trajectories_out.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({**summary, "results": trajectories}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with args.trajectories_out.open("w", encoding="utf-8") as handle:
        for row in trajectories:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
