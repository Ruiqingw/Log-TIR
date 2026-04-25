from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dataset_adapter import (
    TextToSQLExample,
    load_text_to_sql_examples,
    resolve_dataset_root,
    resolve_db_path,
    resolve_split_file,
)
from eval import execution_match
from sft_data import _load_tables, _render_schema, parse_tagged_response


def extract_sql(response: str) -> str:
    try:
        return parse_tagged_response(response)["action"]
    except ValueError:
        return " ".join(response.strip().split())


def build_prompt(
    example: TextToSQLExample,
    schema_text: str,
    include_evidence: bool = True,
) -> str:
    evidence_block = ""
    if include_evidence and example.evidence:
        evidence_block = f"\nEvidence: {example.evidence}"
    return (
        "You are a SQLite Text-to-SQL agent.\n"
        "Given the schema and question, return exactly two tags:\n"
        "<thought>brief reasoning</thought>\n"
        "<action>one SQLite query</action>\n"
        "The <action> section must contain only one executable SQLite query.\n\n"
        f"{schema_text}\n\n"
        f"Question: {example.question}{evidence_block}"
    )


def _schema_texts_for_spider(root: Path) -> dict[str, str]:
    tables_path = root / "tables.json"
    if not tables_path.exists():
        return {}
    return {
        db_id: _render_schema(schema)
        for db_id, schema in _load_tables(tables_path).items()
    }


def _schema_from_sqlite(db_path: Path, db_id: str) -> str:
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        table_rows = conn.execute(
            "select name from sqlite_master where type='table' and name not like 'sqlite_%' order by name"
        ).fetchall()
        lines = [f"Database: {db_id}", "Tables:"]
        for (table_name,) in table_rows:
            column_rows = conn.execute(f'pragma table_info("{table_name}")').fetchall()
            columns = ", ".join(row[1] for row in column_rows) or "(no columns)"
            lines.append(f"- {table_name}: {columns}")
        return "\n".join(lines)
    finally:
        conn.close()


def schema_text_for_example(
    dataset: str,
    root: Path,
    example: TextToSQLExample,
    spider_schemas: dict[str, str],
) -> str:
    if dataset == "spider" and example.db_id in spider_schemas:
        return spider_schemas[example.db_id]
    return _schema_from_sqlite(resolve_db_path(dataset, root, example.db_id), example.db_id)


def _format_chat_prompts(tokenizer: Any, prompts: list[str]) -> list[str]:
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompts
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]


def generate_responses_transformers(
    model_name_or_path: str,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device_map: str,
) -> list[str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        trust_remote_code=True,
    )
    responses: list[str] = []
    for input_text in _format_chat_prompts(tokenizer, prompts):
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return responses


def generate_responses_vllm(
    model_name_or_path: str,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    tensor_parallel_size: int,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int | None,
) -> list[str]:
    from vllm import LLM, SamplingParams

    llm_kwargs: dict[str, Any] = {
        "model": model_name_or_path,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    formatted_prompts = _format_chat_prompts(tokenizer, prompts)
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    outputs = llm.generate(formatted_prompts, sampling_params)
    return [output.outputs[0].text.strip() if output.outputs else "" for output in outputs]


def generate_responses(
    backend: str,
    model_name_or_path: str,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device_map: str,
    tensor_parallel_size: int,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int | None,
) -> list[str]:
    if backend == "transformers":
        return generate_responses_transformers(
            model_name_or_path=model_name_or_path,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device_map=device_map,
        )
    if backend == "vllm":
        return generate_responses_vllm(
            model_name_or_path=model_name_or_path,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
    raise ValueError(f"Unsupported inference backend: {backend}")


def evaluate_responses(
    dataset: str,
    root: Path,
    examples: list[TextToSQLExample],
    responses: list[str],
    timeout_s: float,
) -> dict[str, Any]:
    if len(examples) != len(responses):
        raise ValueError(
            f"Response count {len(responses)} does not match example count {len(examples)}"
        )

    matched = 0
    results: list[dict[str, Any]] = []
    for example, response in zip(examples, responses):
        pred_sql = extract_sql(response)
        db_path = resolve_db_path(dataset, root, example.db_id)
        outcome = execution_match(pred_sql, example.gold_sql, db_path, timeout_s=timeout_s)
        matched += int(outcome["match"])
        results.append(
            {
                "index": example.index,
                "db_id": example.db_id,
                "question": example.question,
                "gold_sql": example.gold_sql,
                "response": response,
                "pred_sql": pred_sql,
                "match": outcome["match"],
                "pred_error": outcome["pred_result"].get("error", ""),
                "gold_error": outcome["gold_result"].get("error", ""),
            }
        )

    total = len(examples)
    return {
        "dataset": dataset,
        "total": total,
        "matched": matched,
        "accuracy": matched / total if total else 0.0,
        "results": results,
    }


def _load_response_file(path: Path) -> list[str]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload and isinstance(payload[0], dict):
            return [
                str(item.get("response", item.get("query", item.get("sql", ""))))
                for item in payload
            ]
        return [str(item) for item in payload]
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone model inference and execution-match evaluation."
    )
    parser.add_argument("--dataset", choices=["spider", "bird"], default="spider")
    parser.add_argument("--data-root", type=Path, default=Path("data/spider"))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--dev-file", type=Path, default=None)
    parser.add_argument("--model", help="Model or checkpoint path to load.")
    parser.add_argument(
        "--backend",
        choices=["vllm", "transformers"],
        default="vllm",
        help="Inference backend. Use vLLM on GPU servers for batched generation.",
    )
    parser.add_argument("--responses-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("logs/infer_eval_results.json"))
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

    root = resolve_dataset_root(args.dataset, args.data_root)
    split_file = resolve_split_file(args.dataset, root, args.split, args.dev_file)
    examples = load_text_to_sql_examples(args.dataset, split_file)
    if args.limit is not None:
        examples = examples[: args.limit]

    if args.responses_file is not None:
        responses = _load_response_file(args.responses_file)
        if args.limit is not None:
            responses = responses[: args.limit]
    else:
        if not args.model:
            raise ValueError("Provide --model or --responses-file")
        spider_schemas = _schema_texts_for_spider(root) if args.dataset == "spider" else {}
        prompts = [
            build_prompt(
                example,
                schema_text_for_example(args.dataset, root, example, spider_schemas),
            )
            for example in examples
        ]
        responses = generate_responses(
            backend=args.backend,
            model_name_or_path=args.model,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device_map=args.device_map,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )

    report = evaluate_responses(args.dataset, root, examples, responses, args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "results"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
