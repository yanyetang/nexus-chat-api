from pathlib import Path

from app.optimization.optimize import compile_optimized_pipeline, save_artifact


def main() -> None:
    dataset_path = Path("tests/eval/golden_dataset.json")
    if not dataset_path.exists():
        raise FileNotFoundError("tests/eval/golden_dataset.json was not found")

    output_path = Path("artifacts/optimization_report.json")
    golden_dataset = __import__("json").loads(dataset_path.read_text(encoding="utf-8"))
    artifact = compile_optimized_pipeline(golden_dataset=golden_dataset)
    save_artifact(artifact=artifact, output_path=output_path)
    print(f"Saved optimization report to {output_path}")


if __name__ == "__main__":
    main()
