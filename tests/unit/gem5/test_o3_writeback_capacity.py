import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def class_constants(path: Path, class_name: str) -> dict[str, int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            values = {}
            for statement in node.body:
                if not isinstance(statement, ast.Assign):
                    continue
                if len(statement.targets) != 1:
                    continue
                target = statement.targets[0]
                if isinstance(target, ast.Name) and isinstance(
                    statement.value, ast.Constant
                ):
                    values[target.id] = statement.value.value
            return values
    raise AssertionError(f"class {class_name} not found in {path}")


def test_d9300_iew_queue_covers_rob():
    cores = (
        ("O3_ARM_Cortex_x4.py", "O3_ARM_Cortex_x4"),
        ("O3_ARM_Cortex_A720.py", "O3_ARM_Cortex_A720"),
    )
    core_dir = ROOT / "sim/gem5/configs/common/cores/arm"

    for filename, class_name in cores:
        values = class_constants(core_dir / filename, class_name)
        capacity = (values["forwardComSize"] + 1) * values["wbWidth"]
        assert capacity >= values["numROBEntries"], (
            f"{class_name} IEW queue capacity {capacity} cannot hold "
            f"all {values['numROBEntries']} in-flight ROB entries"
        )
