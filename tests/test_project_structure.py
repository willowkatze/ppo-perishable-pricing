"""Basic tests for the initial repository structure."""

from pathlib import Path

from src import config


def test_required_directories_exist() -> None:
    required_dirs = [
        config.PROJECT_ROOT / "data" / "raw",
        config.PROJECT_ROOT / "data" / "processed",
        config.PROJECT_ROOT / "notebooks",
        config.PROJECT_ROOT / "src",
        config.PROJECT_ROOT / "tests",
        config.PROJECT_ROOT / "outputs" / "figures",
        config.PROJECT_ROOT / "outputs" / "tables",
        config.PROJECT_ROOT / "outputs" / "models",
        config.PROJECT_ROOT / "docs",
    ]

    for directory in required_dirs:
        assert directory.is_dir(), f"Missing directory: {directory}"


def test_required_source_files_exist() -> None:
    required_files = [
        config.PROJECT_ROOT / "src" / "__init__.py",
        config.PROJECT_ROOT / "src" / "config.py",
        config.PROJECT_ROOT / "src" / "data_processing.py",
        config.PROJECT_ROOT / "src" / "demand_model.py",
        config.PROJECT_ROOT / "src" / "pricing_env.py",
        config.PROJECT_ROOT / "src" / "baselines.py",
        config.PROJECT_ROOT / "src" / "train_ppo.py",
        config.PROJECT_ROOT / "src" / "evaluate.py",
    ]

    for file_path in required_files:
        assert file_path.is_file(), f"Missing file: {file_path}"


def test_configured_paths_are_inside_project_root() -> None:
    configured_paths = [
        config.RAW_DATA_PATH,
        config.PROCESSED_DATA_DIR,
        config.FIGURES_DIR,
        config.TABLES_DIR,
        config.MODELS_DIR,
    ]

    project_root = config.PROJECT_ROOT.resolve()
    for path in configured_paths:
        assert project_root in Path(path).resolve().parents
