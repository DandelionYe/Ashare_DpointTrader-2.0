"""
配置 Schema 定义和验证。

使用 Pydantic 定义严格的配置结构，替代松散的 dict。
提供配置验证、序列化和反序列化功能。
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
import pandas as pd


# =========================================================
# Feature Config Schema
# =========================================================
@dataclass
class FeatureConfig:
    """特征工程配置"""
    windows: List[int] = field(default_factory=lambda: [3, 5, 10, 20])
    use_momentum: bool = True
    use_volatility: bool = True
    use_volume: bool = True
    use_candle: bool = True
    use_turnover: bool = True
    vol_metric: Literal["std", "mad"] = "std"
    liq_transform: Literal["ratio", "zscore"] = "ratio"

    def validate(self) -> List[str]:
        """验证配置，返回错误列表"""
        errors = []
        if not self.windows:
            errors.append("windows cannot be empty")
        if any(w < 1 for w in self.windows):
            errors.append("windows must be positive integers")
        if len(self.windows) > 10:
            errors.append("too many windows (max 10)")
        
        # 至少启用一个特征族
        if not any([self.use_momentum, self.use_volatility, 
                    self.use_volume, self.use_candle, self.use_turnover]):
            errors.append("at least one feature family must be enabled")
        
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeatureConfig:
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =========================================================
# Model Config Schema
# =========================================================
@dataclass
class ModelConfig:
    """模型配置"""
    model_type: Literal["logreg", "sgd", "xgb"] = "logreg"
    
    # Logistic Regression
    C: float = 1.0
    penalty: Literal["l1", "l2", "elasticnet"] = "l2"
    solver: Literal["lbfgs", "liblinear", "saga"] = "lbfgs"
    l1_ratio: Optional[float] = None
    class_weight: Optional[Literal["balanced", None]] = None
    
    # SGD Classifier
    alpha: float = 1e-4
    max_iter: int = 1000
    
    # XGBoost
    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    min_child_weight: float = 1.0

    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        
        if self.model_type == "logreg":
            if self.C <= 0:
                errors.append("C must be positive")
            if self.penalty == "elasticnet" and not (0 <= (self.l1_ratio or 0) <= 1):
                errors.append("l1_ratio must be in [0, 1] for elasticnet")
        
        elif self.model_type == "sgd":
            if self.alpha <= 0:
                errors.append("alpha must be positive")
        
        elif self.model_type == "xgb":
            if self.n_estimators < 1:
                errors.append("n_estimators must be positive")
            if self.max_depth < 1:
                errors.append("max_depth must be positive")
            if not (0 < self.learning_rate <= 1):
                errors.append("learning_rate must be in (0, 1]")
        
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelConfig:
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =========================================================
# Trade Config Schema
# =========================================================
@dataclass
class TradeConfig:
    """交易执行配置"""
    initial_cash: float = 100000.0
    buy_threshold: float = 0.55
    sell_threshold: float = 0.45
    confirm_days: int = 2
    min_hold_days: int = 1
    max_hold_days: int = 20
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    
    # 真实执行参数
    exec_price_model: Literal["same_close_idealized", "next_open", "next_close"] = "next_open"
    slippage_bps: float = 10.0
    commission_rate: float = 0.00025
    commission_min: float = 5.0

    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        
        if self.initial_cash <= 0:
            errors.append("initial_cash must be positive")
        
        if not (0 <= self.buy_threshold <= 1):
            errors.append("buy_threshold must be in [0, 1]")
        
        if not (0 <= self.sell_threshold <= 1):
            errors.append("sell_threshold must be in [0, 1]")
        
        if self.sell_threshold >= self.buy_threshold:
            errors.append("sell_threshold must be < buy_threshold")
        
        if self.confirm_days < 1:
            errors.append("confirm_days must be >= 1")
        
        if self.min_hold_days < 1:
            errors.append("min_hold_days must be >= 1")
        
        if self.max_hold_days < self.min_hold_days:
            errors.append("max_hold_days must be >= min_hold_days")
        
        if self.take_profit is not None and self.take_profit <= 0:
            errors.append("take_profit must be positive")
        
        if self.stop_loss is not None and self.stop_loss <= 0:
            errors.append("stop_loss must be positive")
        
        if self.slippage_bps < 0:
            errors.append("slippage_bps must be non-negative")
        
        if self.commission_rate < 0:
            errors.append("commission_rate must be non-negative")
        
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TradeConfig:
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =========================================================
# Search Config Schema
# =========================================================
@dataclass
class SearchConfig:
    """随机搜索超参数配置"""
    runs: int = 100
    epsilon: float = 0.01
    exploit_ratio: float = 0.7
    top_k: int = 10
    max_features: int = 80
    n_jobs: int = -1  # -1 = all cores, 1 = single thread, >0 = specified cores

    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        
        if self.runs < 1:
            errors.append("runs must be positive")
        
        if self.epsilon < 0:
            errors.append("epsilon must be non-negative")
        
        if not (0 <= self.exploit_ratio <= 1):
            errors.append("exploit_ratio must be in [0, 1]")
        
        if self.top_k < 1:
            errors.append("top_k must be positive")
        
        if self.max_features < 1:
            errors.append("max_features must be positive")
        
        # n_jobs: -1 = all cores, >=1 = specified cores
        if self.n_jobs < -1 or self.n_jobs == 0:
            errors.append("n_jobs must be -1 (all cores) or a positive integer")
        
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SearchConfig:
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =========================================================
# Full Config Schema
# =========================================================
@dataclass
class FullConfig:
    """完整配置（包含所有子配置）"""
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    trade_config: TradeConfig = field(default_factory=TradeConfig)
    search_config: SearchConfig = field(default_factory=SearchConfig)
    split_mode: Literal["walkforward"] = "walkforward"

    # Walk-forward 参数
    n_folds: int = 4
    train_start_ratio: float = 0.5
    wf_min_rows: int = 80

    def validate(self) -> List[str]:
        """验证完整配置"""
        errors = []
        errors.extend(self.feature_config.validate())
        errors.extend(self.model_config.validate())
        errors.extend(self.trade_config.validate())
        errors.extend(self.search_config.validate())

        if self.n_folds < 2:
            errors.append("n_folds must be >= 2")

        if not (0 < self.train_start_ratio < 1):
            errors.append("train_start_ratio must be in (0, 1)")

        if self.wf_min_rows < 10:
            errors.append("wf_min_rows must be >= 10")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "feature_config": self.feature_config.to_dict(),
            "model_config": self.model_config.to_dict(),
            "trade_config": self.trade_config.to_dict(),
            "search_config": self.search_config.to_dict(),
            "split_mode": self.split_mode,
            "n_folds": self.n_folds,
            "train_start_ratio": self.train_start_ratio,
            "wf_min_rows": self.wf_min_rows,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FullConfig:
        """从字典创建"""
        return cls(
            feature_config=FeatureConfig.from_dict(data.get("feature_config", {})),
            model_config=ModelConfig.from_dict(data.get("model_config", {})),
            trade_config=TradeConfig.from_dict(data.get("trade_config", {})),
            search_config=SearchConfig.from_dict(data.get("search_config", {})),
            split_mode=data.get("split_mode", "walkforward"),
            n_folds=data.get("n_folds", 4),
            train_start_ratio=data.get("train_start_ratio", 0.5),
            wf_min_rows=data.get("wf_min_rows", 80),
        )

    def to_json(self, indent: int = 2) -> str:
        """序列化为 JSON"""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> FullConfig:
        """从 JSON 反序列化"""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_json_file(cls, filepath: str) -> FullConfig:
        """从 JSON 文件加载"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 支持两种格式：直接是配置，或包含在 best_config 中
        if "best_config" in data:
            data = data["best_config"]
        return cls.from_dict(data)

    def save_json(self, filepath: str, indent: int = 2):
        """保存到 JSON 文件"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)

    def apply_cli_overrides(
        self,
        runs: Optional[int] = None,
        seed: Optional[int] = None,
        initial_cash: Optional[float] = None,
        exec_price_model: Optional[str] = None,
        slippage_bps: Optional[float] = None,
        commission_rate: Optional[float] = None,
        commission_min: Optional[float] = None,
    ) -> FullConfig:
        """
        应用 CLI 参数覆盖，返回新的 FullConfig 实例。
        只有当 CLI 参数不是默认值时才覆盖。
        """
        # 深拷贝当前配置
        new_config = FullConfig.from_dict(self.to_dict())

        # 覆盖 TradeConfig 中的执行参数
        if exec_price_model is not None:
            new_config.trade_config.exec_price_model = exec_price_model  # type: ignore
        if slippage_bps is not None:
            new_config.trade_config.slippage_bps = slippage_bps  # type: ignore
        if commission_rate is not None:
            new_config.trade_config.commission_rate = commission_rate  # type: ignore
        if commission_min is not None:
            new_config.trade_config.commission_min = commission_min  # type: ignore
        if initial_cash is not None:
            new_config.trade_config.initial_cash = initial_cash  # type: ignore

        return new_config


# =========================================================
# Run Metadata Schema
# =========================================================
@dataclass
class RunMetadata:
    """运行元数据（用于复现和追踪）"""
    run_id: int
    created_at: str
    code_version: str
    python_version: str
    dependency_versions: Dict[str, str]
    data_hash: str
    data_path: str
    random_seed: int
    config: FullConfig
    git_commit: Optional[str] = None
    hostname: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "code_version": self.code_version,
            "python_version": self.python_version,
            "dependency_versions": self.dependency_versions,
            "data_hash": self.data_hash,
            "data_path": self.data_path,
            "random_seed": self.random_seed,
            "config": self.config.to_dict(),
            "git_commit": self.git_commit,
            "hostname": self.hostname,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RunMetadata:
        """从字典创建"""
        return cls(
            run_id=data["run_id"],
            created_at=data["created_at"],
            code_version=data.get("code_version", "unknown"),
            python_version=data.get("python_version", "unknown"),
            dependency_versions=data.get("dependency_versions", {}),
            data_hash=data.get("data_hash", ""),
            data_path=data.get("data_path", ""),
            random_seed=data.get("random_seed", 42),
            config=FullConfig.from_dict(data.get("config", {})),
            git_commit=data.get("git_commit"),
            hostname=data.get("hostname"),
            notes=data.get("notes", []),
        )

    def to_json(self, indent: int = 2) -> str:
        """序列化为 JSON"""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, filepath: str, indent: int = 2):
        """保存到 JSON 文件"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)

    @classmethod
    def from_json_file(cls, filepath: str) -> RunMetadata:
        """从 JSON 文件加载"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# =========================================================
# Utility Functions
# =========================================================
def compute_data_hash(df: pd.DataFrame) -> str:
    """计算 DataFrame 的哈希值"""
    raw = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(raw).hexdigest()


def get_code_version() -> str:
    """获取代码版本（从 git 或文件修改时间）"""
    import subprocess
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            encoding="utf-8"
        ).strip()[:8]
        return commit
    except Exception:
        #  fallback to file modification time
        import os
        main_file = os.path.join(os.path.dirname(__file__), "main_cli.py")
        if os.path.exists(main_file):
            mtime = os.path.getmtime(main_file)
            return datetime.fromtimestamp(mtime).strftime("%Y%m%d")
        return "unknown"


def get_dependency_versions() -> Dict[str, str]:
    """获取关键依赖的版本"""
    import sys
    versions = {}
    
    packages = [
        "pandas", "numpy", "sklearn", "openpyxl", 
        "xlsxwriter", "joblib", "xgboost"
    ]
    
    for pkg in packages:
        try:
            module = sys.modules.get(pkg)
            if module is None:
                __import__(pkg)
                module = sys.modules[pkg]
            
            version = getattr(module, "__version__", "unknown")
            versions[pkg] = str(version)
        except ImportError:
            versions[pkg] = "not installed"
    
    return versions


def get_python_version() -> str:
    """获取 Python 版本"""
    import sys
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_git_commit() -> Optional[str]:
    """获取 git commit hash"""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            encoding="utf-8"
        ).strip()
    except Exception:
        return None


def get_hostname() -> str:
    """获取主机名"""
    import socket
    return socket.gethostname()
