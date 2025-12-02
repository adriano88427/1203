# -*- coding: utf-8 -*-

"""配置与常量模块，集中管理路径/文件相关设置。"""

import os
from typing import Dict, Any, List

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据目录 / 报表目录（相对于项目根目录）
DATA_DIR = os.path.abspath(os.path.join(_BASE_DIR, "..", "shuju"))
DATA_TABLE_DIR = os.path.join(DATA_DIR, "biaoge")
REPORT_OUTPUT_DIR = os.path.abspath(os.path.join(_BASE_DIR, "..", "baogao"))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATA_TABLE_DIR, exist_ok=True)
os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

# =============================
# 用户可配置项（统一集中管理）
# =============================

# 默认数据文件路径设置（方便用户修改）
# 如果有多个年度/批次的数据文件，可在列表中继续追加路径。
DEFAULT_DATA_FILES = [
   
    os.path.join(DATA_TABLE_DIR, "回测详细数据结合股本分析：创业板2025.xlsx"),
    os.path.join(DATA_TABLE_DIR, "回测详细数据结合股本分析：创业板2024.xlsx"),
    os.path.join(DATA_TABLE_DIR, "回测详细数据结合股本分析：创业板2023.xlsx"),
    os.path.join(DATA_TABLE_DIR, "回测详细数据结合股本分析：创业板2022.xlsx"),
    os.path.join(DATA_TABLE_DIR, "回测详细数据结合股本分析：创业板2021.xlsx"),
]
# 为保持兼容性，保留单文件常量（默认取列表中的第一个）
DEFAULT_DATA_FILE = DEFAULT_DATA_FILES[0] if DEFAULT_DATA_FILES else ""

# Excel 解析配置（可根据需要扩展）
DATA_PARSE_CONFIG: Dict[str, Any] = {
    "preferred_engines": ["pyarrow", "openpyxl", "pandas"],
    "sheet_policy": "all",
    "na_values": ["", " ", "NA", "N/A", "-", "—", "--"],
    "column_aliases": {
        "股票代码": ["证券代码", "code", "stock_code"],
        "股票名称": ["证券名称", "name", "stock_name"],
        "信号日期": ["signal_date", "日期", "交易日期"],
        "次日开盘买入持股两日收益率": ["收益率", "return", "return_rate"],
        "当日回调": ["当日回撤", "daily_drawdown"],
        "机构持股比例(%)": ["机构持股比例", "inst_holding"],
        "流通市值(元)": ["流通市值", "market_cap"],
        "财务投资机构合计（投资公司+私募+集合理财+其他理财+员工持股+信托+QFII+券商+基金）": [
            "财务投资机构合计"
        ],
        "朋友合计（企业大股东（大非）+社保+保险）": ["朋友合计"],
        "十大流通个人持股合计": [],
        "高管/大股东持股比例大非": [],
        "普通散户持股比例": [],
        "企业大股东大非（包含国资）": [],
        "企业大股东（包含国资）（小非）": [],
        "前10大流通股东持股比例合计": [],
        "十大流通股东小非合计": [],
        "十大流通股东大非合计": [],
        "十大流通机构大非": [],
        "十大流通机构小非": [],
        "基金持仓占比": [],
        "QFII持仓占比": [],
        "持有基金家数": [],
    },
    # column_types 取值说明：
    # - "percent": 原始值以百分比表示，解析时除以100并存为小数
    # - "amount": 金额/数量，保持原值
    # - "auto": 无法预判类型，按默认逻辑解析并在日志中提示确认
    "column_types": {
        "当日回调": "percent",
        "机构持股比例(%)": "percent",
        "流通市值(元)": "amount",
        "财务投资机构合计（投资公司+私募+集合理财+其他理财+员工持股+信托+QFII+券商+基金）": "percent",
        "朋友合计（企业大股东（大非）+社保+保险）": "percent",
        "十大流通个人持股合计": "percent",
        "高管/大股东持股比例大非": "percent",
        "普通散户持股比例": "percent",
        "企业大股东大非（包含国资）": "percent",
        "企业大股东（包含国资）（小非）": "percent",
        "前10大流通股东持股比例合计": "percent",
        "十大流通股东小非合计": "percent",
        "十大流通股东大非合计": "percent",
        "十大流通机构大非": "percent",
        "十大流通机构小非": "percent",
        "基金持仓占比": "percent",
        "QFII持仓占比": "percent",
        "持有基金家数": "auto",
    },
    # column_enabled: 是否参与后续因子分析（"是" / "否"，默认"是"）
    "column_enabled": {
        "当日回调": "是",
        "机构持股比例(%)": "是",
        "流通市值(元)": "是",
        "财务投资机构合计（投资公司+私募+集合理财+其他理财+员工持股+信托+QFII+券商+基金）": "是",
        "朋友合计（企业大股东（大非）+社保+保险）": "是",
        "十大流通个人持股合计": "是",
        "高管/大股东持股比例大非": "是",
        "普通散户持股比例": "是",
        "企业大股东大非（包含国资）": "是",
        "企业大股东（包含国资）（小非）": "是",
        "前10大流通股东持股比例合计": "是",
        "十大流通股东小非合计": "是",
        "十大流通股东大非合计": "是",
        "十大流通机构大非": "是",
        "十大流通机构小非": "是",
        "基金持仓占比": "是",
        "QFII持仓占比": "是",
        "持有基金家数": "是",
    },
}

# 因子语义 & 处理策略元数据
FACTOR_META: Dict[str, Dict[str, Any]] = {
    "当日回调": {
        "semantic": "percent",
        "display": "percent",
        "scale_hint": 0.01,
        "aliases": ["当日回撤", "daily_drawdown"],
    },
    "次日开盘买入持股两日收益率": {
        "semantic": "percent",
        "display": "percent",
        "scale_hint": 0.01,
        "aliases": ["收益率", "return", "return_rate"],
    },
    "机构持股比例(%)": {
        "semantic": "percent",
        "display": "percent",
        "scale_hint": 0.01,
    },
    "普通散户持股比例": {
        "semantic": "percent",
        "display": "percent",
        "scale_hint": 0.01,
    },
    "基金持仓占比": {
        "semantic": "percent",
        "display": "percent",
        "scale_hint": 0.01,
    },
    "QFII持仓占比": {
        "semantic": "percent",
        "display": "percent",
        "scale_hint": 0.01,
    },
    "流通市值(元)": {
        "semantic": "amount",
        "display": "amount",
        "scale_hint": 1.0,
        "unit": "CNY",
        "aliases": ["流通市值", "market_cap"],
    },
    "持有基金家数": {
        "semantic": "numeric",
        "display": "numeric",
    },
}

# 因子分组规则（针对未知列名自动匹配语义）
FACTOR_GROUP_RULES = [
    {
        "pattern": r"(回调|回撤|收益率)",
        "semantic": "percent",
        "display": "percent",
        "scale_hint": 0.01,
    },
    {
        "pattern": r"(占比|比例|持股)",
        "semantic": "percent",
        "display": "percent",
        "scale_hint": 0.01,
    },
    {
        "pattern": r"(市值|金额|资产|流通)",
        "semantic": "amount",
        "display": "amount",
        "scale_hint": 1.0,
    },
]

PERCENT_STYLE_COLUMNS: List[str] = []

COLUMN_ALIGNMENT_RULES: Dict[str, Dict[str, Any]] = {}

# 需要分析的因子列（非参数 & 带参数分析都会引用）
# 程序会自动对这些列（及收益列）尝试进行字符串/百分比到数值的转换，无需额外配置。
FACTOR_COLUMNS = [
    '当日回调',
    '机构持股比例(%)',
    '流通市值(元)',
    '财务投资机构合计（投资公司+私募+集合理财+其他理财+员工持股+信托+QFII+券商+基金）',
    '朋友合计（企业大股东（大非）+社保+保险）',
    '十大流通个人持股合计',
    '高管/大股东持股比例大非',
    '普通散户持股比例','企业大股东大非（包含国资）','企业大股东（包含国资）（小非）','前10大流通股东持股比例合计','十大流通股东小非合计','十大流通股东大非合计',
    '十大流通机构大非','十大流通机构小非','基金持仓占比','QFII持仓占比','持有基金家数',
]

# 收益率列名称（所有IC/收益分析均依赖此列）
RETURN_COLUMN = '次日开盘买入持股两日收益率'

# =============================
# 系统内置常量（无需用户调整）
# =============================

def build_report_path(filename: str) -> str:
    """将文件名映射到报表输出目录."""
    if not filename:
        return REPORT_OUTPUT_DIR
    if os.path.isabs(filename):
        return filename
    return os.path.join(REPORT_OUTPUT_DIR, filename)


# 板块 & 样本相关常量（内置映射，避免额外配置文件）
SEGMENT_MIN_SAMPLES = 30
SEGMENT_MIN_DAILY = 2
MARKET_SEGMENT_RULES = (
    ("创业板", ("SZ300", "SZ301"), ("300", "301")),
    ("科创板", ("SH688",), ("688",)),
    ("北交所", ("BJ",), tuple(str(code) for code in range(830, 840))),
    ("深圳主板", ("SZ000", "SZ001", "SZ002", "SZ003"), ("000", "001", "002", "003")),
    ("沪主板", ("SH600", "SH601", "SH603", "SH605"), ("600", "601", "603", "605")),
)


# 动态可靠性加权配置
RELIABILITY_CONFIG = {
    'base_weights': {
        'base': 0.35,
        'overall': 0.25,
        'rolling': 0.15,
        'temporal': 0.125,
        'sample': 0.125,
    },
    'scale_bounds': (0.4, 1.6),
    'normalized_bounds': (0.05, 0.55),
    'drop_threshold': 0.35,
}
