# Paper Trader v2

RL + LLM 混合自动交易系统。PPO 强化学习 + DeepSeek 多源新闻分析 + IBKR 模拟交易。

## 架构

```
29 News Sources (Yahoo/Bloomberg/Reuters/WSJ...)
        │
        ▼
DeepSeek LLM (4 Analysts → Bull vs Bear Debate)
        │
        ├──────────────────────┐
        ▼                      ▼
PPO RL Model ───→ Signal Fusion ←── Reflection Tracker
        │              │                │
        ▼              ▼                ▼
   Risk Engine → IBKR Paper Trading → Dashboard
```

## 特性

- **RL + LLM 混合决策**：PPO 模型处理价格模式，DeepSeek 处理新闻/基本面
- **三层信号融合**：RL 注入 → LLM 覆盖 → 辩论加成
- **反思学习**：追踪每个分析师准确率，动态调整权重
- **因子挖掘**：IC/RankIC 驱动的 Alpha 因子自动发现
- **29 新闻来源**：Yahoo Finance 聚合 + Google News RSS + Finnhub
- **增强回测**：滑点模型、交易统计、Alpha/Beta、信息比率
- **增量训练**：每日收盘后用新数据微调 PPO 模型
- **IBKR 集成**：Paper Trading 执行，限价单，风控引擎
- **Streamlit 7-tab Dashboard**：训练/交易/回测/信号/风控/日志/手册
- **FastAPI REST Server**：`/api/status` `/api/portfolio` `/api/signals` `/api/reflection`
- **自动调度**：cron 自动启停 + 健康检查

## 快速开始

### 前置条件

- Python 3.11+
- IB Gateway（用于模拟交易执行）
- DeepSeek API Key（[免费获取](https://platform.deepseek.com)）

### 一键安装

```bash
git clone https://github.com/your-username/paper-trader.git
cd paper-trader
bash scripts/setup.sh
```

### 手动安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install openai fastapi uvicorn

# 下载数据 & 训练模型
make download
make train

# 设置 API Key
export DEEPSEEK_API_KEY="sk-your-key"
```

### 启动 IB Gateway

1. 下载 [IB Gateway](https://www.interactivebrokers.com/en/trading/ibgateway.php)
2. 登录 → 选择 **IBKR Paper Trading**
3. Settings → API → 取消勾选 **Read-Only API**，端口设为 **4001**

### 运行

```bash
source .venv/bin/activate

make dashboard    # 启动控制面板 → http://localhost:8501
make paper        # 启动模拟交易（或点 Dashboard 按钮）
make api          # 启动 REST API → http://localhost:8090/docs
```

## 命令

| 命令 | 功能 |
|------|------|
| `make dashboard` | Streamlit 控制面板 |
| `make paper` | 启动模拟交易 |
| `make train` | 全量训练 PPO |
| `make backtest` | 增强回测 |
| `make refresh` | 增量更新数据 |
| `make analyze` | LLM 多源分析 |
| `make debate` | 多空辩论 |
| `make api` | REST API 服务 |
| `make reflect` | 分析师表现报告 |
| `python -m src.main incremental-train` | 增量训练 |
| `python -m src.main mine-factors` | 因子挖掘 |

## 配置

编辑 `config/settings.yaml`：

```yaml
llm:
  enabled: true
  provider: "deepseek"      # deepseek | anthropic | openai
  model: "deepseek-chat"

risk:
  max_position_pct: 0.40    # 最大单仓位
  max_daily_loss_pct: 0.05  # 日内止损
  max_total_drawdown_pct: 0.15

signal_fusion:
  rl_weight: 0.2            # RL权重(0-1,越低LLM越主导)
  llm_override_threshold: 0.3

paper_trading_capital: 1276  # 模拟本金(美元)
```

## Mac mini 24/7 部署

```bash
# 在新 Mac mini 上
git clone <repo-url> paper-trader
cd paper-trader
bash scripts/setup.sh

# 安装 IB Gateway（手动）
# 设置 DeepSeek API Key
echo "sk-your-key" > ~/.deepseek_key
chmod 600 ~/.deepseek_key

# 验证 cron 任务
crontab -l

# 启动 Dashboard（可选，通过 SSH 端口转发访问）
make dashboard
```

Cron 会自动：
- 周一至五 21:25 启动交易
- 周二至六 04:00 停止交易
- 盘中每 30 分钟健康检查
- 每日 16:30 数据刷新

## 项目结构

```
paper-trader/
├── config/settings.yaml     # 配置文件
├── src/
│   ├── main.py              # CLI 入口
│   ├── agent/               # RL agent + LLM + 辩论 + 融合 + 反思
│   ├── backtest/            # 回测引擎 + 指标 + 验证器
│   ├── data/                # 数据管道 + 因子挖掘 + 新闻汇聚
│   ├── ibkr/                # IBKR 客户端 + 行情 + 订单管理
│   ├── monitor/             # Dashboard + API + 日志
│   ├── nlp/                 # 自然语言策略解析
│   ├── risk/                # 风控引擎
│   └── config/              # 配置加载
├── scripts/
│   ├── setup.sh             # 一键安装
│   └── auto_trade.sh        # 自动交易调度
├── models/                  # 训练好的 PPO 模型 (gitignored)
├── data/                    # 市场数据 (gitignored)
├── logs/                    # 交易日志 (gitignored)
└── tests/                   # 14 单元测试
```

## 许可证

MIT
