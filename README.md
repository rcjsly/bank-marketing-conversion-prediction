# 银行营销转化预测与模型服务

一个端到端机器学习项目：使用真实公开数据完成特征工程、传统模型与 PyTorch 表格 Transformer 训练、独立测试集评估和 FastAPI 推理；使用 SQLite 记录预测，并提供监控指标与部署配置。**这是离线实验和本地服务演示，不宣称真实业务上线。**

## 业务问题

在拨打营销电话**之前**，估计客户订阅定期存款的概率，辅助排序待联系客户。数据使用 [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) 的 `bank-additional-full.csv`，共 41,188 条记录。数据许可、变量释义和引用请查看 UCI 页面；首次训练自动从 UCI 下载。

核心设计：

- 校验输入字段与标签、处理缺失类别；将 `pdays=999` 转为缺失值，同时保留“是否曾联系”的指示特征。
- **删除 `duration`**：通话时长只有通话结束后才知道，用它预测拨打前的转化会造成目标场景的信息泄漏。
- 分层随机拆分训练/验证/测试集（70/15/15）。预处理器仅在训练集拟合；验证集比较逻辑回归和直方图梯度提升树的平均精度 AP，并选择 F1 阈值；只在选型完成后评估测试集。
- 可选 PyTorch 表格 Transformer：将数值和类别特征编码为 token，经一层多头自注意力预测；在相同数据划分上按验证集 AP 早停，与传统模型公平比较。这不是对预训练大语言模型微调。
- FastAPI 启动时加载模型与阈值，`/predict` 返回概率并写入数据库（不存客户原始字段）；`/health` 检查模型和数据库，`/live` 为存活探针，`/metrics` 输出 Prometheus 基础指标。

注意：原始数据没有完整日期/年份，随机拆分不等于跨时间的真实泛化；部署前仍须补充时序回测、数据合规审查、概率校准和线上监控。当前没有真实转化标签回流或模型漂移告警。

## 本地运行（Windows PowerShell）

需要 Python 3.12、网络连接。以下命令在项目根目录运行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m bank_campaign.train
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn bank_campaign.api:app --host 127.0.0.1 --port 8000
```

训练产生 `artifacts/model.joblib` 和 `artifacts/metrics.json`，原始数据保存在 `data/`；这些文件不入 Git。默认使用 `artifacts/predictions.sqlite3` 记录预测摘要；也可通过 `DATABASE_URL=postgresql+psycopg://用户名:密码@主机:5432/数据库` 切换 PostgreSQL。改用自有数据时，提供同名列（包括训练标签 `y`）并用 `--data` 指定路径。检查 `/docs` 或执行：

```powershell
$body = @{
  age=35; job="technician"; marital="single"; education="university.degree"
  default="no"; housing="yes"; loan="no"; contact="cellular"
  month="may"; day_of_week="mon"; campaign=2; pdays=999; previous=0
  poutcome="nonexistent"; "emp.var.rate"=1.1; "cons.price.idx"=93.994
  "cons.conf.idx"=-36.4; euribor3m=4.857; "nr.employed"=5191.0
} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body $body
```

可选训练 Transformer（额外安装 PyTorch，CPU 也能运行）：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m bank_campaign.train_transformer --epochs 12
$env:MODEL_PATH = "artifacts/transformer.joblib"
.\.venv\Scripts\python.exe -m uvicorn bank_campaign.api:app --host 127.0.0.1 --port 8000
```

训练生成 `artifacts/transformer_metrics.json`。默认服务仍加载验证集 AP 更高的梯度提升树模型；要切回它，移除 `MODEL_PATH` 环境变量。

## 数据库与部署

先执行训练生成模型产物，再在项目根目录运行：

```powershell
docker build -t bank-campaign-ml .
docker run --rm -p 8000:8000 bank-campaign-ml
$env:POSTGRES_PASSWORD = "请设置仅用于本地演示的密码"
docker compose up --build
```

独立 Transformer 镜像：先完成训练，再执行 `docker build -f Dockerfile.transformer -t bank-campaign-transformer .`。默认镜像不包含 PyTorch。预测记录表由服务启动时创建，保存时间、模型名、概率、阈值和决策；生产环境应改用数据库迁移、凭据轮换和数据保留策略。

`.github/workflows/ci.yml` 在 PR/主分支自动执行测试、真实数据训练和镜像构建，并在主分支推送按 commit SHA 标记的 GHCR 镜像。代码须先推送 GitHub，开通 Packages 权限；CI 训练依赖 UCI 数据下载可用。`.github/workflows/deploy.yml` 提供**手动触发**的集群滚动发布：配置受保护的 `production` 环境与 `KUBE_CONFIG_BASE64` secret，预先创建集群 Deployment，提供已发布的 40 位 commit SHA。

`k8s/app.yaml` 包含 Deployment、Service、存活/就绪探针与 Prometheus 抓取注解。需要可访问的**外部 PostgreSQL**：先创建 Secret `bank-campaign-db`（键 `url` 的值是 `postgresql+psycopg://...`），将清单中的示例镜像替换为真实 GHCR 镜像及 SHA 标签，然后运行 `kubectl apply -f k8s/app.yaml`。私有镜像另需拉取凭据；`/metrics` 应在部署网络中限制访问。

**验证边界：**本机没有 Docker、kubectl 或 PostgreSQL 服务，镜像构建、PostgreSQL 实例、GitHub Actions 和 Kubernetes 实际发布尚未实测。SQLite 读写、模型推理与 Prometheus 指标输出已在本地验证；目前没有运行 Prometheus/Grafana 或配置告警。

## 实验结果与简历表达

本机实测使用 41,188 条数据，正样本占比 11.27%，训练/验证/测试集分别为 28,831 / 6,178 / 6,179 条。验证集选中梯度提升树（AP 0.4905，F1 0.5192）；**独立测试集 AP 0.4905、ROC-AUC 0.8143、Precision 0.4440、Recall 0.6207、F1 0.5177**。表格 Transformer 的测试集 AP 0.4736、F1 0.5128，未超过树模型。完整结果在 `artifacts/metrics.json` 与 `artifacts/transformer_metrics.json`；测试指标不参与选型。

简历项目名：**银行营销转化预测与模型服务 | Python、scikit-learn、PyTorch、FastAPI、SQLAlchemy**

- 基于 UCI 4.1 万条银行营销记录构建拨打前转化预测流程，设计数据校验、缺失值处理、`pdays=999` 业务特征和类别编码，主动剔除事后才能获取的通话时长，规避场景信息泄漏。
- 采用分层训练/验证/测试划分，比较逻辑回归、梯度提升树与 PyTorch 表格 Transformer；按验证集 AP 选模型并确定分类阈值，独立测试集取得 AP 0.4905、ROC-AUC 0.8143、F1 0.5177。
- 将模型与阈值持久化并封装 FastAPI 推理服务，使用 SQLAlchemy 将预测摘要写入 SQLite，暴露 Prometheus 指标及健康检查；编写 Docker、GitHub Actions 和 Kubernetes 配置（集群发布未实测）。

面试时可讲清楚：为什么删除 `duration`；类别不平衡时为何关注 AP；为何表格 Transformer 不一定优于树模型；随机切分相较时间切分的局限；Prometheus 基础指标与真正的模型效果监控有何区别。请只写自己能解释、能演示的部分。
