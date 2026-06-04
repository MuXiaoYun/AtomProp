# AtomProp Web Platform

基于 **Vue 3** + **Django** 的分子性质预测与模型训练 Web 应用。

## 功能

### 性质预测（`predict_gui.py`）

- 上传 `.pth` 模型、批量 SMILES 预测
- 文本输入 / 拖放 TXT、CSV
- 导出 CSV 结果
- **日间 / 夜间模式**切换
- 纯色表格（无斑马纹）

### 模型训练（`finetune_regression.py` / `finetune.py`）

- 上传含 SMILES 与真实值的 CSV
- 自动识别 SMILES 列，手动选择目标列
- 回归 / 分类任务切换
- 从头训练、预训练权重、或从已有 `.pth` 继续训练
- 可配置 Epoch 数与学习率（其余参数沿用 `config_reg.py` / `config_finetune.py`）
- 训练进度条、Loss 曲线、模型下载

## 快速开始

```bash
# 后端（AtomProp 环境）
pip install -r web_predict/backend/requirements.txt
cd web_predict/backend && python manage.py runserver 0.0.0.0:8000

# 前端
cd web_predict/frontend && npm install && npm run dev
```

- 预测：http://localhost:5173/
- 训练：http://localhost:5173/train

## 训练 API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/train/defaults/` | 默认 epoch、学习率 |
| POST | `/api/train/dataset/upload/` | 上传训练 CSV |
| POST | `/api/train/init-model/upload/` | 上传初始 .pth |
| POST | `/api/train/start/` | 启动训练任务 |
| GET | `/api/train/jobs/<id>/` | 轮询进度与 loss |
| GET | `/api/train/jobs/<id>/model/` | 下载训练好的模型 |
