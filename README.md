# 南开大学-信息检索系统原理-HW3 Academic Evidence Finder

面向 IR/RAG 研究入门者的轻量学术论文检索与证据展示 demo。系统使用本地小规模论文语料，提供关键词检索、语义近似检索、混合排序、主题/年份筛选、证据片段高亮、论文详情和相似论文推荐。

## Features

- 自然语言检索：支持英文 query，也内置了常见中文术语到英文检索词的映射。
- 扩展论文语料：本地索引约 70 篇经典 IR、神经检索、RAG、评测和 2024 年代表性 RAG 论文。
- 混合检索：BM25 负责精确匹配，TF-IDF 风格向量余弦相似度负责语义近似匹配。
- 可解释重排序：Hybrid 模式叠加字段权重、短语/主题 boost 和 MMR 去冗余。
- 证据展示：每个结果返回最相关 chunk，并高亮命中的关键词。
- 排序解释：结果卡片展示 matched phrases、topic match 和 ranking reasons。
- 检索工作台 UI：推荐查询、语料规模、筛选器、信号分数条、证据片段和详情弹窗。
- 筛选与排序：支持年份、主题、检索模式筛选。
- 详情页：展示论文元数据、可引用片段和相似论文。
- 离线可运行：默认版本只依赖 Python 标准库。

## Quick Start

```powershell
python app\server.py
```

然后打开：

```text
http://127.0.0.1:8000
```

网站说明页：

```text
http://127.0.0.1:8000/help.html
```

## API

- `GET /api/search?q=&year=&topic=&mode=`
  - `mode`: `hybrid`, `bm25`, `semantic`
- `GET /api/papers/{id}`
- `GET /api/topics`
- `GET /api/evaluate`

## Verification

```powershell
python -m unittest discover tests
python app\evaluate.py
```

当前评价脚本会对比 `bm25`、`semantic`、`hybrid` 三种模式，并输出 Precision@5、Recall@10、MRR。Hybrid 模式用于展示完整的可解释排序 pipeline；扩展语料后评价集也加入了更多真实相关论文，避免新增语料被误判为无关结果。

## Optional FastAPI Version

如果环境中可以安装依赖，可使用 FastAPI 包装层：

```powershell
pip install -r requirements.txt
uvicorn app.fastapi_app:app --reload
```

默认提交不包含大型数据集、模型文件或参数文件。
