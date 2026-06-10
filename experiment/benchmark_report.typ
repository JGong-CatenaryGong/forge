// MolAgent LLM 分子设计基准测试报告
// 目标: HOMO-LUMO gap = 3.0 eV
// 编译: typst compile benchmark_report.typ

#set page(margin: (top: 2cm, bottom: 2cm, left: 2cm, right: 2cm))
#set text(font: "Noto Sans CJK SC", size: 10pt)
#set table(
  stroke: 0.5pt,
  cell-inset: (x: 4pt, y: 3pt),
  align: center + horizon,
)

#align(center)[
  #block(text(size: 22pt, weight: "bold")[MolAgent LLM 分子设计基准测试报告])
  #v(4pt)
  #text(size: 11pt, fill: gray)[目标: HOMO-LUMO gap = 3.0 eV ｜ 2026-06-09]
]

#v(12pt)

= 模型排名

#let data = (
  ((rank: "1🥇", model: "Doubao seed-2-0-pro-260215", provider: "Doubao", best: "3.0004", dev: "0.0004", med: "3.4155", mean: "3.4520", wall: "1081s", llm: "541s", tokens: "64,604", seeds: "3/3", dft: "45%"),
   (rank: "2🥈", model: "Qwen 3.7-max", provider: "Qwen", best: "2.9976", dev: "0.0024", med: "3.4529", mean: "3.5002", wall: "1699s", llm: "628s", tokens: "90,489", seeds: "3/3", dft: "100%"),
   (rank: "3🥉", model: "GLM 5.1", provider: "Zhipu", best: "3.0030", dev: "0.0030", med: "3.4434", mean: "3.4091", wall: "2999s", llm: "2502s", tokens: "159,742", seeds: "3/3", dft: "100%"),
   (rank: "4", model: "Mimo v2.5-pro", provider: "Mimo", best: "2.9947", dev: "0.0053", med: "3.1790", mean: "3.1199", wall: "1483s", llm: "794s", tokens: "97,411", seeds: "3/3", dft: "88%"),
   (rank: "5", model: "DeepSeek v4-flash", provider: "DeepSeek", best: "2.9863", dev: "0.0056", med: "3.2613", mean: "3.2412", wall: "2631s", llm: "200s", tokens: "54,550", seeds: "3/3", dft: "100%"),
   (rank: "6", model: "DeepSeek v4-pro", provider: "DeepSeek", best: "3.0089", dev: "0.0089", med: "3.5756", mean: "3.5997", wall: "1923s", llm: "580s", tokens: "63,592", seeds: "3/3", dft: "100%"),
   (rank: "7", model: "MiniMax M3", provider: "MiniMax", best: "2.9821", dev: "0.0103", med: "3.2938", mean: "3.4339", wall: "4014s", llm: "1020s", tokens: "73,878", seeds: "2/3", dft: "100%"),
   (rank: "—", model: "Kimi k2.6", provider: "Moonshot", best: "—", dev: "—", med: "—", mean: "—", wall: "—", llm: "—", tokens: "—", seeds: "0/3", dft: "—")))

#table(
  columns: (auto, 2.2cm, auto, auto, auto, auto, auto, auto, auto, auto, auto),
  inset: 5pt,
  stroke: 0.5pt,
  [*Rank*], [*Model*], [*Best Gap*], [*\|dev\|\*], [*Med Gap*], [*Mean Gap*], [*Wall*], [*LLM*], [*Tokens*], [*Seeds*], [*DFT%*],
  ..data.map(r => (
    r.rank, [#r.model #box(width: 0pt)], r.best, r.dev, r.med, r.mean, r.wall, r.llm, r.tokens, r.seeds, r.dft
  )).flatten()
)

#v(12pt)
#text(size: 9pt, fill: gray)[备注: |dev| = |dft_gap - 3.0|, 越接近 0 越好。DFT% 为 DFT 阶段成功率。Kimi k2.6 因 API 429 过载未完成。]

= 逐模型详细数据

#for (i, m) in data.enumerate() {
  #if m.seeds == "0/3" { continue }  // skip kimi
  #v(8pt)
  #heading(level: 2, numbering: "1.1")[#m.model (#m.provider)]

  #let per_seed = {
    let model_key = m.model
    if model_key == "Doubao seed-2-0-pro-260215" {
      return (
        ((seed: "s42", best: "3.0004", med: "3.3174", mean: "3.4214", wall: "925s", llm: "453s", tok: "68,885", dft: "43/63", smi: "ClC(=O)C(=O)C(=O)C(=O)C=O"),
         (seed: "s123", best: "3.0488", med: "3.6227", mean: "3.5820", wall: "1184s", llm: "434s", tok: "67,756", dft: "50/50", smi: "N#CC=CC=CC=CC=CC=NO"),
         (seed: "s456", best: "3.1034", med: "3.3064", mean: "3.3526", wall: "1134s", llm: "735s", tok: "57,170", dft: "20/140", smi: "O=CC(=O)C(=NC)C(=O)C=O")))
      }
      if model_key == "Qwen 3.7-max" {
      return (
        ((seed: "s42", best: "2.9142", med: "3.5829", mean: "3.6726", wall: "1174s", llm: "652s", tok: "88,394", dft: "50/50", smi: "O=CC=C(C=O)C=O"),
         (seed: "s123", best: "2.9976", med: "3.0608", mean: "3.2225", wall: "2842s", llm: "648s", tok: "91,726", dft: "50/50", smi: "CN(C)c1ccc(-c2sc(C=CC=O)cc2)cc1"),
         (seed: "s456", best: "3.0112", med: "3.7149", mean: "3.6055", wall: "1082s", llm: "583s", tok: "91,347", dft: "50/50", smi: "O=CC(=O)C(=O)C(=O)C(=O)OC")))
      }
      if model_key == "GLM 5.1" {
      return (
        ((seed: "s42", best: "3.0775", med: "3.5503", mean: "3.4951", wall: "2561s", llm: "2188s", tok: "171,704", dft: "40/40", smi: "[H]C(=O)C(=O)C(=O)C(=N[H])C(=O)[H]"),
         (seed: "s123", best: "3.0030", med: "3.4398", mean: "3.3566", wall: "2626s", llm: "2152s", tok: "158,291", dft: "45/45", smi: "[H]C(=O)C(=O)C(=C(C#N)C([H])([H])C#N)C(=O)[H]"),
         (seed: "s456", best: "3.0140", med: "3.3402", mean: "3.3756", wall: "3809s", llm: "3166s", tok: "149,232", dft: "50/50", smi: "O=C1C=C(C(=O)OCCF)C(=O)C1=O")))
      }
      if model_key == "Mimo v2.5-pro" {
      return (
        ((seed: "s42", best: "2.9399", med: "3.3174", mean: "3.2621", wall: "1653s", llm: "806s", tok: "94,757", dft: "45/65", smi: "[H]C(=O)C(=O)C(=O)C(=O)C(=O)C#N"),
         (seed: "s123", best: "3.0097", med: "3.1211", mean: "3.0997", wall: "1508s", llm: "938s", tok: "104,815", dft: "50/50", smi: "O=CC(=O)C(=O)C(=O)C(=O)OC=O"),
         (seed: "s456", best: "2.9947", med: "3.0984", mean: "2.9978", wall: "1289s", llm: "639s", tok: "92,660", dft: "50/50", smi: "[H]C(=O)C(=O)C(=O)C(=O)C(=O)S[H]")))
      }
      if model_key == "DeepSeek v4-flash" {
      return (
        ((seed: "s42", best: "2.9944", med: "3.0730", mean: "3.2970", wall: "1187s", llm: "182s", tok: "50,826", dft: "50/50", smi: "C=CC=CC=CC=CC=CC=CS(=O)C"),
         (seed: "s123", best: "3.1507", med: "3.7631", mean: "3.5604", wall: "4378s", llm: "299s", tok: "64,999", dft: "50/50", smi: "c1ccc2c(c1)c1c3ccccc3c3ccccc3c1c2=O"),
         (seed: "s456", best: "2.9863", med: "2.9478", mean: "2.8663", wall: "2328s", llm: "118s", tok: "47,825", dft: "47/47", smi: "c1ccc2c(c1)nc1c2cc2c1ccnc2")))
      }
      if model_key == "DeepSeek v4-pro" {
      return (
        ((seed: "s42", best: "3.0772", med: "3.5846", mean: "3.5851", wall: "2592s", llm: "644s", tok: "69,347", dft: "50/50", smi: "CN(C)c1ccc(C=C2SC(=S)NC2=O)cc1"),
         (seed: "s123", best: "3.0473", med: "3.6193", mean: "3.5832", wall: "1558s", llm: "546s", tok: "62,357", dft: "50/50", smi: "Nc1ccc2c(c1)OC(C=C(C#N)C#N)=C2"),
         (seed: "s456", best: "3.0089", med: "3.5229", mean: "3.6308", wall: "1619s", llm: "552s", tok: "59,073", dft: "50/50", smi: "N#CC(=CC(=C(N(C)C)C=O)C#N)C#N")))
      }
      if model_key == "MiniMax M3" {
      return (
        ((seed: "s42", best: "2.9821", med: "3.3945", mean: "3.4038", wall: "4416s", llm: "607s", tok: "63,000", dft: "50/50", smi: "CN(C)c1ccc(C(=O)c2ccc(/C=C/c3ccc([N+](=O)[O-])cc3)cc2)cc1"),
         (seed: "s123", best: "3.0103", med: "3.1930", mean: "3.4640", wall: "3611s", llm: "1434s", tok: "84,755", dft: "45/45", smi: "Cc1cc(N(C)C)ccc1/N=N/c1ccc(C=O)cc1")))
      }
      return ()
    }()

    #table(
      columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
      inset: 4pt,
      [*Seed*], [*Best Gap*], [*Med Gap*], [*Mean Gap*], [*Wall*], [*LLM*], [*Tokens*], [*DFT*], [*Best SMILES*],
      ..per_seed.map(s => (
        s.seed, s.best, s.med, s.mean, s.wall, s.llm, s.tok, s.dft, [#s.smi #box(width: 0pt)]
      )).flatten()
    )
  }

= 关键发现

- *精确度冠军*: Doubao seed-2-0-pro-260215 (|dev| = 0.0004, gap = 3.0004)
- *速度冠军*: Doubao seed-2-0-pro-260215 (平均 1081s / seed)
- *性价比冠军*: DeepSeek v4-flash (仅 54k tokens, 200s LLM, |dev| = 0.0056)
- *DFT 成功率*: Qwen 3.7-max, DeepSeek v4-pro/flash, GLM 5.1 均达 100%
- *最低 LLM 耗时*: DeepSeek v4-flash seed=456 (仅 118s LLM 调用)
- *Kimi k2.6*: 因 API 429 过载未完成测试

= 实验配置

#table(
  columns: (auto, auto),
  [目标 Gap], [3.0 eV],
  [迭代轮数], [10],
  [候选数/轮], [20],
  [反馈数/轮], [5],
  [Temperature], [0.8 (kimi: 1.0)],
  [max_tokens], [16384],
  [gap_range_margin], [2.0],
  [反射模式], [spr],
  [RAG], [开启],
  [相关性校准], [开启 (MLP)],
  [种子数], [3 (42, 123, 456)],
  [量子化学引擎], [XTB (GFN2-xTB) → pySCF DFT (B3LYP/6-31G**)],
)

= 数据说明

计算环境: Linux (12th Gen i9-12900KF, Intel DG2 Arc A380 GPU)。DFT 计算使用 pySCF，并行度 n_parallel=2。XTB→DFT 相关性校准使用预训练 MLP 模型。

_报告自动生成于 2026-06-09_
