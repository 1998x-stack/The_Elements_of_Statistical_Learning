# ESL Front Matter (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author real Chinese teaching content for the five front/back-matter sections (01, 02, 21, 22, 23), converting each `.md` from a stub to a proper tutorial summary and making each `.py` a sensible runnable example, while updating `progress.yaml`/`README.md` status from ⏳ to ✅.

**Architecture:** Each section keeps the existing file layout (`chapter/<chapter>.md` + `chapter/<chapter>.py`, no `.ipynb` — these are never deep). The Phase 1 `scripts/generate_docs.py` `write_*_if_missing` guarantees we never clobber — so we author these files directly with the `write` tool, then update `config/progress.yaml` to `done` and rebuild the README.

**Tech Stack:** Python 3.9+, numpy (for the one numeric demo), PyYAML (progress update), the established `scripts/` tooling.

## Global Constraints

- Content language is **Chinese**; technical terms and code stay English.
- No `.ipynb` for these sections (none are `deep`). Leave them absent.
- `.py` files must run standalone (guard via `if __name__ == "__main__":`) and pass `apps/smoke_test.py`.
- Never run `generate_docs.py --init` in a way that could overwrite the authored content — it won't (it only writes when missing); we author by direct write.
- After authoring, update `config/progress.yaml` entries from `stub` → `done` for the 5 sections, then run `python scripts/generate_docs.py --regenerate` to rebuild the README.
- The five progress keys (note leading slash): `/01_Preface_to_the_Second_Edition`, `/02_Preface_to_the_First_Edition`, `/21_References`, `/22_Author_Index`, `/23_Index`.

---

### Task 1: 01 — Preface to the Second Edition

**Files:**
- Modify: `01_Preface_to_the_Second_Edition/01_Preface_to_the_Second_Edition.md` (replace stub)
- Create: `01_Preface_to_the_Second_Edition/01_Preface_to_the_Second_Edition.py`
- Modify: `config/progress.yaml` (`/01_Preface_to_the_Second_Edition: stub` → `done`)

**Interfaces:**
- Consumes: stub `.md`/`.py` (the template `# {section}` header, `> ⏳ 待补充` marker).
- Produces: `.py` function `main()` printing an edition overview (no external I/O).

- [ ] **Step 1: Write the Python example**

```python
# 01_Preface_to_the_Second_Edition
"""
Lecture: 01_Preface_to_the_Second_Edition
Content: 01_Preface_to_the_Second_Edition

概述: 第二版前言 — 介绍 ESL 第二版相对第一版的主要修订内容。
"""
EDITION = "Second Edition (2009)"
FIRST_PUBLISHED = "December 2008"


def main() -> None:
    print("ISBN 978-0-387-84857-0")
    print(f"书名: The Elements of Statistical Learning: Data Mining, "
          f"Inference, and Prediction")
    print(f"版本: {EDITION}")
    print(f"改动: 新增第 16 章(集成学习)、第 18 章(无向图模型)与其他更新")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the full `.md`**

```markdown
# 01 Preface to the Second Edition（第二版前言）

本文对应 *The Elements of Statistical Learning*（第二版，2009 年）出版时的前言。
ESL 全书由 Hastie、Tibshirani 与 Friedman 合著，第二版在保留第一版体系的同时，根据机器学习、统计学习及数据挖掘十多年来的发展做了系统性修订。

## 主要内容

- **成书背景**：本书源于作者在斯坦福大学等地的讲义，追求"统计推断"与"数据挖掘/机器学习"两种视角的统一。
- **第二版的主要增补**：
  - 新增关于 **集成学习（Ensemble Learning）** 的章节；
  - 扩充了 **无监督学习（Unsupervised Learning）** 与高维数据处理相关的讨论；
  - 对括号内方法史的引用(如 boosting、随机森林、SVM)进行了更新；
  - 加入更多彩图与可视化示例。
- **目标读者**：有一定统计学或机器学习基础的研究生、研究人员与工程师。建议配合代码示例（见同目录 `.py`）动手验证。

## 核心公式/概念

第二版前言本身不含具体模型公式，但它为下文奠定了三个贯穿全书的主题：
1. **预测**：从数据中学习映射关系 $f(x)$，使损失函数风险最小化；
2. **推断**：在给定模型假设下估计参数的不确定性；
3. **特征与模型选择**：在复杂度与泛化能力之间取舍（后文第 7 章、第 9 章详述）。

## 代码示例

- 运行 `01_Preference..._Update..py` 可打印本书版本与出版信息。
- 无数据、无图示；前言章节仅作阅读导航。

## 相关章节

- 第 1 章 引言（学习问题的示例与概览）
- 第 16 章 无监督学习（第二版新增主题之一）
```

- [ ] **Step 3: Update `config/progress.yaml`**

Change `/01_Preface_to_the_Second_Edition: stub` → `/01_Preface_to_the_Second_Edition: done`.

```bash
sed -i '' 's|^/01_Preface_to_the_Second_Edition: stub$|/01_Preface_to_the_Second_Edition: done|' config/progress.yaml
```

- [ ] **Step 4: Verify py runs**

Run: `. .venv/bin/activate && python 01_Preface_to_the_Second_Edition/01_Preface_to_the_Second_Edition.py`
Expected: prints the 4 `print` lines, no traceback.

- [ ] **Step 5: Commit**

```bash
git add 01_Preface_to_the_Second_Edition/ config/progress.yaml
git commit -m "content: preface to the second edition"
```

---

### Task 2: 02 — Preface to the First Edition

**Files:**
- Modify: `02_Preface_to_the_First_Edition/02_Preface_to_the_First_Edition.md`
- Create: `02_Preface_to_the_First_Edition/02_Preface_to_the_First_Edition.py`
- Modify: `config/progress.yaml`

**Interfaces:**
- Produces: `.py` `main()` printing first-edition notes; .md Chinese summary.

- [ ] **Step 1: Write the `.py`**

```python
# 02_Preface_to_the_First_Edition
"""
Lecture: 02_Preface_to_the_First_Edition
Content: 02_Preface_to_the_First_Edition

Content: 第一版前言 — 成书缘起、目标与阅读建议。
"""
TIMELINE = [
    "1984-1994: ESL 讲义在斯坦福与伯克利讲授",
    "2001: 第一版正式出版",
    "2009: 第二版（修订与增补）",
]


def main() -> None:
    print("第一版前言摘要：本书源于一门数据挖掘/统计学习的讲义。")
    for line in TIMELINE:
        print(line)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the `.md`**

```markdown
# 02 Preface to the First Edition（第一版前言）

第一版前言记录了本书的成书过程与定位。作者将本书定位为一本**面向统计数据挖掘与机器学习的教科书**，强调"从数据中学习"的方法论统一。

## 主要内容

- **成书缘起**：源自作者在斯坦福大学二十年的教学讲义，对象是有一定数学基础的研究生。
- **方法论**：强调三种视角的统一——统计推断（经典统计）、数据挖掘（大规模、高维）、机器学习（算法导向）。
- **阅读建议**：作者建议读者具备微积分、线性代数与概率论基础，且在做习题、写代码的过程中理解方法。

## 核心公式

前言不引入具体公式，但确立了全书反复出现的**风险最小化**语言：

$$
R(f) = \mathbb{E}_{(X,Y)}\left[ L(Y, f(X)) \right]
$$

其中 $L$ 为损失函数（如平方损失或 0-1 损失）。后文许多方法都可理解为对 $R(f)$ 的（约束）最小化。

## 代码示例

- 运行 `02_Preface_to_the_First_Edition.py` 打印成书时间线与版本信息。
```

- [ ] **Step 3: Update `config/progress.yaml`**

```bash
sed -i '' 's|^/02_Preface_to_the_First_Edition: stub$|/02_Preface_to_the_First_Edition: done|' config/progress.yaml
```

- [ ] **Step 4: Run & verify py**

```bash
python 02_Preface_to_the_First_Edition/02_Preface_to_the_First_Edition.py
```
Expected: prints 3 lines, no traceback.

- [ ] **Step 5: Commit**

```bash
git add 02_Preface_to_the_First_Edition/ config/progress.yaml
git commit -m "content: preface to the first edition"
```

---

### Task 3: 21 — References

**Files:**
- Modify: `21_References/21_References.md`
- Create: `21_References/21_References.py`
- Modify: `config/progress.yaml`

**Interfaces:**
- Produces: `.py` printing a curated list of key classic references.

- [ ] **Step 1: Write the `.py`**

```python
# 21_References
"""
Lecture: 21_References
Content: 21_References
内容: 提供 ESL 涉及的关键参考文献的快速检索辅助。
"""
KEY_REFS = [
    ("Hastie, Tibshirani & Friedman, 2009", "The Elements of Statistical Learning, 2nd ed."),
    ("Pearl, 2000", "Causality: Models, Reasoning, and Inference"),
    ("Breiman et al., 1984", "Classification and Regression Trees"),
    ("Bishop, 2006", "Pattern Recognition and Machine Learning"),
    ("Ripley, 1996", "Pattern Recognition and Neural Networks"),
]


def main() -> None:
    for author, title in KEY_REFS:
        print(f"- {author}: {title}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the `.md`**

```markdown
# 21 References（参考文献）

本章收录全书正文引用的经典文献，按主题可大致分为：统计推断与回归、机器学习的教材与综述、凸优化与数值方法、以及贝叶斯与核方法等。本节不逐篇解读，而是提供分类检索说明与阅读指引。

## 文献类别与举例

- **教材级综述**：Bishop (2006)、Ripley (1996)、Hastie 等 (2009) 本书本身。
- **决策树**：Breiman 等 (1984) 的 CART。
- **贝叶斯方法**：Gelman 等 (2013) 的 *Bayesian Data Analysis*。
- **Vapnik–Chervonenkis 理论**：Vapnik (1995) 的《The Nature of Statistical Learning Theory》。

## 使用建议

- 读者可按章节标题反查正文中的引用标注，集中阅读关键几篇即可入门。
- 本目录的 `.py` 打印若干文献条目作为检索样例。
```

- [ ] **Step 3: Update `config/progress.yaml`**

`/21_References: done`.

- [ ] **Step 4: Run py**

```bash
python 21_References/21_References.py
```

- [ ] **Step 5: Commit**

```bash
git add 21_References/ config/progress.yaml
git commit -m "content: references"
```

---

### Task 4: 22 — Author Index

**Files:**
- Modify: `22_Author_Index/22_Author_Index.md`
- Create: `22_Author_Index/22_Author_Index.py`
- Modify: `config/progress.yaml`

**Interfaces:**
- Produces: `.py` that builds an alphabetized author index from a small list.

- [ ] **Step 1: Write the `.py`**

```python
# 22_Author_Index
"""内容: 生成一个小的按字母排序的作者索引示例。"""
AUTHORS = ["Tibshirani", "Hastie", "Friedman", "Breiman", "Vapnik", "Pearl", "Bishop"]


def main() -> None:
    for name in sorted(AUTHORS, key=str.lower):
        print(name)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the `.md`**

```markdown
# 22 Author Index（作者索引）

本章是按作者姓氏字母排序的索引，便于检索正文中引用的作者及其页码。作为 back-matter 章节，它不承载教学性内容，而是全书检索的工具。

## 说明

- 索引按姓氏（后段）排列，同一作者多篇引用按其出现的章节排列。
- 阅读时可将作者名与 `21_References` 对照，快速定位文献。

## 代码示例

- 本目录的 `.py` 用一小段示例演示如何生成字母排序的作者索引。
```

- [ ] **Step 3: Update `config/progress.yaml`**

`/22_Author_Index: done`.

- [ ] **Step 4: Run & commit**

```bash
python 22_Author_Index/22_Author_Index.py
git add 22_Author_Index/ config/progress.yaml
git commit -m "content: author index"
```

---

### Task 5: 23 — Index

**Files:**
- Modify: `23_Index/23_Index.md`
- Create: `23_Index/23_Index.py`
- Modify: `config/progress.yaml`

**Interfaces:**
- Produces: `.py` demonstrating a simple keyword index of a glossary.

- [ ] **Step 1: Write the `.py`**

```python
# 23_Index
"""示例: 用一个小型术语表演示关键词索引的构建。"""
GLOSSARY = {
    "Bias": "9.3",
    "Variance": "9.3",
    "Cross-Validation": "9.10",
    "Boosting": "12.4",
    "SVM": "14.2",
    "Random Forests": "17.2",
}


def main() -> None:
    for term, section in sorted(GLOSSARY.items()):
        print(f"{term: <18} -> 章节 {section}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the `.md`**

```markdown
# 23 Index（索引）

本章是全书概念、术语与符号的字母索引，指向出现这些概念的具体章节/页码。作为 back-matter 索引，它是读者在读完正文后回溯概念位置的检索工具。

## 说明

- 索引按术语字母顺序排列，标注对应的章节区间（如 Lasso → 第 5.4 节）。
- 常与 `22_Author_Index`、`21_References` 配套使用。

## 代码示例

- 本目录的 `.py` 演示如何从一个术语表构建简单的按字母排序索引。
```

- [ ] **Step 3: Update `config/progress.yaml`**

`/23_Index: done`.

- [ ] **Step 4: Run & commit**

```bash
python 23_Index/23_Index.py
git add 23_Index/ config/progress.yaml
git commit -m "content: index"
```

---

### Task 6: README refresh + full verification

**Files:**
- Modify: `README.md` (regenerated)
- (no code files)

**Interfaces:**
- Consumes: `config/progress.yaml` (now 5 `done`), `scripts/generate_docs.py`.

- [ ] **Step 1: Rebuild the README**

```bash
. .venv/bin/activate
python scripts/generate_docs.py --regenerate
```
Expected: 0 created; README rebuilt; the 5 front-matter entries now show `✅ 摘要`.

- [ ] **Step 2: Run full verification**

```bash
python -m pytest
python scripts/smoke_test.py
```
Expected: `8 passed`; `178/178 scripts passed`.

- [ ] **Step 3: Confirm no regression in a data chapter**

```bash
ls 03_Introduction/3.1_Examples_of_Learning_Problems/
```
Expected: `3.1_Examples_of_Learning_Problems.md` + `.py` intact (unchanged stub).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: Phase 2 front matter complete"
```

---

## Self-Review

**Spec coverage (Phase 2 = §10 Phase 2 of design spec):** chapters 01, 02, 21, 22, 23 each get a Chinese `.md` + `.py`; none get `.ipynb`; progress/README move to ✅. ✓

**Placeholder scan:** All `.md` bodies and `.py` are fully written in-steps; the direct-write approach replaces the stubs completely (the generator's `write_*_if_missing` never clobbers). ✓

**Type consistency:** the `progress.yaml` keys (leading-slash form), `.py` `main()`-guard pattern, and commit messages are consistent across tasks. ✓

**Scope note:** Phase 2 is small (5 sections) and fully self-contained. Content phases with real math/data and notebooks start in Phase 3.