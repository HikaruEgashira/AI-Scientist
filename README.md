<h1 align="center">
  <a href="https://github.com/SakanaAI/AI-Scientist/blob/main/docs/logo_2.png">
    <img src="docs/logo_2.png" width="215" /></a><br>
  <b>The AI Scientist: Towards Fully Automated</b><br>
  <b>Open-Ended Scientific Discovery 🧑‍🔬</b><br>
</h1>

<p align="center">
  📚 <a href="https://arxiv.org/abs/2408.06292">[Paper]</a> |
  📝 <a href="https://sakana.ai/ai-scientist/">[Blog Post]</a> |
  📂 <a href="https://drive.google.com/drive/folders/1G7A0wTqfXVa-cpexjk0oaXakaSJwffEt">[Drive Folder]</a>
</p>

One of the grand challenges of artificial intelligence is developing agents capable of conducting scientific research and discovering new knowledge. While frontier models have already been used to aid human scientists—for example, for brainstorming ideas or writing code—they still require extensive manual supervision or are heavily constrained to specific tasks.

We're excited to introduce **The AI Scientist**, the first comprehensive system for fully automatic scientific discovery, enabling Foundation Models such as Large Language Models (LLMs) to perform research independently.

We provide all runs and data from our paper [here](https://drive.google.com/drive/folders/1G7A0wTqfXVa-cpexjk0oaXakaSJwffEt?usp=sharing), where we run each base model on each template for approximately 50 ideas. We *highly* recommend reading through some of the [Claude papers](https://drive.google.com/drive/folders/1Mmpz6M1FK4q8e-SewgZcUzdeD0Q2zC39?usp=sharing) to get a sense of the system's strengths and weaknesses. Here are some example papers generated by **The AI Scientist** 📝:

1. [DualScale Diffusion: Adaptive Feature Balancing for Low-Dimensional Generative Models](https://github.com/SakanaAI/AI-Scientist/blob/main/example_papers/adaptive_dual_scale_denoising.pdf)
2. [Multi-scale Grid Noise Adaptation: Enhancing Diffusion Models For Low-dimensional Data](https://github.com/SakanaAI/AI-Scientist/blob/main/example_papers/grid_based_noise_adaptation.pdf)
3. [GAN-Enhanced Diffusion: Boosting Sample Quality and Diversity](https://github.com/SakanaAI/AI-Scientist/blob/main/example_papers/gan_diffusion.pdf)
4. [DualDiff: Enhancing Mode Capture in Low-dimensional Diffusion Models via Dual-expert Denoising](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/dual_expert_denoiser.pdf) 
5. [StyleFusion: Adaptive Multi-style Generation in Character-Level Language Models](https://github.com/SakanaAI/AI-Scientist/blob/main/example_papers/multi_style_adapter.pdf)
6. [Adaptive Learning Rates for Transformers via Q-Learning](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/rl_lr_adaptation.pdf)
7. [Unlocking Grokking: A Comparative Study of Weight Initialization Strategies in Transformer Models](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/weight_initialization_grokking.pdf)
8. [Grokking Accelerated: Layer-wise Learning Rates for Transformer Generalization](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/layerwise_lr_grokking.pdf)
9. [Grokking Through Compression: Unveiling Sudden Generalization via Minimal Description Length](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/mdl_grokking_correlation.pdf)
10. [Accelerating Mathematical Insight: Boosting Grokking Through Strategic Data Augmentation](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/data_augmentation_grokking.pdf)

> **Note:**  
> **Caution!** This codebase will execute LLM-written code. There are various risks and challenges associated with this autonomy, including the use of potentially dangerous packages, web access, and potential spawning of processes. Use at your own discretion. Please make sure to [containerize](#containerization) and restrict web access appropriately.

<p align="center">
  <a href="https://github.com/SakanaAI/AI-Scientist/blob/main/example_papers/adaptive_dual_scale_denoising/adaptive_dual_scale_denoising.pdf"><img src="https://github.com/SakanaAI/AI-Scientist/blob/main/docs/anim-ai-scientist.gif" alt="Adaptive Dual Scale Denoising" width="80%" />
</a></p>

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
   - [Installation](#installation)
   - [Supported Models and API Keys](#supported-models-and-api-keys)
3. [Setting Up the Templates](#setting-up-the-templates)
   - [NanoGPT Template](#nanogpt-template)
   - [2D Diffusion Template](#2d-diffusion-template)
   - [Grokking Template](#grokking-template)
4. [Run AI Scientist Paper Generation Experiments](#run-ai-scientist-paper-generation-experiments)
5. [Getting an LLM-Generated Paper Review](#getting-an-llm-generated-paper-review)
6. [Making Your Own Template](#making-your-own-template)
   - [Community-Contributed Templates](#community-contributed-templates)
7. [Template Resources](#template-resources)
8. [Citing The AI Scientist](#citing-the-ai-scientist)
9. [Frequently Asked Questions](#frequently-asked-questions)
10. [Containerization](#containerization)

## Introduction

We provide three templates, which were used in our paper, covering the following domains: **NanoGPT**, **2D Diffusion**, and **Grokking**. These templates enable The AI Scientist to generate ideas and conduct experiments in these areas. We accept contributions of new templates from the community, but please note that they are not maintained by us. All other templates beyond the three provided are community contributions.

## Requirements

This code is designed to run on Linux with NVIDIA GPUs using CUDA and PyTorch. Support for other GPU architectures may be possible by following the [PyTorch guidelines](https://pytorch.org/get-started/locally/). The current templates would likely take an infeasible amount of time on CPU-only machines. Running on other operating systems may require significant adjustments.

### Installation

```bash
conda create -n ai_scientist python=3.11
conda activate ai_scientist
# Install pdflatex
sudo apt-get install texlive-full

# Install PyPI requirements
pip install -r requirements.txt
```

**Note:** Installing `texlive-full` can take a long time. You may need to [hold Enter](https://askubuntu.com/questions/956006/pregenerating-context-markiv-format-this-may-take-forever) during the installation.

### Supported Models and API Keys

We support a wide variety of models, including open-weight and API-only models. In general, we recommend using only frontier models above the capability of the original GPT-4. To see a full list of supported models, see [here](https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/llm.py).

#### OpenAI API (GPT-4o, GPT-4o-mini, o1 models)

By default, this uses the `OPENAI_API_KEY` environment variable.

#### Anthropic API (Claude Sonnet 3.5)

By default, this uses the `ANTHROPIC_API_KEY` environment variable.

##### Claude Models via Bedrock

For Claude models provided by [Amazon Bedrock](https://aws.amazon.com/bedrock/), please install these additional packages:

```bash
pip install anthropic[bedrock]
```

Next, specify a set of valid [AWS Credentials](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-envvars.html) and the target [AWS Region](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html):

Set the environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME`.

##### Claude Models via Vertex AI

For Claude models provided by [Vertex AI Model Garden](https://cloud.google.com/model-garden?hl=en), please install these additional packages:

```bash
pip install google-cloud-aiplatform
pip install anthropic[vertex]
```

Next, set up valid authentication for a [Google Cloud project](https://cloud.google.com/vertex-ai/docs/authentication), for example by providing the region and project ID:

```bash
export CLOUD_ML_REGION="REGION"           # for Model Garden call
export ANTHROPIC_VERTEX_PROJECT_ID="PROJECT_ID"  # for Model Garden call
export VERTEXAI_LOCATION="REGION"         # for Aider/LiteLLM call
export VERTEXAI_PROJECT="PROJECT_ID"      # for Aider/LiteLLM call
```

#### DeepSeek API (DeepSeek-Coder-V2)

By default, this uses the `DEEPSEEK_API_KEY` environment variable.

#### OpenRouter API (Llama3.1)

By default, this uses the `OPENROUTER_API_KEY` environment variable.

#### Semantic Scholar API (Literature Search)

Our code can also optionally use a Semantic Scholar API Key (`S2_API_KEY`) for higher throughput [if you have one](https://www.semanticscholar.org/product/api), though it should work without it in principle. If you have problems with Semantic Scholar, you can skip the literature search and citation phases of paper generation.

Be sure to provide the key for the model used for your runs, e.g.:

```bash
export OPENAI_API_KEY="YOUR KEY HERE"
export S2_API_KEY="YOUR KEY HERE"
```

## Setting Up the Templates

This section provides instructions for setting up each of the three templates used in our paper. Before running The AI Scientist experiments, please ensure you have completed the setup steps for the templates you are interested in.

### NanoGPT Template

**Description:** This template investigates transformer-based autoregressive next-token prediction tasks.

**Setup Steps:**

1. **Prepare the data:**

   ```bash
   python data/enwik8/prepare.py
   python data/shakespeare_char/prepare.py
   python data/text8/prepare.py
   ```

2. **Create baseline runs (machine dependent):**

   ```bash
   # Set up NanoGPT baseline run
   # NOTE: YOU MUST FIRST RUN THE PREPARE SCRIPTS ABOVE!
   cd templates/nanoGPT
   python experiment.py --out_dir run_0
   python plot.py
   ```

### 2D Diffusion Template

**Description:** This template studies improving the performance of diffusion generative models on low-dimensional datasets.

**Setup Steps:**

1. **Install dependencies:**

   ```bash
   # Set up 2D Diffusion
   git clone https://github.com/gregversteeg/NPEET.git
   cd NPEET
   pip install .
   pip install scikit-learn
   ```

2. **Create baseline runs:**

   ```bash
   # Set up 2D Diffusion baseline run
   cd templates/2d_diffusion
   python experiment.py --out_dir run_0
   python plot.py
   ```

### Grokking Template

**Description:** This template investigates questions about generalization and learning speed in deep neural networks.

**Setup Steps:**

1. **Install dependencies:**

   ```bash
   # Set up Grokking
   pip install einops
   ```

2. **Create baseline runs:**

   ```bash
   # Set up Grokking baseline run
   cd templates/grokking
   python experiment.py --out_dir run_0
   python plot.py
   ```

## Run AI Scientist Paper Generation Experiments

**Note:** Please ensure the setup steps above are completed before running these experiments.

```bash
conda activate ai_scientist
# Run the paper generation.
python launch_scientist.py --model "gpt-4o-2024-05-13" --experiment nanoGPT_lite --num-ideas 2
python launch_scientist.py --model "claude-3-5-sonnet-20241022" --experiment nanoGPT_lite --num-ideas 2
```

If you have more than one GPU, use the `--parallel` option to parallelize ideas across multiple GPUs.

## Getting an LLM-Generated Paper Review

```python
import openai
from ai_scientist.perform_review import load_paper, perform_review

client = openai.OpenAI()
model = "gpt-4o-2024-05-13"

# Load paper from PDF file (raw text)
paper_txt = load_paper("report.pdf")

# Get the review dictionary
review = perform_review(
    paper_txt,
    model,
    client,
    num_reflections=5,
    num_fs_examples=1,
    num_reviews_ensemble=5,
    temperature=0.1,
)

# Inspect review results
review["Overall"]    # Overall score (1-10)
review["Decision"]   # 'Accept' or 'Reject'
review["Weaknesses"] # List of weaknesses (strings)
```

To run batch analysis:

```bash
cd review_iclr_bench
python iclr_analysis.py --num_reviews 500 --batch_size 100 --num_fs_examples 1 --num_reflections 5 --temperature 0.1 --num_reviews_ensemble 5
```

## Making Your Own Template

If there is an area of study you would like **The AI Scientist** to explore, it is straightforward to create your own templates. In general, follow the structure of the existing templates, which consist of:

- `experiment.py` — This is the main script where the core content is. It takes an argument `--out_dir`, which specifies where it should create the folder and save the relevant information from the run.
- `plot.py` — This script takes the information from the `run` folders and creates plots. The code should be clear and easy to edit.
- `prompt.json` — Put information about your template here.
- `seed_ideas.json` — Place example ideas here. You can also try to generate ideas without any examples and then pick the best one or two to put here.
- `latex/template.tex` — We recommend using our LaTeX folder but be sure to replace the pre-loaded citations with ones that you expect to be more relevant.

The key to making new templates work is matching the base filenames and output JSONs to the existing format; everything else is free to change.
You should also ensure that the `template.tex` file is updated to use the correct citation style / base plots for your template.

### Community-Contributed Templates

We welcome community contributions in the form of new templates. While these are not maintained by us, we are delighted to highlight your templates to others. Below, we list community-contributed templates along with links to their pull requests (PRs):

- Infectious Disease Modeling (`seir`) - [PR #137](https://github.com/SakanaAI/AI-Scientist/pull/137)
- Image Classification with MobileNetV3 (`mobilenetV3`) - [PR #141](https://github.com/SakanaAI/AI-Scientist/pull/141)
- Sketch RNN (`sketch_rnn`) - [PR #143](https://github.com/SakanaAI/AI-Scientist/pull/143)

*This section is reserved for community contributions. Please submit a pull request to add your template to the list! Please describe the template in the PR description, and also show examples of the generated papers.*

## Template Resources

We provide three templates, which heavily use code from other repositories, credited below:

- **NanoGPT Template** uses code from [NanoGPT](https://github.com/karpathy/nanoGPT) and this [PR](https://github.com/karpathy/nanoGPT/pull/254).
- **2D Diffusion Template** uses code from [tiny-diffusion](https://github.com/tanelp/tiny-diffusion), [ema-pytorch](https://github.com/lucidrains/ema-pytorch), and [Datasaur](https://www.research.autodesk.com/publications/same-stats-different-graphs/).
- **Grokking Template** uses code from [Sea-Snell/grokking](https://github.com/Sea-Snell/grokking) and [danielmamay/grokking](https://github.com/danielmamay/grokking).

We would like to thank the developers of the open-source models and packages for their contributions and for making their work available.

## Citing The AI Scientist

If you use **The AI Scientist** in your research, please cite it as follows:

```
@article{lu2024aiscientist,
  title={The {AI} {S}cientist: Towards Fully Automated Open-Ended Scientific Discovery},
  author={Lu, Chris and Lu, Cong and Lange, Robert Tjarko and Foerster, Jakob and Clune, Jeff and Ha, David},
  journal={arXiv preprint arXiv:2408.06292},
  year={2024}
}
```

## Frequently Asked Questions

We recommend reading our paper first for any questions you have on The AI Scientist.

**Why am I missing files when running The AI Scientist?**

Ensure you have completed all the setup and preparation steps before the main experiment script.

**Why has a PDF or a review not been generated?**

The AI Scientist finishes an idea with a success rate that depends on the template, the base foundation model, and the complexity of the idea. We advise referring to our main paper. The highest success rates are observed with Claude Sonnet 3.5. Reviews are best done with GPT-4o; all other models have issues with positivity bias or failure to conform to required outputs.

**What is the cost of each idea generated?**

Typically less than $15 per paper with Claude Sonnet 3.5. We recommend DeepSeek Coder V2 for a much more cost-effective approach. A good place to look for new models is the [Aider leaderboard](https://aider.chat/docs/leaderboards/).

**How do I change the base conference format associated with the write-ups?**

Change the base `template.tex` files contained within each template.

**How do I run The AI Scientist for different subject fields?**

Please refer to the instructions for different templates. In this current iteration, this is restricted to ideas that can be expressed in code. However, lifting this restriction would represent exciting future work! :)

**How do I add support for a new foundation model?**

You may modify `ai_scientist/llm.py` to add support for a new foundation model. We do not advise using any model that is significantly weaker than GPT-4 level for **The AI Scientist**.

**Why do I need to run the baseline runs myself?**

These appear as `run_0` and should be run per machine you execute **The AI Scientist** on for accurate run-time comparisons due to hardware differences.

**What if I have problems accessing the Semantic Scholar API?**

We use the Semantic Scholar API to check ideas for novelty and collect citations for the paper write-up. You may be able to skip these phases if you don't have an API key or the API is slow to access.

## Containerization

We include a [community-contributed](https://github.com/SakanaAI/AI-Scientist/pull/21) Docker image that may assist with your containerization efforts in `experimental/Dockerfile`.

You can use this image like this:

```bash
# Endpoint Script
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -v `pwd`/templates:/app/AI-Scientist/templates <AI_SCIENTIST_IMAGE> \
  --model gpt-4o-2024-05-13 \
  --experiment 2d_diffusion \
  --num-ideas 2
```

```bash
# Interactive
docker run -it -e OPENAI_API_KEY=$OPENAI_API_KEY \
  --entrypoint /bin/bash \
  <AI_SCIENTIST_IMAGE>
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=SakanaAI/AI-Scientist&type=Date)](https://star-history.com/#SakanaAI/AI-Scientist&Date)

## 目次

1. [はじめに](#はじめに)
2. [要件](#要件)
   - [インストール](#インストール)
   - [サポートされているモデルとAPIキー](#サポートされているモデルとAPIキー)
3. [テンプレートの設定](#テンプレートの設定)
   - [NanoGPTテンプレート](#nanogptテンプレート)
   - [2D Diffusionテンプレート](#2d-diffusionテンプレート)
   - [Grokkingテンプレート](#grokkingテンプレート)
4. [AI Scientistの論文生成実験を実行する](#ai-scientistの論文生成実験を実行する)
5. [LLM生成の論文レビューを取得する](#llm生成の論文レビューを取得する)
6. [独自のテンプレートを作成する](#独自のテンプレートを作成する)
   - [コミュニティが提供するテンプレート](#コミュニティが提供するテンプレート)
7. [テンプレートリソース](#テンプレートリソース)
8. [AI Scientistの引用](#ai-scientistの引用)
9. [よくある質問](#よくある質問)
10. [コンテナ化](#コンテナ化)

## はじめに

私たちは、論文で使用した3つのテンプレートを提供しています。これらは、**NanoGPT**、**2D Diffusion**、および**Grokking**の領域をカバーしています。これらのテンプレートを使用して、AI Scientistがこれらの分野でアイデアを生成し、実験を行うことができます。コミュニティからの新しいテンプレートの貢献を受け付けていますが、これらは私たちによって維持されていないことに注意してください。提供された3つのテンプレート以外のすべてのテンプレートはコミュニティの貢献です。

## 要件

このコードは、CUDAとPyTorchを使用してNVIDIA GPUを搭載したLinuxで実行するように設計されています。他のGPUアーキテクチャのサポートは、[PyTorchのガイドライン](https://pytorch.org/get-started/locally/)に従うことで可能です。現在のテンプレートは、CPUのみのマシンでは実行に非常に長い時間がかかる可能性があります。他のオペレーティングシステムでの実行には、かなりの調整が必要です。

### インストール

```bash
conda create -n ai_scientist python=3.11
conda activate ai_scientist
# pdflatexをインストール
sudo apt-get install texlive-full

# PyPIの要件をインストール
pip install -r requirements.txt
```

**注:** `texlive-full`のインストールには時間がかかることがあります。インストール中に[Enterキーを押し続ける](https://askubuntu.com/questions/956006/pregenerating-context-markiv-format-this-may-take-forever)必要があるかもしれません。

### サポートされているモデルとAPIキー

私たちは、オープンウェイトモデルとAPI専用モデルを含むさまざまなモデルをサポートしています。一般的に、元のGPT-4の能力を超えるフロンティアモデルのみを使用することをお勧めします。サポートされているモデルの完全なリストについては、[こちら](https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/llm.py)を参照してください。

#### OpenAI API (GPT-4o, GPT-4o-mini, o1モデル)

デフォルトでは、`OPENAI_API_KEY`環境変数を使用します。

#### Anthropic API (Claude Sonnet 3.5)

デフォルトでは、`ANTHROPIC_API_KEY`環境変数を使用します。

##### Bedrock経由のClaudeモデル

[Amazon Bedrock](https://aws.amazon.com/bedrock/)が提供するClaudeモデルの場合、以下の追加パッケージをインストールしてください：

```bash
pip install anthropic[bedrock]
```

次に、有効な[AWS認証情報](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-envvars.html)とターゲット[AWSリージョン](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html)を指定します：

環境変数を設定します：`AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`、`AWS_REGION_NAME`。

##### Vertex AI経由のClaudeモデル

[Vertex AI Model Garden](https://cloud.google.com/model-garden?hl=en)が提供するClaudeモデルの場合、以下の追加パッケージをインストールしてください：

```bash
pip install google-cloud-aiplatform
pip install anthropic[vertex]
```

次に、有効な[Google Cloudプロジェクト](https://cloud.google.com/vertex-ai/docs/authentication)の認証を設定します。たとえば、リージョンとプロジェクトIDを提供します：

```bash
export CLOUD_ML_REGION="REGION"           # Model Gardenの呼び出し用
export ANTHROPIC_VERTEX_PROJECT_ID="PROJECT_ID"  # Model Gardenの呼び出し用
export VERTEXAI_LOCATION="REGION"         # Aider/LiteLLMの呼び出し用
export VERTEXAI_PROJECT="PROJECT_ID"      # Aider/LiteLLMの呼び出し用
```

#### DeepSeek API (DeepSeek-Coder-V2)

デフォルトでは、`DEEPSEEK_API_KEY`環境変数を使用します。

#### OpenRouter API (Llama3.1)

デフォルトでは、`OPENROUTER_API_KEY`環境変数を使用します。

#### Semantic Scholar API (文献検索)

私たちのコードは、オプションでSemantic Scholar APIキー（`S2_API_KEY`）を使用して、[高スループット](https://www.semanticscholar.org/product/api)を実現できますが、原則としてそれなしでも動作するはずです。Semantic Scholarに問題がある場合は、文献検索と論文生成の引用フェーズをスキップできます。

使用するモデルのキーを必ず提供してください。例：

```bash
export OPENAI_API_KEY="YOUR KEY HERE"
export S2_API_KEY="YOUR KEY HERE"
```

## テンプレートの設定

このセクションでは、論文で使用した3つのテンプレートの設定手順を提供します。AI Scientistの実験を実行する前に、興味のあるテンプレートの設定手順を完了してください。

### NanoGPTテンプレート

**説明:** このテンプレートは、トランスフォーマーベースの自己回帰次トークン予測タスクを調査します。

**設定手順:**

1. **データの準備:**

   ```bash
   python data/enwik8/prepare.py
   python data/shakespeare_char/prepare.py
   python data/text8/prepare.py
   ```

2. **ベースラインランの作成（マシン依存）:**

   ```bash
   # NanoGPTベースラインランの設定
   # 注: まず上記の準備スクリプトを実行する必要があります！
   cd templates/nanoGPT
   python experiment.py --out_dir run_0
   python plot.py
   ```

### 2D Diffusionテンプレート

**説明:** このテンプレートは、低次元データセットでの拡散生成モデルの性能向上を研究します。

**設定手順:**

1. **依存関係のインストール:**

   ```bash
   # 2D Diffusionの設定
   git clone https://github.com/gregversteeg/NPEET.git
   cd NPEET
   pip install .
   pip install scikit-learn
   ```

2. **ベースラインランの作成:**

   ```bash
   # 2D Diffusionベースラインランの設定
   cd templates/2d_diffusion
   python experiment.py --out_dir run_0
   python plot.py
   ```

### Grokkingテンプレート

**説明:** このテンプレートは、ディープニューラルネットワークにおける一般化と学習速度に関する質問を調査します。

**設定手順:**

1. **依存関係のインストール:**

   ```bash
   # Grokkingの設定
   pip install einops
   ```

2. **ベースラインランの作成:**

   ```bash
   # Grokkingベースラインランの設定
   cd templates/grokking
   python experiment.py --out_dir run_0
   python plot.py
   ```

## AI Scientistの論文生成実験を実行する

**注:** これらの実験を実行する前に、上記の設定手順を完了してください。

```bash
conda activate ai_scientist
# 論文生成を実行
python launch_scientist.py --model "gpt-4o-2024-05-13" --experiment nanoGPT_lite --num-ideas 2
python launch_scientist.py --model "claude-3-5-sonnet-20241022" --experiment nanoGPT_lite --num-ideas 2
```

複数のGPUを持っている場合は、`--parallel`オプションを使用してアイデアを複数のGPUに並列化します。

## LLM生成の論文レビューを取得する

```python
import openai
from ai_scientist.perform_review import load_paper, perform_review

client = openai.OpenAI()
model = "gpt-4o-2024-05-13"

# PDFファイルから論文を読み込む（生テキスト）
paper_txt = load_paper("report.pdf")

# レビューディクショナリを取得
review = perform_review(
    paper_txt,
    model,
    client,
    num_reflections=5,
    num_fs_examples=1,
    num_reviews_ensemble=5,
    temperature=0.1,
)

# レビュー結果を確認
review["Overall"]    # 全体スコア（1-10）
review["Decision"]   # 'Accept'または'Reject'
review["Weaknesses"] # 弱点のリスト（文字列）
```

バッチ分析を実行するには：

```bash
cd review_iclr_bench
python iclr_analysis.py --num_reviews 500 --batch_size 100 --num_fs_examples 1 --num_reflections 5 --temperature 0.1 --num_reviews_ensemble 5
```

## 独自のテンプレートを作成する

**The AI Scientist**に探求してほしい研究分野がある場合は、独自のテンプレートを作成するのは簡単です。一般的に、既存のテンプレートの構造に従います。これらは次のように構成されています：

- `experiment.py` — これは、コアコンテンツが含まれるメインスクリプトです。`--out_dir`引数を取り、実行結果を保存するフォルダを指定します。
- `plot.py` — このスクリプトは、`run`フォルダから情報を取得し、プロットを作成します。コードは明確で編集しやすいものであるべきです。
- `prompt.json` — テンプレートに関する情報をここに記載します。
- `seed_ideas.json` — ここに例のアイデアを配置します。例がなくてもアイデアを生成し、最良のものを1つか2つ選んでここに配置することもできます。
- `latex/template.tex` — 私たちのLaTeXフォルダを使用することをお勧めしますが、事前に読み込まれた引用を期待されるものに置き換えることを確認してください。

新しいテンプレートを機能させるための鍵は、基本的なファイル名と出力JSONを既存の形式に一致させることです。それ以外のすべては自由に変更できます。
また、`template.tex`ファイルがテンプレートに適した正しい引用スタイル/基本プロットを使用するように更新されていることを確認する必要があります。

### コミュニティが提供するテンプレート

私たちは、新しいテンプレートの形でのコミュニティの貢献を歓迎します。これらは私たちによって維持されていませんが、他の人々にあなたのテンプレートを紹介することを喜んでいます。以下に、コミュニティが提供するテンプレートとそのプルリクエスト（PR）へのリンクを示します：

- 感染症モデリング（`seir`） - [PR #137](https://github.com/SakanaAI/AI-Scientist/pull/137)
- MobileNetV3を使用した画像分類（`mobilenetV3`） - [PR #141](https://github.com/SakanaAI/AI-Scientist/pull/141)
- Sketch RNN（`sketch_rnn`） - [PR #143](https://github.com/SakanaAI/AI-Scientist/pull/143)

*このセクションはコミュニティの貢献のために予約されています。テンプレートをリストに追加するためにプルリクエストを提出してください！PRの説明にテンプレートを説明し、生成された論文の例も示してください。*

## テンプレートリソース

私たちは、他のリポジトリからのコードを多用している3つのテンプレートを提供しています。以下にクレジットを示します：

- **NanoGPTテンプレート**は、[NanoGPT](https://github.com/karpathy/nanoGPT)とこの[PR](https://github.com/karpathy/nanoGPT/pull/254)のコードを使用しています。
- **2D Diffusionテンプレート**は、[tiny-diffusion](https://github.com/tanelp/tiny-diffusion)、[ema-pytorch](https://github.com/lucidrains/ema-pytorch)、および[Datasaur](https://www.research.autodesk.com/publications/same-stats-different-graphs/)のコードを使用しています。
- **Grokkingテンプレート**は、[Sea-Snell/grokking](https://github.com/Sea-Snell/grokking)および[danielmamay/grokking](https://github.com/danielmamay/grokking)のコードを使用しています。

オープンソースモデルとパッケージの開発者に感謝し、その作業を利用できるようにしてくれたことに感謝します。

## AI Scientistの引用

**The AI Scientist**を研究に使用する場合は、次のように引用してください：

```
@article{lu2024aiscientist,
  title={The {AI} {S}cientist: Towards Fully Automated Open-Ended Scientific Discovery},
  author={Lu, Chris and Lu, Cong and Lange, Robert Tjarko and Foerster, Jakob and Clune, Jeff and Ha, David},
  journal={arXiv preprint arXiv:2408.06292},
  year={2024}
}
```

## よくある質問

The AI Scientistに関する質問がある場合は、まず私たちの論文を読むことをお勧めします。

**The AI Scientistを実行する際にファイルが見つからないのはなぜですか？**

メインの実験スクリプトを実行する前に、すべての設定と準備手順を完了していることを確認してください。

**PDFやレビューが生成されていないのはなぜですか？**

The AI Scientistは、テンプレート、基礎となるモデル、およびアイデアの複雑さに応じて、成功率が異なります。私たちのメインの論文を参照することをお勧めします。最も高い成功率はClaude Sonnet 3.5で観察されます。レビューはGPT-4oで行うのが最適です。他のすべてのモデルは、ポジティブバイアスや必要な出力に従わない問題があります。

**生成されるアイデアのコストはどれくらいですか？**

通常、Claude Sonnet 3.5で1論文あたり15ドル未満です。よりコスト効果の高いアプローチとして、DeepSeek Coder V2をお勧めします。新しいモデルを探す良い場所は、[Aiderリーダーボード](https://aider.chat/docs/leaderboards/)です。

**書き起こしに関連する基本的な会議フォーマットを変更するにはどうすればよいですか？**

各テンプレートに含まれる基本的な`template.tex`ファイルを変更します。

**異なる分野の研究をThe AI Scientistで実行するにはどうすればよいですか？**

異なるテンプレートの指示を参照してください。この現在のバージョンでは、コードで表現できるアイデアに制限されています。ただし、この制限を解除することは、将来のエキサイティングな作業を表しています！ :)

**新しい基礎モデルのサポートを追加するにはどうすればよいですか？**

新しい基礎モデルのサポートを追加するには、`ai_scientist/llm.py`を変更することができます。**The AI Scientist**には、GPT-4レベルよりも大幅に弱いモデルを使用することはお勧めしません。

**なぜベースラインランを自分で実行する必要があるのですか？**

これらは`run_0`として表示され、ハードウェアの違いによる実行時間の比較のために、**The AI Scientist**を実行する各マシンで実行する必要があります。

**Semantic Scholar APIにアクセスする際に問題がある場合はどうすればよいですか？**

私たちは、アイデアの新規性を確認し、論文の書き起こしのための引用を収集するためにSemantic Scholar APIを使用しています。APIキーがない場合やAPIへのアクセスが遅い場合は、これらのフェーズをスキップできるかもしれません。

## コンテナ化

`experimental/Dockerfile`に、コンテナ化の取り組みに役立つ[コミュニティが提供する](https://github.com/SakanaAI/AI-Scientist/pull/21)Dockerイメージを含めています。

このイメージを次のように使用できます：

```bash
# エンドポイントスクリプト
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -v `pwd`/templates:/app/AI-Scientist/templates <AI_SCIENTIST_IMAGE> \
  --model gpt-4o-2024-05-13 \
  --experiment 2d_diffusion \
  --num-ideas 2
```

```bash
# インタラクティブ
docker run -it -e OPENAI_API_KEY=$OPENAI_API_KEY \
  --entrypoint /bin/bash \
  <AI_SCIENTIST_IMAGE>
```

## スターヒストリー

[![Star History Chart](https://api.star-history.com/svg?repos=SakanaAI/AI-Scientist&type=Date)](https://star-history.com/#SakanaAI/AI-Scientist&Date)
