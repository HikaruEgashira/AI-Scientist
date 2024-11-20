<h1 align="center">
  <a href="https://github.com/SakanaAI/AI-Scientist/blob/main/docs/logo_2.png">
    <img src="docs/logo_2.png" width="215" /></a><br>
  <b>AIサイエンティスト: 完全自動化された</b><br>
  <b>オープンエンドの科学的発見 🧑‍🔬</b><br>
</h1>

<p align="center">
  📚 <a href="https://arxiv.org/abs/2408.06292">[論文]</a> |
  📝 <a href="https://sakana.ai/ai-scientist/">[ブログ記事]</a> |
  📂 <a href="https://drive.google.com/drive/folders/1G7A0wTqfXVa-cpexjk0oaXakaSJwffEt">[ドライブフォルダ]</a>
</p>

人工知能の大きな課題の一つは、科学研究を行い、新しい知識を発見する能力を持つエージェントを開発することです。最先端のモデルはすでに人間の科学者を支援するために使用されていますが（例えば、アイデアのブレインストーミングやコードの作成など）、依然として広範な手動監督が必要であったり、特定のタスクに厳しく制約されています。

私たちは、**AIサイエンティスト**を紹介できることを嬉しく思います。これは、基盤モデル（例えば、大規模言語モデル（LLM））が独立して研究を行うことを可能にする、完全自動化された科学的発見のための最初の包括的なシステムです。

私たちの論文からのすべての実行とデータを[こちら](https://drive.google.com/drive/folders/1G7A0wTqfXVa-cpexjk0oaXakaSJwffEt?usp=sharing)で提供しています。各ベースモデルを各テンプレートで約50のアイデアに対して実行しています。システムの強みと弱みを理解するために、いくつかの[Claude論文](https://drive.google.com/drive/folders/1Mmpz6M1FK4q8e-SewgZcUzdeD0Q2zC39?usp=sharing)を読むことを強くお勧めします。以下は**AIサイエンティスト**によって生成された例の論文です 📝:

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

> **注意:**  
> **警告!** このコードベースはLLMによって書かれたコードを実行します。この自律性には、潜在的に危険なパッケージの使用、ウェブアクセス、およびプロセスの生成の可能性など、さまざまなリスクと課題が伴います。自己責任で使用してください。適切に[コンテナ化](#containerization)し、ウェブアクセスを制限することをお勧めします。

<p align="center">
  <a href="https://github.com/SakanaAI/AI-Scientist/blob/main/example_papers/adaptive_dual_scale_denoising/adaptive_dual_scale_denoising.pdf"><img src="https://github.com/SakanaAI/AI-Scientist/blob/main/docs/anim-ai-scientist.gif" alt="Adaptive Dual Scale Denoising" width="80%" />
</a></p>

## 目次

1. [イントロダクション](#introduction)
2. [要件](#requirements)
   - [インストール](#installation)
   - [サポートされているモデルとAPIキー](#supported-models-and-api-keys)
3. [テンプレートの設定](#setting-up-the-templates)
   - [NanoGPTテンプレート](#nanogpt-template)
   - [2D拡散テンプレート](#2d-diffusion-template)
   - [Grokkingテンプレート](#grokking-template)
4. [AIサイエンティスト論文生成実験の実行](#run-ai-scientist-paper-generation-experiments)
5. [LLM生成論文のレビュー取得](#getting-an-llm-generated-paper-review)
6. [独自テンプレートの作成](#making-your-own-template)
   - [コミュニティが提供するテンプレート](#community-contributed-templates)
7. [テンプレートリソース](#template-resources)
8. [AIサイエンティストの引用](#citing-the-ai-scientist)
9. [よくある質問](#frequently-asked-questions)
10. [コンテナ化](#containerization)
## イントロダクション

私たちは、論文で使用した3つのテンプレートを提供しています。これらは、**NanoGPT**、**2D Diffusion**、および**Grokking**のドメインをカバーしています。これらのテンプレートは、AIサイエンティストがこれらの分野でアイデアを生成し、実験を行うことを可能にします。コミュニティからの新しいテンプレートの貢献も受け付けていますが、それらは私たちによって維持されていません。提供された3つのテンプレート以外のすべてのテンプレートはコミュニティの貢献です。

## 要件

このコードは、CUDAおよびPyTorchを使用してNVIDIA GPU上でLinuxで実行するように設計されています。他のGPUアーキテクチャのサポートは、[PyTorchガイドライン](https://pytorch.org/get-started/locally/)に従うことで可能かもしれません。現在のテンプレートは、CPUのみのマシンでは実行に非常に長い時間がかかる可能性があります。他のオペレーティングシステムでの実行には、かなりの調整が必要です。

### インストール

```bash
conda create -n ai_scientist python=3.11
conda activate ai_scientist
# pdflatexをインストール
sudo apt-get install texlive-full

# PyPIの要件をインストール
pip install -r requirements.txt
```

**注意:** `texlive-full`のインストールには長い時間がかかることがあります。インストール中に[Enterを押し続ける](https://askubuntu.com/questions/956006/pregenerating-context-markiv-format-this-may-take-some-time-takes-forever)必要があるかもしれません。

### サポートされているモデルとAPIキー

私たちは、オープンウェイトモデルおよびAPI専用モデルを含むさまざまなモデルをサポートしています。一般的に、元のGPT-4の能力を超えるフロンティアモデルのみを使用することをお勧めします。サポートされているモデルの完全なリストについては、[こちら](https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/llm.py)を参照してください。

#### OpenAI API (GPT-4o, GPT-4o-mini, o1モデル)

デフォルトでは、`OPENAI_API_KEY`環境変数を使用します。

#### Anthropic API (Claude Sonnet 3.5)

デフォルトでは、`ANTHROPIC_API_KEY`環境変数を使用します。

##### Bedrock経由のClaudeモデル

[Amazon Bedrock](https://aws.amazon.com/bedrock/)が提供するClaudeモデルについては、以下の追加パッケージをインストールしてください：

```bash
pip install anthropic[bedrock]
```

次に、有効な[AWSクレデンシャル](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-envvars.html)とターゲット[AWSリージョン](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html)を指定します：

環境変数を設定します：`AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`、`AWS_REGION_NAME`。

##### Vertex AI経由のClaudeモデル

[Vertex AI Model Garden](https://cloud.google.com/model-garden?hl=en)が提供するClaudeモデルについては、以下の追加パッケージをインストールしてください：

```bash
pip install google-cloud-aiplatform
pip install anthropic[vertex]
```

次に、有効な[Google Cloudプロジェクト](https://cloud.google.com/vertex-ai/docs/authentication)の認証を設定します。例えば、リージョンとプロジェクトIDを指定します：

```bash
export CLOUD_ML_REGION="REGION"           # Model Garden呼び出し用
export ANTHROPIC_VERTEX_PROJECT_ID="PROJECT_ID"  # Model Garden呼び出し用
export VERTEXAI_LOCATION="REGION"         # Aider/LiteLLM呼び出し用
export VERTEXAI_PROJECT="PROJECT_ID"      # Aider/LiteLLM呼び出し用
```
#### DeepSeek API (DeepSeek-Coder-V2)

デフォルトでは、`DEEPSEEK_API_KEY`環境変数を使用します。

#### OpenRouter API (Llama3.1)

デフォルトでは、`OPENROUTER_API_KEY`環境変数を使用します。

#### Semantic Scholar API (文献検索)

私たちのコードは、オプションでSemantic Scholar APIキー（`S2_API_KEY`）を使用してスループットを向上させることができます（[こちら](https://www.semanticscholar.org/product/api)から取得できます）。ただし、原則としてキーがなくても動作するはずです。Semantic Scholarに問題がある場合は、文献検索と引用フェーズをスキップすることができます。

実行に使用するモデルのキーを必ず提供してください。例えば：

```bash
export OPENAI_API_KEY="YOUR KEY HERE"
export S2_API_KEY="YOUR KEY HERE"
```

## テンプレートの設定

このセクションでは、論文で使用した3つのテンプレートの設定手順を提供します。AIサイエンティストの実験を実行する前に、興味のあるテンプレートの設定手順を完了してください。

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
   # 注意: まず上記の準備スクリプトを実行してください！
   cd templates/nanoGPT
   python experiment.py --out_dir run_0
   python plot.py
   ```

### 2D拡散テンプレート

**説明:** このテンプレートは、低次元データセットに対する拡散生成モデルの性能向上を研究します。

**設定手順:**

1. **依存関係のインストール:**

   ```bash
   # 2D拡散の設定
   git clone https://github.com/gregversteeg/NPEET.git
   cd NPEET
   pip install .
   pip install scikit-learn
   ```

2. **ベースラインランの作成:**

   ```bash
   # 2D拡散ベースラインランの設定
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

## AIサイエンティスト論文生成実験の実行

**注意:** これらの実験を実行する前に、上記のセットアップ手順が完了していることを確認してください。

```bash
conda activate ai_scientist
# 論文生成を実行します。
python launch_scientist.py --model "gpt-4o-2024-05-13" --experiment nanoGPT_lite --num-ideas 2
python launch_scientist.py --model "claude-3-5-sonnet-20241022" --experiment nanoGPT_lite --num-ideas 2
```

複数のGPUを持っている場合は、`--parallel`オプションを使用してアイデアを複数のGPUに並列化できます。

## LLM生成論文のレビュー取得

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
review["Overall"]    # 全体のスコア（1-10）
review["Decision"]   # 'Accept' または 'Reject'
review["Weaknesses"] # 弱点のリスト（文字列）
```

バッチ分析を実行するには：

```bash
cd review_iclr_bench
python iclr_analysis.py --num_reviews 500 --batch_size 100 --num_fs_examples 1 --num_reflections 5 --temperature 0.1 --num_reviews_ensemble 5
```

## 独自テンプレートの作成

**The AI Scientist**に探求してほしい研究分野がある場合、独自のテンプレートを作成するのは簡単です。一般的に、既存のテンプレートの構造に従ってください。これらは以下で構成されています：

- `experiment.py` — これは主要なスクリプトで、コアコンテンツが含まれています。`--out_dir`引数を取り、実行結果を保存するフォルダを指定します。
- `plot.py` — このスクリプトは`run`フォルダから情報を取得し、プロットを作成します。コードは明確で編集しやすいはずです。
- `prompt.json` — テンプレートに関する情報をここに記載します。
- `seed_ideas.json` — ここに例のアイデアを配置します。例を使わずにアイデアを生成し、最良のものを選んでここに配置することもできます。
- `latex/template.tex` — LaTeXフォルダを使用することをお勧めしますが、事前に読み込まれた引用を期待されるものに置き換えてください。

新しいテンプレートを機能させる鍵は、基本ファイル名と出力JSONを既存の形式に一致させることです。それ以外は自由に変更できます。また、`template.tex`ファイルが正しい引用スタイル/基本プロットを使用するように更新されていることを確認してください。

### コミュニティ提供テンプレート

私たちは、新しいテンプレートの形でのコミュニティの貢献を歓迎します。これらは私たちによって維持されていませんが、他の人々にあなたのテンプレートを紹介できることを嬉しく思います。以下に、コミュニティ提供のテンプレートとそのプルリクエスト（PR）へのリンクを示します：

- 感染症モデリング（`seir`） - [PR #137](https://github.com/SakanaAI/AI-Scientist/pull/137)
- MobileNetV3を使用した画像分類（`mobilenetV3`） - [PR #141](https://github.com/SakanaAI/AI-Scientist/pull/141)
- Sketch RNN（`sketch_rnn`） - [PR #143](https://github.com/SakanaAI/AI-Scientist/pull/143)

*このセクションはコミュニティの貢献に予約されています。テンプレートをリストに追加するためにプルリクエストを提出してください！PRの説明でテンプレートを説明し、生成された論文の例も示してください。*

## テンプレートリソース

私たちは、他のリポジトリからのコードを多用した3つのテンプレートを提供しています。以下にクレジットを示します：

- **NanoGPTテンプレート**は[NanoGPT](https://github.com/karpathy/nanoGPT)とこの[PR](https://github.com/karpathy/nanoGPT/pull/254)のコードを使用しています。
- **2D Diffusionテンプレート**は[tiny-diffusion](https://github.com/tanelp/tiny-diffusion)、[ema-pytorch](https://github.com/lucidrains/ema-pytorch)、および[Datasaur](https://www.research.autodesk.com/publications/same-stats-different-graphs/)のコードを使用しています。
- **Grokkingテンプレート**は[Sea-Snell/grokking](https://github.com/Sea-Snell/grokking)および[danielmamay/grokking](https://github.com/danielmamay/grokking)のコードを使用しています。

オープンソースモデルやパッケージの開発者に感謝し、その貢献と彼らの仕事を利用できることに感謝します。

## AIサイエンティストの引用

**The AI Scientist**を研究に使用する場合は、以下のように引用してください：


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

**The AI Scientistを実行するときにファイルが見つからないのはなぜですか？**

メインの実験スクリプトの前に、すべてのセットアップと準備手順を完了していることを確認してください。

**PDFやレビューが生成されないのはなぜですか？**

The AI Scientistは、テンプレート、基盤モデル、およびアイデアの複雑さに依存して、成功率が異なります。私たちのメイン論文を参照することをお勧めします。最も高い成功率はClaude Sonnet 3.5で観察されています。レビューはGPT-4oで行うのが最適です。他のモデルは、ポジティブバイアスや必要な出力に従わない問題があります。

**各アイデアの生成コストはどれくらいですか？**

通常、Claude Sonnet 3.5で1論文あたり15ドル未満です。よりコスト効果の高いアプローチとして、DeepSeek Coder V2をお勧めします。新しいモデルを探す良い場所は[Aiderリーダーボード](https://aider.chat/docs/leaderboards/)です。

**書き込みのベース会議フォーマットを変更するにはどうすればよいですか？**

各テンプレート内に含まれるベースの`template.tex`ファイルを変更してください。

**異なる分野のアイデアに対してThe AI Scientistを実行するにはどうすればよいですか？**

異なるテンプレートの指示を参照してください。この現在のバージョンでは、コードで表現できるアイデアに制限されています。しかし、この制限を解除することは将来的に興味深い課題です！ :)

**新しい基盤モデルのサポートを追加するにはどうすればよいですか？**

`ai_scientist/llm.py`を変更して、新しい基盤モデルのサポートを追加できます。The AI Scientistには、GPT-4レベルよりも著しく弱いモデルの使用はお勧めしません。

**なぜベースラインランを自分で実行する必要があるのですか？**

これらは`run_0`として表示され、ハードウェアの違いによる正確な実行時間の比較のために、The AI Scientistを実行する各マシンで実行する必要があります。

**Semantic Scholar APIにアクセスする際に問題がある場合はどうすればよいですか？**

私たちは、アイデアの新規性を確認し、論文の引用を収集するためにSemantic Scholar APIを使用しています。APIキーがない場合やAPIへのアクセスが遅い場合は、これらのフェーズをスキップできるかもしれません。

## コンテナ化

`experimental/Dockerfile`に、コンテナ化の取り組みに役立つ[コミュニティ提供](https://github.com/SakanaAI/AI-Scientist/pull/21)のDockerイメージを含めています。

このイメージは以下のように使用できます：

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
