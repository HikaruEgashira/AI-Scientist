import os
import numpy as np
import json
from pypdf import PdfReader
import pymupdf
import pymupdf4llm
from ai_scientist.llm import (
    get_response_from_llm,
    get_batch_responses_from_llm,
    extract_json_between_markers,
)

# レビュアーのシステムプロンプトの基本
reviewer_system_prompt_base = (
    "あなたは、著名なML会議に提出された論文をレビューしているAI研究者です。"
    "慎重かつ批判的に判断してください。"
)

# 否定的なレビュアーのシステムプロンプト
reviewer_system_prompt_neg = (
    reviewer_system_prompt_base
    + "論文が悪い場合や不明な場合は、低いスコアを付けて拒否してください。"
)

# 肯定的なレビュアーのシステムプロンプト
reviewer_system_prompt_pos = (
    reviewer_system_prompt_base
    + "論文が良い場合や不明な場合は、高いスコアを付けて受け入れてください。"
)

# テンプレートの指示
template_instructions = """
以下の形式で応答してください:

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

<THOUGHT>では、評価の直感と理由を簡単に説明します。
高レベルの議論、必要な選択肢、レビューの望ましい結果を詳細に説明します。
ここでは一般的なコメントをせず、現在の論文に特化した具体的なコメントをしてください。
これをレビューのメモ取りフェーズと考えてください。

<JSON>では、次のフィールドを順番に含むJSON形式でレビューを提供します:
- "Summary": 論文の内容と貢献の要約。
- "Strengths": 論文の強みのリスト。
- "Weaknesses": 論文の弱みのリスト。
- "Originality": 1から4の評価（低、中、高、非常に高い）。
- "Quality": 1から4の評価（低、中、高、非常に高い）。
- "Clarity": 1から4の評価（低、中、高、非常に高い）。
- "Significance": 1から4の評価（低、中、高、非常に高い）。
- "Questions": 著者に回答を求める明確な質問のセット。
- "Limitations": 研究の限界と潜在的な負の社会的影響のセット。
- "Ethical Concerns": 倫理的懸念があるかどうかを示すブール値。
- "Soundness": 1から4の評価（低、公平、良い、優れた）。
- "Presentation": 1から4の評価（低、公平、良い、優れた）。
- "Contribution": 1から4の評価（低、公平、良い、優れた）。
- "Overall": 1から10の評価（非常に強い拒否から受賞品質まで）。
- "Confidence": 1から5の評価（低、中、高、非常に高い、絶対的）。
- "Decision": 受け入れまたは拒否のいずれかの決定。

"Decision"フィールドでは、弱い受け入れ、境界線の受け入れ、境界線の拒否、強い拒否を使用しないでください。代わりに、受け入れまたは拒否のみを使用してください。
このJSONは自動的に解析されるため、形式が正確であることを確認してください。
"""

# NeurIPSのレビュー形式
neurips_form = (
    """
## レビュー形式
以下は、各論文に対してレビュー形式で尋ねられる質問の説明と、これらの質問に回答する際に考慮すべきガイドラインです。
レビューを書く際には、決定が行われた後、受け入れられた論文とオプトインされた拒否された論文のレビューとメタレビューが公開されることを念頭に置いてください。

1. 要約: 論文とその貢献を簡単に要約してください。ここは論文を批判する場所ではありません。著者はよく書かれた要約に一般的に同意するはずです。
  - 強みと弱み: 論文の強みと弱みを各次元に触れながら徹底的に評価してください:
  - 独創性: タスクや方法は新しいですか？よく知られた技術の新しい組み合わせですか？（これは価値があります！）この作業が以前の貢献とどのように異なるかが明確ですか？関連する作業は適切に引用されていますか？
  - 質: 提出物は技術的に健全ですか？主張は理論的分析や実験結果によって十分に裏付けられていますか？使用される方法は適切ですか？これは完全な作業ですか、それとも進行中の作業ですか？著者は自分の作業の強みと弱みを評価する際に慎重で正直ですか？
  - 明確さ: 提出物は明確に書かれていますか？よく整理されていますか？（そうでない場合は、明確さを改善するための建設的な提案をしてください。）読者に十分な情報を提供していますか？（優れた論文は、専門家の読者がその結果を再現するのに十分な情報を提供します。）
  - 重要性: 結果は重要ですか？他の研究者や実務者がアイデアを使用したり、それに基づいて構築する可能性はありますか？提出物は以前の作業よりも優れた方法で困難なタスクに取り組んでいますか？それは実証可能な方法で最先端を進めていますか？それは独自のデータ、既存データに関する��自の結論、または独自の理論的または実験的アプローチを提供していますか？

2. 質問: 著者に対する質問や提案をリストアップし、慎重に説明してください。著者からの回答が意見を変えたり、混乱を解消したり、制限を解決したりする可能性があることを考えてください。これは、著者との生産的な反論と議論のフェーズにとって非常に重要です。

3. 限界: 著者は自分の作業の限界と潜在的な負の社会的影響に十分に対処していますか？そうでない場合は、改善のための建設的な提案を含めてください。
一般的に、著者が自分の作業の限界と潜在的な負の社会的影響について率直であることを奨励し、それに対して罰するのではなく報いるべきです。欠けている重要な点があるかどうかを考え、それを著者へのフィードバックとして提供してください。

4. 倫理的懸念: この論文に倫理的な問題がある場合は、倫理レビューのために論文をフラグしてください。これが適切な場合のガイダンスについては、NeurIPSの倫理ガイドラインを確認してください。

5. 健全性: 技術的な主張、実験および研究方法論の健全性、および論文の中心的な主張が十分に裏付けられているかどうかを示すために、次のスケールで論文に数値評価を付けてください。
  4: 優れた
  3: 良い
  2: 公平
  1: 低い

6. プレゼンテーション: プレゼンテーションの質を示すために、次のスケールで論文に数値評価を付けてください。これには、執筆スタイルと明確さ、および以前の作業に対する文脈化が含まれます。
  4: 優れた
  3: 良い
  2: 公平
  1: 低い

7. 貢献: 研究分野に対する全体的な貢献の質を示すために、次のスケールで論文に数値評価を付けてください。質問は重要ですか？論文はアイデアや実行の独創性をもたらしていますか？結果はNeurIPSコミュニティと共有する価値がありますか？
  4: 優れた
  3: 良い
  2: 公平
  1: 低い

8. 全体: この提出物に対する「全体スコア」を提供してください。選択肢:
  10: 受賞品質: 技術的に欠陥のない論文で、AIの1つ以上の分野に画期的な影響を与え、評価、再現性、リソースが非常に強力であり、未解決の倫理的考慮事項がない。
  9: 非常に強い受け入れ: 技術的に欠陥のない論文で、少なくとも1つのAI分野に画期的な影響を与え、複数のAI分野に優れた影響を与え、評価、リソース、再現性が完璧であり、未解決の倫理的考慮事項がない。
  8: 強い受け入れ: 技術的に強力な論文で、新しいアイデアを持ち、少なくとも1つのAI分野に優れた影響を与え、複数のAI分野に高から非常に高い影響を与え、評価、リソース、再現性が優れており、未解決の倫理的考慮事項がない。
  7: 受け入れ: 技術的に堅実な論文で、少なくとも1つのAIサブエリアに高い影響を与え、複数のAI分野に中から高い影響を与え、評価、リソース、再現性が良から優れており、未解決の倫理的考慮事項がない。
  6: 弱い受け入れ: 技術的に堅実で、中から高い影響を持つ論文で、評価、リソース、再現性、倫理的考慮事項に関して大きな懸念がない。
  5: 境界線の受け入れ: 技術的に堅実な論文で、受け入れる理由が拒否する理由を上回る場合（例: 限られた評価）。慎重に使用してください。
  4: 境界線の拒否: 技術的に堅実な論文で、拒否する理由が受け入れる理由を上回る場合（例: 良い評価）。慎重に使用してください。
  3: 拒否: 例えば、技術的な欠陥、弱い評価、不十分な再現性、未解決の倫理的考慮事項がある論文。
  2: 強い拒否: 例えば、重大な技術的欠陥、評価の不十分さ、影響の限定、再現性の低さ、ほとんど未解決の倫理的考慮事項がある論文。
  1: 非常に強い拒否: 例えば、些細な結果や未解決の倫理的考慮事項がある論文。

9. 自信: この提出物に対する評価の自信スコアを提供してください。選択肢:
  5: 評価に絶対的な自信があります。関連する作業に非常に精通しており、数学/その他の詳細を慎重に確認しました。
  4: 評価に自信がありますが、絶対的ではありません。提出物の一部を理解していない可能性や、関連する作業に精通していない可能性は低いですが、存在します。
  3: 評価にかなりの自信があります。提出物の一部を理解していない可能性や、関連する作業に精通していない可能性があります。数学/その他の詳細は慎重に確認されていません。
  2: 評価を擁護する意思はありますが、提出物の中心部分を理解していない可能性や、関連する作業に精通していない可能性が高いです。数学/その他の詳細は慎重に確認されていません。
  1: 評価は教育的な推測です。提出物は自分の専門分野ではないか、理解が難しいものでした。数学/その他の詳細は慎重に確認されていません。
"""
    + template_instructions
)


def perform_review(
    text,
    model,
    client,
    num_reflections=1,
    num_fs_examples=1,
    num_reviews_ensemble=1,
    temperature=0.75,
    msg_history=None,
    return_msg_history=False,
    reviewer_system_prompt=reviewer_system_prompt_neg,
    review_instruction_form=neurips_form,
):
    if num_fs_examples > 0:
        fs_prompt = get_review_fewshot_examples(num_fs_examples)
        base_prompt = review_instruction_form + fs_prompt
    else:
        base_prompt = review_instruction_form

    base_prompt += f"""
ここにレビューするように求められた論文があります:
```
{text}
```"""

    if num_reviews_ensemble > 1:
        llm_review, msg_histories = get_batch_responses_from_llm(
            base_prompt,
            model=model,
            client=client,
            system_message=reviewer_system_prompt,
            print_debug=False,
            msg_history=msg_history,
            # 多様性を促進するために高い温度設定
            temperature=0.75,
            n_responses=num_reviews_ensemble,
        )
        parsed_reviews = []
        for idx, rev in enumerate(llm_review):
            try:
                parsed_reviews.append(extract_json_between_markers(rev))
            except Exception as e:
                print(f"アンサンブルレビュー {idx} に失敗しました: {e}")
        parsed_reviews = [r for r in parsed_reviews if r is not None]
        review = get_meta_review(model, client, temperature, parsed_reviews)

        # メタレビュアーが失敗した場合、最初の有効なレビューを使用
        if review is None:
            review = parsed_reviews[0]

        # 数値スコアをアンサンブルの平均値に置き換え
        for score, limits in [
            ("Originality", (1, 4)),
            ("Quality", (1, 4)),
            ("Clarity", (1, 4)),
            ("Significance", (1, 4)),
            ("Soundness", (1, 4)),
            ("Presentation", (1, 4)),
            ("Contribution", (1, 4)),
            ("Overall", (1, 10)),
            ("Confidence", (1, 5)),
        ]:
            scores = []
            for r in parsed_reviews:
                if score in r and limits[1] >= r[score] >= limits[0]:
                    scores.append(r[score])
            review[score] = int(round(np.mean(scores)))

        # 有効なレビューと新しい集計レビューでメッセージ履歴を再作成
        msg_history = msg_histories[0][:-1]
        msg_history += [
            {
                "role": "assistant",
                "content": f"""
THOUGHT:
以前取得した{num_reviews_ensemble}人のレビュアーの意見を集約します。

REVIEW JSON:
```json
{json.dumps(review)}
```
""",
            }
        ]
    else:
        llm_review, msg_history = get_response_from_llm(
            base_prompt,
            model=model,
            client=client,
            system_message=reviewer_system_prompt,
            print_debug=False,
            msg_history=msg_history,
            temperature=temperature,
        )
        review = extract_json_between_markers(llm_review)

    if num_reflections > 1:
        for j in range(num_reflections - 1):
            # print(f"Relection: {j + 2}/{num_reflections}")
            text, msg_history = get_response_from_llm(
                reviewer_reflection_prompt,
                client=client,
                model=model,
                system_message=reviewer_system_prompt,
                msg_history=msg_history,
                temperature=temperature,
            )
            review = extract_json_between_markers(text)
            assert review is not None, "LLM出力からJSONの抽出に失敗しました"

            if "I am done" in text:
                # print(f"Review generation converged after {j + 2} iterations.")
                break

    if return_msg_history:
        return review, msg_history
    else:
        return review


# レビュアーの反省プロンプト
reviewer_reflection_prompt = """ラウンド {current_round}/{num_reflections}.
考えの中で、まず作成したレビューの正確性と健全性を慎重に考慮してください。
論文を評価する際に重要だと思う他の要素も含めてください。
レビューが明確で簡潔であり、JSONが正しい形式であることを確認してください。
物事を過度に複雑にしないでください。
次の試行では、レビューを改善し、改善するように努めてください。
重大な問題がない限り、元のレビューの精神に従ってください。

以前と同じ形式で応答してください:
THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

改善することがない場合は、単に前のJSONを正確に繰り返し、考えの最後に「I am done」を含めてください。
変更を加えない場合にのみ「I am done」を含めてください。"""


def load_paper(pdf_path, num_pages=None, min_size=100):
    try:
        if num_pages is None:
            text = pymupdf4llm.to_markdown(pdf_path)
        else:
            reader = PdfReader(pdf_path)
            min_pages = min(len(reader.pages), num_pages)
            text = pymupdf4llm.to_markdown(pdf_path, pages=list(range(min_pages)))
        if len(text) < min_size:
            raise Exception("テキストが短すぎます")
    except Exception as e:
        print(f"pymupdf4llmでエラーが発生しました。pymupdfにフォールバックします: {e}")
        try:
            doc = pymupdf.open(pdf_path)  # ドキュメントを開く
            if num_pages:
                doc = doc[:num_pages]
            text = ""
            for page in doc:  # ドキュメントのページを反復処理
                text = text + page.get_text()  # UTF-8としてエンコードされたプレーンテキストを取得
            if len(text) < min_size:
                raise Exception("テキストが短すぎます")
        except Exception as e:
            print(f"pymupdfでエラーが発生しました。pypdfにフォールバックします: {e}")
            reader = PdfReader(pdf_path)
            if num_pages is None:
                text = "".join(page.extract_text() for page in reader.pages)
            else:
                text = "".join(page.extract_text() for page in reader.pages[:num_pages])
            if len(text) < min_size:
                raise Exception("テキストが短すぎます")

    return text


def load_review(path):
    with open(path, "r") as json_file:
        loaded = json.load(json_file)
    return loaded["review"]


# このファイルのディレクトリを取得
dir_path = os.path.dirname(os.path.realpath(__file__))

fewshot_papers = [
    os.path.join(dir_path, "fewshot_examples/132_automated_relational.pdf"),
    os.path.join(dir_path, "fewshot_examples/attention.pdf"),
    os.path.join(dir_path, "fewshot_examples/2_carpe_diem.pdf"),
]

fewshot_reviews = [
    os.path.join(dir_path, "fewshot_examples/132_automated_relational.json"),
    os.path.join(dir_path, "fewshot_examples/attention.json"),
    os.path.join(dir_path, "fewshot_examples/2_carpe_diem.json"),
]


def get_review_fewshot_examples(num_fs_examples=1):
    fewshot_prompt = """
以下は、以前の機械学習会議からコピーされたサンプルレビューです。
各レビューはレビュアーのスタイルに応じて異なる形式で書かれていますが、レビューはよく構造化されており、ナビゲートしやすいです。
"""
    for paper, review in zip(
        fewshot_papers[:num_fs_examples], fewshot_reviews[:num_fs_examples]
    ):
        txt_path = paper.replace(".pdf", ".txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                paper_text = f.read()
        else:
            paper_text = load_paper(paper)
        review_text = load_review(review)
        fewshot_prompt += f"""
論文:

```
{paper_text}
```

レビュー:

```
{review_text}
```
"""

    return fewshot_prompt


# メタレビュアーのシステムプロンプト
meta_reviewer_system_prompt = """あなたは機械学習会議のエリアチェアです。
あなたは{reviewer_count}人のレビュアーによってレビューされた論文のメタレビューを担当しています。
あなたの仕事は、レビューを1つのメタレビューに集約することです。
慎重かつ批判的に判断し、コンセンサスを見つけ、すべてのレビュアーの意見を尊重してください。"""


def get_meta_review(model, client, temperature, reviews):
    # 個々のレビューのセットからメタレビューを作成
    review_text = ""
    for i, r in enumerate(reviews):
        review_text += f"""
レビュー {i + 1}/{len(reviews)}:
```
{json.dumps(r)}
```
"""
    base_prompt = neurips_form + review_text

    llm_review, msg_history = get_response_from_llm(
        base_prompt,
        model=model,
        client=client,
        system_message=meta_reviewer_system_prompt.format(reviewer_count=len(reviews)),
        print_debug=False,
        msg_history=None,
        temperature=temperature,
    )
    meta_review = extract_json_between_markers(llm_review)
    return meta_review


def perform_improvement(review, coder):
    improvement_prompt = '''以下のレビューがあなたの研究論文に対して作成されました:
"""
{review}
"""

レビューを使用してテキストを改善してください。'''.format(
        review=json.dumps(review)
    )
    coder_out = coder.run(improvement_prompt)
