import json
import os
import os.path as osp
import time
from typing import List, Dict, Union

import backoff
import requests

from ai_scientist.llm import get_response_from_llm, extract_json_between_markers, create_client, AVAILABLE_LLMS

S2_API_KEY = os.getenv("S2_API_KEY")

idea_first_prompt = """{task_description}
<experiment.py>
{code}
</experiment.py>

Here are the ideas that you have already generated:

'''
{prev_ideas_string}
'''

Come up with the next impactful and creative idea for research experiments and directions you can feasibly investigate with the code provided.
Note that you will not have access to any additional resources or datasets.
Make sure any idea is not overfit the specific training dataset or model, and has wider significance.

Respond in the following format:

THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and motivations for the idea. Detail your high-level plan, necessary design choices and ideal outcomes of the experiments. Justify how the idea is different from the existing ones.

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A title for the idea, will be used for the report writing.
- "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
- "Interestingness": A rating from 1 to 10 (lowest to highest).
- "Feasibility": A rating from 1 to 10 (lowest to highest).
- "Novelty": A rating from 1 to 10 (lowest to highest).

Be cautious and realistic on your ratings.
This JSON will be automatically parsed, so ensure the format is precise.
You will have {num_reflections} rounds to iterate on the idea, but do not need to use them all.
"""

idea_reflection_prompt = """Round {current_round}/{num_reflections}.
In your thoughts, first carefully consider the quality, novelty, and feasibility of the idea you just created.
Include any other factors that you think are important in evaluating the idea.
Ensure the idea is clear and concise, and the JSON is the correct format.
Do not make things overly complicated.
In the next attempt, try and refine and improve your idea.
Stick to the spirit of the original idea unless there are glaring issues.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

If there is nothing to improve, simply repeat the previous JSON EXACTLY after the thought and include "I am done" at the end of the thoughts but before the JSON.
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES。"""

# アイデアを生成する関数
def generate_ideas(
        base_dir,
        client,
        model,
        skip_generation=False,
        max_num_generations=20,
        num_reflections=5,
):
    if skip_generation:
        # 既存のアイデアをファイルから読み込む
        try:
            with open(osp.join(base_dir, "ideas.json"), "r") as f:
                ideas = json.load(f)
            print("既存のアイデアを読み込みました:")
            for idea in ideas:
                print(idea)
            return ideas
        except FileNotFoundError:
            print("既存のアイデアが見つかりません。新しいアイデアを生成します。")
        except json.JSONDecodeError:
            print("既存のアイデアのデコードエラー。新しいアイデアを生成します。")

    idea_str_archive = []
    with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
        seed_ideas = json.load(f)
    for seed_idea in seed_ideas:
        idea_str_archive.append(json.dumps(seed_idea))

    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()

    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)

    idea_system_prompt = prompt["system"]

    for _ in range(max_num_generations):
        print()
        print(f"アイデアを生成中 {_ + 1}/{max_num_generations}")
        try:
            prev_ideas_string = "\n\n".join(idea_str_archive)

            msg_history = []
            print(f"反復 1/{num_reflections}")
            text, msg_history = get_response_from_llm(
                idea_first_prompt.format(
                    task_description=prompt["task_description"],
                    code=code,
                    prev_ideas_string=prev_ideas_string,
                    num_reflections=num_reflections,
                ),
                client=client,
                model=model,
                system_message=idea_system_prompt,
                msg_history=msg_history,
            )
            ## 出力を解析
            json_output = extract_json_between_markers(text)
            assert json_output is not None, "LLM出力からJSONの抽出に失敗しました"
            print(json_output)

            # アイデアを反復的に改善
            if num_reflections > 1:
                for j in range(num_reflections - 1):
                    print(f"反復 {j + 2}/{num_reflections}")
                    text, msg_history = get_response_from_llm(
                        idea_reflection_prompt.format(
                            current_round=j + 2, num_reflections=num_reflections
                        ),
                        client=client,
                        model=model,
                        system_message=idea_system_prompt,
                        msg_history=msg_history,
                    )
                    ## 出力を解析
                    json_output = extract_json_between_markers(text)
                    assert (
                            json_output is not None
                    ), "LLM出力からJSONの抽出に失敗しました"
                    print(json_output)

                    if "I am done" in text:
                        print(f"アイデア生成は {j + 2} 回の反復後に収束しました。")
                        break

            idea_str_archive.append(json.dumps(json_output))
        except Exception as e:
            print(f"アイデアの生成に失敗しました: {e}")
            continue

    ## アイデアを保存
    ideas = []
    for idea_str in idea_str_archive:
        ideas.append(json.loads(idea_str))

    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(ideas, f, indent=4)

    return ideas


# オープンエンドのアイデアを生成する関数
def generate_next_idea(
        base_dir,
        client,
        model,
        prev_idea_archive=[],
        num_reflections=5,
        max_attempts=10,
):
    idea_archive = prev_idea_archive
    original_archive_size = len(idea_archive)

    print(f"アイデアを生成中 {original_archive_size + 1}")

    if len(prev_idea_archive) == 0:
        print(f"最初の反復、シードアイデアを使用")
        # 最初の実行時に事前存在するアイデアでアーカイブをシード
        with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
            seed_ideas = json.load(f)
        for seed_idea in seed_ideas[:1]:
            idea_archive.append(seed_idea)
    else:
        with open(osp.join(base_dir, "experiment.py"), "r") as f:
            code = f.read()
        with open(osp.join(base_dir, "prompt.json"), "r") as f:
            prompt = json.load(f)
        idea_system_prompt = prompt["system"]

        for _ in range(max_attempts):
            try:
                idea_strings = []
                for idea in idea_archive:
                    idea_strings.append(json.dumps(idea))
                prev_ideas_string = "\n\n".join(idea_strings)

                msg_history = []
                print(f"反復 1/{num_reflections}")
                text, msg_history = get_response_from_llm(
                    idea_first_prompt.format(
                        task_description=prompt["task_description"],
                        code=code,
                        prev_ideas_string=prev_ideas_string,
                        num_reflections=num_reflections,
                    )
                    + """
Completed ideas have an additional "Score" field which indicates the assessment by an expert ML reviewer.
This is on a standard 1-10 ML conference scale.
Scores of 0 indicate the idea failed either during experimentation, writeup or reviewing.
""",
                    client=client,
                    model=model,
                    system_message=idea_system_prompt,
                    msg_history=msg_history,
                )
                ## 出力を解析
                json_output = extract_json_between_markers(text)
                assert json_output is not None, "LLM出力からJSONの抽出に失敗しました"
                print(json_output)

                # アイデアを反復的に改善
                if num_reflections > 1:
                    for j in range(num_reflections - 1):
                        print(f"反復 {j + 2}/{num_reflections}")
                        text, msg_history = get_response_from_llm(
                            idea_reflection_prompt.format(
                                current_round=j + 2, num_reflections=num_reflections
                            ),
                            client=client,
                            model=model,
                            system_message=idea_system_prompt,
                            msg_history=msg_history,
                        )
                        ## 出力を解析
                        json_output = extract_json_between_markers(text)
                        assert (
                                json_output is not None
                        ), "LLM出力からJSONの抽出に失敗しました"
                        print(json_output)

                        if "I am done" in text:
                            print(
                                f"アイデア生成は {j + 2} 回の反復後に収束しました。"
                            )
                            break

                idea_archive.append(json_output)
                break
            except Exception as e:
                print(f"アイデアの生成に失敗しました: {e}")
                continue

    ## アイデアを保存
    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(idea_archive, f, indent=4)

    return idea_archive


def on_backoff(details):
    print(
        f"関数 {details['target'].__name__} の呼び出しで {details['tries']} 回の試行後に {details['wait']:0.1f} 秒間のバックオフ "
        f" {time.strftime('%X')} に"
    )


@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def search_for_papers(query, result_limit=10) -> Union[None, List[Dict]]:
    if not query:
        return None
    rsp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        headers={"X-API-KEY": S2_API_KEY},
        params={
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
        },
    )
    print(f"レスポンスステータスコード: {rsp.status_code}")
    print(
        f"レスポンスコンテンツ: {rsp.text[:500]}"
    )  # レスポンスコンテンツの最初の500文字を表示
    rsp.raise_for_status()
    results = rsp.json()
    total = results["total"]
    time.sleep(1.0)
    if not total:
        return None

    papers = results["data"]
    return papers


novelty_system_msg = """あなたは、フィールドに大きく貢献する論文を発表しようとしている意欲的なAI博士課程の学生です。
アイデアが新規かどうかを確認したいと考えています。つまり、既存の文献と大きく重複していないか、すでに十分に探求されているかどうかを確認します。
新規性に対して厳しい批評家であり、新しい会議やワークショップの論文に十分な貢献があることを確認してください。
Semantic Scholar APIにアクセスできるようになり、文献を調査し、アイデアの決定を支援するための関連論文を見つけることができます。
検索クエリの上位10件の結果が、要約とともに提示されます。

{num_rounds} 回の決定を行うことができますが、すべてを使用する必要はありません。
任意のラウンドで早期に終了し、アイデアの新規性について決定することができます。
十分な検索を行った後、アイデアが既存の論文と大きく重複していない場合、新規であると判断します。
アイデアが既存の論文と大きく重複している場合、新規でないと判断します。

{task_description}
<experiment.py>
{code}
</experiment.py>
"""

novelty_prompt = '''ラウンド {current_round}/{num_rounds}.
このアイデアがあります:

"""
{idea}
"""

最後のクエリの結果は次のとおりです（最初のラウンドでは空です）:
"""
{last_query_results}
"""

次の形式で応答してください:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

<THOUGHT>では、まずアイデアを簡単に考察し、決定を支援するためのクエリを特定します。
決定を下した場合は、"Decision made: novel." または "Decision made: not novel." を考えに追加します。

<JSON>では、次のフィールドのみを含むJSON形式で応答します:
- "Query": 文献を検索するためのオプションの検索クエリ（例: attention is all you need）。このラウンドで決定していない場合はクエリを作成する必要があります。

クエリは、探している論文の正確な名前や著者を思い出すことができる場合に最適に機能します。
このJSONは自動的に解析されるため、形式が正確であることを確認してください。'''


def check_idea_novelty(
        ideas,
        base_dir,
        client,
        model,
        max_num_iterations=10,
):
    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()
    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)
        task_description = prompt["task_description"]

    for idx, idea in enumerate(ideas):
        if "novel" in idea:
            print(f"アイデア {idx} をスキップします。すでに確認済みです。")
            continue

        print(f"\nアイデア {idx} の新規性を確認中: {idea['Name']}")

        novel = False
        msg_history = []
        papers_str = ""

        for j in range(max_num_iterations):
            try:
                text, msg_history = get_response_from_llm(
                    novelty_prompt.format(
                        current_round=j + 1,
                        num_rounds=max_num_iterations,
                        idea=idea,
                        last_query_results=papers_str,
                    ),
                    client=client,
                    model=model,
                    system_message=novelty_system_msg.format(
                        num_rounds=max_num_iterations,
                        task_description=task_description,
                        code=code,
                    ),
                    msg_history=msg_history,
                )
                if "decision made: novel" in text.lower():
                    print("ラウンド後に決定: 新規", j)
                    novel = True
                    break
                if "decision made: not novel" in text.lower():
                    print("ラウンド後に決定: 新規でない", j)
                    break

                ## 出力を解析
                json_output = extract_json_between_markers(text)
                assert json_output is not None, "LLM出力からJSONの抽出に失敗しました"

                ## 論文を検索
                query = json_output["Query"]
                papers = search_for_papers(query, result_limit=10)
                if papers is None:
                    papers_str = "論文が見つかりませんでした。"

                paper_strings = []
                for i, paper in enumerate(papers):
                    paper_strings.append(
                        """{i}: {title}. {authors}. {venue}, {year}.\n引用数: {cites}\n要約: {abstract}""".format(
                            i=i,
                            title=paper["title"],
                            authors=paper["authors"],
                            venue=paper["venue"],
                            year=paper["year"],
                            cites=paper["citationCount"],
                            abstract=paper["abstract"],
                        )
                    )
                papers_str = "\n\n".join(paper_strings)

            except Exception as e:
                print(f"エラー: {e}")
                continue

        idea["novel"] = novel

    # 結果をJSONファイルに保存
    results_file = osp.join(base_dir, "ideas.json")
    with open(results_file, "w") as f:
        json.dump(ideas, f, indent=4)

    return ideas


if __name__ == "__main__":
    MAX_NUM_GENERATIONS = 32
    NUM_REFLECTIONS = 5
    import argparse

    parser = argparse.ArgumentParser(description="AI科学者のアイデアを生成する")
    # 実験の種類を追加 (nanoGPT, Boston, etc.)
    parser.add_argument(
        "--experiment",
        type=str,
        default="nanoGPT",
        help="AI Scientistを実行する実験。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=AVAILABLE_LLMS,
        help="AI Scientistに使用するモデル。",
    )
    parser.add_argument(
        "--skip-idea-generation",
        action="store_true",
        help="アイデア生成をスキップし、既存のアイデアを使用します。",
    )
    parser.add_argument(
        "--check-novelty",
        action="store_true",
        help="アイデアの新規性を確認します。",
    )
    args = parser.parse_args()

    # クライアントを作成
    client, client_model = create_client(args.model)

    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)
    ideas = generate_ideas(
        base_dir,
        client=client,
        model=client_model,
        skip_generation=args.skip_idea_generation,
        max_num_generations=MAX_NUM_GENERATIONS,
        num_reflections=NUM_REFLECTIONS,
    )
    if args.check_novelty:
        ideas = check_idea_novelty(
            ideas,
            base_dir=base_dir,
            client=client,
            model=client_model,
        )
