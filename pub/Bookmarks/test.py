import re

from bs4 import BeautifulSoup, Comment


# 半角スペースの連続後に#があるパターンにマッチし、置換する関数
def replace_spaces(match):
    # スペースの連続を取得
    spaces = match.group(0)
    # スペースの数を4で割り、その商だけ#に置き換える
    replacement = "\n" + "#" * (len(spaces) // 4)
    return replacement


def html_to_markdown(html_file_path, output_md_path, input_md_path):
    with open(html_file_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    html_content = (
        html_content.replace("    ", "$ ").replace("$ $", "$$").replace("$ $", "$$")
    )

    soup = BeautifulSoup(html_content, "html.parser")

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()

    # TITLE、H1タグを取り除く
    for tag in soup.find_all("TITLE"):
        tag.decompose()

    # DT、DL、Pタグの中身を保持してタグ自体は取り除く
    for tag in soup.find_all(["dt", "dl", "p"]):
        tag.unwrap()

    # ファイルを開いて内容を読み込む
    with open(input_md_path, "r", encoding="utf-8") as file:
        markdown_content = file.read()

    for element in soup.contents:
        if element.name == "h3":
            markdown_content += "#" + " " + element.get_text().strip() + "\n"
        elif element.name == "a":
            # リンクをマークダウン形式に変換
            link_text = element.get_text()
            href = element.get("href", "")
            icon_html = ""
            if element.has_attr("icon"):
                print("iconあります")
                # icon要素がある場合、HTMLイメージタグを使用して行頭に表示
                icon_src = element["icon"]
            else:
                icon_src = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAAsTAAALEwEAmpwYAAACC0lEQVR4nO2Wy0pcQRCGv8WYTcaJ2SfgwnGEbGJGs84beAHX+g7esnQhCVmFiBshJuYZsorjKL6CkJgIIoIukxAVzWYcKfgHGrH7dJ9RUPCHhsOp6uq/q7oucI87hgowC6wB28CJln3XgBmg9yYOHgDqQDNyGcHqdRzcASwC5zL8G1gGhuSNh8ADDwnbswAU8h7+GFiXsVNgHih5dLO80ZXn5nUZOAQGr9B5lhCSTXkqGovaeAA88ejMJRCw9SHlwZ3L7aGH9D2RQAN4EUOgrg0Wcx+aOZelahAV57U/Ig2xISmHjLyW0kfyI4vAdGhzTUqW5yFY/n8G9oAVYBzojvTCasjwjpR6Ajp9wFZi7JFN+/4ZInAkpU6PfEK1P3SYeWEM2AWWnEwqSn4UQ6Dokcfc9pVnb0nyfyECv6TU2waBjUDomuqcmY9wuA0CbtxdjOr/txCBGSl98shn2/DCF8kmYwrRn0AhmsvxDqwb/o0pRKh9muIb4hDj/ncx7m+hqmZ0psaUhcuHW4Fy8VK2rBk9JxILEe34cuo21SGtSrbwVPOEyd6TgIITikPdwocl6Z1oSGmhH9h3umDyaNblkDAXvvWMVlXpTDj7LOb/ndqfPJK1UNAk03Cyw2I8osJS1OrTvxXntTfk9txDqQtz7deEIlSLnX5SUVY/X1VJPdb6oRSbyuik97h9uAA+3jTLypXd7wAAAABJRU5ErkJggg=="
            icon_html = f"![with: 16px;]({icon_src})"
            markdown_content += f"- [{icon_html}{link_text}]({href})\n"
        elif element.name is None and element.string is not None:
            markdown_content += element.string.strip()
        elif element.name not in ["h3", "a"]:
            markdown_content += element.get_text(separator="\n", strip=True) + "\n"

        markdown_content = markdown_content.replace("$", "    ")

        # 正規表現で置換
        markdown_content = re.sub(r"\n {4,}(?=#)", replace_spaces, markdown_content)
        markdown_content = re.sub(r"\n {4}", "\n", markdown_content)

    # MDファイルとして出力
    with open(output_md_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_content)


# 使用例
html_file_path = "input.html"  # 入力HTMLファイルのパス
output_md_path = "output.md"  # 出力するMarkdownファイルのパス
input_md_path = "input.md"
html_to_markdown(html_file_path, output_md_path, input_md_path)
