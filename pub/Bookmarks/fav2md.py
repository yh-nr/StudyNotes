import re
from datetime import datetime

from bs4 import BeautifulSoup, Comment


# 半角スペースの連続後に#があるパターンにマッチし、置換する関数
def replace_spaces(match):
    spaces = match.group(0)
    replacement = "\n---\n\n" + "#" * (len(spaces) - 3)
    return replacement


def html_to_markdown(html_file_path, output_md_path, input_md_path):
    with open(html_file_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    lines = html_content.split("\n")
    html_content = "\n".join(lines[1:])

    # 4つのスペースを "$ " に置換する
    html_content = html_content.replace("    ", "$ ")
    while html_content.count("$ $"):
        html_content = html_content.replace("$ $", "$$")

    soup = BeautifulSoup(html_content, "html.parser")

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()

    # TITLE、H1タグを取り除く
    for tag in soup.find_all(["title", "h1"]):
        tag.extract()

    # DT、DL、Pタグの中身を保持してタグ自体は取り除く
    for tag in soup.find_all(["dt", "dl", "p"]):
        tag.unwrap()

    markdown_content = ""

    for element in soup.contents:
        if element.name == "h3":
            markdown_content += "#" + " " + element.get_text().strip() + "\n"
        elif element.name == "a":
            # リンクをマークダウン形式に変換
            link_text = element.get_text()
            href = element.get("href", "")
            icon_html = ""
            if element.has_attr("icon"):
                # icon要素がある場合、HTMLイメージタグを使用して行頭に表示
                icon_src = element["icon"]
            else:
                icon_src = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAAsTAAALEwEAmpwYAAACC0lEQVR4nO2Wy0pcQRCGv8WYTcaJ2SfgwnGEbGJGs84beAHX+g7esnQhCVmFiBshJuYZsorjKL6CkJgIIoIukxAVzWYcKfgHGrH7dJ9RUPCHhsOp6uq/q7oucI87hgowC6wB28CJln3XgBmg9yYOHgDqQDNyGcHqdRzcASwC5zL8G1gGhuSNh8ADDwnbswAU8h7+GFiXsVNgHih5dLO80ZXn5nUZOAQGr9B5lhCSTXkqGovaeAA88ejMJRCw9SHlwZ3L7aGH9D2RQAN4EUOgrg0Wcx+aOZelahAV57U/Ig2xISmHjLyW0kfyI4vAdGhzTUqW5yFY/n8G9oAVYBzojvTCasjwjpR6Ajp9wFZi7JFN+/4ZInAkpU6PfEK1P3SYeWEM2AWWnEwqSn4UQ6Dokcfc9pVnb0nyfyECv6TU2waBjUDomuqcmY9wuA0CbtxdjOr/txCBGSl98shn2/DCF8kmYwrRn0AhmsvxDqwb/o0pRKh9muIb4hDj/ncx7m+hqmZ0psaUhcuHW4Fy8VK2rBk9JxILEe34cuo21SGtSrbwVPOEyd6TgIITikPdwocl6Z1oSGmhH9h3umDyaNblkDAXvvWMVlXpTDj7LOb/ndqfPJK1UNAk03Cyw2I8osJS1OrTvxXntTfk9txDqQtz7deEIlSLnX5SUVY/X1VJPdb6oRSbyuik97h9uAA+3jTLypXd7wAAAABJRU5ErkJggg=="
            icon_html = f"![width:24px]({icon_src})"
            markdown_content += f"- [{icon_html}{link_text}]({href})"
        elif element.name is None and element.string is not None:
            markdown_content += "".join(element.string)

        # 正規表現で置換
        markdown_content2 = re.sub(r"\n\$+ #", replace_spaces, markdown_content)
        markdown_content3 = markdown_content2.replace("$", "")

    # 文字列を改行で分割してリストにする
    lines = markdown_content3.split("\n")
    non_blank_lines = [line for line in lines if line.strip()]
    toc_lines = []

    for index, line in enumerate(non_blank_lines, start=1):
        if line.startswith("## "):
            toc_item = line.replace("## ", "")
            toc_lines.append(f"- [{toc_item}](#{index})")

    toc_content = "\n".join(toc_lines)
    markdown_content4 = "\n".join(non_blank_lines[3:])

    # ファイルを開いて内容を読み込む
    with open(input_md_path, "r", encoding="utf-8") as file:
        markdown_content_header = file.read()

    # MDファイルとして出力
    with open(output_md_path, "w", encoding="utf-8") as md_file:
        md_file.write(
            markdown_content_header + "\n" + toc_content + "\n\n" + markdown_content4
        )


# 現在の日付を取得
current_date = datetime.now()
# フォーマットを指定して日付文字列を生成
date_string = current_date.strftime("%Y_%m_%d")

# 使用例
html_file_path = f"bookmarks_{date_string}.html"  # 入力HTMLファイルのパス
output_md_path = "bookmarks.md"  # 出力するMarkdownファイルのパス
input_md_path = "inputmd.txt"
html_to_markdown(html_file_path, output_md_path, input_md_path)
