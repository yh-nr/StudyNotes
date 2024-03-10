from datetime import datetime

# 現在の日付を取得
current_date = datetime.now()

# フォーマットを指定して日付文字列を生成
date_string = current_date.strftime("%Y_%m_%d")

# 文字列を表示
print(f"bookmarks_{date_string}.html")
