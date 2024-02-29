---
marp: true
theme: Marpでアウトプット量産
headingDivider: 2
paginate: true
math: katex
---

# Marpでアウトプット量産
<!-- _paginate: false -->
<!-- _class : title -->

## Todo

#### Marp関連でやりたい事

- Qiitaにアウトプット
- MDエディタ
- mermaid試す（何ができるか）
- 16/9テーマ作成
- 背景画像テーマ作成

# Marpとは

## Marp: Markdown Presentation Ecosystem

#### Marpはマークダウン形式を用いてプレゼンテーションを作成するツールです

1. **シンプル:** マークダウン形式で直感的にスライドを作成。
2. **ビジュアルエディタ:** リアルタイムでスライドの見た目を確認。
3. **多機能:** PDFやHTMLへエクスポート可能、テーマ選択やカスタマイズ可能。
4. **対応OS:** Windows, MacOS, Linux等。

<style 'scoped'>section h3{text-align:center}</style>

#### Marpは、見栄えの良いスライドを簡単、迅速に作成するツールです

## Marpの使い方について

#### 参考リンク集

- [Marp公式](https://marp.app/)
- [Directives](https://marpit.marp.app/directives)
- [よろしく](https://zenn.dev/cota_hu/books/marp-beginner-advanced/viewer/develop-7#pasteimage.defaultname)
- [Marp入門〜応用｜markdownでプレゼン資料を楽に素早く作って発表しよう](https://zenn.dev/cota_hu/books/marp-beginner-advanced)
- [【VS Code + Marp】Markdownから爆速・自由自在なデザインで、プレゼンスライドを作る](https://qiita.com/tomo_makes/items/aafae4021986553ae1d8)
- [VSCode と Marp で A4 マニュアルを作成する方法](https://zenn.dev/ashitaka0789/articles/8a558b279e16a6)
- [Markdownでプレゼンを作ってGitHubで自動公開するフローを整えた](https://blog.cosnomi.com/posts/marp-github-actions/)

# ２カラムレイアウトのテスト

## こんなふうに2カラムにできる

<div class="twocolumnview"><div>

#### 左側のコンテンツ

![w:400 h:380px](images/image-3.png)

</div><div>

#### 右側のコンテンツ

- あいうえお
- かきくけこ

</div></div>

これはどうなる？

# Tips:画像の挿入について

## VScodeの設定次第でCtrl＋Cで画像を挿入する事が可能

- "markdown.editor.filePaste.enabled": true など
- ファイルをコピーする事も可能
- 下記のようにフィルター設定をかける事もできる
  ![h:100](images/image-4.png)
  ![blur h:100](images/image-4.png)
  ![brightness:80% h:100](images/image-4.png)
  ![contrast:200% h:100](images/image-4.png)
  ![drop-shadow h:100](images/image-4.png)
  ![grayscale h:100](images/image-4.png)
  ![hue-rotate h:100](images/image-4.png)
  ![invert h:100](images/image-4.png)
  ![opacity:0.5 h:100](images/image-4.png)
  ![saturate h:100](images/image-4.png)
  ![sepia h:100](images/image-4.png)
- 詳しくは[こちら](https://marpit.marp.app/image-syntax?id=image-filters)を参照

# 目次について

## 目次の作り方

 Tips:目次の作り方

- 拡張機能[Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one)を使用。

  1. 目次を挿入したい場所にカーソルを移動する。
  2. コマンドの表示と実行(Ctrl+Shift+P)から「Markdown All in One:目次(TOC)の作成」を選択し実行する。

## テスト目次

<style 'scoped'> H2 + ul{font-size: 5px;}</style>

# 背景画像の指定について

## Marpでの背景画像指定

 [WIP]Tips:背景画像について

![bg](blank)
![bg blur:5px opacity:.1](images/icon.png)

# フリーフォントを利用する

## Google Web fontsの使用方法

 [WIP] Tips:Google Web fontsの使い方

1. [Google Web fonts](https://fonts.google.com/?subset=japanese)で、好きなフォントを選ぶ。
    - a
2.
