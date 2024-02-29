---
marp: true
theme: Marpでアウトプット量産
headingDivider: 2
paginate: true
math: katex
---


# Mermaidサンプル

## Mermaidサンプル
<!-- _style: div.mermaid { all: unset; } -->

<script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10.0.0/dist/mermaid.esm.min.mjs';
    mermaid.initialize({
        startOnLoad: true,
        'theme': 'black'
        });
    window.addEventListener(
        'vscode.markdown.updateContent',
        function() { mermaid.init() }
        );
</script>

- [公式解説](https://mermaid.js.org/syntax/mindmap.html)
- [Live Editor](https://mermaid.live/)
- [テーマ設定について](https://mermaid.js.org/config/theming.html)

<div class="mermaid" style="display: flex;justify-content: center;">
mindmap
  root((mindmap))
    Origins
      Long history
      ::icon(fa fa-book)
      Popularisation
        British popular psychology author Tony Buzan
    Research
      On effectivness<br/>and features
      On Automatic creation
        Uses
            Creative techniques
            Strategic planning
            Argument mapping
    Tools
      Pen and paper
      Mermaid
    Tools
      Pen and paper
      Mermaid
</div>
