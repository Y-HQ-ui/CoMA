import re
from typing import List, Tuple

# === 配置 ===
SEPARATORS = ['|', '|]', '||', '[|', '|:', ':|', '::']
NEWSEP = '<|>'  # 列分隔符（允许两侧有任意空白）

def normalize_spacing(text: str) -> str:
    """
    归一化小节线与邻接字符的空白，尽量不改变音乐记号顺序：
    - 保证每个已知小节线两侧有合适空格
    - 修正引号与换行邻接的小节线粘连
    不做音乐性改写。
    """
    for sep in SEPARATORS:
        # 在 " sep X" 的位置，确保 sep 后面至少有一个空格（但如果 X 是 :, |, ] 则不加空格）
        def subfunc(m: re.Match) -> str:
            nxt = m.group(1)
            if nxt in [':', '|', ']']:
                return f' {sep}{nxt}'
            return f' {sep} {nxt}'
        pattern = r' ' + re.escape(sep) + r'(.{1})'
        text = re.sub(pattern, subfunc, text)

        # 行首/引号相邻的边界情况
        text = text.replace('\n' + sep + '"', '\n ' + sep + ' "')
        text = text.replace(' ' + sep + '\n', ' ' + sep + ' \n')

    return text

def split_head_and_body(src: str) -> Tuple[str, str]:
    """
    将文本按首次出现的 <|>（允许两侧有空白）切为 (head, body)。
    若找不到分隔符，则 body 为空。
    """
    m = re.search(r'\s*' + re.escape(NEWSEP) + r'\s*', src)
    if not m:
        return src, ''  # 视为没有多列
    return src[:m.start()], src[m.start():]

def _protect_barlines_for_column_split(s: str) -> str:
    """
    将已知小节线模式替换为占位符，方便之后按 '<=>' 切分列内容，再还原。
    具体做法：
    - 把 ' {sep} ' 替换为 ' <SEP_i><=> '，其中 <SEP_i> 是编号占位
    - 之后再按 '<=>' 切分为列片段，再恢复占位符为原小节线
    """
    # 生成映射：' sep ' -> ' <i>'
    sep_dict = {' ' + sep + ' ': f' <SEP{i}>' for i, sep in enumerate(SEPARATORS, start=1)}
    out = s
    for sep, tok in sep_dict.items():
        out = out.replace(sep, tok + '<=> ')
    return out

def _restore_barlines(s: str) -> str:
    """恢复 _protect_barlines_for_column_split() 放入的占位符为原小节线。"""
    for i, sep in enumerate(SEPARATORS, start=1):
        s = s.replace(f' <SEP{i}>', f' {sep} ')
    return s

def split_to_voices(src: str) -> Tuple[str, List[str]]:
    """
    核心：按 <|> 列切分 → 多声部。
    返回 (head, voices)，其中 voices[i] 是第 i 个声部的原始文本拼接结果。
    规则：
    - 先把 <n> → 换行，再做 spacing 归一
    - 以首个 <|> 为界，之前是头部 head
    - 之后部分按正则 r'\\s*<\\|>\\s*' 切出“列单元”序列 units
    - 每个单元再用占位保护已知小节线 → 按 '<=>' 切分 → 得到本单元的多列片段
      （若某单元列数与已有声部数不一致，自动扩展或对齐）
    - 拼回每个声部的全文
    """
    # 1) <n> -> newline，并做 spacing
    text = src.replace('<n>', '\n')
    text = normalize_spacing(text)

    # 2) 头部与主体
    head, body = split_head_and_body(text)
    if not body:
        # 没有列分隔，视为单声部
        return head.strip(), [text.strip()]

    # 3) 按 <|>（含可变空白）切分单元
    units = re.split(r'\s*' + re.escape(NEWSEP) + r'\s*', body)
    # 去掉空串单元（常见于 “<|> <|>” 连续导致的空单元）
    units = [u for u in units if u and u.strip()]

    voices: List[List[str]] = []  # 收集每个声部的片段列表

    for u in units:
        # 保护小节线 → 按 '<=>' 切分（每个 '<=>' 之前的占位代表一个列/声部片段）
        protected = _protect_barlines_for_column_split(u)
        cols = [c for c in re.split(r'<=>', protected) if c and c.strip()]
        # 还原小节线占位
        cols = [_restore_barlines(c) for c in cols]

        # 初始化/扩展声部容器
        if not voices:
            voices = [[] for _ in range(len(cols))]
        elif len(cols) > len(voices):
            voices.extend([[] for _ in range(len(cols) - len(voices))])

        # 逐列累加到对应声部
        for i, frag in enumerate(cols):
            voices[i].append(frag.strip())

    # 把各声部片段拼成字符串
    voice_texts = [' '.join(v).strip() for v in voices]
    return head.strip(), voice_texts

def reconstruct_abc(src: str, add_v_tags: bool = True) -> str:
    """
    将 split_to_voices 的结果重组为 ABC 文本：
    - 保留 head 原样（不强制修正 L/Q/M/K 的占位值）
    - 每个声部前加 V:1/2/…（可关闭）
    """
    head, voices = split_to_voices(src)
    lines = []
    if head:
        lines.append(head.strip())
    for i, vtext in enumerate(voices, start=1):
        if add_v_tags and len(voices) > 1:
            lines.append(f'V:{i}')
        lines.append(vtext)
    return '\n'.join(lines).rstrip()
    
def split_abc_columns(raw_text: str, add_v_tags: bool = True) -> str:
    """
    主接口函数：从原始含 <|> / <n> 的 ABC 编码中分离出多声部文本。

    参数：
        raw_text (str): 原始带 <|> / <n> 占位的 ABC 源字符串。
        add_v_tags (bool): 是否在输出中添加 V:1 / V:2 / ... 标记（默认 True）。

    返回：
        str: 处理后完整的 ABC 文本（多声部已展开）。
    """
    return reconstruct_abc(raw_text, add_v_tags=add_v_tags)


if __name__ == "__main__":
    piece = r'''X:1<n>L:<n>Q:<n>M:4/4<n>K:Gmin<n>|:"Gm" BGdB gdBG | <|> <|> "Cm" (3cdc (3BAG"D" ^F3 D | <|> <|> "Gm" (3BAG (3BAG"Cm" (3BAG (3BAG | <|> <|> <n>"D" ^F3 G (3AGF"Cm" (3FED | <|> <|> "Gm" (3BAG (3BAG"Cm" (3BAG (3BAG | <|> <|> "D" ^F3 G (3AGF"Cm" (3FED | <|> <|> <n>"D" ^F3 G"Gm" (3AGF (3FED | <|> <|> "Cm" CE"D"^FE'''
    # 只做结构分离（不改内容）
    print(reconstruct_abc(piece, add_v_tags=True))
