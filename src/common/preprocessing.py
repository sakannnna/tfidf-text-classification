"""共享邮件头清洗规则：保留标题及原样正文。"""
import re

HEADER_SEPARATOR = re.compile(r"\r?\n[ \t]*\r?\n")
HEADER_FIELD = re.compile(r"^([!-9;-~]+):(.*)$")


def parse_header(text):
    """返回 (字段列表, 原样正文)；边界或字段结构不明确则返回 None。"""
    separator = HEADER_SEPARATOR.search(text)
    if separator is None:
        return None

    fields = []
    for line in text[:separator.start()].splitlines():
        if line.startswith((" ", "\t")):
            if not fields:  # 没有前一个字段，无法判断此续行属于哪里。
                return None
            name, value = fields[-1]
            fields[-1] = (name, value + " " + line.strip())
        else:
            match = HEADER_FIELD.fullmatch(line)
            if match is None:
                return None
            fields.append((match.group(1).lower(), match.group(2).strip()))

    # 本数据每篇都有标题；缺少标题时保守地保留原文。
    if not any(name == "subject" for name, _ in fields):
        return None
    return fields, text[separator.end():]


def clean_headers(text):
    """保留完整标题（含续行、Re:）和原样正文，删除其他开头字段。"""
    parsed = parse_header(text)
    if parsed is None:
        return text
    fields, body = parsed
    subjects = [value for name, value in fields if name == "subject"]
    return "\n".join(subjects) + "\n\n" + body


def check_cleaning(texts, name):
    """检查整个数据集的解析覆盖率及正文保真情况。"""
    fallback = 0
    for text in texts:
        parsed = parse_header(text)
        cleaned = clean_headers(text)
        if parsed is None:
            fallback += 1
            assert cleaned == text
        else:
            fields, body = parsed
            title = "\n".join(value for key, value in fields if key == "subject")
            assert cleaned[len(title) + 2:] == body
        assert cleaned.strip(), "清洗产生了空文本"
    print(f"{name}：{len(texts)} 条；保留原文回退：{fallback} 条；正文完整性检查通过")
