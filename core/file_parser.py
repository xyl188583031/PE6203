"""简历/JD 文件解析：从 PDF / Word(.docx) / 纯文本中抽取可读文本。

设计原则：
- 只依赖轻量、纯 Python 的库（pypdf、python-docx），缺失时给出可执行的提示而不是崩栈；
- 所有函数接收 bytes，返回 (文本, 错误信息)，错误信息为 None 表示成功；
- 不写盘、不外发，仅在内存中处理。
"""

import io
import os
from typing import Optional, Tuple

# 允许上传的扩展名（不含点）
SUPPORTED_EXTENSIONS = ["pdf", "docx", "txt", "md"]

# 上传控件里放宽到 .doc，便于给出「另存为 .docx」的明确提示
UPLOAD_EXTENSIONS = ["pdf", "doc", "docx", "txt", "md"]

# 抽取文本的长度上限，防止异常大的文件撑爆 prompt
MAX_CHARS = 20000


def _decode_bytes(data: bytes) -> str:
    """按常见编码依次尝试解码纯文本。"""
    for enc in ("utf-8", "utf-8-sig", "gb18030", "big5", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("未安装 pypdf，请执行：pip install pypdf")

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)


def _extract_docx_via_zip(data: bytes) -> str:
    """不依赖第三方库的 .docx 抽取（docx 本质是 zip + XML）。

    直接读取 word/document.xml，按段落 <w:p> 切分，拼接其中的 <w:t> 文本。
    表格单元格同样是 <w:p>，因此表格内容也能被抽到。
    """
    import re
    import zipfile

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        target = "word/document.xml"
        if target not in names:
            candidates = [n for n in names if n.endswith("document.xml")]
            if not candidates:
                raise RuntimeError("无效的 .docx 文件：缺少 word/document.xml")
            target = candidates[0]
        xml = zf.read(target).decode("utf-8", errors="ignore")

    paragraphs = re.findall(r"<w:p[ >].*?</w:p>|<w:p/>", xml, flags=re.S)
    out = []
    for para in paragraphs:
        # <w:tab/> 视为制表符，<w:br/> 视为换行
        para = para.replace("<w:tab/>", "\t")
        texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", para, flags=re.S)
        line = "".join(texts)
        line = (
            line.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&apos;", "'")
        )
        out.append(line)
    return "\n".join(out)


def extract_docx(data: bytes) -> str:
    try:
        import docx  # python-docx（可选，装了就用它，表格处理更细）
    except ImportError:
        return _extract_docx_via_zip(data)

    document = docx.Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs]

    # 简历常用表格排版，表格内容也要抽取
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            line = "\t".join(c for c in cells if c)
            if line:
                parts.append(line)

    return "\n".join(parts)


def _clean(text: str) -> str:
    """压缩连续空行、去掉首尾空白。"""
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    out = []
    blank = 0
    for ln in lines:
        if ln.strip():
            blank = 0
            out.append(ln)
        else:
            blank += 1
            if blank <= 1:  # 最多保留一个空行
                out.append("")
    return "\n".join(out).strip()


def extract_text(
    data: bytes, filename: str = "", max_chars: int = MAX_CHARS
) -> Tuple[str, Optional[str]]:
    """从文件字节流抽取文本。

    返回 (文本, 错误描述)。成功时错误为 None；失败时文本为 ""。
    """
    if not data:
        return "", "文件内容为空"

    ext = os.path.splitext(filename or "")[1].lower().lstrip(".")

    try:
        if ext == "pdf":
            raw = extract_pdf(data)
        elif ext in ("docx", "doc"):
            if ext == "doc":
                return "", "不支持旧版 .doc 格式，请在 Word 中「另存为 .docx」后重新上传"
            raw = extract_docx(data)
        elif ext in ("txt", "md", ""):
            raw = _decode_bytes(data)
        else:
            return "", f"暂不支持 .{ext} 格式，支持：PDF / Word(.docx) / txt / md"
    except RuntimeError as e:
        return "", str(e)
    except Exception as e:  # 损坏文件、加密文件等
        return "", f"解析失败（文件可能损坏或已加密）：{e}"

    text = _clean(raw)
    if not text:
        hint = "（可能是扫描件或图片型 PDF，需要 OCR）" if ext == "pdf" else ""
        return "", f"未能从文件中提取到文本{hint}"

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n...（内容过长，已截断至 {max_chars} 字）"
    return text, None


def extract_text_from_upload(uploaded_file) -> Tuple[str, Optional[str]]:
    """Streamlit UploadedFile 的便捷封装。"""
    if uploaded_file is None:
        return "", "未选择文件"
    try:
        data = uploaded_file.getvalue()
    except Exception:
        data = uploaded_file.read()
    return extract_text(data, getattr(uploaded_file, "name", ""))
