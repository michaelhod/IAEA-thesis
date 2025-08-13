import sys
sys.path.insert(1, r"C:/Users/micha/Documents/Imperial Courses/Thesis/IAEA-thesis")
from Stage1.tree_helpers import *

BLOCKS = {'p','li','dt','dd','blockquote','pre','h1','h2','h3','h4','h5','h6','figcaption'}
LEFTOVER = {'td'}
SKIP   = {'script','style','noscript','template'}

# --- helpers ---------------------------------------------------------------

def style_dict(el):
    """Parse inline style into a dict with lowercase keys/values."""
    s = el.get('style') or ''
    items = [x.strip() for x in s.split(';') if x.strip()]
    out = {}
    for it in items:
        if ':' in it:
            k, v = it.split(':', 1)
            out[k.strip().lower()] = v.strip().lower()
    return out

def hidden_by_inline_style(el):
    s = style_dict(el)
    if s.get('display') == 'none': return True
    if s.get('visibility') == 'hidden': return True
    return False

def non_empty_text(el):
    txt = get_node_text(el)
    return bool(txt)

def xpath_of(el):
    return el.getroottree().getpath(el)

def is_inside_any(el, containers_set):
    """Return True if el has any ancestor in containers_set."""
    p = el.getparent()
    while p is not None:
        if p in containers_set:
            return True
        p = p.getparent()
    return False

def is_ancestor_of_any(el, block_containers):
    return any(is_inside_any(bc, {el}) for bc in block_containers)


def anchor_is_block_like(a, block_containers):
    """
    Decide if <a> should be treated as a block of text by itself.
    Rules:
      - style display != inline
      - followed by a <br>
      - lone element child of parent
      - has no parent in block
    """
    if hidden_by_inline_style(a):
        return False

    # 1) inline style: display != inline
    disp = style_dict(a).get('display')
    if disp and disp != 'inline':
        return True

    # 2) <a> followed by <br> (next *element* sibling)
    sib = a.getnext()
    if sib is not None and isinstance(sib.tag, str) and sib.tag.lower() == 'br':
        return True

    # 3) lone element child
    parent = a.getparent()
    if parent is not None:
        elem_kids = [c for c in parent if isinstance(c.tag, str)]
        if len(elem_kids) == 1 and elem_kids[0] is a:
            return True
        
    # 4) Is not inside a block
    if not is_inside_any(a, block_containers):
        return True

    return False

def is_deepest_td_with_text(td_el):
    """True if td_el has text and no descendant <td> with text (ignoring hidden)."""
    if not non_empty_text(td_el):
        return False
    for d in td_el.findall('.//td'):
        if not isinstance(d.tag, str):
            continue
        if hidden_by_inline_style(d):
            continue
        if non_empty_text(d):
            return False
    return True

# --- main extractor --------------------------------------------------------

def extract_chunk_xpaths(html_path, safeurl, include_text=False):
    """
    Return XPaths of chunk-defining elements:
      - block containers (p, li, blockquote, pre, headings, figcaption)
      - 'line-like' anchors as defined in anchor_is_block_like()
    """
    # with io.open(html_path, "r", encoding="utf-8", errors="ignore") as f:
    #     html_text = f.read()
    # doc = html.fromstring(html_text)

    tree = load_html_as_tree(html_path)

    #remove_external_a(tree, safeurl)

    results = []
    seen = set()            # for XPath de-dupe
    block_containers = set()  # store element objects for containment checks

    def push(el, kind):
        xp = xpath_of(el)
        if xp in seen:
            return
        seen.add(xp)
        if include_text:
            txt = ' '.join(el.itertext()).strip()
            results.append({'xpath': xp, 'type': kind, 'text': txt})
        else:
            results.append(xp)

    # 1) Block containers
    for tag in BLOCKS:
        for el in tree.findall('.//' + tag):
            if not isinstance(el.tag, str):     # skip comments/PIs
                continue
            if el.tag.lower() in SKIP:
                continue
            if hidden_by_inline_style(el):
                continue
            if not non_empty_text(el):
                continue
            block_containers.add(el)
            push(el, 'block')

    # 1.5) Leftovers IF they havent been picked up
    for tag in LEFTOVER:
        for el in tree.findall('.//' + tag):
            if is_inside_any(el, block_containers):
                continue
            if is_ancestor_of_any(el, block_containers):
                continue
            if el.tag.lower() == "td" and not is_deepest_td_with_text(el):
                continue
            if el.tag.lower() == "span":
                print(get_node_text(el))
            block_containers.add(el)
            push(el, 'l-over')

    # 2) "Line-like" anchors outside the captured blocks
    for a in tree.findall('.//a'):
        if not isinstance(a.tag, str):
            continue
        if hidden_by_inline_style(a):
            continue
        if is_inside_any(a, block_containers):
            continue
        if not non_empty_text(a):
            continue
        if anchor_is_block_like(a, block_containers):
            push(a, 'line')

    return results

# --- example usage ---------------------------------------------------------
if __name__ == "__main__":
    path = "C:\\Users\\micha\\Documents\\Imperial Courses\\Thesis\\IAEA-thesis\\data\\swde\\sourceCode\\sourceCode\\movie\\movie\\movie-allmovie(2000)\\0000.htm"
    xps = extract_chunk_xpaths(path, include_text=True)
    for item in xps:
        print(item['type'].ljust(6), item['xpath'], "=>", item['text'][:90])
