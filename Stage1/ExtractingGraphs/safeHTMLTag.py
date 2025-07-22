"""
This file is needed as swde gives us broken html (example in univerisity-princetonreview(615)/0595.htm, there is a <b... tag)
"""

import re

ELLIPSIS_RE  = re.compile(r'\u2026|\.{3}')

# XML element-name syntax  (https://www.w3.org/TR/xml/#NT-Name)
XML_NAME_RE  = re.compile(r'^[A-Za-z_][\w\-]*$')
NAME_START_RE = re.compile(r'^[A-Za-z_]')
NAME_CHARS_RE = re.compile(r'[A-Za-z0-9_\-]')

def safe_name(raw: str) -> str:
    """
    Return *raw* if it is a legal XML name, otherwise
    return the longest legal prefix, or '*' as a last resort.
    """
    #Strips ellipsis as singular . are allowed but ellipsis are not
    raw = ELLIPSIS_RE.sub('', raw) 
    if XML_NAME_RE.match(raw):
        return raw

    if NAME_START_RE.match(raw):
        # keep characters until we hit something illegal
        legal = ''.join(ch for ch in raw if NAME_CHARS_RE.match(ch))
        if legal:
            return legal  # e.g. 'b' from 'b...<'
    return '*'
