'''visual_vocabulary.py
--------------------
Shared vocabulary of visual-descriptor words used by both the gating-strategy
comparison script and the downstream preprocessing pipeline.

History: the original list lived inline in `score_visual_signal.py` (used for
initial product-candidate scoring). That copy is now frozen and should be
considered legacy. New code should import from here.

A review 'passes' the vocab gate if it contains at least
`VISUAL_WORD_THRESHOLD` distinct words from `ALL_VISUAL_WORDS`.
Whole-word matching (regex \\b boundaries) prevents false positives like
'red' matching 'reduced'.
'''

import re


## COLOR ##

COLOR_WORDS = {
    # Basic colors
    'black', 'white', 'gray', 'grey', 'silver', 'gold', 'copper', 'bronze',
    'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown',
    'beige', 'tan', 'cream', 'navy', 'teal', 'turquoise', 'burgundy', 'maroon',
    'olive', 'mint', 'coral', 'lavender', 'magenta', 'crimson', 'indigo',
    # Finishes
    'matte', 'glossy', 'shiny', 'metallic', 'chrome', 'brushed', 'polished',
    'anodized', 'powder-coated', 'translucent', 'transparent', 'opaque',
    'iridescent', 'holographic', 'pearlescent', 'satin',
}

# NEW: modifiers that sharpen a color description ('dark walnut', 'pale blue').
COLOR_MODIFIER_WORDS = {
    'dark', 'light', 'bright', 'pale', 'deep', 'rich', 'vivid', 'muted',
    'two-tone', 'multi-color',
    # 'faded' is already in APPEARANCE_WORDS; intentionally omitted here.
}


## MATERIAL ##

MATERIAL_WORDS = {
    'wood', 'wooden', 'metal', 'metallic', 'plastic', 'rubber', 'silicone',
    'leather', 'fabric', 'cotton', 'polyester', 'nylon', 'denim', 'suede',
    'cashmere', 'wool', 'canvas', 'mesh',
    'stainless', 'steel', 'aluminum', 'iron', 'titanium', 'brass',
    'glass', 'ceramic', 'porcelain', 'clay',
    'bamboo', 'oak', 'pine', 'walnut', 'mahogany', 'rosewood', 'cherry',
    'acrylic', 'resin', 'foam', 'vinyl',
}


## SHAPE / SIZE ##

SHAPE_SIZE_WORDS = {
    'tall', 'short', 'wide', 'narrow', 'thin', 'thick', 'slim', 'bulky',
    'compact', 'small', 'large', 'huge', 'tiny', 'medium',
    'round', 'square', 'rectangular', 'curved', 'angular', 'oval', 'cylindrical',
    'sleek', 'streamlined', 'chunky', 'slender',
    'inch', 'inches', 'cm', 'mm', 'foot', 'feet',
    'ounce', 'ounces', 'oz', 'pound', 'lb', 'lbs', 'gram', 'grams',
    'gallon', 'liter', 'liters', 'ml',
}


## APPEARANCE (visual verbs, aesthetic adjectives, condition/finish) ##

APPEARANCE_WORDS = {
    # Visual verbs
    'looks', 'looked', 'looking', 'appear', 'appears', 'appeared', 'appearing',
    'seem', 'seems', 'seemed', 'visible', 'shows', 'showing', 'display', 'displays',
    # Photo/picture references
    'picture', 'pictures', 'pictured', 'photo', 'photos', 'photograph',
    'image', 'images', 'pic', 'pics', 'depicted', 'shown',
    # Design/aesthetic
    'design', 'designed', 'style', 'styling', 'stylish', 'aesthetic', 'elegant',
    'beautiful', 'gorgeous', 'pretty', 'attractive', 'ugly', 'hideous',
    'cute', 'classy', 'tacky', 'cheesy', 'cheap-looking',
    # Quality visual descriptors
    'pristine', 'flawless', 'spotless', 'clean-looking',
    'scratched', 'chipped', 'faded', 'worn', 'weathered', 'distressed',
    'dented', 'cracked', 'scuffed', 'smudged',
    # Finish condition
    'fingerprints', 'smudges', 'glare', 'reflection', 'reflective',
    'textured', 'smooth', 'rough', 'bumpy', 'grainy',
    'pattern', 'patterned', 'printed', 'embossed', 'engraved', 'etched',
}


## SPATIAL / STRUCTURAL (where things are; parts of the object) ##

SPATIAL_STRUCTURAL_WORDS = {
    'board', 'top', 'bottom', 'front', 'back', 'side', 'sides',
    'edge', 'edges', 'corner', 'corners', 'center', 'middle',
    'inside', 'outside', 'surface', 'base',
    'handle', 'handles', 'strap', 'straps',
    'zipper', 'zippers', 'pocket', 'pockets', 'button', 'buttons',
    'seam', 'seams', 'lid', 'cap',
    'logo', 'label', 'mark', 'marking',
    'stitching', 'stitched', 'trim', 'trimmed',
    'backing', 'lining', 'lined',
    'waistband',
}


## FORM / QUALITY (how the object is built/decorated) ##

FORM_QUALITY_WORDS = {
    'detail', 'details', 'carved', 'ornate', 'plain', 'simple', 'fancy',
    'decorative', 'functional', 'sturdy', 'solid', 'flimsy', 'delicate',
    # 'etched', 'engraved', 'patterned' already in APPEARANCE_WORDS; omitted.
}


## FIT / GARMENT (needed for the jeans product specifically) ##

FIT_GARMENT_WORDS = {
    'fit', 'fitted', 'loose', 'tight', 'baggy', 'skinny',
    'flared', 'bootcut', 'stretchy', 'hem', 'straight',
}


## PATTERN / TEXTURE (picks up visual detail not covered above) ##

PATTERN_TEXTURE_WORDS = {
    'striped', 'stripes', 'woven', 'checkered', 'embroidered',
}


## PRODUCT-SPECIFIC (chess + water-bottle anatomy not in other groups) ##

PRODUCT_SPECIFIC_WORDS = {
    # Chess
    'pieces', 'compartment', 'inlay', 'felt',
    # Water bottle
    'mouth', 'spout', 'straw',
}


## UNION + REGEX ##

ALL_VISUAL_WORDS = (
    COLOR_WORDS
    | COLOR_MODIFIER_WORDS
    | MATERIAL_WORDS
    | SHAPE_SIZE_WORDS
    | APPEARANCE_WORDS
    | SPATIAL_STRUCTURAL_WORDS
    | FORM_QUALITY_WORDS
    | FIT_GARMENT_WORDS
    | PATTERN_TEXTURE_WORDS
    | PRODUCT_SPECIFIC_WORDS
)

# Sort longest-first so multi-char tokens match before shorter substrings
# (defense-in-depth alongside \b; doesn't change semantics but is safer).
_SORTED_WORDS = sorted(ALL_VISUAL_WORDS, key=len, reverse=True)

VISUAL_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(w) for w in _SORTED_WORDS) + r')\b',
    flags=re.IGNORECASE,
)

# Threshold for the vocab gate in the comparison script and downstream
# preprocessing. NOTE: the legacy score_visual_signal.py uses its own
# (stricter) threshold of 3 for a different purpose (candidate scoring).
VISUAL_WORD_THRESHOLD = 2


def visual_word_set(text: str) -> set:
    '''Return the set of distinct visual descriptor words found in `text`.'''
    if not text:
        return set()
    return {m.lower() for m in VISUAL_PATTERN.findall(text)}


def passes_vocab_gate(text: str) -> tuple[bool, set]:
    '''Return (passes, matched_words). Passes if >= VISUAL_WORD_THRESHOLD.'''
    matched = visual_word_set(text)
    return (len(matched) >= VISUAL_WORD_THRESHOLD, matched)


if __name__ == '__main__':
    # Self-check: print group sizes and total.
    groups = [
        ('COLOR_WORDS', COLOR_WORDS),
        ('COLOR_MODIFIER_WORDS', COLOR_MODIFIER_WORDS),
        ('MATERIAL_WORDS', MATERIAL_WORDS),
        ('SHAPE_SIZE_WORDS', SHAPE_SIZE_WORDS),
        ('APPEARANCE_WORDS', APPEARANCE_WORDS),
        ('SPATIAL_STRUCTURAL_WORDS', SPATIAL_STRUCTURAL_WORDS),
        ('FORM_QUALITY_WORDS', FORM_QUALITY_WORDS),
        ('FIT_GARMENT_WORDS', FIT_GARMENT_WORDS),
        ('PATTERN_TEXTURE_WORDS', PATTERN_TEXTURE_WORDS),
        ('PRODUCT_SPECIFIC_WORDS', PRODUCT_SPECIFIC_WORDS),
    ]
    raw_sum = 0
    for name, s in groups:
        print(f'  {name:<28} {len(s):>4}')
        raw_sum += len(s)
    print(f'  {"-"*28} {"----"}')
    print(f'  {"raw sum (with overlaps)":<28} {raw_sum:>4}')
    print(f'  {"ALL_VISUAL_WORDS (dedup)":<28} {len(ALL_VISUAL_WORDS):>4}')
    print(f'  threshold: >= {VISUAL_WORD_THRESHOLD} distinct visual words')
