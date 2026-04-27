'''Cell-by-cell verification of Tables 4-9 in the report against summary.csv.'''
import csv

rows = list(csv.DictReader(open('eval_results/summary.csv')))
data = {(r['product'], r['model'], r['config']): r for r in rows}

def f(x):
    return float(x) if x else None

cmap = {'v1': 'v1_title_clip',
        'v2': 'v2_initial_prompt_clip',
        'v3': 'v3_initial_prompt_features'}

# Reported values from each per-product table in the report
reported = {
    # Table 4 - Chess Set
    ('chess_set', 'flux',  'v1'): {'iq': 0.324, 'clip': 0.867, 'dino': 0.741, 'sig': 0.839, 'fc': 0.730},
    ('chess_set', 'flux',  'v2'): {'iq': 0.319, 'clip': 0.867, 'dino': 0.741, 'sig': 0.839, 'fc': 0.730},
    ('chess_set', 'flux',  'v3'): {'iq': 0.429, 'clip': 0.830, 'dino': 0.728, 'sig': 0.826, 'fc': 0.812},
    ('chess_set', 'gpt',   'v1'): {'iq': 0.324, 'clip': 0.864, 'dino': 0.734, 'sig': 0.856, 'fc': 0.728},
    ('chess_set', 'gpt',   'v2'): {'iq': 0.324, 'clip': 0.864, 'dino': 0.734, 'sig': 0.856, 'fc': 0.728},
    ('chess_set', 'gpt',   'v3'): {'iq': 0.423, 'clip': 0.861, 'dino': 0.741, 'sig': 0.851, 'fc': 0.737},
    # Table 5 - Water Bottle
    ('water_bottle', 'flux', 'v1'): {'iq': 0.277, 'clip': 0.754, 'dino': 0.413, 'sig': 0.660, 'fc': 0.579},
    ('water_bottle', 'flux', 'v2'): {'iq': 0.280, 'clip': 0.689, 'dino': 0.420, 'sig': 0.613, 'fc': 0.636},
    ('water_bottle', 'flux', 'v3'): {'iq': 0.524, 'clip': 0.689, 'dino': 0.420, 'sig': 0.613, 'fc': 0.636},
    ('water_bottle', 'gpt',  'v1'): {'iq': 0.306, 'clip': 0.741, 'dino': 0.436, 'sig': 0.743, 'fc': 0.786},
    ('water_bottle', 'gpt',  'v2'): {'iq': 0.312, 'clip': 0.741, 'dino': 0.436, 'sig': 0.743, 'fc': 0.786},
    ('water_bottle', 'gpt',  'v3'): {'iq': 0.488, 'clip': 0.741, 'dino': 0.436, 'sig': 0.743, 'fc': 0.786},
    # Table 6 - Jeans
    ('jeans', 'flux', 'v1'): {'iq': 0.282, 'clip': 0.895, 'dino': 0.480, 'sig': 0.760, 'fc': 0.694},
    ('jeans', 'flux', 'v2'): {'iq': 0.326, 'clip': 0.853, 'dino': 0.439, 'sig': 0.742, 'fc': 0.497},
    ('jeans', 'flux', 'v3'): {'iq': 0.503, 'clip': 0.853, 'dino': 0.439, 'sig': 0.742, 'fc': 0.497},
    ('jeans', 'gpt',  'v1'): {'iq': 0.294, 'clip': 0.882, 'dino': 0.557, 'sig': 0.786, 'fc': 0.594},
    ('jeans', 'gpt',  'v2'): {'iq': 0.330, 'clip': 0.889, 'dino': 0.584, 'sig': 0.791, 'fc': 0.617},
    ('jeans', 'gpt',  'v3'): {'iq': 0.450, 'clip': 0.873, 'dino': 0.547, 'sig': 0.751, 'fc': 0.571},
    # Table 7 - Backpack (no in_loop_best_q reported)
    ('backpack', 'flux', 'v1'): {'clip': 0.813, 'dino': 0.551, 'sig': 0.649, 'fc': 0.156},
    ('backpack', 'flux', 'v2'): {'clip': 0.813, 'dino': 0.551, 'sig': 0.649, 'fc': 0.156},
    ('backpack', 'flux', 'v3'): {'clip': 0.813, 'dino': 0.551, 'sig': 0.649, 'fc': 0.156},
    ('backpack', 'gpt',  'v1'): {'clip': 0.824, 'dino': 0.633, 'sig': 0.729, 'fc': 0.555},
    ('backpack', 'gpt',  'v2'): {'clip': 0.824, 'dino': 0.633, 'sig': 0.729, 'fc': 0.555},
    ('backpack', 'gpt',  'v3'): {'clip': 0.824, 'dino': 0.633, 'sig': 0.729, 'fc': 0.555},
}

print('TABLES 4-7: per-cell verification')
errors_47 = []
for (p, m, ver), rep in reported.items():
    row = data[(p, m, cmap[ver])]
    actual = {
        'iq':   f(row['in_loop_best_q']),
        'clip': f(row['clip_img_vs_gt_mean']),
        'dino': f(row['dinov2_vs_gt_mean']),
        'sig':  f(row['siglip_vs_gt_mean']),
        'fc':   f(row['gen_features_vs_ground_truth']),
    }
    for col in ['iq', 'clip', 'dino', 'sig', 'fc']:
        if col not in rep:
            continue
        a = round(actual[col], 3)
        r = rep[col]
        diff = round(a - r, 4)
        if abs(diff) > 0.001:
            errors_47.append(f"  {p}/{m}/{ver} col={col}: reported={r:.3f}  actual={a:.3f}  diff={diff:+.4f}")

if errors_47:
    print('  DISCREPANCIES:')
    for e in errors_47:
        print(e)
else:
    print('  All 110 numerical cells in Tables 4-7 match summary.csv to 3dp.')

# Table 8: Feat. Agr. GT
print('\nTABLE 8: Feat. Agr. GT cells')
table8 = {
    ('chess_set',    'flux', 'v1'): 0.730,
    ('chess_set',    'flux', 'v2'): 0.730,
    ('chess_set',    'flux', 'v3'): 0.812,
    ('chess_set',    'gpt',  'v1'): 0.728,
    ('chess_set',    'gpt',  'v2'): 0.728,
    ('chess_set',    'gpt',  'v3'): 0.737,
    ('water_bottle', 'flux', 'v1'): 0.579,
    ('water_bottle', 'flux', 'v2'): 0.636,
    ('water_bottle', 'flux', 'v3'): 0.636,
    ('water_bottle', 'gpt',  'v1'): 0.786,
    ('water_bottle', 'gpt',  'v2'): 0.786,
    ('water_bottle', 'gpt',  'v3'): 0.786,
    ('jeans',        'flux', 'v1'): 0.694,
    ('jeans',        'flux', 'v2'): 0.497,
    ('jeans',        'flux', 'v3'): 0.497,
    ('jeans',        'gpt',  'v1'): 0.594,
    ('jeans',        'gpt',  'v2'): 0.617,
    ('jeans',        'gpt',  'v3'): 0.571,
    ('backpack',     'flux', 'v1'): 0.156,
    ('backpack',     'flux', 'v2'): 0.156,
    ('backpack',     'flux', 'v3'): 0.156,
    ('backpack',     'gpt',  'v1'): 0.555,
    ('backpack',     'gpt',  'v2'): 0.555,
    ('backpack',     'gpt',  'v3'): 0.555,
}
errors_8 = []
for (p, m, v), rep in table8.items():
    actual = round(f(data[(p, m, cmap[v])]['gen_features_vs_ground_truth']), 3)
    if abs(actual - rep) > 0.001:
        errors_8.append(f"  {p}/{m}/{v}: reported={rep}  actual={actual}")
if errors_8:
    print('  DISCREPANCIES:')
    for e in errors_8:
        print(e)
else:
    print('  All 24 cells in Table 8 match.')

# Table 9: averages across v1/v2/v3
print('\nTABLE 9: CLIP avg and DINOv2 avg per (product, model)')
table9 = {
    ('chess_set',    'flux'): {'clip': 0.855, 'dino': 0.737},
    ('chess_set',    'gpt'):  {'clip': 0.863, 'dino': 0.736},
    ('water_bottle', 'flux'): {'clip': 0.711, 'dino': 0.418},
    ('water_bottle', 'gpt'):  {'clip': 0.741, 'dino': 0.436},
    ('jeans',        'flux'): {'clip': 0.867, 'dino': 0.453},
    ('jeans',        'gpt'):  {'clip': 0.881, 'dino': 0.563},
    ('backpack',     'flux'): {'clip': 0.813, 'dino': 0.551},
    ('backpack',     'gpt'):  {'clip': 0.824, 'dino': 0.633},
}
errors_9 = []
for (p, m), rep in table9.items():
    clips = [f(data[(p, m, cmap[v])]['clip_img_vs_gt_mean']) for v in ['v1', 'v2', 'v3']]
    dinos = [f(data[(p, m, cmap[v])]['dinov2_vs_gt_mean']) for v in ['v1', 'v2', 'v3']]
    cavg = round(sum(clips) / 3, 3)
    davg = round(sum(dinos) / 3, 3)
    if abs(cavg - rep['clip']) > 0.001:
        errors_9.append(f"  {p}/{m}: CLIP reported={rep['clip']:.3f}  actual={cavg:.3f}")
    if abs(davg - rep['dino']) > 0.001:
        errors_9.append(f"  {p}/{m}: DINOv2 reported={rep['dino']:.3f}  actual={davg:.3f}")
if errors_9:
    print('  DISCREPANCIES:')
    for e in errors_9:
        print(e)
else:
    print('  All 16 cells in Table 9 match computed averages.')
