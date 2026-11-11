"""Generate Jak 1's cinematic draw adapter in an isolated source project."""
import argparse
import json
from pathlib import Path


EFFECTS = {
    'eichar-lod0': [3, 1, 2, 0],
    'sidekick-lod0': [0, 0, 0, 0, 0, 2, 0],
    'sage-lod0': [0, 0, 2, 3, 0, 1, 0],
    'assistant-lod0': [2, 1, 0, 3],
}

# Reflected cinematic materials store reflection strength in texture alpha.
# Native draw-bones sets ignore-alpha for their primary pass. That property
# follows the cinematic material, independently of gameplay visibility controls.
IGNORE_ALPHA = {
    'eichar-lod0': 0b0111,
    'sidekick-lod0': 0b0111111,
    'sage-lod0': 0b0111101,
    'assistant-lod0': 0b1010,
}


def patch(source, effects, reflection_effects=None, alpha_masks=None):
    text = source
    reflection_effects = reflection_effects or effects
    alpha_masks = alpha_masks or IGNORE_ALPHA

    def replace(old, new, expected=1):
        nonlocal text
        count = text.count(old)
        if count != expected:
            raise ValueError(f'Unsupported source anchor ({count}, expected {expected}): {old[:90]}')
        text = text.replace(old, new)

    def predicate(draw):
        return '(or ' + ' '.join(f'(string= (-> {draw} mgeo name) "{name}")' for name in effects) + ')'

    count = '(cond ' + ' '.join(f'((string= (-> dc mgeo name) "{name}") {len(mapping)})'
                              for name, mapping in effects.items()) + ' (else 0))'
    replace('(qwc-total 0))', f'(qwc-total 0) (cinematic-effect-count {count}))')
    matrices = '''          (when (nonzero? cinematic-effect-count)
            (dotimes (mat-idx (+ (-> dc mgeo num-joints) 3))
              (when (and (< mat-idx 128) (zero? (-> transferred-matrices mat-idx)))
                (set! (-> transferred-matrices mat-idx) 1)
                (set! (-> matrix-slot-string matrix-out-idx) mat-idx)
                (+! matrix-out-idx 1)
                (set! (-> (the (pointer pointer) dma-buf)) (&+ matrix-buf (* 128 mat-idx)))
                (&+! dma-buf 16)
                (+! qwc-total 1))))
'''
    replace('          ;; end matrix string\n', matrices + '          ;; end matrix string\n')
    masks = ''
    for name, mapping in effects.items():
        masks += f'''          (when (string= (-> dc mgeo name) "{name}")
            (let ((original-enable enable-mask) (original-alpha ignore-alpha-mask))
              (set! enable-mask 0)
              (set! ignore-alpha-mask 0)
'''
        for destination, native in enumerate(mapping):
            masks += f'              (when (logtest? original-enable {1 << native}) (logior! enable-mask {1 << destination}))\n'
            masks += f'              (when (logtest? original-alpha {1 << native}) (logior! ignore-alpha-mask {1 << destination}))\n'
        authored_alpha = alpha_masks.get(name, 0)
        if authored_alpha >= 1 << len(mapping):
            raise ValueError(f'Authored alpha mask exceeds effect count for {name}')
        if authored_alpha:
            masks += f'              (logior! ignore-alpha-mask {authored_alpha})\n'
        masks += '            ))\n'
    replace('          ;; flags\n', masks + '          ;; flags\n')
    effect_count = '(if (nonzero? cinematic-effect-count) cinematic-effect-count (-> merc-ctrl header effect-count))'
    replace('(set! (-> flags effect-count) (-> merc-ctrl header effect-count))',
            f'(set! (-> flags effect-count) {effect_count})')
    for kind, original in (
        ('fade', '''            (dotimes (i (-> merc-ctrl header effect-count))
              (set! (-> fades i) (the-as uint (-> *merc-bucket-info* effect i color-fade))))'''),
        ('ptr', '''            (dotimes (i (-> merc-ctrl header effect-count))
              (set! (-> merc-ptrs i) (-> merc-ctrl effect i)))'''),
    ):
        replacement = '            (cond\n'
        for name, mapping in effects.items():
            replacement += f'              ((string= (-> dc mgeo name) "{name}")\n'
            controls = reflection_effects[name] if kind == 'fade' else mapping
            if len(controls) != len(mapping):
                raise ValueError(f'Reflection group count differs for {name}')
            for destination, native in enumerate(controls):
                if kind == 'fade':
                    replacement += f'                (set! (-> fades {destination}) (the-as uint (-> *merc-bucket-info* effect {native} color-fade)))\n'
                else:
                    replacement += f'                (set! (-> merc-ptrs {destination}) (-> merc-ctrl effect {native}))\n'
            replacement += '              )\n'
        replacement += '              (else\n' + original + '              ))'
        replace(original, replacement)
    replace('(let ((num-fades (/ (+ (-> merc-ctrl header effect-count) 3) 4)))',
            f'(let ((num-fades (/ (+ {effect_count} 3) 4)))', expected=2)
    replace('              ;; loop over effects, and set them up/pick renderers.\n',
            '              (dotimes (i 40) (set! (-> blerc-weights i) 0.0))\n'
            '              ;; loop over effects, and set them up/pick renderers.\n')
    replace('                          (if *use-fp-blerc*\n',
            f'                          (if (or *use-fp-blerc* {predicate("arg0")})\n')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--effects', type=Path)
    parser.add_argument('--materials', type=Path, help='Provenance-checked report from build_materials.py')
    args = parser.parse_args()
    effects = json.loads(args.effects.read_text()) if args.effects else EFFECTS
    material_path = args.materials or args.project.parent / 'material-report.json'
    material_report = json.loads(material_path.read_text()) if material_path.exists() else []
    material_rows = {row['model']: row for row in material_report}
    reflection_effects = {}
    alpha_masks = {}
    for name, mapping in effects.items():
        if not mapping or len(mapping) > 64 or any(not isinstance(i, int) or i < 0 or i > 3 for i in mapping):
            raise ValueError(f'Invalid native effect mapping for {name}')
        row = material_rows.get(name, {})
        if row and row['nativeEffectMap'] != mapping:
            raise ValueError(f'Runtime mapping differs from rebuilt material groups for {name}')
        reflection_effects[name] = row.get('nativeReflectionEffectMap', mapping)
        alpha_masks[name] = row.get('authoredIgnoreAlphaMask', IGNORE_ALPHA.get(name, 0))
        if len(reflection_effects[name]) != len(mapping) or any(i < 0 or i > 3 for i in reflection_effects[name]):
            raise ValueError(f'Invalid reflection effect mapping for {name}')
    relative = Path('goal_src/jak1/engine/gfx/foreground/bones.gc')
    original = (args.data / relative).read_text()
    output = patch(original, effects, reflection_effects, alpha_masks)
    (args.project / relative).write_text(output)
    (args.project.parent / 'bones-cinematic.gc').write_text(output)
    (args.project.parent / 'runtime-effect-map.json').write_text(json.dumps(effects, indent=2) + '\n')
    (args.project.parent / 'runtime-alpha-map.json').write_text(json.dumps(alpha_masks, indent=2) + '\n')
    (args.project.parent / 'runtime-reflection-map.json').write_text(json.dumps(reflection_effects, indent=2) + '\n')
    print('Generated isolated Jak 1 cinematic draw adapter')


if __name__ == '__main__':
    main()
