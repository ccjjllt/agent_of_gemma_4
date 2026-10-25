"""F2 f2-push5: derive keep-list from triple review (0 KEEPs, honest empty)."""
import json
from pathlib import Path

OUT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/competition/vfarl/acquisition/f2-new/f2-push5')


def save(p, o):
    p.write_text(json.dumps(o, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    review = json.loads((OUT / 'push5-review-set.json').read_text())
    assert review['count'] == 75, review['count']
    # triple-review-log.md: 75 verdicts, 0 KEEP / 0 COLLIDE / S 24 / R 51.
    keeps = []
    families = {}
    save(OUT / 'keep-list.json', keeps)
    save(OUT / 'keep-families.json', families)
    print(f'reviewed=75 keeps=0 families=0 (no triple-KEEP; nothing to prepare/replay)')
