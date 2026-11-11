"""Restore authored alpha and isolate custom texture names from global animation names."""
import json,os,struct,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from jak1_paths import DATA,WORK,RELEASE
os.environ['CINEMATIC_BUILD_WORK']=str(WORK)
sys.path.insert(0,str(RELEASE/'stages'))
from fr3_models import Reader,decompress,compress
from inspect_transformations import textures

def main():
    donors=[];report={}
    for name in ('introcst','hiphog','lintcstb'):
        donors+=textures(decompress(DATA/f'out/jak2/fr3/{name}.fr3'))[0]
    # Mixed-source Keira feet use the user's original Jak1 skin/sandal images.
    for archive in ('GAME','village1'):
        donors+=textures(decompress(WORK/f'{archive}-original.fr3'))[0]
    for archive in ('GAME','village1'):
        raw=decompress(WORK/f'{archive}-faces.fr3');target,end=textures(raw);restored=[]
        for t in target:
            if t['page']!='custom-level':continue
            matches=[s for s in donors if (s['name'],s['width'],s['height'])==(t['name'],t['width'],t['height'])
                      and np.array_equal(np.frombuffer(s['pixels'],'u1').reshape(-1,4)[:,:3],np.frombuffer(t['pixels'],'u1').reshape(-1,4)[:,:3])]
            variants={bytes(s['pixels']) for s in matches}
            if len(variants)==1:t['pixels']=next(iter(variants))
            elif t['name']!='white':raise ValueError(('Missing or ambiguous donor texture',t['name']))
            restored.append(t['name']);t['name']='j1-cinematic/'+t['name']
        r=Reader(raw,2);r.take(r.num('Q'));out=bytearray(raw[:r.pos])+struct.pack('<Q',len(target))
        for t in target:
            out+=struct.pack('<HHI',t['width'],t['height'],t['combo'])+struct.pack('<Q',len(t['pixels'])//4)+t['pixels']
            for key in ('name','page'):
                encoded=t[key].encode();out+=struct.pack('<Q',len(encoded))+encoded
            out+=t['flag']
        out+=raw[end:];compress(WORK/f'{archive}-cinematic.fr3',bytes(out))
        assert textures(bytes(out))[0]==target
        report[archive]={'restoredTextures':restored,'nativeGlobalNamesPreserved':True}
    (WORK/'texture-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:len(v['restoredTextures']) for k,v in report.items()})

if __name__=='__main__':main()
