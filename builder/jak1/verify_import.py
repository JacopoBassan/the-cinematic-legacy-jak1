"""Verify imported geometry, disambiguating model names from texture-page names."""
import json,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from fr3_models import Reader,DRAW,decompress
from jak1_paths import WORK

def read_model(raw,name):
    needle=struct.pack('<Q',len(name))+name.encode();pos=0;found=[]
    while (pos:=raw.find(needle,pos))>=0:
        start=pos;pos+=1
        try:
            r=Reader(raw,start+len(needle));count=r.num('Q')
            if not 0<count<100:continue
            effects=[r.effect() for _ in range(count)];r.take(20)
            draws=[DRAW.unpack(d) for e in effects for d in e['draws']]
            if not draws or any(d[4]>10000000 or d[5]>1000000 for d in draws):continue
            found.append({'name':name,'start':start,'end':r.pos,'effects':effects})
        except (AssertionError,struct.error,OverflowError):continue
    assert len(found)==1,(name,len(found))
    return found[0]

def main():
    rows=[]
    for file,name,expected in [('GAME.fr3','eichar-lod0',9886),('GAME.fr3','sidekick-lod0',9510),
                              ('village1.fr3','sage-lod0',10434),('village1.fr3','assistant-lod0',10022)]:
        record=read_model(decompress(WORK/'import/out/jak1/fr3'/file),name)
        count=sum(DRAW.unpack(d)[5] for e in record['effects'] for d in e['draws'])
        assert count==expected,(name,count,expected)
        rows.append({'model':name,'triangles':count,'importVerified':True})
    report={'models':rows,'runtimeValidated':False}
    (WORK/'import-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
