"""Restore cinematic effect boundaries and PS2 draw modes in the private Jak 1 build."""
import copy,hashlib,io,json,os,struct,sys
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from jak1_paths import DATA,WORK,RELEASE
os.environ['CINEMATIC_BUILD_WORK']=str(WORK)
sys.path[:0]=[str(RELEASE/'jak1'),str(RELEASE/'stages')]
from native_io import activate
activate()
from fr3_models import DRAW,compress,decompress,model,serialize
from dax_effects import restore
from inspect_transformations import textures

# Native gameplay visibility and reflection controls have different meanings.
# Cinematic accessories that belong to the gameplay body still need the native
# metal reflection control rather than the body's intentionally zero fade.
METAL_NATIVE={
 ('eichar-lod0',2):2,
 ('sidekick-lod0',1):3,
 ('sidekick-lod0',2):3,
 ('sage-lod0',0):3,
 ('sage-lod0',3):3,
}
METAL_TEXTURES={
 ('eichar-lod0',2):{'jak-orig-armor','jak-orig-clips','jak-orig-goggles','jak-orig-lenscover'},
 ('sidekick-lod0',1):{'daxterlense'},
 ('sidekick-lod0',2):{'daxtergoggles','daxterbolt'},
 ('sage-lod0',0):{'samos-metal'},
 ('sage-lod0',3):{'samos-metal'},
}

def restore_keira_manifest(raw,cinematic,baseline,glb_path,manifest_path):
    """Restore mixed-source draw groups without assuming donor primitive indices."""
    name='assistant-lod0';target=model(raw,name);donor=model(cinematic,'keira-highres-lod0');native=model(baseline,name)
    manifest=json.loads(manifest_path.read_text());rows=manifest['primitives']
    glb=glb_path.read_bytes();length=struct.unpack_from('<I',glb,12)[0]
    doc=json.loads(glb[20:20+length]);binary=glb[28+length:];primitives=doc['meshes'][0]['primitives']
    assert len(primitives)==len(rows)==29,('Unsupported Keira footwear manifest',len(rows))
    assert [r['outputPrimitiveIndex'] for r in rows]==list(range(len(rows)))
    draws=sorted([d for e in target['effects'] for d in e['draws']],key=lambda d:DRAW.unpack(d)[3])
    assert len(draws)==len(rows),'Import the manifest GLB before restoring its materials'
    texture_list,_=textures(raw)
    groups={};native_draws=0
    for row,primitive,draw in zip(rows,primitives,draws):
        origin=row['origin'];assert origin in ('native','donor')
        source_model=native if origin=='native' else donor
        source_effect=source_model['effects'][row['sourceEffectIndex']]
        source_draw=DRAW.unpack(source_effect['draws'][row['sourceDrawIndex']])
        assert source_draw[0]==row['sourceDrawMode'] and source_draw[1]==row['sourceTextureIndex']
        assert bool(source_effect['tail'][8])==row['sourceHasEnvmap']
        assert row['triangles']==doc['accessors'][primitive['indices']]['count']//3==DRAW.unpack(draw)[5]
        assert len(row['sourceTriangleIndices'])==row['triangles']
        # Native sandals/calves belong to the same non-reflective gameplay body
        # control as the retained cinematic body, so no new runtime group is needed.
        if origin=='native':
            assert row['sourceEffectIndex']==row['nativeEffectIndex']==0 and not row['sourceHasEnvmap']
            group=3;native_draws+=1
        else:group=row['sourceEffectIndex']
        assert group in (1,2,3,4)
        authored=donor['effects'][group]
        assert bool(authored['tail'][8])==row['sourceHasEnvmap']
        if group not in groups:
            template=next(e for e in target['effects'] if bool(e['tail'][8])==bool(authored['tail'][8]))
            result=copy.deepcopy(template);result['draws']=[]
            reflection=0
            if authored['tail'][8]:
                material=doc['materials'][primitive['material']]
                ti=material['extensions']['KHR_materials_specular']['specularColorTexture']['index']
                image=doc['images'][doc['textures'][ti]['source']];view=doc['bufferViews'][image['bufferView']]
                start=view.get('byteOffset',0);decoded=Image.open(io.BytesIO(binary[start:start+view['byteLength']])).convert('RGBA')
                pixels=bytearray(decoded.tobytes());pixels[3::4]=bytes(a>>1 for a in pixels[3::4])
                matches=[i for i,t in enumerate(texture_list) if t['name']==image['name'] and t['page']=='custom-level'
                         and (t['width'],t['height'])==decoded.size and t['pixels']==pixels]
                assert matches,('Missing Keira reflection image',image['name'])
                reflection=matches[0]
            result['tail']=authored['tail'][:4]+struct.pack('<I',reflection)+authored['tail'][8:9]+b'\0'
            groups[group]=result
        values=list(DRAW.unpack(draw));values[0]=source_draw[0];groups[group]['draws'].append(DRAW.pack(*values))
    retained=sorted(groups);target['effects']=[groups[i] for i in retained]
    target['tail']=target['tail'][:8]+native['tail'][8:]
    fixed=raw[:target['start']]+serialize(target)+raw[target['end']:]
    counts=[sum(DRAW.unpack(d)[5] for d in e['draws']) for e in target['effects']]
    assert sum(counts)==manifest['totalTriangles']==10022 and native_draws==5
    return fixed,retained,counts,dict(nativeFootwearDraws=native_draws,triangles=sum(counts),
        provenanceManifestSha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        nativeFootwearMaterialsAndGameplayControlsPreserved=True)

def restore_gameplay_metal(raw,baseline,name,target,retained,mapping):
    native=model(baseline,name)
    native_textures,_=textures(baseline)
    target_textures,_=textures(raw)
    reflection_mapping=list(mapping);changes=[]
    for dst,donor_effect in enumerate(retained):
        source=METAL_NATIVE.get((name,donor_effect))
        if source is None:continue
        effect=target['effects'][dst];native_effect=native['effects'][source]
        affected={target_textures[DRAW.unpack(d)[1]]['name'] for d in effect['draws']}
        assert affected==METAL_TEXTURES[(name,donor_effect)],('Metal surface provenance changed',name,donor_effect,affected)
        assert effect['tail'][8] and native_effect['tail'][8],(name,donor_effect,source)
        native_mode,native_texture=struct.unpack_from('<II',native_effect['tail'])
        old_mode,old_texture=struct.unpack_from('<II',effect['tail'])
        original=target_textures[old_texture];wanted=native_textures[native_texture]
        matches=[i for i,t in enumerate(target_textures) if
                 (t['combo'],t['name'],t['page'],t['width'],t['height'],t['pixels'])==
                 (wanted['combo'],wanted['name'],wanted['page'],wanted['width'],wanted['height'],wanted['pixels'])]
        assert matches,('Native metal environment texture provenance changed',name,source)
        # Jak1 extraction can retain several identical global texture aliases.
        selected=native_texture if native_texture in matches else matches[0]
        effect['tail']=struct.pack('<II',native_mode,selected)+effect['tail'][8:]
        reflection_mapping[dst]=source
        changes.append(dict(effect=dst,donorEffect=donor_effect,nativeReflectionEffect=source,
           previousReflectionEffect=mapping[dst],previousEnvironmentTexture=original['name'],
           environmentTexture=wanted['name'],environmentPage=wanted['page'],
           previousEnvironmentMode=old_mode,environmentMode=native_mode,
           environmentTextureSha256=hashlib.sha256(wanted['pixels']).hexdigest(),
           nativeTextureProvenanceVerified=True,identicalNativeTextureAliases=len(matches),materials=sorted(affected)))
    return reflection_mapping,changes

JOBS={
 'GAME': [('eichar-lod0','jakone-highres-lod0','introcst',[3,1,2,0]),
          ('sidekick-lod0','daxter-highres-lod0','hiphog',[0,0,0,0,0,2,0])],
 'village1':[('sage-lod0','samos-highres-lod0','lintcstb',[0,0,2,3,0,1,0]),
             ('assistant-lod0','keira-highres-lod0','lintcstb',[0,2,1,0,3,2])]}

def main():
    report=[]
    for archive,jobs in JOBS.items():
        raw=decompress(WORK/f'{archive}-import.fr3')
        baseline=decompress(WORK/f'{archive}-original.fr3')
        for name,source,donor,mapping in jobs:
            cinematic=decompress(DATA/f'out/jak2/fr3/{donor}.fr3');original=model(cinematic,source)
            manifest_path=WORK/f'rig-prototypes/{name}.provenance.json'
            if name=='assistant-lod0' and manifest_path.exists():
                raw,retained,counts,footwear=restore_keira_manifest(raw,cinematic,baseline,
                    WORK/f'rig-prototypes/{name}.glb',manifest_path)
                target=model(raw,name);mapping=[mapping[i] for i in retained]
                reflection_mapping,metal_changes=restore_gameplay_metal(raw,baseline,name,target,retained,mapping)
                raw=raw[:target['start']]+serialize(target)+raw[target['end']:]
                report.append({'model':name,'effects':len(counts),'triangles':sum(counts),'nativeEffectMap':mapping,
                    'nativeReflectionEffectMap':reflection_mapping,
                    'authoredIgnoreAlphaMask':sum(1<<i for i,e in enumerate(target['effects']) if e['tail'][8]),
                    'metalEnvironmentOverrides':metal_changes,'footwear':footwear})
                continue
            omitted=(0,1,27) if name=='assistant-lod0' else ()
            pi=0;counts=[]
            for effect in original['effects']:
                kept=[]
                for draw in effect['draws']:
                    if pi not in omitted:kept.append(draw)
                    pi+=1
                counts.append(sum(DRAW.unpack(d)[5] for d in kept))
            raw=restore(raw,cinematic,(WORK/f'rig-prototypes/{name}.glb').read_bytes(),
                        name=name,donor_name=source,expected_triangles=counts,omitted_primitive_indices=omitted)
            target=model(raw,name)
            target['tail']=target['tail'][:8]+model(baseline,name)['tail'][8:]
            pi=0
            for new,old in zip(target['effects'],original['effects']):
                kept=[]
                for draw in old['draws']:
                    if pi not in omitted:kept.append(draw)
                    pi+=1
                for i,(nd,od) in enumerate(zip(new['draws'],kept)):
                    values=list(DRAW.unpack(nd));values[0]=DRAW.unpack(od)[0];new['draws'][i]=DRAW.pack(*values)
            # Merc2 identifies imported models through the first group's first
            # draw, so omitted accessories must not leave empty effect records.
            retained=[i for i,e in enumerate(target['effects']) if e['draws']]
            target['effects']=[target['effects'][i] for i in retained]
            mapping=[mapping[i] for i in retained]
            counts=[counts[i] for i in retained]
            reflection_mapping,metal_changes=restore_gameplay_metal(raw,baseline,name,target,retained,mapping)
            raw=raw[:target['start']]+serialize(target)+raw[target['end']:]
            report.append({'model':name,'effects':len(counts),'triangles':sum(counts),'nativeEffectMap':mapping,
                           'nativeReflectionEffectMap':reflection_mapping,
                           'authoredIgnoreAlphaMask':sum(1<<i for i,e in enumerate(target['effects']) if e['tail'][8]),
                           'metalEnvironmentOverrides':metal_changes})
        compress(WORK/f'{archive}-materials.fr3',raw)
    (WORK/'material-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
