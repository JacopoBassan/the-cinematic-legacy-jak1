"""Attach native Jak 1 expression channels, deforming eyelids and live eye textures."""
import json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from jak1_paths import DATA,WORK,RELEASE
os.environ['CINEMATIC_BUILD_WORK']=str(WORK)
sys.path[:0]=[str(RELEASE/'jak1'),str(RELEASE/'stages')]
from native_io import activate
activate()
from fr3_models import decompress,compress,serialize,model,DRAW
from inspect_faces import glb,buffers
from fix_faces import closest_surface,fix_model,vec

def eye_parts(doc):
    result={}
    for pi,p in enumerate(doc['meshes'][0]['primitives']):
        mat=doc['materials'][p['material']]
        texture=mat.get('pbrMetallicRoughness',{}).get('baseColorTexture',{}).get('index')
        if texture is None:continue
        name=doc['images'][doc['textures'][texture]['source']]['name']
        if name in ('programmer_eye_left','programmer_eye_right'):result[pi]=int(name.endswith('right'))
    assert sorted(result.values())==[0,1]
    return result

def transfer(p,ha,sources,la):
    triangles=[];position=[];normal=[]
    for source in sources:
        ids=la(source['indices']).reshape(-1,3)
        triangles.append(la(source['attributes']['POSITION'])[ids])
        position.append(np.array([la(t['POSITION'])[ids] for t in source['targets']]))
        normal.append(np.array([la(t['NORMAL'])[ids] for t in source['targets']]))
    ti,bary,_=closest_surface(ha(p['attributes']['POSITION']),np.concatenate(triangles))
    return (np.einsum('tnij,ni->tnj',np.concatenate(position,axis=1)[:,ti],bary),
            np.einsum('tnij,ni->tnj',np.concatenate(normal,axis=1)[:,ti],bary))

def main():
    report=[]
    for archive,names in [('GAME',[('eichar-lod0','common'),('sidekick-lod0','common')]),
                          ('village1',[('sage-lod0','village1'),('assistant-lod0','village1')])]:
        raw=decompress(WORK/f'{archive}-materials.fr3');baseline=decompress(WORK/f'{archive}-original.fr3')
        start,indices,_,original_vertices=buffers(raw,names[0][0]);vertices=bytearray(original_vertices);new_indices=indices;updates=[]
        for name,folder in names:
            ld,la=glb(DATA/f'decompiler_out/jak1/levels/{folder}/{name}.glb')
            hd,ha=glb(WORK/f'rig-prototypes/{name}.glb')
            lp,hp=ld['meshes'][0]['primitives'],hd['meshes'][0]['primitives']
            count=len(lp[0]['targets']);coefficient=np.load(WORK/f'face-candidates/{name}-coefficients.npy')
            assert coefficient.shape==(count,len(hp[0]['targets']))
            deformations={}
            for pi,p in enumerate(hp):
                dp=np.einsum('ct,tnj->cnj',coefficient,np.array([ha(t['POSITION']) for t in p['targets']]))
                dn=np.einsum('ct,tnj->cnj',coefficient,np.array([ha(t['NORMAL']) for t in p['targets']]))
                if np.max(np.abs(dp))>1e-7:deformations[pi]=(dp,dn)
            if name=='sidekick-lod0':
                from native_jaw import apply_native_jaw
                jaw_report=apply_native_jaw(lp,la,hp,ha,deformations)
                (WORK/'face-candidates/native-jaw-report.json').write_text(json.dumps(jaw_report,indent=2)+'\n')
            high_eyes=eye_parts(hd);low_eyes={side:pi for pi,side in eye_parts(ld).items()}
            for pi,side in high_eyes.items():
                # Native Jak 1 blink/expression channels also move the eye surface.
                deformations[pi]=transfer(hp[pi],ha,[lp[low_eyes[side]]],la)
            native=model(baseline,name)
            ids=sorted({DRAW.unpack(d)[2] for e in native['effects'] for d in e['draws'] if DRAW.unpack(d)[2]!=255})
            assert len(ids)==2 and ids[0]%2==0 and ids[1]==ids[0]+1,(name,ids)
            eye_uv_report=None
            native_uv_data=(ld,la,low_eyes)
            if name=='eichar-lod0':
                from eye_alignment import calibrated_jak_eye_uvs,write_eye_uvs
                calibrated_uvs,eye_uv_report=calibrated_jak_eye_uvs(lp,la,hp,ha,high_eyes,low_eyes)
                write_eye_uvs(raw,name,hd,ha,new_indices,vertices,calibrated_uvs)
                # The generic native XY projection would overwrite the shared
                # cinematic gaze calibration. Live eye IDs and the native eye
                # compositor remain active through eye_fn and eye_slot below.
                native_uv_data=None
            elif name=='sidekick-lod0':
                from eye_alignment import authored_daxter_eye_uvs,write_eye_uvs
                authored_uvs,eye_uv_report=authored_daxter_eye_uvs(hp,ha,high_eyes)
                write_eye_uvs(raw,name,hd,ha,new_indices,vertices,authored_uvs)
                # Keep this projection in the copied modified eye buffer too.
                # Native blink/expression geometry above remains unchanged.
                native_uv_data=None
            def adapter(*_):return hd,ha,deformations,count,[]
            target,added,detail=fix_model(raw,name,name,{},new_indices,vertices,retarget_fn=adapter,
                 eye_fn=lambda _:high_eyes,native_eye_data=native_uv_data,eye_slot=ids[0]//2)
            target['tail']=target['tail'][:8]+native['tail'][8:]
            new_indices=np.concatenate([new_indices]+added);updates.append(target)
            report.append(detail|{'nativeEyeIds':ids,'nativeEyeGeometryChannelsTransferred':True,
                                  'eyeUVCalibration':eye_uv_report})
        fixed=raw[:start]+vec(new_indices.astype('<u4').tobytes(),4)+vec(vertices,64)+raw[-2:]
        for target in sorted(updates,key=lambda x:x['start'],reverse=True):
            fixed=fixed[:target['start']]+serialize(target)+fixed[target['end']:]
        _,check_indices,_,check_vertices=buffers(fixed,names[0][0]);after=np.frombuffer(check_vertices,'u1').reshape(-1,64)
        before=np.frombuffer(original_vertices,'u1').reshape(-1,64)
        assert np.array_equal(check_indices[:len(indices)],indices)
        assert np.array_equal(before[:,:48],after[:,:48]) and np.array_equal(before[:,56:],after[:,56:])
        compress(WORK/f'{archive}-faces.fr3',fixed)
    (WORK/'native-faces-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
