"""Build four private cross-game rigs from local disc extractions, without assets in source."""
import argparse,hashlib,json,os,struct,sys
from pathlib import Path
import numpy as np
RELEASE=Path(__file__).resolve().parents[1]
ROOT=RELEASE.parent
sys.path.insert(0,str(RELEASE))
from retarget_glb import retarget,load
sys.path.insert(0,str(Path(__file__).resolve().parent))
from rig_validation import verify
from jak1_paths import DATA,WORK

OUT=WORK/'rig-prototypes'
JOBS=[
 ('Jak','common/eichar-lod0','introcst/jakone-highres-lod0',{'extra':'main'}),
 ('Daxter','common/sidekick-lod0','hiphog/daxter-highres-lod0',
  {'tailFirst':'tailBase',**{x:'head' for x in ['tongueBase','tongueMaster','tongueMidA','tongueMidB','tongueTip','uvula']},
   **{s+f+p:s+'middle'+p for s in ['L','R'] for f in ['pinky','ring'] for p in ['A','B']}}),
 ('Samos','village1/sage-lod0','lintcstb/samos-highres-lod0',
  {'beardDriver':'head','Birdfeather':'BIRDhead1','Birdjaw':'BIRDhead1','BIRDR2':'BIRDR1','BIRDL2':'BIRDL1','Birdtail':'BIRD1','RhairBASE':'head','Rhair':'head'}),
 ('Keira','village1/assistant-lod0','lintcstb/keira-highres-lod0',
  {'rEara':'rEar1','rEarb':'rEar2','lEara':'lEar1','lEarb':'lEar2',
   'Rmidhaira':'Rhair1','Rmidhairb':'Rhair2','Lmidhaira':'Lhair1','Lmidhairb':'Lhair2',
   **{s+'bang'+a:'flip'+s+str(i) for s in ['L','R'] for i,a in enumerate('abc',1)},
   'lTopStrap':'lTopStrap1','rTopStrap':'rTopStrap1','lBotStrap':'lBotStrap1','rBotStrap':'rBotStrap1',
   'Lanklestrap':'LfootFlaps','Ranklestrap':'RfootFlaps',
   # Original Keira never weights the trailing pant-flap joints. They can be
   # outside the native merc joint-evaluation range, unlike the knee controls.
   'LpantFlap':'Lknee','RpantFlap':'Rknee',
   # These donor-only joints are removed with their mask surfaces after retargeting.
   'maskstrap':'prejoint','mask':'prejoint'})]

# Jak II Keira has a welding mask that the Jak 1 NPC never uses. Keep her
# animated Jak 1 goggles, rather than leaving donor mask parts at the origin.
KEIRA_OMITTED_PRIMITIVES=(0,1,27)
KEIRA_TORCH_PRIMITIVES=(2,3,4,5)
# Both games have a torch joint, but the accessory's authored bind axes differ.
# Donor nozzle +Z becomes native nozzle -X; torchTIP stays in the native rig.
KEIRA_TORCH_ROTATION=np.asarray([[0,0,-1],[0,1,0],[1,0,0]],dtype='<f4')


def correct_keira_accessories(output):
    """Apply deliberate accessory corrections after verifying donor provenance."""
    doc,binary,acc=load(output)
    binary=bytearray(binary)
    primitives=doc['meshes'][0]['primitives']
    assert len(primitives)==28,'Keira donor primitive layout changed'
    names=[doc['materials'][p['material']]['name'] for p in primitives]
    assert [names[i] for i in KEIRA_OMITTED_PRIMITIVES]==['keira-brownstraps-new','keira-maskbolt','keira-mask']
    assert [names[i] for i in KEIRA_TORCH_PRIMITIVES]==['keira-torch-guard-01','keira-torch-nozzle-01','keira-brownstraps-new','keira-torch-nozzle-02']
    written=set()
    def rotate(ai):
        if ai in written:return
        written.add(ai)
        entry=doc['accessors'][ai];view=doc['bufferViews'][entry['bufferView']]
        assert entry['componentType']==5126 and entry['type']=='VEC3'
        values=np.asarray(acc(ai)@KEIRA_TORCH_ROTATION.T,dtype='<f4')
        start=view.get('byteOffset',0)+entry.get('byteOffset',0)
        assert view.get('byteStride',12)==12
        binary[start:start+values.nbytes]=values.tobytes()
        entry.update(min=values.min(0).tolist(),max=values.max(0).tolist())
    for i in KEIRA_TORCH_PRIMITIVES:
        primitive=primitives[i]
        for attribute in ('POSITION','NORMAL'):
            if attribute in primitive['attributes']:rotate(primitive['attributes'][attribute])
        for target in primitive.get('targets',[]):
            for ai in target.values():rotate(ai)
    removed_triangles=sum(doc['accessors'][primitives[i]['indices']]['count']//3 for i in KEIRA_OMITTED_PRIMITIVES)
    doc['meshes'][0]['primitives']=[p for i,p in enumerate(primitives) if i not in KEIRA_OMITTED_PRIMITIVES]
    encoded=json.dumps(doc,separators=(',',':')).encode();encoded+=b' '*(-len(encoded)%4)
    binary+=b'\0'*(-len(binary)%4)
    output.write_bytes(struct.pack('<III',0x46546c67,2,28+len(encoded)+len(binary))+struct.pack('<II',len(encoded),0x4e4f534a)+encoded+struct.pack('<II',len(binary),0x004e4942)+binary)
    return dict(omittedDonorPrimitiveIndices=list(KEIRA_OMITTED_PRIMITIVES),omittedTriangles=removed_triangles,
                rotatedDonorPrimitiveIndices=list(KEIRA_TORCH_PRIMITIVES),torchRotation=KEIRA_TORCH_ROTATION.tolist())


def verify_keira_corrections(low_path,high_path,output):
    low,_,la=load(low_path);high,_,ha=load(high_path);result,_,ra=load(output)
    assert [result['nodes'][i]['name'] for i in result['skins'][0]['joints']]==[low['nodes'][i]['name'] for i in low['skins'][0]['joints']]
    assert np.array_equal(la(low['skins'][0]['inverseBindMatrices']),ra(result['skins'][0]['inverseBindMatrices']))
    rows=[(i,p) for i,p in enumerate(high['meshes'][0]['primitives']) if i not in KEIRA_OMITTED_PRIMITIVES]
    assert len(rows)==len(result['meshes'][0]['primitives'])
    triangles=vertices=0
    result_names=[result['nodes'][i]['name'] for i in result['skins'][0]['joints']]
    for (i,source),dest in zip(rows,result['meshes'][0]['primitives']):
        used,inverse=np.unique(ha(source['indices']).ravel(),return_inverse=True)
        assert np.array_equal(inverse,ra(dest['indices']).ravel())
        expected=ha(source['attributes']['POSITION'])[used]
        if i in KEIRA_TORCH_PRIMITIVES:expected=expected@KEIRA_TORCH_ROTATION.T
        assert np.array_equal(expected,ra(dest['attributes']['POSITION']))
        assert np.array_equal(ha(source['attributes']['TEXCOORD_0'])[used],ra(dest['attributes']['TEXCOORD_0']))
        assert len(source.get('targets',[]))==len(dest.get('targets',[]))
        for st,dt in zip(source.get('targets',[]),dest.get('targets',[])):
            for attribute in st:
                expected=ha(st[attribute])[used]
                if i in KEIRA_TORCH_PRIMITIVES:expected=expected@KEIRA_TORCH_ROTATION.T
                assert np.array_equal(expected,ra(dt[attribute]))
        weights=ra(dest['attributes']['WEIGHTS_0']);joints=ra(dest['attributes']['JOINTS_0'])
        assert np.isfinite(weights).all() and (weights>=0).all() and np.allclose(weights.sum(1),1,atol=1e-6)
        assert ((weights>0).sum(1)<=3).all() and joints.max()<len(result['skins'][0]['joints'])
        assert not np.any((joints==result_names.index('prejoint')) & (weights>0)),'Uncontrolled donor mask geometry remains'
        triangles+=len(inverse)//3;vertices+=len(used)
    def surface_points(doc,accessor,material_name):
        primitive=next(p for p in doc['meshes'][0]['primitives'] if doc['materials'][p['material']]['name']==material_name)
        used=np.unique(accessor(primitive['indices']).ravel())
        return accessor(primitive['attributes']['POSITION'])[used]
    native_guard=surface_points(low,la,'assis-torch-guard-01-envmap')
    candidate_guard=surface_points(result,ra,'keira-torch-guard-01')
    guard_distances=np.linalg.norm(native_guard[:,None]-candidate_guard[None,:],axis=-1)
    guard_error=float(max(guard_distances.min(0).max(),guard_distances.min(1).max()))
    assert guard_error<0.005,'Corrected torch guard is not aligned with native animation attachment'
    return dict(output=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),drawVertices=vertices,
                triangles=triangles,nativeJoints=len(result['skins'][0]['joints']),
                authoredMorphTargets=max(len(p.get('targets',[])) for p in result['meshes'][0]['primitives']),
                exactGeometryUVTopologyAndMorphs=False,documentedAccessoryCorrectionsVerified=True,
                unchangedSurfacesUVTopologyAndMorphsExact=True,nativeJointOrderAndBindMatrices=True,
                nativeTorchGuardAlignmentMaxMeters=guard_error,uncontrolledRootAccessoryVertices=0,
                geometryProvenance='Exact donor geometry except documented torch bind-axis rotation and omitted donor-only welding mask')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--only',choices=[j[0] for j in JOBS])
    args=parser.parse_args();output=args.output;data=args.data
    output.mkdir(parents=True,exist_ok=True);os.environ['CINEMATIC_BUILD_WORK']=str(output)
    results=[]
    for label,low,high,aliases in JOBS:
        if args.only and label!=args.only:continue
        lp=data/f'decompiler_out/jak1/levels/{low}.glb';hp=data/f'decompiler_out/jak2/levels/{high}.glb'
        dest=output/(Path(low).name+'.glb')
        retarget(lp,hp,dest,donor_path=data/f'out/jak2/fr3/{high.split("/")[0]}.fr3',joint_aliases=aliases,
                 preserve_morph_targets=True)
        native,_,_=load(lp)
        verification=verify(lp,hp,dest)
        corrections={}
        if label=='Jak':
            from native_face_rig import restore_native_face_rig,restore_native_pants_rig
            corrections['nativeFacialRig']=restore_native_face_rig(dest,lp)
            corrections['nativeClothRig']=restore_native_pants_rig(dest,lp)
            verification=verify(lp,hp,dest)
        if label=='Keira':
            corrections=correct_keira_accessories(dest)
            verification=verify_keira_corrections(lp,hp,dest)
            # Jak II's enclosed shoes change Keira's Jak 1 outfit. Exchange
            # complete connected feet and calves underneath the pant hems.
            from keira_feet import restore_open_feet
            feet=restore_open_feet(dest,lp,hp,data/'out/jak1/fr3/village1.fr3',data/'out/jak2/fr3/lintcstb.fr3')
            corrections['openFeet']={k:v for k,v in feet.items() if k not in ('primitives','verification')}
            verification.update(feet['verification'])
            verification.update(sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                                geometryProvenance='Donor upper body with corrected torch axes, omitted welding mask, cinematic cuffs fitted and articulated to native Jak 1 design, and original open feet and calves',
                                primitiveProvenance=str(dest.with_suffix('.provenance.json')))
        results.append(verification|{'character':label,'aliases':aliases,'accessoryCorrections':corrections,
                      'nativeMorphChannels':max(len(p.get('targets',[])) for p in native['meshes'][0]['primitives'])})
    report={'status':'Private rig prototypes; not installed or gameplay-ready','subjects':results,
            'remaining':['Native facial-channel fitting and live eyes','Samos secondary hair and bird motion',
                         'Keira clothing, goggles and torch motion','Gameplay capture and regression checks','EU/USA launcher packaging']}
    (output/'prototype-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
