"""Fit native Jak 1 channels in each donor's authored shape basis.

Channel numbers are not assumed to correspond between games. Mouth centerline
landmarks distinguish the upper and lower surfaces of the closed donor mouths.
Eye geometry and live gaze are integrated separately by build_faces.py.
"""
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from jak1_paths import DATA as DATA_ROOT,WORK,RELEASE
OUT=WORK/'face-candidates'
DATA=DATA_ROOT/'decompiler_out/jak1/levels'
sys.path.insert(0,str(RELEASE))
from retarget_glb import load

# Donor indices refer to compact rig-prototypes, while native indices refer to
# the original Jak 1 accessors. Coordinates are checked to fail safely if the
# extraction layout changes. Match deltas, preserving the donor neutral face.
JOBS=[
 dict(name='eichar-lod0',folder='common',parts={8:[0],9:[2],10:[0]},
      landmarks=[
        ('upper-lip',0,5445,8,626,[0,2.3109,.3330],[0,2.3170,.3334]),
        ('lower-lip',0,5409,8,730,[0,2.2973,.3330],[0,2.3051,.3266]),
        ('upper-teeth',2,5682,9,55,[0,2.3415,.3229],[0,2.3477,.3096]),
        ('lower-teeth',2,5677,9,8,[0,2.2837,.3127],[0,2.2915,.2943])],
      eyes={2:[8],3:[9]}),
 dict(name='sidekick-lod0',folder='common',
      parts={4:[2,10],5:[0,1,2],6:[1,2,4,5,6,10],15:[5,6]},
      landmarks=[
        ('upper-lip',2,8737,6,413,[0,1.0489,.2047],[0,1.0486,.2069]),
        ('lower-lip',1,8688,6,533,[0,1.0215,.1754],[0,1.0297,.1651]),
        ('inner-lower-lip',4,8837,6,1094,[0,1.0416,.1699],[0,1.0436,.1612]),
        ('upper-teeth',6,8863,15,16,[0,1.0434,.1773],[0,1.0595,.1502]),
        ('lower-teeth',6,8876,15,63,[0,1.0270,.1699],[0,1.0217,.1433])],
      eyes={13:[11],14:[12]}),
 dict(name='sage-lod0',folder='village1',parts={6:[2,3],10:[1,2],11:[4]},
      landmarks=[
        ('inner-upper-lip',1,4119,10,601,[0,1.5204,.2180],[0,1.5459,.2166]),
        ('lower-lip-top',1,4164,10,36,[0,1.5349,.2936],[0,1.5459,.2937]),
        ('lower-lip',1,4171,10,694,[0,1.5030,.2994],[0,1.5095,.3002]),
        ('upper-teeth',4,4296,11,2,[0,1.5320,.2355],[0,1.5738,.2230]),
        ('lower-teeth',4,4328,11,42,[0,1.5030,.2297],[0,1.4988,.2144])],
      eyes={7:[23],8:[24]}),
 # The corrected Keira prototype omits donor mask hardware primitives 0, 1,
 # and 27 because their independent animation does not exist in Jak 1.
 dict(name='assistant-lod0',folder='village1',parts={7:[0],8:[1]},
      landmarks=[
        ('upper-lip',0,85,7,145,[0,2.2705,.3493],[0,2.2665,.3555]),
        ('lower-lip',0,30,7,665,[0,2.2490,.3305],[0,2.2441,.3351]),
        ('upper-teeth',1,543,8,43,[0,2.2705,.3251],[0,2.2675,.3351]),
        ('lower-teeth',1,532,8,55,[0,2.2517,.3224],[0,2.2410,.3290])],
      eyes={5:[14],6:[15]})]


def validate_landmarks(low,high,job):
    ld,_,la=load(low);hd,_,ha=load(high)
    lp=ld['meshes'][0]['primitives'];hp=hd['meshes'][0]['primitives']
    landmarks=[];metadata=[]
    for label,li,lv,hi,hv,expected_low,expected_high in job['landmarks']:
        native=la(lp[li]['attributes']['POSITION'])[lv]
        donor=ha(hp[hi]['attributes']['POSITION'])[hv]
        assert lv in la(lp[li]['indices']) and hv in ha(hp[hi]['indices'])
        assert np.allclose(native,expected_low,atol=7e-5),f'Unsupported native {job["name"]} {label}'
        assert np.allclose(donor,expected_high,atol=7e-5),f'Unsupported donor {job["name"]} {label}'
        landmarks.append((li,lv,hi,hv))
        metadata.append(dict(label=label,nativePart=li,nativeIndex=lv,cinematicPart=hi,
                             cinematicIndex=hv,nativePosition=native.tolist(),cinematicPosition=donor.tolist()))
    return landmarks,metadata


def fit(low_path,high_path,parts,landmarks):
    """Choose the strongest useful mouth constraint within a surface-error guard.

    A fixed, large landmark weight can invent an expression that matches a few
    lip points while the cheeks distort. Bound the complete fitted surface error
    to 15 percent above the unconstrained optimum, and cap total displacement.
    """
    from fix_faces import closest_surface
    ld,_,la=load(low_path);hd,_,ha=load(high_path)
    lp=ld['meshes'][0]['primitives'];hp=hd['meshes'][0]['primitives']
    native_count=len(lp[0]['targets']);donor_count=len(hp[0]['targets'])
    matrices=[];targets=[];correspondence=[]
    for hi,source_parts in parts.items():
        p=hp[hi];used=np.unique(ha(p['indices']))
        position=ha(p['attributes']['POSITION'])[used]
        basis=np.array([ha(t['POSITION'])[used] for t in p['targets']])
        moving=np.linalg.norm(basis,axis=2).max(0)>1e-6
        triangles=[];deltas=[]
        for li in source_parts:
            q=lp[li];ids=la(q['indices']).reshape(-1,3)
            triangles.append(la(q['attributes']['POSITION'])[ids])
            deltas.append(np.array([la(t['POSITION'])[ids] for t in q['targets']]))
        triangle,bary,distance=closest_surface(position,np.concatenate(triangles))
        desired=np.einsum('tnij,ni->tnj',np.concatenate(deltas,axis=1)[:,triangle],bary)
        matrices.append(basis[:,moving].reshape(donor_count,-1).T)
        targets.append(desired[:,moving].reshape(native_count,-1).T)
        correspondence.append(dict(cinematicPart=hi,gameplayParts=source_parts,
                                   fittedVertices=int(moving.sum()),maxDistance=float(distance[moving].max())))
    A=np.concatenate(matrices);B=np.concatenate(targets)
    LA=np.concatenate([np.array([ha(t['POSITION'])[hv] for t in hp[hi]['targets']]).T
                       for li,lv,hi,hv in landmarks])
    LB=np.concatenate([np.array([la(t['POSITION'])[lv] for t in lp[li]['targets']]).T
                       for li,lv,hi,hv in landmarks])
    regularizer=np.eye(donor_count)*.025
    coefficients=[];errors=[]
    for channel in range(native_count):
        desired=B[:,channel];magnitude=max(np.linalg.norm(desired),1e-12)
        initial=lsq_linear(np.vstack([A,regularizer]),np.r_[desired,np.zeros(donor_count)],
                           bounds=(0,1.5),tol=1e-9)
        assert initial.success
        initial_error=np.linalg.norm(A@initial.x-desired)
        native_max=np.linalg.norm(desired.reshape(-1,3),axis=1).max()
        chosen=initial.x;weight=0;best=np.linalg.norm(LA@initial.x-LB[:,channel])
        for trial_weight in [1,2,4,8,16,32]:
            result=lsq_linear(np.vstack([A,regularizer,LA*trial_weight]),
                              np.r_[desired,np.zeros(donor_count),LB[:,channel]*trial_weight],
                              bounds=(0,1.5),tol=1e-9)
            assert result.success
            actual=A@result.x;mouth_error=np.linalg.norm(LA@result.x-LB[:,channel])
            surface_ok=np.linalg.norm(actual-desired)<=initial_error*1.15+1e-9
            displacement_ok=np.linalg.norm(actual.reshape(-1,3),axis=1).max()<=max(native_max*1.25,1e-8)
            if surface_ok and displacement_ok and mouth_error<best:
                chosen=result.x;weight=trial_weight;best=mouth_error
        actual=A@chosen;coefficients.append(chosen)
        errors.append(dict(channel=channel,landmarkWeight=weight,
                           mouthErrorBefore=float(np.linalg.norm(LA@initial.x-LB[:,channel])),
                           mouthErrorAfter=float(np.linalg.norm(LA@chosen-LB[:,channel])),
                           relativeSurfaceFitErrorBefore=float(initial_error/magnitude),
                           relativeSurfaceFitError=float(np.linalg.norm(actual-desired)/magnitude),
                           nativeMaxDisplacement=float(native_max),
                           candidateMaxDisplacement=float(np.linalg.norm(actual.reshape(-1,3),axis=1).max())))
    coefficients=np.array(coefficients)
    assert np.isfinite(coefficients).all()
    return coefficients,dict(gameplayChannels=native_count,cinematicShapes=donor_count,
                             facialChannels=list(range(native_count)),correspondence=correspondence,
                             errors=errors,mouthLandmarks=landmarks,
                             surfaceErrorLimitRatio=1.15,
                             status='Numerical candidate; expression and movement validation required')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--model',action='append',choices=[job['name'] for job in JOBS],
                        help='Rebuild only selected models and retain other existing report entries')
    args=parser.parse_args();output=args.output
    output.mkdir(parents=True,exist_ok=True);os.environ['CINEMATIC_BUILD_WORK']=str(output)
    sys.path.insert(0,str(RELEASE/'stages'))
    report_path=output/'fit-report.json'
    report=json.loads(report_path.read_text()) if args.model and report_path.exists() else {}
    for job in JOBS:
        native=job['name'];folder=job['folder']
        if args.model and native not in args.model:continue
        low=DATA/f'{folder}/{native}.glb';high=OUT.parent/f'rig-prototypes/{native}.glb'
        doc,_,_=load(low);count=len(doc['meshes'][0]['primitives'][0]['targets'])
        landmarks,metadata=validate_landmarks(low,high,job)
        coefficients,detail=fit(low,high,job['parts'],landmarks)
        np.save(output/f'{native}-coefficients.npy',coefficients)
        detail['mouthLandmarkDetails']=metadata
        detail['eyeGeometryTransfer']=job['eyes']
        detail['nativeGLBSha256']=hashlib.sha256(low.read_bytes()).hexdigest()
        detail['prototypeGLBSha256']=hashlib.sha256(high.read_bytes()).hexdigest()
        detail['limitations']=['Different neutral face designs prevent identical deformation everywhere',
          'Native eyelid geometry must be transferred separately; gaze uses the native compositor',
          'Native asymmetric expressions must be retained; in-game expression and motion checks remain']
        report[native]=detail
        print(native,'channels',count,'cinematic shapes',coefficients.shape[1],
              'mouth error',round(sum(x['mouthErrorBefore'] for x in detail['errors']),5),'to',
              round(sum(x['mouthErrorAfter'] for x in detail['errors']),5),flush=True)
    report_path.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
