from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

CLASS_NAMES = ["GRASP", "IDLE", "PICKUP", "PLACE", "TRANSPORT", "RELEASE"]
CLASS_FOLDER_NAMES = [f"{i}_{name}" for i, name in enumerate(CLASS_NAMES)]
DEFAULT_CLIP_ROOT = r"F:\sih\temporal_clips"
DEFAULT_YOLO_MODEL = r"F:\sih\runs\detect\train\weights\best.pt"
DEFAULT_HAND_MODEL = r"F:\sih\models\hand_landmarker.task"
DEFAULT_OUTPUT_DIR = r"F:\sih\temporal_model_1dcnn"
SEQ_LEN = 20  # temporal window length in sampled timesteps
WINDOW_STRIDE = 2  # overlap: next window starts 5 timesteps later
FEATURE_FPS = 20.0
FEATURE_VERSION = 3  # sliding-window temporal samples  # physical-time velocity normalization
YOLO_CONF = 0.25
HAND_DETECTION_CONF = 0.50
HAND_PRESENCE_CONF = 0.50
HAND_TRACKING_CONF = 0.50
BATCH_SIZE = 16
MAX_EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 12
SEED = 42
HAND_LANDMARK_COUNT = 21
HAND_TIP_IDS = [4, 8, 12, 16, 20]


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def point_to_box_distance(px, py, box):
    x1, y1, x2, y2 = box
    return float(math.hypot(max(x1 - px, 0.0, px - x2), max(y1 - py, 0.0, py - y2)))


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def box_diag(box):
    x1, y1, x2, y2 = box
    return max(1.0, math.hypot(x2 - x1, y2 - y1))


@dataclass
class HandState:
    present: float
    landmarks_px: np.ndarray
    wrist_xy_norm: tuple[float, float]
    scale_px: float
    bbox: tuple[float, float, float, float] | None
    wrist_velocity_norm: tuple[float, float]


def _hand_scale(pts_xy):
    return max(1.0, float(np.linalg.norm(pts_xy[9] - pts_xy[0])))


def _hand_bbox(pts_xy):
    return (float(np.min(pts_xy[:, 0])), float(np.min(pts_xy[:, 1])), float(np.max(pts_xy[:, 0])), float(np.max(pts_xy[:, 1])))


def parse_hands(hand_result, frame_w, frame_h, previous_wrists, dt_seconds):
    raw = []
    if hand_result.hand_landmarks:
        for hand_landmarks in hand_result.hand_landmarks:
            pts = np.array([[p.x * frame_w, p.y * frame_h, p.z * frame_w] for p in hand_landmarks], dtype=np.float32)
            scale = _hand_scale(pts[:, :2])
            wrist_norm = (float(pts[0, 0] / max(1, frame_w)), float(pts[0, 1] / max(1, frame_h)))
            raw.append((pts, _hand_bbox(pts[:, :2]), scale, wrist_norm))
    raw.sort(key=lambda item: item[0][0, 0])
    hands = []
    for slot in range(2):
        if slot < len(raw):
            pts, bbox, scale, wrist_norm = raw[slot]
            prev = previous_wrists[slot]
            vel = (0.0, 0.0) if prev is None or dt_seconds <= 0 else ((wrist_norm[0] - prev[0]) / dt_seconds, (wrist_norm[1] - prev[1]) / dt_seconds)
            hands.append(HandState(1.0, pts, wrist_norm, scale, bbox, vel))
        else:
            hands.append(HandState(0.0, np.zeros((21, 3), np.float32), (0.0, 0.0), 1.0, None, (0.0, 0.0)))
    return hands


class InteractionFeatureExtractor:
    def __init__(self, yolo_model, hand_detector, object_class_names):
        self.yolo_model = yolo_model
        self.hand_detector = hand_detector
        self.object_class_names = object_class_names
        self.num_objects = len(object_class_names)
        self.reset_temporal_state()

    def reset_temporal_state(self):
        self.prev_object_centers = {i: None for i in range(self.num_objects)}
        self.prev_wrists = [None, None]

    def _predict_objects(self, frame, hands):
        result = self.yolo_model.predict(source=frame, conf=YOLO_CONF, verbose=False)[0]
        by_class = {i: [] for i in range(self.num_objects)}
        if result.boxes is not None:
            for b in result.boxes:
                cls_id = int(b.cls[0].item())
                if 0 <= cls_id < self.num_objects:
                    by_class[cls_id].append({"box": tuple(float(v) for v in b.xyxy[0].tolist()), "confidence": float(b.conf[0].item())})
        fingertips = [(float(h.landmarks_px[t, 0]), float(h.landmarks_px[t, 1])) for h in hands if h.present for t in HAND_TIP_IDS]
        chosen = {}
        for cls_id, candidates in by_class.items():
            if not candidates: chosen[cls_id] = None; continue
            if fingertips:
                candidates.sort(key=lambda d: (min(point_to_box_distance(px, py, d["box"]) for px, py in fingertips), -d["confidence"]))
            else:
                candidates.sort(key=lambda d: -d["confidence"])
            chosen[cls_id] = candidates[0]
        return chosen

    def _hand_features(self, hands, frame_w, frame_h):
        out = []
        for hand in hands:
            out.append(float(hand.present))
            if not hand.present:
                out.extend([0.0] * 68)
                continue
            out.extend([hand.wrist_xy_norm[0], hand.wrist_xy_norm[1], hand.scale_px / max(frame_w, frame_h), hand.wrist_velocity_norm[0], hand.wrist_velocity_norm[1]])
            wrist = hand.landmarks_px[0]; scale = max(1.0, hand.scale_px)
            for p in hand.landmarks_px:
                out.extend([float((p[0] - wrist[0]) / scale), float((p[1] - wrist[1]) / scale), float(np.clip(p[2] / scale, -3.0, 3.0))])
        return out

    def _object_features(self, objects, hands, frame_w, frame_h, dt_seconds):
        out = []
        for cls_id in range(self.num_objects):
            item = objects.get(cls_id)
            if item is None:
                out.extend([0.0] * 11); self.prev_object_centers[cls_id] = None; continue
            box = item["box"]; conf = item["confidence"]; x1, y1, x2, y2 = box; cx, cy = box_center(box); bw, bh = max(0.0, x2-x1), max(0.0, y2-y1)
            min_dist = 1.0; max_iou = 0.0
            for hand in hands:
                if not hand.present or hand.bbox is None: continue
                min_dist = min(min_dist, min(point_to_box_distance(float(hand.landmarks_px[t,0]), float(hand.landmarks_px[t,1]), box) for t in HAND_TIP_IDS) / box_diag(box))
                max_iou = max(max_iou, box_iou(hand.bbox, box))
            prev = self.prev_object_centers[cls_id]
            vx, vy = (0.0, 0.0) if prev is None or dt_seconds <= 0 else (((cx-prev[0])/frame_w) / dt_seconds, ((cy-prev[1])/frame_h) / dt_seconds)
            self.prev_object_centers[cls_id] = (cx, cy)
            out.extend([1.0, cx/frame_w, cy/frame_h, bw/frame_w, bh/frame_h, float(conf), float(np.clip(min_dist,0,5)), max_iou, float(np.clip(vx,-1,1)), float(np.clip(vy,-1,1)), 1.0 if min_dist < 0.15 else 0.0])
        return out

    def extract(self, frame, timestamp_ms, duration_seconds, dt_seconds):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hand_result = self.hand_detector.detect_for_video(mp_image, timestamp_ms)
        hands = parse_hands(hand_result, w, h, self.prev_wrists, dt_seconds)
        self.prev_wrists = [h.wrist_xy_norm if h.present else None for h in hands]
        objects = self._predict_objects(frame, hands)
        feat = self._hand_features(hands, w, h) + self._object_features(objects, hands, w, h, dt_seconds)
        return np.asarray(feat, dtype=np.float32), hands, objects


def sample_video_features(video_path, extractor, target_fps=FEATURE_FPS):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); duration = count / fps if count > 0 else 0.0
    stride = max(1, int(round(fps / target_fps)))
    extractor.reset_temporal_state(); features=[]; idx=0; prev_sample_time=None
    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % stride == 0:
            sample_time = idx / float(fps)
            dt = 0.0 if prev_sample_time is None else max(1e-6, sample_time - prev_sample_time)
            vec, _, _ = extractor.extract(frame, int(round(sample_time*1000)), duration, dt)
            features.append(vec)
            prev_sample_time = sample_time
        idx += 1
    cap.release()
    if not features: raise RuntimeError(f"No frames in {video_path}")
    return np.stack(features).astype(np.float32), duration


def make_sliding_windows(sequence, window_size=SEQ_LEN, stride=WINDOW_STRIDE):
    """Return overlapping [window_size, features] windows from one clip.

    This does NOT cut or rewrite the original video. It only creates overlapping
    temporal feature windows in memory for CNN training. Every window keeps the
    parent clip's label because the supplied clips are assumed to be clean,
    single-action clips.
    """
    t = sequence.shape[0]
    if t < window_size:
        return []

    return [
        sequence[start:start + window_size]
        for start in range(0, t - window_size + 1, stride)
    ]


def discover_clips(root):
    exts={".mp4",".avi",".mov",".mkv",".webm"}; clips=[]
    for cid, folder in enumerate(CLASS_FOLDER_NAMES):
        path=root/folder
        if not path.exists(): print(f"WARNING missing {path}"); continue
        for p in sorted(path.iterdir()):
            if p.is_file() and p.suffix.lower() in exts: clips.append((p,cid))
    if not clips: raise RuntimeError(f"No clips found under {root}")
    return clips


def stratified_split(clips, seed=SEED, train_ratio=.70, val_ratio=.15):
    rng=random.Random(seed); groups={i:[] for i in range(6)}
    for x in clips: groups[x[1]].append(x)
    train=[]; val=[]; test=[]
    for cid, items in groups.items():
        rng.shuffle(items); n=len(items); nt=max(1,int(round(n*train_ratio))); nv=max(1,int(round(n*val_ratio))); nt=min(nt,n-2); nv=min(nv-nt,n-nt-1)
        train += items[:nt]; val += items[nt:nt+nv]; test += items[nt+nv:]
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test); return train,val,test


def file_signature(path):
    st=path.stat(); return hashlib.sha1(f"{path.resolve()}::{st.st_size}::{st.st_mtime_ns}".encode()).hexdigest()


def build_cache(items, extractor, cache_file, meta_file, object_names, rebuild=False):
    config={
        "feature_version":FEATURE_VERSION,
        "seq_len":SEQ_LEN,
        "window_stride":WINDOW_STRIDE,
        "feature_fps":FEATURE_FPS,
        "classes":CLASS_NAMES,
        "objects":object_names
    }

    if cache_file.exists() and meta_file.exists() and not rebuild:
        try:
            meta=json.loads(meta_file.read_text())
            data=np.load(cache_file,allow_pickle=True)
            paths=data['paths'].tolist()
            if meta.get('config')==config and meta.get('signatures')==[file_signature(Path(p)) for p in paths]:
                print(f"Using feature cache: {cache_file}")
                return (data['X'].astype(np.float32), data['y'].astype(np.int64), paths, data['window_starts'].astype(np.int64))
        except Exception as exc:
            print(f"Cache ignored: {exc}")

    X=[]; y=[]; paths=[]; starts=[]; sig=[]

    for i,(path,label) in enumerate(items,1):
        print(f"[{i}/{len(items)}] {CLASS_NAMES[label]:10s} {path.name}")
        clip_hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=DEFAULT_HAND_MODEL),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=HAND_DETECTION_CONF,
                min_hand_presence_confidence=HAND_PRESENCE_CONF,
                min_tracking_confidence=HAND_TRACKING_CONF,
            )
        )
        clip_extractor = InteractionFeatureExtractor(extractor.yolo_model, clip_hand_detector, object_names)
        try:
            seq,_=sample_video_features(path,clip_extractor)
            windows=make_sliding_windows(seq,SEQ_LEN,WINDOW_STRIDE)

            if not windows:
                print(f"  WARNING: {path.name} is shorter than {SEQ_LEN} sampled timesteps; skipped.")
                continue

            signature=file_signature(path)
            for start_idx, window in zip(
                range(0, len(seq)-SEQ_LEN+1, WINDOW_STRIDE),
                windows
            ):
                X.append(window)
                y.append(label)
                paths.append(str(path))
                starts.append(start_idx)
                sig.append(signature)

            print(f"  -> {len(windows)} sliding windows")

        except Exception as exc:
            print(f"  ERROR: {exc}")
        finally:
            clip_hand_detector.close()

    if not X:
        raise RuntimeError(
            f"No usable sliding windows. Each clip needs at least {SEQ_LEN} sampled timesteps."
        )

    X=np.stack(X).astype(np.float32)
    y=np.asarray(y,np.int64)
    starts=np.asarray(starts,np.int64)

    np.savez_compressed(
        cache_file,
        X=X,
        y=y,
        paths=np.asarray(paths,dtype=object),
        window_starts=starts,
    )
    meta_file.write_text(
        json.dumps({"config":config,"signatures":sig},indent=2)
    )
    return X,y,paths,starts


def fit_standardizer(X):
    flat=X.reshape(-1,X.shape[-1]); mean=flat.mean(0).astype(np.float32); std=flat.std(0).astype(np.float32); std[std<1e-6]=1.0; return mean,std


def standardize(X,mean,std): return ((X-mean)/std).astype(np.float32)


class TemporalDataset(Dataset):
    def __init__(self,X,y,training=False): self.X=X; self.y=y; self.training=training
    def __len__(self): return len(self.y)
    def __getitem__(self,i):
        x=torch.from_numpy(self.X[i]).float(); y=torch.tensor(self.y[i]).long()
        if self.training:
            if torch.rand(())<.5: x=x+0.01*torch.randn_like(x)
            if torch.rand(())<.2:
                start=int(torch.randint(0,max(1,x.shape[0]-2),(1,)).item()); x[start:start+min(2,x.shape[0]-start)]=0
        return x.transpose(0,1),y


class TemporalConvBlock(nn.Module):
    def __init__(self,inch,outch,dilation,drop=.2):
        super().__init__(); pad=dilation*2
        self.c1=nn.Conv1d(inch,outch,5,padding=pad,dilation=dilation); self.b1=nn.BatchNorm1d(outch)
        self.c2=nn.Conv1d(outch,outch,5,padding=pad,dilation=dilation); self.b2=nn.BatchNorm1d(outch); self.drop=nn.Dropout(drop)
        self.skip=nn.Conv1d(inch,outch,1) if inch!=outch else nn.Identity()
    def forward(self,x):
        r=self.skip(x); y=torch.relu(self.b1(self.c1(x))); y=self.drop(y); y=self.b2(self.c2(y)); return torch.relu(y+r)


class TemporalCNN(nn.Module):
    def __init__(self,input_features,num_classes=6):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv1d(input_features,64,5,padding=2),nn.BatchNorm1d(64),nn.ReLU())
        self.b1=TemporalConvBlock(64,64,1,.15); self.b2=TemporalConvBlock(64,96,2,.20); self.b3=TemporalConvBlock(96,128,4,.25)
        self.pool=nn.AdaptiveAvgPool1d(8)
        self.fc=nn.Sequential(nn.Flatten(),nn.Linear(128*8,128),nn.ReLU(),nn.Dropout(.35),nn.Linear(128,num_classes))
    def forward(self,x): return self.fc(self.pool(self.b3(self.b2(self.b1(self.stem(x))))))


def cm_np(ytrue,ypred,n=6):
    cm=np.zeros((n,n),dtype=np.int64)
    for a,b in zip(ytrue,ypred): cm[int(a),int(b)]+=1
    return cm


def macro_f1(cm):
    fs=[]
    for i in range(cm.shape[0]):
        tp=cm[i,i]; fp=cm[:,i].sum()-tp; fn=cm[i,:].sum()-tp; p=tp/max(1,tp+fp); r=tp/max(1,tp+fn); fs.append(0.0 if p+r==0 else 2*p*r/(p+r))
    return float(np.mean(fs))


@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); yt=[]; yp=[]
    for X,y in loader:
        logits=model(X.to(device)); yp.extend(logits.argmax(1).cpu().numpy()); yt.extend(y.numpy())
    yt=np.asarray(yt); yp=np.asarray(yp); cm=cm_np(yt,yp); acc=float((yt==yp).mean()) if len(yt) else 0.0; return acc,macro_f1(cm),cm


def save_artifacts(out,mean,std,object_names,feature_count):
    np.savez_compressed(out/'feature_standardizer.npz',mean=mean,std=std)
    (out/'metadata.json').write_text(json.dumps({"class_names":CLASS_NAMES,"object_class_names":object_names,"seq_len":SEQ_LEN,"window_stride":WINDOW_STRIDE,"feature_fps":FEATURE_FPS,"input_features":feature_count,"feature_version":FEATURE_VERSION},indent=2))


def train(args):
    seed_everything(args.seed); root=Path(args.clips); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    yolo=YOLO(args.yolo_model); object_names=[str(yolo.names[i]) for i in range(len(yolo.names))]
    if len(object_names)!=8: print(f"WARNING: expected 8 YOLO classes, found {len(object_names)}: {object_names}")
    hand=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=args.hand_model),running_mode=vision.RunningMode.VIDEO,num_hands=2,min_hand_detection_confidence=HAND_DETECTION_CONF,min_hand_presence_confidence=HAND_PRESENCE_CONF,min_tracking_confidence=HAND_TRACKING_CONF))
    ext=InteractionFeatureExtractor(yolo,hand,object_names); clips=discover_clips(root); train_items,val_items,test_items=stratified_split(clips,args.seed)
    all_items=train_items+val_items+test_items
    Xall,yall,paths,window_starts=build_cache(all_items,ext,out/'feature_cache.npz',out/'feature_cache_meta.json',object_names,args.rebuild_cache)
    train_sources={str(p) for p,_ in train_items}
    val_sources={str(p) for p,_ in val_items}
    test_sources={str(p) for p,_ in test_items}
    tr=np.asarray([i for i,p in enumerate(paths) if p in train_sources],dtype=np.int64)
    va=np.asarray([i for i,p in enumerate(paths) if p in val_sources],dtype=np.int64)
    te=np.asarray([i for i,p in enumerate(paths) if p in test_sources],dtype=np.int64)
    Xtr,ytr=Xall[tr],yall[tr]; Xv,yv=Xall[va],yall[va]; Xt,yt=Xall[te],yall[te]
    mean,std=fit_standardizer(Xtr); Xtr=standardize(Xtr,mean,std); Xv=standardize(Xv,mean,std); Xt=standardize(Xt,mean,std); save_artifacts(out,mean,std,object_names,Xtr.shape[-1])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=TemporalCNN(Xtr.shape[-1]).to(device)
    train_loader=DataLoader(TemporalDataset(Xtr,ytr,True),batch_size=BATCH_SIZE,shuffle=True,num_workers=0,pin_memory=device.type=='cuda'); val_loader=DataLoader(TemporalDataset(Xv,yv),batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
    opt=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY); crit=nn.CrossEntropyLoss(label_smoothing=.05); sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=4,min_lr=1e-6)
    best=-1; wait=0; ck=out/'temporal_1dcnn_best.pt'
    for epoch in range(1,MAX_EPOCHS+1):
        model.train(); losses=[]
        for X,y in train_loader:
            X,y=X.to(device),y.to(device); opt.zero_grad(set_to_none=True); loss=crit(model(X),y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); losses.append(loss.item())
        va_acc,va_f1,_=evaluate(model,val_loader,device); score=.5*(va_acc+va_f1); sched.step(score)
        print(f"Epoch {epoch:03d} loss={np.mean(losses):.4f} val_acc={va_acc:.4f} val_f1={va_f1:.4f} lr={opt.param_groups[0]['lr']:.2e}")
        if score>best:
            best=score; wait=0; torch.save({'model_state_dict':model.state_dict(),'input_features':Xtr.shape[-1],'num_classes':6,'seq_len':SEQ_LEN},ck)
        else:
            wait+=1
            if wait>=EARLY_STOPPING_PATIENCE: break
    model.load_state_dict(torch.load(ck,map_location=device)['model_state_dict']); test_loader=DataLoader(TemporalDataset(Xt,yt),batch_size=BATCH_SIZE,shuffle=False,num_workers=0); acc,f1,cm=evaluate(model,test_loader,device)
    print(f"\nTEST accuracy={acc:.4f} macro_f1={f1:.4f}\nConfusion matrix:"); print(cm)
    np.savetxt(out/'confusion_matrix.csv',cm,fmt='%d',delimiter=','); hand.close()


def open_source(s): return cv2.VideoCapture(int(s)) if s.isdigit() else cv2.VideoCapture(s)


def load_inference_artifacts(out,yolo):
    meta=json.loads((out/'metadata.json').read_text()); z=np.load(out/'feature_standardizer.npz'); current=[str(yolo.names[i]) for i in range(len(yolo.names))]
    if current!=meta['object_class_names']: raise RuntimeError(f"YOLO class mismatch. Trained: {meta['object_class_names']} Current: {current}")
    model=TemporalCNN(int(meta['input_features']),6); ck=torch.load(out/'temporal_1dcnn_best.pt',map_location='cpu'); model.load_state_dict(ck['model_state_dict']); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); return model.to(device).eval(),device,meta,z['mean'],z['std']


def draw_hands(frame,hands):
    for h in hands:
        if not h.present: continue
        pts=h.landmarks_px[:,:2].astype(np.int32)
        for a,b in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]: cv2.line(frame,tuple(pts[a]),tuple(pts[b]),(0,220,0),1)
        for p in pts: cv2.circle(frame,tuple(p),2,(0,255,0),-1)


def draw_objects(frame,objects,names):
    for cid,item in objects.items():
        if item is None: continue
        x1,y1,x2,y2=map(int,item['box']); c=item['confidence']; cv2.rectangle(frame,(x1,y1),(x2,y2),(255,150,0),2); cv2.putText(frame,f"{names[cid]} {c:.2f}",(x1,max(20,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2)


def infer(args):
    out=Path(args.output); yolo=YOLO(args.yolo_model); model,device,meta,mean,std=load_inference_artifacts(out,yolo); hand=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=args.hand_model),running_mode=vision.RunningMode.VIDEO,num_hands=2,min_hand_detection_confidence=HAND_DETECTION_CONF,min_hand_presence_confidence=HAND_PRESENCE_CONF,min_tracking_confidence=HAND_TRACKING_CONF)); ext=InteractionFeatureExtractor(yolo,hand,meta['object_class_names']); cap=open_source(args.source)
    if not cap.isOpened(): raise RuntimeError(f"Could not open source: {args.source}")
    src_fps=cap.get(cv2.CAP_PROP_FPS) or 30.0; interval=1.0/float(meta['feature_fps']); last_sample=-1e9; frame_idx=0; buf=deque(maxlen=int(meta['seq_len'])); ema=None; alpha=.25; last_t=time.perf_counter(); fps_sm=0.0
    while True:
        ok,frame=cap.read();
        if not ok: break
        t=frame_idx/src_fps
        debug=None
        if t-last_sample>=interval:
            dt=0.0 if last_sample < 0 else max(1e-6, t-last_sample)
            vec,hands,objects=ext.extract(frame,int(round(t*1000)),0.0,dt); buf.append(vec); debug=(hands,objects); last_sample=t
            if len(buf)==buf.maxlen:
                x=standardize(np.stack(buf),mean,std); X=torch.from_numpy(x).float().unsqueeze(0).transpose(1,2).to(device)
                with torch.no_grad(): p=torch.softmax(model(X),1)[0].cpu().numpy()
                ema=p if ema is None else (1-alpha)*ema+alpha*p
        if debug:
            draw_hands(frame,debug[0]); draw_objects(frame,debug[1],meta['object_class_names'])
        label='WARMING UP'; conf=0.0
        if ema is not None:
            i=int(np.argmax(ema)); conf=float(ema[i]); label=CLASS_NAMES[i] if conf>=.45 else 'UNCERTAIN'
        cv2.putText(frame,f"ACTION: {label} {conf:.1%}",(20,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2)
        cv2.putText(frame,f"BUFFER: {len(buf)}/{buf.maxlen}",(20,65),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        y0=100
        for name,p in zip(CLASS_NAMES,ema if ema is not None else np.zeros(6)):
            cv2.putText(frame,f"{name:9s} {p:6.1%}",(20,y0),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1); y0+=24
        now=time.perf_counter(); inst=1.0/max(1e-6,now-last_t); fps_sm=inst if fps_sm==0 else .9*fps_sm+.1*inst; last_t=now
        cv2.putText(frame,f"FPS: {fps_sm:.1f}",(20,y0+10),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        cv2.imshow('SIH 2026 - 1D CNN H-OI',frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break
        frame_idx+=1
    cap.release(); hand.close(); cv2.destroyAllWindows()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['train','infer'],required=True); p.add_argument('--clips',default=DEFAULT_CLIP_ROOT); p.add_argument('--output',default=DEFAULT_OUTPUT_DIR); p.add_argument('--yolo-model',default=DEFAULT_YOLO_MODEL); p.add_argument('--hand-model',default=DEFAULT_HAND_MODEL); p.add_argument('--source',default='1'); p.add_argument('--rebuild-cache',action='store_true'); p.add_argument('--seed',type=int,default=SEED); a=p.parse_args(); train(a) if a.mode=='train' else infer(a)

if __name__=='__main__': main()
