"""
Structural Topology Kernel (STK-50): 50 deterministic, training-free structural
coordinates for grayscale fields (SEM micrographs, EBSD-derived maps, binary masks).

Layers: L1 geometry (placement, spread, orientation, spectrum), L3 radial
compliance, L4 gradient-field cohesion (SCI), L6 multi-island resistance graph,
L8 GLCM texture, L9 blob microstructure, plus spectral/edge and tonal channels.
Color-specific machinery is not part of this kernel.

FROZEN: this file is the research kernel (June 2026) with only this docstring and
one comment reworded. Numbers are pinned by tests/test_golden.py.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt, label as nd_label
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import jensenshannon
from scipy.spatial import ConvexHull
from skimage.filters import threshold_local, threshold_otsu
from scipy.ndimage import binary_opening, binary_closing
from skimage.morphology import disk, remove_small_objects
from skimage.segmentation import watershed
from skimage.feature import peak_local_max, canny, blob_dog
from skimage import measure
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# ── Canonical constants ───────────────────────────────────────────────────
VOID_THRESH = 0.18
PERIPH_BAND = 0.15
G4_SCALES   = [1, 8, 32]
BLOB_MIN_SIGMA, BLOB_MAX_SIGMA, BLOB_THRESHOLD = 3.0, 20.0, 0.04  # min_sigma lowered for fine microstructure

def norm01(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)

def sobel_grads(field):
    Gx = np.gradient(field, axis=1); Gy = np.gradient(field, axis=0)
    return Gx, Gy, np.sqrt(Gx**2 + Gy**2)

# ── L1 kernel primitives (operate on the gradient/mass field) ──────────────
def compute_centroid(field):
    nf = norm01(field); total = nf.sum()
    if total < 1e-8: return 0.0, 0.0
    H, W = nf.shape
    cx = (nf.sum(0) @ np.arange(W)) / total; cy = (nf.sum(1) @ np.arange(H)) / total
    return float((cx-(W-1)/2)/W), float((cy-(H-1)/2)/H)

def compute_void_ratio(field):
    nf = norm01(field); return float((nf < VOID_THRESH).sum() / nf.size)

def compute_cohesion(field):
    nf = norm01(field); total = nf.sum()
    if total < 1e-8: return 0.0
    return float(nf[nf >= np.percentile(nf, 75)].sum() / total)

def compute_rho_r(field):
    nf = norm01(field); active = nf >= VOID_THRESH; ma = active.sum()
    if ma < 3: return 0.0
    ys, xs = np.where(active); pts = np.column_stack([xs, ys])
    if len(pts) > 1500: pts = pts[np.linspace(0, len(pts)-1, 1500, dtype=int)]
    try:
        hv = ConvexHull(pts).volume
        return float(min(150, 100*ma/hv)) if hv >= 1 else 0.0
    except Exception: return 0.0

def compute_xp(field):
    nf = norm01(field); H, W = nf.shape; total = nf.sum()
    if total < 1e-8: return 0.0
    bh, bw = int(H*PERIPH_BAND), int(W*PERIPH_BAND)
    edge = np.zeros((H, W), bool); edge[:bh,:]=edge[H-bh:,:]=edge[:,:bw]=edge[:,W-bw:]=True
    return float(nf[edge].sum()/total)

def compute_theta(field, Gx, Gy):
    nf = norm01(field); active = nf >= VOID_THRESH
    if active.sum() < 2: return 0.0
    ga, gb = Gx[active], Gy[active]; mag = np.sqrt(ga**2+gb**2)+1e-8
    ang = (np.arctan2(gb, ga)+np.pi) % (2*np.pi)
    hist,_ = np.histogram(ang, bins=8, range=(0,2*np.pi), weights=mag)
    p = hist/(hist.sum()+1e-8); Hent = -np.sum(p[p>0]*np.log2(p[p>0]))
    return float(1 - Hent/np.log2(8))

def compute_ds(field):
    nf = norm01(field); H, W = nf.shape; active = nf >= VOID_THRESH
    if active.sum() < 2: return 0.0
    return float(2*distance_transform_edt(active)[active].mean()/min(W, H))

def compute_sdi(field):
    nf = norm01(field); H, W = nf.shape; diag = np.sqrt(W**2+H**2); active = nf >= VOID_THRESH
    if active.sum() < 2: return 0.0
    ys, xs = np.where(active)
    return float(np.sqrt((xs-xs.mean())**2+(ys-ys.mean())**2).mean()/diag)

# ── L4 VCLI-G ──────────────────────────────────────────────────────────────
def _orient_entropy(field):
    Gx = np.gradient(field, axis=1); Gy = np.gradient(field, axis=0)
    mag = np.sqrt(Gx**2+Gy**2); ang = (np.arctan2(Gy, Gx)+np.pi) % (2*np.pi)
    hist,_ = np.histogram(ang.ravel(), bins=36, range=(0,2*np.pi), weights=mag.ravel())
    p = hist/(hist.sum()+1e-8); Hent = -np.sum(p[p>0]*np.log2(p[p>0]))
    return float(Hent/np.log2(36)*5.0)

def compute_g1(L):  # centroid wander across scale
    H, W = L.shape; pts=[]
    for s in [2,4,8,16,32]:
        b = gaussian_filter(L, s); t = b.sum()
        if t < 1e-8: pts.append([0.5,0.5])
        else: pts.append([(b.sum(0)@np.arange(W))/t/W, (b.sum(1)@np.arange(H))/t/H])
    pts = np.array(pts); d = np.diff(pts, 0)
    plen = float(np.sqrt((d**2).sum(1)).sum())
    v = d/(np.sqrt((d**2).sum(1,keepdims=True))+1e-8)
    ca = [np.arccos(np.clip(v[i]@v[i+1],-1,1)) for i in range(len(v)-1)]
    return plen, float(np.mean(ca)) if ca else 0.0

def compute_g2(L):  # void topology
    H, W = L.shape; g = gaussian_filter(L, 3.0)
    bs = max(31, (min(H,W)//10)|1)
    binary = g > threshold_local(g, bs, offset=-0.02); voids = ~binary
    voids = binary_closing(binary_opening(voids, disk(2)), disk(3))
    voids = remove_small_objects(voids, min_size=max(64, H*W//500))
    reg = measure.regionprops(measure.label(voids))
    ar = [r.axis_major_length/r.axis_minor_length for r in reg if r.axis_minor_length>0]
    areas = [r.area for r in reg]
    return float(voids.mean()), len(reg), float(np.mean(ar)) if ar else 1.0, float(max(areas)/(H*W)) if areas else 0.0

def compute_g3(L):  # contour curvature
    lo = float(np.percentile((L*255).astype(np.uint8),85))/255.0
    hi = float(np.percentile((L*255).astype(np.uint8),97))/255.0
    edges = canny(L, sigma=1.0, low_threshold=lo, high_threshold=hi)
    kv, idn = [], []
    for c in measure.find_contours(edges.astype(float), 0.5):
        if len(c) < 5: continue
        d = np.diff(c,0); tg = d/(np.sqrt((d**2).sum(1,keepdims=True))+1e-8)
        a = [np.arccos(np.clip(tg[i]@tg[i+1],-1,1)) for i in range(len(tg)-1)]
        if not a: continue
        kv.append(float(np.var(a)))
        idn.append(int(np.sum(np.diff(np.sign(np.array(a)-np.mean(a)))!=0))/(len(c)+1e-8))
    return (float(np.mean(kv)) if kv else 0.0, float(np.mean(idn)) if idn else 0.0)

def compute_g4(L):  # orientation entropy slope across scale
    ents=[]
    for s in G4_SCALES:
        ents.append(_orient_entropy(L if s<=1 else gaussian_filter(L, s)))
    span = np.log(G4_SCALES[-1])-np.log(G4_SCALES[0])
    return float(ents[0]), float((ents[-1]-ents[0])/(span if span>0 else 1.0))

def compute_sci(L, Gx, Gy):
    H, W = L.shape; mag = np.sqrt(Gx**2+Gy**2)
    quads = [(slice(None,H//2),slice(None,W//2)),(slice(None,H//2),slice(W//2,None)),
             (slice(H//2,None),slice(None,W//2)),(slice(H//2,None),slice(W//2,None))]
    regional = 1/(1+np.var([mag[q].mean() for q in quads])*1000)
    ang = (np.arctan2(Gy,Gx)+np.pi)%(2*np.pi)
    h18,_ = np.histogram(ang.ravel(),bins=18,range=(0,2*np.pi),weights=mag.ravel())
    p18 = h18/(h18.sum()+1e-8); align = 1 - (-np.sum(p18[p18>0]*np.log2(p18[p18>0])))/np.log2(18)
    es = [np.sqrt(np.gradient(gaussian_filter(L,s),axis=1)**2+np.gradient(gaussian_filter(L,s),axis=0)**2).mean() for s in [1,2,4]]
    scale_cons = 1/(1+np.var(es)*100)
    flat = L.ravel()-L.mean(); n=min(1000,len(flat))
    ac = np.correlate(flat[:n],flat[:n],'full'); ac=ac/(ac.max()+1e-8)
    rhythm = float(np.mean(np.abs(ac))*2)
    return float(np.clip((0.30*regional+0.30*align+0.25*scale_cons+0.15*rhythm)*5.0,0,5))

# ── L3 radial compliance ────────────────────────────────────────────────────
def _radial_profile(field, center, nbins=60):
    H, W = field.shape; cx, cy = center; Y, X = np.mgrid[0:H,0:W]
    R = np.sqrt((X-cx)**2+(Y-cy)**2); Rm = np.sqrt(max(cx,W-1-cx)**2+max(cy,H-1-cy)**2)+1e-8
    Rn = R/Rm; hist = np.zeros(nbins); e = np.linspace(0,1,nbins+1)
    for i in range(nbins):
        m = (Rn>=e[i]) & (Rn<e[i+1]); hist[i] = field[m].sum()
    return hist/(hist.sum()+1e-8)

def _fit_rc(profile, nbins=60):
    r = np.linspace(0,1,nbins)
    def loss(a):
        q = np.exp(-a*r); q/=q.sum()+1e-8; return float(np.mean((q-profile)**2))
    a = minimize_scalar(loss, bounds=(0.1,200.0), method='bounded').x
    q = np.exp(-a*r); q=(q+1e-12)/(q+1e-12).sum(); p=(profile+1e-12)/(profile+1e-12).sum()
    return float(1-jensenshannon(p,q))

def compute_radial_compliance(field):
    nf = norm01(field); H, W = nf.shape
    rc_f = _fit_rc(_radial_profile(nf, ((W-1)/2,(H-1)/2)))
    dx, dy = compute_centroid(nf)
    rc_s = _fit_rc(_radial_profile(nf, (dx*W+(W-1)/2, dy*H+(H-1)/2)))
    return float(rc_s-rc_f), float(np.sqrt(dx**2+dy**2)/np.sqrt(0.5))

# ── L8 GLCM texture (luminance, 2 scales) ───────────────────────────────────
def _glcm_stats(gq, levels, distances):
    H, W = gq.shape; g = gq.astype(np.int32); cv=[]; ev=[]; hv=[]; nv=[]
    for d in distances:
        for (dy,dx) in [(0,d),(d,0)]:
            glcm = np.zeros((levels,levels)); iv = g[:H-dy,:W-dx].ravel(); jv = g[dy:,dx:].ravel()
            ok = (iv>=0)&(iv<levels)&(jv>=0)&(jv<levels); iv,jv = iv[ok],jv[ok]
            np.add.at(glcm,(iv,jv),1); glcm += glcm.T; tot=glcm.sum()
            if tot<1: continue
            p = glcm/tot; ii,jj = np.meshgrid(np.arange(levels),np.arange(levels),indexing='ij'); nz=p>0
            cv.append(np.sum(p*(ii-jj)**2)); ev.append(-np.sum(p[nz]*np.log2(p[nz])))
            hv.append(np.sum(p/(1+np.abs(ii-jj)))); nv.append(np.sum(p**2))
    if not cv: return dict(contrast=np.nan,entropy=np.nan,homogeneity=np.nan,energy=np.nan)
    return dict(contrast=float(np.mean(cv)),entropy=float(np.mean(ev)),homogeneity=float(np.mean(hv)),energy=float(np.mean(nv)))

def compute_glcm_texture(L, levels=32):
    gq_fine = (gaussian_filter(L,2.0)*(levels-1)).clip(0,levels-1).astype(np.uint8)
    gq_coarse = (gaussian_filter(L,16.0)*(levels-1)).clip(0,levels-1).astype(np.uint8)
    f = _glcm_stats(gq_fine, levels, [1]); c = _glcm_stats(gq_coarse, levels, [1])
    return {**{f'tex_{k}_fine':v for k,v in f.items()}, **{f'tex_{k}_coarse':v for k,v in c.items()}}

# ── L9 blob microstructure (grain/precipitate detection) ────────────────────
def compute_blob_metrics(field):
    try:
        H, W = field.shape
        blobs = blob_dog(norm01(field), min_sigma=BLOB_MIN_SIGMA, max_sigma=BLOB_MAX_SIGMA,
                         sigma_ratio=1.6, threshold=BLOB_THRESHOLD)
        n = len(blobs); area = (H*W)/10000.0
        if n == 0: return dict(blob_count=0, blob_density=0.0, blob_size_mean=0.0, blob_size_cv=0.0)
        radii = blobs[:,2]*np.sqrt(2)
        return dict(blob_count=int(n), blob_density=float(n/area), blob_size_mean=float(radii.mean()),
                    blob_size_cv=float(radii.std()/(radii.mean()+1e-8)) if n>1 else 0.0)
    except Exception:
        return dict(blob_count=-1, blob_density=np.nan, blob_size_mean=np.nan, blob_size_cv=np.nan)

# ── L6 multi-island resistance graph ────────────────────────────────────────
def compute_islands(gmag_n, L, rv, rho_r):
    try:
        pack = float(np.clip(rho_r/100.0, 0.05, 1.0))
        sig = float(np.clip(4.0+(1.0-pack)*24.0, 4.0, 18.0))
        lm = gaussian_filter(L.astype(np.float64), sig*0.5)
        ps = norm01(gmag_n.astype(np.float64)/(lm+0.15)); dens = norm01(gaussian_filter(ps, sig))
        vf = float(np.clip(rv,0.1,0.9)); seg = float(np.clip(0.25+(1-vf)*0.25+pack*0.15, 0.20, 0.72))
        H, W = dens.shape; md = int(min(W,H)*0.10)
        pk = peak_local_max(dens, min_distance=md, threshold_abs=float(threshold_otsu(dens)))
        if len(pk) == 0: return dict(n_islands=0, island_size_cv=0.0, island_spread=0.0, resistance_mean=np.nan, drain_confidence=np.nan)
        markers = np.zeros(dens.shape, int)
        for i,(py,px) in enumerate(pk, 1): markers[py,px]=i
        lab = watershed(1.0-dens, markers, mask=dens>seg*0.6)
        reg = measure.regionprops(lab, intensity_image=dens)
        if len(reg) == 0: return dict(n_islands=0, island_size_cv=0.0, island_spread=0.0, resistance_mean=np.nan, drain_confidence=np.nan)
        areas = np.array([r.area for r in reg]); cents = np.array([r.centroid for r in reg])
        size_cv = float(areas.std()/(areas.mean()+1e-8)) if len(areas)>1 else 0.0
        spread = float(np.sqrt(((cents-cents.mean(0))**2).sum(1)).mean()/np.sqrt(H**2+W**2)) if len(cents)>1 else 0.0
        # directed resistance graph: corridor cost between island centroids
        res=[]
        for i in range(len(reg)):
            for j in range(len(reg)):
                if i==j: continue
                y0,x0 = cents[i]; y1,x1 = cents[j]; nstep=20
                ys = np.linspace(y0,y1,nstep).astype(int); xs = np.linspace(x0,x1,nstep).astype(int)
                ys = np.clip(ys,0,H-1); xs = np.clip(xs,0,W-1)
                res.append(1.0 - dens[ys,xs].mean())
        res = np.array(res)
        # drain confidence: how dominant the single lowest-resistance sink is
        if len(reg)>1:
            inbound = np.zeros(len(reg)); k=0
            for i in range(len(reg)):
                for j in range(len(reg)):
                    if i==j: continue
                    inbound[j]+=res[k]; k+=1
            inbound/= (len(reg)-1)
            drain_conf = float((inbound.max()-inbound.min())/(inbound.max()+1e-8))
        else: drain_conf = 0.0
        return dict(n_islands=int(len(reg)), island_size_cv=size_cv, island_spread=spread,
                    resistance_mean=float(res.mean()) if len(res) else np.nan, drain_confidence=drain_conf)
    except Exception:
        return dict(n_islands=-1, island_size_cv=np.nan, island_spread=np.nan, resistance_mean=np.nan, drain_confidence=np.nan)

# ── Extended coords (v3.1 spectral/edge + L5 tonal + L7 gap) ────────────────
def compute_extended(L, Gx, Gy, gmag):
    H, W = L.shape; out = {}
    # spectral band decomposition (v3.1)
    power = np.abs(np.fft.fftshift(np.fft.fft2(L)))**2
    fy, fx = np.mgrid[-H//2:H//2, -W//2:W//2]; fr = np.sqrt(fx**2+fy**2)
    mf = np.sqrt((H//2)**2+(W//2)**2); tp = power.sum()+1e-9
    out['ds_low']  = float(power[fr < mf/3].sum()/tp)
    out['ds_mid']  = float(power[(fr>=mf/3)&(fr<2*mf/3)].sum()/tp)
    out['ds_high'] = float(power[fr >= 2*mf/3].sum()/tp)
    # NILS / LER / edge density (v3.1, Canny)
    if _HAS_CV2:
        u8 = (L*255).clip(0,255).astype(np.uint8); edges = cv2.Canny(u8, 30, 90); em = edges.astype(bool)
        out['edge_density'] = float(em.sum()/(H*W)); si = L.copy(); si[si<1e-6]=1e-6
        out['nils'] = float(np.clip(np.median(gmag[em]/si[em]) if em.sum()>0 else 0, 0, 10))
        cnts,_ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE); devs=[]
        for c in cnts:
            p = c.squeeze()
            if p.ndim<2 or len(p)<8: continue
            win = max(3, len(p)//10); xc=p[:,0].astype(float); yc=p[:,1].astype(float)
            d = np.sqrt((xc-np.convolve(xc,np.ones(win)/win,'same'))**2+(yc-np.convolve(yc,np.ones(win)/win,'same'))**2)
            t=win//2; devs.extend(d[t:-t].tolist())
        out['ler'] = float(np.std(devs)/np.sqrt(H**2+W**2)) if len(devs)>=10 else 0.0
        # L7 gap field: Sobel fires, Canny does not
        gapf = (gmag*((gmag>np.percentile(gmag,75))&~em)).astype(np.float64)
        out['gap_rv'] = compute_void_ratio(gapf); out['gap_theta'] = compute_theta(gapf, Gx, Gy)
    else:
        out.update(edge_density=np.nan, nils=np.nan, ler=np.nan, gap_rv=np.nan, gap_theta=np.nan)
    # L5 tonal mass fractions (grayscale-valid)
    out['tonal_shadow'] = float((L<0.33).mean()); out['tonal_mid'] = float(((L>=0.33)&(L<0.66)).mean())
    out['tonal_high'] = float((L>=0.66).mean())
    return out

# ── Full kernel ─────────────────────────────────────────────────────────────
def compute_full_kernel(gray):
    """Compute the full grayscale-valid kernel on a normalized [0,1] grayscale image.
    Returns a flat dict of ~30 named coordinates. Field conventions:
      - L1 geometry on the gradient/boundary field (gmag_n); VCLI-G on the luminance (L_star=gray).
    """
    L = norm01(gray.astype(np.float64))
    Gx, Gy, gmag = sobel_grads(L); gmag_n = norm01(gmag)
    out = {}
    # L1 (on gradient field)
    dx, dy = compute_centroid(gmag_n); out['delta_x'], out['delta_y'] = dx, dy
    out['r_v'] = compute_void_ratio(gmag_n); out['mu'] = compute_cohesion(gmag_n)
    out['rho_r'] = compute_rho_r(gmag_n); out['x_p'] = compute_xp(gmag_n)
    out['theta'] = compute_theta(gmag_n, Gx, Gy); out['d_s'] = compute_ds(gmag_n)
    out['sdi'] = compute_sdi(gmag_n)
    # L4 (on luminance)
    out['g1_pathlen'], out['g1_curvature'] = compute_g1(L)
    out['g2_void'], out['g2_nvoids'], out['g2_AR'], out['g2_cut'] = compute_g2(L)
    out['g3_kvar'], out['g3_infl'] = compute_g3(L)
    out['g4_oent'], out['g4_slope'] = compute_g4(L)
    out['sci'] = compute_sci(L, Gx, Gy)
    # L3
    out['rc_drc'], out['rc_dr'] = compute_radial_compliance(gmag_n)
    # L8 texture
    out.update(compute_glcm_texture(L))
    # L9 blobs
    out.update(compute_blob_metrics(L))
    # L6 islands
    out.update(compute_islands(gmag_n, L, out['r_v'], out['rho_r']))
    # extended: v3.1 spectral/edge + L5 tonal + L7 gap
    out.update(compute_extended(L, Gx, Gy, gmag))
    return out

KERNEL_COORDS = ['delta_x','delta_y','r_v','mu','rho_r','x_p','theta','d_s','sdi',
    'g1_pathlen','g1_curvature','g2_void','g2_nvoids','g2_AR','g2_cut','g3_kvar','g3_infl',
    'g4_oent','g4_slope','sci','rc_drc','rc_dr',
    'tex_contrast_fine','tex_entropy_fine','tex_homogeneity_fine','tex_energy_fine',
    'tex_contrast_coarse','tex_entropy_coarse','tex_homogeneity_coarse','tex_energy_coarse',
    'blob_count','blob_density','blob_size_mean','blob_size_cv',
    'n_islands','island_size_cv','island_spread','resistance_mean','drain_confidence',
    'ds_low','ds_mid','ds_high','edge_density','nils','ler','gap_rv','gap_theta',
    'tonal_shadow','tonal_mid','tonal_high']

if __name__ == '__main__':
    import numpy as _np
    t = _np.random.default_rng(0).random((256,256))
    k = compute_full_kernel(t)
    print(f"{len(k)} coords computed:"); [print(f"  {c}: {k.get(c)}") for c in KERNEL_COORDS]
