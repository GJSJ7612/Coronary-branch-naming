"""
CCTA冠状动脉分支命名系统 - 后端服务
运行: uvicorn main:app --reload --port 8000
"""
import json, uuid, shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI(title="CCTA Coronary Artery Labeling System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("tmp/uploads")
OUTPUT_DIR = Path("tmp/outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 挂载前端静态文件
app.mount("/static", StaticFiles(directory="./frontend"), name="static")


# ─────────────────────────────────────────────
# 工具函数：mask → 三角网格（OBJ格式顶点+面）
# ─────────────────────────────────────────────

def mask_to_mesh(mask_npy, label_value=None, center=None, scale=None):
    """
    用 scikit-image marching_cubes 把体素mask转成三角网格。
    返回 {"vertices": [[x,y,z],...], "faces": [[i,j,k],...]}
    """
    from skimage.measure import marching_cubes

    if label_value is not None:
        binary = (mask_npy == label_value).astype(np.uint8)
    else:
        binary = (mask_npy > 0).astype(np.uint8)

    if binary.sum() == 0:
        return None

    verts, faces, _, _ = marching_cubes(binary, level=0.5, step_size=2)
    verts = verts.astype(float)
    if center is None:
        center = verts.mean(axis=0)
        scale = np.abs(verts - center).max()
    verts -= center
    if scale > 0:
        verts /= scale

    return {"vertices": verts.tolist(), "faces": faces.tolist()}


def _get_save_suffix(filename: str) -> str:
    """根据上传文件名推断保存后缀，支持 .nrrd / .nii.gz / .nii"""
    name = (filename or "").lower()
    if name.endswith(".nrrd"):
        return ".nrrd"
    elif name.endswith(".nii.gz"):
        return ".nii.gz"
    elif name.endswith(".nii"):
        return ".nii"
    else:
        raise ValueError(f"不支持的格式：{filename}，请上传 .nrrd 或 .nii.gz 文件")


def load_volume(path: str) -> np.ndarray:
    """
    加载体积文件，自动识别格式：
      .nrrd        → pynrrd
      .nii / .nii.gz → nibabel
    """
    from src.data_process import image_resample

    p = str(path).lower()
    if p.endswith(".nrrd") or p.endswith(".nii.gz"):
        try:
            import SimpleITK as sitk
        except ImportError:
            raise RuntimeError("读取 .nrrd 需要安装 pynrrd：pip install pynrrd")
        data = sitk.ReadImage(path)
    else:
        raise ValueError(f"无法识别的文件格式：{path}")

    data = image_resample(data, new_spacing=(0.5, 0.5, 0.5))
    data = sitk.GetArrayFromImage(data).transpose(2, 1, 0)
    
    return data


# ─────────────────────────────────────────────
# 路由
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("./frontend/index.html")


@app.post("/api/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    ccta_file: UploadFile = File(..., description="CCTA影像 (.nii.gz 或 .nrrd)"),
    mask_file: UploadFile = File(..., description="血管mask文件 (.nii.gz 或 .nrrd)"),
):
    """
    第一步：上传文件，立即返回原始网格用于预览。
    支持格式：.nrrd / .nii.gz / .nii
    """
    # 校验并确定保存后缀
    try:
        ccta_suffix = _get_save_suffix(ccta_file.filename)
        mask_suffix = _get_save_suffix(mask_file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    task_id = str(uuid.uuid4())[:8]
    task_dir = UPLOAD_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # 保存上传文件（保留原始格式后缀）
    ccta_path = task_dir / f"ccta{ccta_suffix}"
    mask_path = task_dir / f"mask{mask_suffix}"

    with open(ccta_path, "wb") as f:
        shutil.copyfileobj(ccta_file.file, f)
    with open(mask_path, "wb") as f:
        shutil.copyfileobj(mask_file.file, f)

    # 记录路径，供 predict 接口查找
    meta_path = task_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump({"ccta": str(ccta_path), "mask": str(mask_path)}, f)

    try:
        mask_data = load_volume(str(mask_path))
        print(f"original_mask 唯一值: {np.unique(mask_data)}")
        print(f"original_mask 非零体素数: {mask_data.sum()}")
        print(f"original_mask shape: {mask_data.shape}")
        mesh = mask_to_mesh(mask_data)
        if mesh is None:
            raise HTTPException(status_code=422, detail="Mask文件中未找到有效血管区域")

        # 保存原始网格
        mesh_path = OUTPUT_DIR / f"{task_id}_raw.json"
        with open(mesh_path, "w") as f:
            json.dump(mesh, f)

        return JSONResponse({
            "task_id": task_id,
            "status": "uploaded",
            "raw_mesh": mesh,  # 直接返回，省去二次请求
            "message": "文件上传成功，原始网格已生成"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"网格生成失败: {str(e)}")


@app.post("/api/predict/{task_id}")
async def predict(task_id: str):
    """
    第二步：运行GNN模型，返回带颜色标签的多分支网格。
    """
    task_dir = UPLOAD_DIR / task_id
    meta_path = task_dir / "meta.json"

    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="任务不存在，请重新上传")

    with open(meta_path) as f:
        meta = json.load(f)
    mask_path = meta["mask"]
    ccta_path = meta["ccta"]

    try:
        print(f"开始处理任务 {task_id}，CCTA路径: {ccta_path}, Mask路径: {mask_path}")

        # 调用真实模型流水线
        # pipeline/predict.py 负责：图构建 → 模型推理 → 边级标签 → 体素级 labeled_mask
        # labeled_mask 值：0=背景 1-RCA、2-LAD、3-LCX、4-LM、5-Other
        from backend.predict import run_pipeline
        labeled_mask = run_pipeline(ccta_path, mask_path)
        print(f"labeled_mask 唯一值: {np.unique(labeled_mask)}")
        print(f"labeled_mask shape: {labeled_mask.shape}")

        # 用整体非零区域算归一化基准
        from skimage.measure import marching_cubes
        base_binary = (labeled_mask > 0).astype(np.uint8)
        print(f"labeled_mask 非零体素数: {base_binary.sum()}")
        base_verts, _, _, _ = marching_cubes(
            (labeled_mask > 0).astype(np.uint8), level=0.5, step_size=2
        )
        base_verts = base_verts.astype(float)
        shared_center = base_verts.mean(axis=0)
        shared_scale = np.abs(base_verts - shared_center).max()
        print(f"shared_center: {shared_center}")
        print(f"shared_scale: {shared_scale}")

        # 为每个分支单独提取网格
        LABEL_MAP = {
            1: {"name": "RCA",   "color": "#E87A5D"},
            2: {"name": "LAD",   "color": "#E8C76A"},
            3: {"name": "LCX",   "color": "#7DE89A"},
            4: {"name": "LM",    "color": "#5DB8E8"},
            5: {"name": "Other", "color": "#A0A0B0"},
        }

        branches = []
        for label_val, info in LABEL_MAP.items():
            mesh = mask_to_mesh(labeled_mask, label_value=label_val,
                                center=shared_center, scale=shared_scale)
            if mesh is not None:
                branches.append({
                    "label": label_val,
                    "name": info["name"],
                    "color": info["color"],
                    "mesh": mesh
                })

        return JSONResponse({
            "task_id": task_id,
            "status": "predicted",
            "branches": branches,
            "message": f"预测完成，共识别 {len(branches)} 个分支"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型推理失败: {str(e)}")

@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    task_dir = UPLOAD_DIR / task_id
    if not task_dir.exists():
        return {"exists": False}
    files = list(task_dir.iterdir())
    return {"exists": True, "files": [f.name for f in files]}
