import json
import uuid
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.http import FileResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

import configs.config_reg as cfg_reg

from .parsers import SMILES_COLUMN_NAMES, detect_smiles_column
from .services.job_manager import create_job, get_job, run_in_background, update_job

_datasets: dict[str, dict] = {}
_init_models: dict[str, str] = {}


def _default_epochs(task_type: str) -> int:
    return cfg_reg.num_epochs


def _default_lr() -> float:
    return float(cfg_reg.lr_head)


@csrf_exempt
@require_http_methods(["GET"])
def train_defaults(_request):
    return JsonResponse(
        {
            "num_epochs": _default_epochs("regression"),
            "learning_rate": _default_lr(),
            "lr_embedding_ratio": cfg_reg.lr_embedding_layer_backbone / cfg_reg.lr_head,
            "tolerance": cfg_reg.tolerance,
            "batch_size": cfg_reg.batch_size,
            "base_model_path": cfg_reg.pretrained_path,
            "use_lora": getattr(cfg_reg, "use_lora", False),
            "lora_rank": getattr(cfg_reg, "lora_rank", 8),
            "lora_alpha": getattr(cfg_reg, "lora_alpha", 8.0),
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def upload_dataset(request):
    if "file" not in request.FILES:
        return JsonResponse({"error": "请上传 CSV 文件"}, status=400)

    uploaded = request.FILES["file"]
    if not uploaded.name.lower().endswith(".csv"):
        return JsonResponse({"error": "仅支持 CSV 文件"}, status=400)

    datasets_dir = Path(settings.MEDIA_ROOT) / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = str(uuid.uuid4())[:12]
    dest = datasets_dir / f"{dataset_id}_{Path(uploaded.name).name}"
    with open(dest, "wb") as f:
        for chunk in uploaded.chunks():
            f.write(chunk)

    try:
        for encoding in ("utf-8", "gbk", "latin-1"):
            try:
                df = pd.read_csv(dest, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            df = pd.read_csv(dest, encoding="utf-8", errors="replace")
    except Exception as e:
        return JsonResponse({"error": f"无法解析 CSV: {e}"}, status=400)

    columns = df.columns.tolist()
    smiles_col = detect_smiles_column(columns)
    if smiles_col is None:
        return JsonResponse({"error": "未找到 SMILES 列，请确保列名包含 smiles/smi"}, status=400)

    target_candidates = [c for c in columns if c != smiles_col]
    preview_rows = df.head(5).fillna("").astype(str).to_dict(orient="records")

    _datasets[dataset_id] = {
        "path": str(dest),
        "filename": uploaded.name,
        "smiles_column": smiles_col,
        "columns": columns,
        "row_count": len(df),
    }

    return JsonResponse(
        {
            "dataset_id": dataset_id,
            "filename": uploaded.name,
            "smiles_column": smiles_col,
            "target_columns": target_candidates,
            "columns": columns,
            "row_count": len(df),
            "preview": preview_rows,
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def upload_init_model(request):
    if "model" not in request.FILES:
        return JsonResponse({"error": "请上传模型文件"}, status=400)
    uploaded = request.FILES["model"]
    if not uploaded.name.lower().endswith(".pth"):
        return JsonResponse({"error": "仅支持 .pth 文件"}, status=400)

    models_dir = Path(settings.MEDIA_ROOT) / "init_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_id = str(uuid.uuid4())[:8]
    dest = models_dir / f"{model_id}_{Path(uploaded.name).name}"
    with open(dest, "wb") as f:
        for chunk in uploaded.chunks():
            f.write(chunk)
    _init_models[model_id] = str(dest)
    return JsonResponse({"model_id": model_id, "model_name": uploaded.name})


@csrf_exempt
@require_http_methods(["POST"])
def start_training(request):
    try:
        if request.content_type and "multipart" in request.content_type:
            data = request.POST
            init_model_file = request.FILES.get("init_model")
        else:
            data = json.loads(request.body)
            init_model_file = None

        dataset_id = data.get("dataset_id")
        target_column = data.get("target_column")
        task_type = data.get("task_type", "regression")
        init_mode = data.get("init_mode", "scratch")
        num_epochs = int(data.get("num_epochs", _default_epochs(task_type)))
        learning_rate = float(data.get("learning_rate", _default_lr()))
        init_model_id = data.get("init_model_id")

        if not dataset_id or dataset_id not in _datasets:
            return JsonResponse({"error": "无效的数据集，请重新上传"}, status=400)
        if not target_column:
            return JsonResponse({"error": "请选择目标列"}, status=400)
        if task_type not in ("regression", "classification"):
            return JsonResponse({"error": "task_type 须为 regression 或 classification"}, status=400)

        ds = _datasets[dataset_id]
        if target_column not in ds["columns"]:
            return JsonResponse({"error": "目标列不存在"}, status=400)

        checkpoint_path = None
        if init_mode == "checkpoint":
            if init_model_file:
                models_dir = Path(settings.MEDIA_ROOT) / "init_models"
                models_dir.mkdir(parents=True, exist_ok=True)
                tmp_id = str(uuid.uuid4())[:8]
                dest = models_dir / f"{tmp_id}_{Path(init_model_file.name).name}"
                with open(dest, "wb") as f:
                    for chunk in init_model_file.chunks():
                        f.write(chunk)
                checkpoint_path = str(dest)
            elif init_model_id and init_model_id in _init_models:
                checkpoint_path = _init_models[init_model_id]
            else:
                return JsonResponse({"error": "继续训练需上传 .pth 模型"}, status=400)
        elif init_mode == "pretrain":
            pass
        else:
            init_mode = "scratch"

        from .services.trainer import TrainParams

        job = create_job(task_type, num_epochs)
        out_dir = Path(settings.MEDIA_ROOT) / "training" / job.job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        params = TrainParams(
            csv_path=ds["path"],
            smiles_column=ds["smiles_column"],
            target_column=target_column,
            task_type=task_type,
            init_mode=init_mode,
            checkpoint_path=checkpoint_path,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            output_dir=str(out_dir),
            job_id=job.job_id,
            use_lora=data.get("use_lora", "false") in ("true", "True", True, "1"),
            lora_rank=int(data.get("lora_rank", 8)),
            lora_alpha=float(data.get("lora_alpha", 8.0)),
            lora_dropout=float(data.get("lora_dropout", 0.0)),
        )

        def on_epoch(epoch: int, total: int, train_loss: float, val_loss: float):
            progress = int(epoch / total * 100)
            job_ref = get_job(job.job_id)
            train_losses = list(job_ref.train_losses) if job_ref else []
            val_losses = list(job_ref.val_losses) if job_ref else []
            train_losses.append(round(train_loss, 6))
            val_losses.append(round(val_loss, 6))
            update_job(
                job.job_id,
                epoch=epoch,
                progress=progress,
                train_losses=train_losses,
                val_losses=val_losses,
                message=f"Epoch {epoch}/{total} — train {train_loss:.4f}, val {val_loss:.4f}",
                best_metric=val_loss,
            )

        def train_task():
            from .services.trainer import run_training

            model_path = run_training(params, on_epoch)
            update_job(job.job_id, model_path=model_path, progress=100)

        run_in_background(job.job_id, train_task)

        return JsonResponse({"job_id": job.job_id, "message": "训练已启动"})
    except json.JSONDecodeError:
        return JsonResponse({"error": "无效的 JSON"}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def job_status(request, job_id: str):
    job = get_job(job_id)
    if not job:
        return JsonResponse({"error": "任务不存在"}, status=404)
    return JsonResponse(job.to_dict())


@csrf_exempt
@require_http_methods(["GET"])
def download_model(request, job_id: str):
    job = get_job(job_id)
    if not job or not job.model_path:
        return JsonResponse({"error": "模型尚未就绪"}, status=404)
    path = Path(job.model_path)
    if not path.is_file():
        return JsonResponse({"error": "模型文件不存在"}, status=404)
    return FileResponse(
        path.open("rb"),
        as_attachment=True,
        filename=f"atomprop_model_{job_id}.pth",
    )
