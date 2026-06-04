import json
import uuid
from pathlib import Path

from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .parsers import parse_smiles_from_file, parse_smiles_from_text
from .services.predictor import predict_smiles

# In-memory session: model_id -> path (single-user local tool)
_loaded_models: dict[str, dict] = {}


def _current_model() -> Path | None:
    default_id = getattr(settings, "DEFAULT_MODEL_ID", None)
    if default_id and default_id in _loaded_models:
        return Path(_loaded_models[default_id]["path"])
    if _loaded_models:
        first = next(iter(_loaded_models.values()))
        return Path(first["path"])
    return None


@csrf_exempt
@require_http_methods(["GET"])
def health(request):
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model = _current_model()
    return JsonResponse(
        {
            "status": "ok",
            "device": device,
            "model_loaded": model is not None,
            "model_name": model.name if model else None,
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def upload_model(request):
    if "model" not in request.FILES:
        return JsonResponse({"error": "No model file provided"}, status=400)
    uploaded = request.FILES["model"]
    if not uploaded.name.lower().endswith(".pth"):
        return JsonResponse({"error": "Only .pth model files are supported"}, status=400)

    models_dir = Path(settings.MEDIA_ROOT) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_id = str(uuid.uuid4())[:8]
    safe_name = Path(uploaded.name).name
    dest = models_dir / f"{model_id}_{safe_name}"
    with open(dest, "wb") as f:
        for chunk in uploaded.chunks():
            f.write(chunk)

    _loaded_models[model_id] = {
        "path": str(dest),
        "name": safe_name,
        "id": model_id,
    }
    settings.DEFAULT_MODEL_ID = model_id  # type: ignore[attr-defined]

    return JsonResponse(
        {
            "model_id": model_id,
            "model_name": safe_name,
            "message": "Model loaded successfully",
        }
    )


@csrf_exempt
@require_http_methods(["GET"])
def model_status(request):
    model = _current_model()
    models = [
        {"id": m["id"], "name": m["name"], "active": m["id"] == getattr(settings, "DEFAULT_MODEL_ID", None)}
        for m in _loaded_models.values()
    ]
    return JsonResponse(
        {
            "loaded": model is not None,
            "model_name": model.name if model else None,
            "models": models,
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def predict(request):
    model_path = _current_model()
    if model_path is None:
        return JsonResponse({"error": "Please upload a model (.pth) first"}, status=400)

    try:
        if request.content_type and "application/json" in request.content_type:
            body = json.loads(request.body)
            smiles_list = body.get("smiles", [])
            if isinstance(smiles_list, str):
                smiles_list = parse_smiles_from_text(smiles_list)
        else:
            text = request.POST.get("smiles", "")
            smiles_list = parse_smiles_from_text(text) if text else []

        if not smiles_list:
            return JsonResponse({"error": "No SMILES provided"}, status=400)

        results = predict_smiles(smiles_list, model_path)
        return JsonResponse(
            {
                "results": results,
                "count": len(results),
                "input_count": len(smiles_list),
            }
        )
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def predict_file(request):
    model_path = _current_model()
    if model_path is None:
        return JsonResponse({"error": "Please upload a model (.pth) first"}, status=400)

    if "file" not in request.FILES:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    uploaded = request.FILES["file"]
    name = uploaded.name.lower()
    if not (name.endswith(".txt") or name.endswith(".csv")):
        return JsonResponse({"error": "Only .txt and .csv files are supported"}, status=400)

    try:
        smiles_list = parse_smiles_from_file(uploaded, uploaded.name)
        if not smiles_list:
            return JsonResponse({"error": "No valid SMILES found in file"}, status=400)

        results = predict_smiles(smiles_list, model_path)
        return JsonResponse(
            {
                "results": results,
                "count": len(results),
                "input_count": len(smiles_list),
                "source_file": uploaded.name,
            }
        )
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def export_csv(request):
    try:
        body = json.loads(request.body)
        results = body.get("results", [])
        if not results:
            return JsonResponse({"error": "No results to export"}, status=400)

        import csv
        import io

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=["SMILES", "predicted_value"])
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "SMILES": row.get("smiles", row.get("SMILES", "")),
                    "predicted_value": row.get("predicted_value", ""),
                }
            )

        response = HttpResponse(buffer.getvalue(), content_type="text/csv; charset=utf-8")
        response["Content-Disposition"] = 'attachment; filename="predictions.csv"'
        return response
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)
