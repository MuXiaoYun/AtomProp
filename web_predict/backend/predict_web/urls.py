from pathlib import Path

from django.conf import settings
from django.conf.urls.static import static
from django.http import FileResponse, HttpResponse
from django.urls import include, path, re_path
from django.views.static import serve as static_serve


def spa_index(_request):
    index = Path(settings.BASE_DIR) / "static" / "dist" / "index.html"
    if index.is_file():
        return FileResponse(index.open("rb"), content_type="text/html; charset=utf-8")
    return HttpResponse(
        "<h1>AtomProp Web</h1><p>请先构建前端: <code>cd web_predict/frontend && npm install && npm run build</code></p>"
        "<p>开发模式请同时运行 Vite (npm run dev) 与 Django (python manage.py runserver)。</p>",
        content_type="text/html; charset=utf-8",
        status=503,
    )


urlpatterns = [
    path("api/", include("api.urls")),
    re_path(
        r"^assets/(?P<path>.*)$",
        static_serve,
        {"document_root": Path(settings.BASE_DIR) / "static" / "dist" / "assets"},
    ),
    path("", spa_index, name="index"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
