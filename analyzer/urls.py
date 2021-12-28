from django.urls import path,include
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='index'),
    path('analyze', views.Analyze.as_view(), name='analyze'),
    path('categories', views.CategoriesList.as_view(), name='categories'),
    path('category/add', views.CategoriesAdd.as_view(), name='category_add'),
    path('category/remove/<int:id>', views.CategoriesDelete.as_view(), name='category_remove'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

