from django.shortcuts import render,redirect,get_object_or_404
from django.views import View
from django.contrib import messages
from django.http import HttpResponse
from .analyzer import AnalyzeEngine
from .models import Category,Comment
import json
# Create your views here.

def index(request):
    return redirect('analyze')

class Analyze(View):
    def get(self,request, *args, **kwargs):
        cs = Comment.objects.all()
        comments=[]
        for comment in cs:
            phrases = []
            categories=[]
            sentiment=0
            if comment.analyze :
                analyze= json.loads(comment.analyze)
                sentiment=analyze['sentiment']
                data= analyze['phrases']
                for i,d in enumerate(data):
                    categories.append({"index":i,"text":d['category'],"sentiment":d['sentiment']})
                    phrases.append({"index":i,"text":d['phrase'].replace(d['root'],'<b style="text-decoration:underline">'+d['root']+'</b>'),"sentiment":d['sentiment']})
            comments.append({
                "id":comment.id,
                "phrases":phrases,
                "categories":categories,
                "sentiment":sentiment,
                "text":comment.text
            })
        return render(request, 'analyzer/dashboard/index.html',{'comments':comments[::-1]})

    def post(self, request, *args, **kwargs):
        text = request.POST.get('text')
        if not text:
            messages.error(request, "text field is required")
        else:
            try:
                categories=[c.name for c in Category.objects.all()]
                engine = AnalyzeEngine()
                res = engine.analyze([text],categories)[0]
                sentiment = engine.get_sentiment(text)
                Comment.objects.create(text=text,analyze=json.dumps({
                    'sentiment':sentiment,
                    'phrases':res
                }))
                messages.success(request, "comment analyzed you can see result at top of list.")
            except Exception as e:
                messages.error(request, "error in analyzing :\n"+str(e))
        return redirect('index')

class CategoriesList(View):
    def get(self, request, *args, **kwargs):
        clusters = Category.objects.all()
        return render(request, 'analyzer/categories/index.html',{'clusters':clusters})

class CategoriesAdd(View):
    def post(self, request, *args, **kwargs):
        name = request.POST.get('category')
        if not name:
            messages.error(request, "name is required")
        else:
            try:
                category = Category.objects.create(name=name)
                messages.success(request, "cluster saved.")
            except Exception as e:
                messages.error(request, "error in adding new cluster :\n"+str(e))
        return redirect('categories')

class CategoriesDelete(View):
    def get(self, request,id=None,*args, **kwargs):
        category = get_object_or_404(Category,pk=id)
        try:
            category.delete()
        except Exception as e:
            messages.error(request,"error in removing cluster :\n"+str(e))
        return redirect('categories')