from django.contrib import admin
from .models import Movie,Rating

@admin.register(Movie)
class movieAdmin(admin.ModelAdmin):
    list_display=('id','title','genres','year','image','movieduration' )
                  

@admin.register(Rating)
class ratingAdmin(admin.ModelAdmin):
    list_display = ( 'user','movie','rating','rated_date',)
# Register your models here.
