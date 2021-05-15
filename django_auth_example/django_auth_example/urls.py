
from django.conf.urls import url, include
from django.contrib import admin
from users import views
from users.views import insert
urlpatterns = [
    url(r'^admin/', admin.site.urls),
    # 别忘记在顶部引入 include 函数
    url(r'^users/', include('users.urls')),
    url(r'^users/', include('django.contrib.auth.urls')),
    url(r'^$', views.index, name='index'),
    url(r'^insert/$', insert),
    url(r'^users/recommend1/$', views.recommend1),
    url(r'^users/recommend2/$', views.recommend2),
    url(r'^users/recommend1/users/recommend1/recommend2/$', views.recommend2),
    # url(r'^users/showmessage/$', views.showmessage),
]
