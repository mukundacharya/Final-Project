from os import name
from django.urls import path
from . import views

from django.views.decorators.csrf import csrf_exempt

urlpatterns=[
path('',views.index,name='index'),
path('webcam_feed',views.webcam_feed,name='webcam_feed'),
path('ip_feed',views.ip_feed,name='ip_feed'),
path('login',views.login,name='login'),
path('logout',views.logout,name='logout'),
path('dashboard',views.dashboard,name='dashboard'),
path('ViewStudents',views.ViewStudents,name='ViewStudents'),
path('studentRegister',views.studentRegister,name='studentRegister'),
path('ViewAllCameras',views.ViewAllCameras,name='ViewAllCameras'),
path('ViewIPStream',views.ViewIPStream,name='ViewIPStream'),
path('ViewWebcamStream',views.ViewWebcamStream,name='ViewWebcamStream'),
path('DeleteStudent/<usn>',views.DeleteStudent,name='DeleteStudent'),
path('ViewStudentDetails/<usn>',views.ViewStudentDetails,name='ViewStudentDetails'),
path('SearchStudent',csrf_exempt(views.SearchStudent),name='SearchStudent'),
path('notifications',views.notifications,name='notifications'),
path('ViolatorDetails/<usn>',views.ViolatorDetails,name='ViolatorDetails'),
path('webcam1',views.webcam1,name='webcam1'),
path('webcam2',views.webcam2,name='webcam2'),
]
