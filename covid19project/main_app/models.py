from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import os
import datetime

# Create your models here.

class StudentDetails(models.Model):
    usn=models.CharField(primary_key=True,max_length=10)
    first_name=models.CharField(max_length=50)
    last_name=models.CharField(max_length=50)
    gender=models.CharField(max_length=15)
    dob=models.DateField()
    phone=models.CharField(max_length=10)
    emailid=models.CharField(max_length=50)
    branch=models.CharField(max_length=50)
    section=models.CharField(max_length=10)
    photo=models.ImageField(upload_to="pics/", null=False, blank=True)

    def __str__(self):
        return self.usn

    def delete(self, *args, **kwargs):
        # object is being removed from db, remove the file from storage first
        self.photo.delete()
        return super(StudentDetails, self).delete(*args, **kwargs)


class Violators(models.Model):
    usn=models.CharField(max_length=10,default="UNKNOWN")
    timestamp=models.DateTimeField(default=timezone.now)
    status=models.CharField(max_length=10,default='unknown')
    pic=models.FileField(upload_to='violators/', null=False)

    def delete(self, *args, **kwargs):
        # object is being removed from db, remove the file from storage first
        self.pic.delete()
        return super(Violators, self).delete(*args, **kwargs)

    # def dateonly(self):
    #     return self.timestamp.strftime('%B %d %Y')
    #
    # def timeonly(self):
    #     return self.timestamp.strftime("%H:%M:%S")
