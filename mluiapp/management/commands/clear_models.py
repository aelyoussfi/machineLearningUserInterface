from django.core.management.base import BaseCommand
from mluiapp.models import inferenceJob, fittingJob

class Command(BaseCommand):
    def handle(self, *args, **options):
        inferenceJob.objects.all().delete()
        fittingJob.objects.all().delete()