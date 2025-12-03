from django.core.management.base import BaseCommand
from api.views import initialize_data
from rest_framework.test import APIRequestFactory

class Command(BaseCommand):
    help = 'Initialize scheduling database with sample data'

    def handle(self, *args, **kwargs):
        factory = APIRequestFactory()
        request = factory.post('/api/initialize-data/')
        response = initialize_data(request)
        
        if response.status_code in [200, 201]:
            self.stdout.write(self.style.SUCCESS(response.data['message']))
        else:
            self.stdout.write(self.style.ERROR(f"Error: {response.data}"))