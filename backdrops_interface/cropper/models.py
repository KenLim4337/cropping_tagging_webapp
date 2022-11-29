from django.db import models

# Create your models here.

class Images(models.Model):
    photo_id = models.CharField(max_length=500)
    file_path = models.CharField(max_length=500)
    image_embeddings_full_image = models.TextField(null=True)
    image_embeddings_background_image = models.TextField(null=True)
    # embeddings, keypoints, descriptors, phash