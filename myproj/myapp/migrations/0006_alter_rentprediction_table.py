# Generated by Django 4.2.7 on 2023-12-13 16:05

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0005_rentprediction_delete_prediction'),
    ]

    operations = [
        migrations.AlterModelTable(
            name='rentprediction',
            table='rent_prediction',
        ),
    ]
