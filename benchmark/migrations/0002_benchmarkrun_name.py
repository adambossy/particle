# Generated by Django 5.1.7 on 2025-03-11 19:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('benchmark', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='benchmarkrun',
            name='name',
            field=models.CharField(blank=True, help_text='Random word identifier for this benchmark run', max_length=100),
        ),
    ]
