"""
Django management command to fix instrumental_delivery data.
For each month, ensures maximum 7 instrumental deliveries.
If a month has more than 7, randomly selects 7 to keep as True, sets rest to False.
"""

import random
from django.core.management.base import BaseCommand
from django.db.models import Count
from django.db.models.functions import TruncMonth
from patients.models import Patient


class Command(BaseCommand):
    help = 'Fix instrumental_delivery: max 7 per month, randomly select if more than 7'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be changed without actually updating the database',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        
        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN MODE - No changes will be saved'))
        
        # Get all patients with instrumental_delivery=True, grouped by month
        instrumental_patients = Patient.objects.filter(
            instrumental_delivery=True,
            time_of_admission__isnull=False
        ).annotate(
            month=TruncMonth('time_of_admission')
        )
        
        # Group by month and count
        monthly_counts = (
            instrumental_patients
            .values('month')
            .annotate(count=Count('id'))
            .order_by('month')
        )
        
        total_updated = 0
        months_processed = 0
        
        self.stdout.write(f'\nProcessing {monthly_counts.count()} months...\n')
        
        for month_data in monthly_counts:
            month = month_data['month']
            count = month_data['count']
            
            if month is None:
                continue
            
            month_str = month.strftime('%Y-%m')
            
            # If count is 2-7, no changes needed
            if 2 <= count <= 7:
                self.stdout.write(
                    f'Month {month_str}: {count} instrumental deliveries (OK, within range 2-7)'
                )
                continue
            
            # If count < 2, leave as is (user didn't specify what to do)
            if count < 2:
                self.stdout.write(
                    f'Month {month_str}: {count} instrumental deliveries (below minimum 2, leaving as is)'
                )
                continue
            
            # If count > 7, need to randomly select between 2-7 and set rest to False
            if count > 7:
                # Randomly choose how many to keep (between 2 and 7)
                target_count = random.randint(2, 7)
                excess_count = count - target_count
                self.stdout.write(
                    f'Month {month_str}: {count} instrumental deliveries (EXCEEDS MAX 7, randomly keeping {target_count}, reducing by {excess_count})'
                )
                
                # Get all patients for this month with instrumental_delivery=True
                month_patients = list(
                    instrumental_patients.filter(month=month)
                )
                
                # Randomly shuffle to ensure random selection
                random.shuffle(month_patients)
                
                # Keep first target_count as True, set rest to False
                keep_patients = month_patients[:target_count]
                remove_patients = month_patients[target_count:]
                
                self.stdout.write(f'  Keeping {len(keep_patients)} patients as True')
                self.stdout.write(f'  Setting {len(remove_patients)} patients to False')
                
                # Update patients to set instrumental_delivery=False
                for patient in remove_patients:
                    if dry_run:
                        self.stdout.write(
                            f'    Would update Patient ID {patient.id} (File: {patient.file_number}): '
                            f'instrumental_delivery from True to False'
                        )
                    else:
                        patient.instrumental_delivery = False
                        patient.save(update_fields=['instrumental_delivery'])
                        self.stdout.write(
                            f'    Updated Patient ID {patient.id} (File: {patient.file_number}): '
                            f'instrumental_delivery from True to False'
                        )
                    total_updated += 1
                
                months_processed += 1
        
        # Summary
        self.stdout.write(f'\n=== Summary ===')
        self.stdout.write(f'Months processed: {months_processed}')
        self.stdout.write(f'Total patients updated: {total_updated}')
        
        if dry_run:
            self.stdout.write(self.style.SUCCESS(
                f'\n✓ DRY RUN: Would update {total_updated} patients'
            ))
            self.stdout.write(self.style.WARNING(
                'Run without --dry-run to apply changes'
            ))
        else:
            # Verify results
            final_monthly_counts = (
                Patient.objects.filter(
                    instrumental_delivery=True,
                    time_of_admission__isnull=False
                )
                .annotate(month=TruncMonth('time_of_admission'))
                .values('month')
                .annotate(count=Count('id'))
                .order_by('month')
            )
            
            max_count = 0
            months_over_limit = 0
            for month_data in final_monthly_counts:
                if month_data['month'] and month_data['count'] > max_count:
                    max_count = month_data['count']
                if month_data['month'] and month_data['count'] > 7:
                    months_over_limit += 1
            
            if max_count <= 7 and months_over_limit == 0:
                self.stdout.write(self.style.SUCCESS(
                    f'\n✓ Successfully fixed! Maximum instrumental deliveries per month: {max_count} (target: ≤7)'
                ))
            else:
                self.stdout.write(self.style.WARNING(
                    f'\n⚠ Maximum instrumental deliveries per month: {max_count}'
                ))
                if months_over_limit > 0:
                    self.stdout.write(self.style.WARNING(
                        f'⚠ {months_over_limit} months still exceed the limit. You may need to run this command again.'
                    ))

