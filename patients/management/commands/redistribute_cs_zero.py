"""
Django management command to redistribute CS patients with total_number_of_cs=0
if they exceed 13% of all CS patients.
"""

import random
from django.core.management.base import BaseCommand
from patients.models import Patient


class Command(BaseCommand):
    help = 'Redistribute CS patients with total_number_of_cs=0 if they exceed 13%'

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
        
        # Get all CS patients
        cs_patients = Patient.objects.filter(mode_of_delivery='cs')
        total_cs = cs_patients.count()
        
        if total_cs == 0:
            self.stdout.write(self.style.ERROR('No CS patients found in database'))
            return
        
        # Get patients with total_number_of_cs = 0
        cs_zero_patients = cs_patients.filter(total_number_of_cs='0')
        cs_zero_count = cs_zero_patients.count()
        
        # Calculate percentage
        current_percentage = (cs_zero_count / total_cs) * 100 if total_cs > 0 else 0
        
        self.stdout.write(f'Total CS patients: {total_cs}')
        self.stdout.write(f'CS patients with total_number_of_cs=0: {cs_zero_count}')
        self.stdout.write(f'Current percentage: {current_percentage:.2f}%')
        
        target_percentage = 13.0
        target_count = int(total_cs * target_percentage / 100)
        
        if current_percentage <= target_percentage:
            self.stdout.write(self.style.SUCCESS(
                f'Current percentage ({current_percentage:.2f}%) is already at or below target ({target_percentage}%). No changes needed.'
            ))
            return
        
        # Calculate how many need to be redistributed
        excess_count = cs_zero_count - target_count
        self.stdout.write(f'Target percentage: {target_percentage}%')
        self.stdout.write(f'Target count: {target_count}')
        self.stdout.write(f'Excess count to redistribute: {excess_count}')
        
        if excess_count <= 0:
            self.stdout.write(self.style.SUCCESS('No redistribution needed'))
            return
        
        # Get list of patients to redistribute
        patients_to_redistribute = list(cs_zero_patients[:excess_count])
        
        # Randomly assign new values (1, 2, or 3)
        redistribution_values = [1, 2, 3]
        updated_count = 0
        
        self.stdout.write(f'\nRedistributing {len(patients_to_redistribute)} patients...')
        
        for patient in patients_to_redistribute:
            # Randomly choose 1, 2, or 3
            new_value = random.choice(redistribution_values)
            
            if dry_run:
                self.stdout.write(
                    f'  Would update Patient ID {patient.id} (File: {patient.file_number}): '
                    f'total_number_of_cs from "0" to "{new_value}"'
                )
            else:
                patient.total_number_of_cs = str(new_value)
                patient.save(update_fields=['total_number_of_cs'])
                self.stdout.write(
                    f'  Updated Patient ID {patient.id} (File: {patient.file_number}): '
                    f'total_number_of_cs from "0" to "{new_value}"'
                )
            updated_count += 1
        
        # Verify the results
        if not dry_run:
            # Recalculate after update
            cs_zero_after = cs_patients.filter(total_number_of_cs='0').count()
            new_percentage = (cs_zero_after / total_cs) * 100 if total_cs > 0 else 0
            
            self.stdout.write(f'\n=== Results ===')
            self.stdout.write(f'Patients updated: {updated_count}')
            self.stdout.write(f'CS patients with total_number_of_cs=0 after: {cs_zero_after}')
            self.stdout.write(f'New percentage: {new_percentage:.2f}%')
            
            if new_percentage <= target_percentage:
                self.stdout.write(self.style.SUCCESS(
                    f'✓ Successfully redistributed! New percentage ({new_percentage:.2f}%) is at or below target ({target_percentage}%)'
                ))
            else:
                self.stdout.write(self.style.WARNING(
                    f'⚠ New percentage ({new_percentage:.2f}%) is still above target ({target_percentage}%). '
                    f'You may need to run this command again.'
                ))
        else:
            self.stdout.write(self.style.SUCCESS(
                f'\n✓ DRY RUN: Would update {updated_count} patients'
            ))
            self.stdout.write(self.style.WARNING(
                'Run without --dry-run to apply changes'
            ))

