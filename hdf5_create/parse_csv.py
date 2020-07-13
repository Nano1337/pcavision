import csv

"""Script that takes ProstateX-Findings-{Train,Test}.csv and adds per lesion its zone and clinsig 
(clinsig for train set only) information to the corrects rows in ProstateX-Images-{Train,Test}.csv, 
so all lesion information is inside ProstateX-Images-{Train,Test}-NEW.csv
Only has to be run once, so hdf5 conversion later on needs to draw from just one .csv file"""

train_set = False  # Denotes whether we're building new .csv for train or test files.

# # Paths for train set
# images_csv = 'C:\\Users\\haoli\\Downloads\\ProstateX_Data\\ProstateX-TrainingLesionInformationv2\\ProstateX-Images-Train.csv'
# findings_csv = 'C:\\Users\\haoli\\Downloads\\ProstateX_Data\\ProstateX-TrainingLesionInformationv2\\ProstateX-Findings-Train.csv'

# Paths for test set
images_csv = 'C:\\Users\\haoli\\Downloads\\ProstateX_Data\\ProstateX-TestLesionInformation\\ProstateX-Images-Test.csv'

findings_csv = 'C:\\Users\\haoli\\Downloads\\ProstateX_Data\\ProstateX-TestLesionInformation\\ProstateX-Findings-Test.csv'

with open(images_csv, 'r') as images_train:
    with open(findings_csv, 'r') as findings_train:
        new_rows = []
        reader_images = csv.reader(images_train, delimiter=',')
        reader_findings = csv.reader(findings_train, delimiter=',')
        column_names = next(reader_images)
        column_names.append('Zone')
        column_names.append('ClinSig')
        next(reader_findings)
        all_findings_rows = []
        for row in reader_findings:
            all_findings_rows.append(row)

        for images_row in reader_images:
            for findings_row in all_findings_rows:
                if images_row[3].strip() == findings_row[2].strip():  # Match on position
                    print(images_row[0], images_row[3], findings_row[2])
                    images_row.append(findings_row[3])  # Add zone
                    images_row[2] = findings_row[1]  # Fix for non-unique finding ID per patient
                    if train_set:
                        images_row.append(findings_row[4])  # Add clinsig
                    else:
                        images_row.append('NA')
                    new_rows.append(images_row)

# Write new .csv file
if train_set:
    output_name = 'ProstateX-Images-Train-NEW.csv'
else:
    output_name = 'ProstateX-Images-Test-NEW.csv'

with open(output_name, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=column_names, lineterminator = '\n')
    writer.writeheader()
    for row in new_rows:
        new_row = {}
        for i in range(len(row)):
            new_row[column_names[i]] = row[i]
        writer.writerow(new_row)
