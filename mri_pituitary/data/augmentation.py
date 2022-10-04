import albumentations as albu


# Train transforms
TRN_TFMS = albu.Compose([
    albu.HorizontalFlip(),
    albu.Rotate(10),
    albu.Normalize(),
    ToTensor(),
])

# Test transforms
VAL_TFMS = albu.Compose([
    albu.Normalize(),
    ToTensor(),
])

