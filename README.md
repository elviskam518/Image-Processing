# Restoring Driving Images for Rain and Snow Classification

Can classical image processing make corrupted driving images clearer and more useful to an existing classifier?

For this image-processing coursework project, I built an **OpenCV restoration pipeline** for 100 supplied driving images: 50 labelled snow and 50 labelled rain. The work combines geometric correction, missing-region repair, sharpening, denoising, brightness adjustment and colour balancing.

My contribution is the preprocessing pipeline and its experimental evaluation. The pretrained classifier and evaluation script are supplied coursework components.

**[Read the coursework report](cqst66%20.pdf)** · **[Browse processed images](Results/)**

## What I implemented

The current `main.py` applies the following sequence:

```text
Corrupted image
  → inpaint dark / missing regions
  → correct perspective warping
  → sharpen
  → median and bilateral filtering
  → adaptive gamma correction
  → Gray World white balance
  → save restored image
```

- **Inpainting:** constructs a mask from pixels with low grayscale intensity and uses OpenCV's Navier–Stokes method.
- **Perspective correction:** uses four fixed source points for the shared distortion in the coursework images.
- **Sharpening:** applies unsharp masking by combining the image with a Gaussian-blurred version.
- **Denoising:** combines a 3 × 3 median filter with bilateral filtering.
- **Brightness correction:** chooses a bounded gamma adjustment from the image's average luminance.
- **White balance:** rescales colour channels using the Gray World assumption.

The script processes a directory of images and saves the results with their original filenames. An optional helper plots the combined luminance histogram.

## Experimental findings from the report

The report compares processing choices using the supplied classifier:

| Comparison | Reported accuracy |
| --- | ---: |
| Without sharpening | 89% |
| Sharpening before denoising | **93%** |
| Sharpening after denoising | 87% |
| Without colour balancing | 88% |
| With colour balancing | **93%** |

These are separate configuration comparisons, not cumulative improvements. The denoising sweep also contains a 94% configuration; the report selects a 93% configuration as a balance between noise suppression and classification performance.

The results illustrate that the **order and strength of processing operations matter**, and that visually smoother images do not necessarily produce better classification.

The report describes warping before inpainting, while the submitted code performs inpainting first. The pipeline above follows the actual code. Reported scores have not been reproduced for this README.

## Skills demonstrated

**Python · OpenCV · NumPy · Matplotlib · Classical image restoration · Parameter comparison · Downstream evaluation**

The project demonstrates constructing a multi-stage processing pipeline, examining intermediate effects and using both visual inspection and a fixed classifier to evaluate design choices.

## Scope and limitations

The fixed perspective points and dark-region mask are tailored to the supplied corruption pattern. Naturally dark image areas may be mistaken for missing regions, and the same settings may not generalise to unrelated images.

The experiments use a small, fixed coursework dataset. Classification accuracy measures usefulness to the supplied model; it is not a complete measure of restoration quality or evidence of performance in a deployed driving system.

## Repository guide

| File or folder | Purpose |
| --- | --- |
| [`main.py`](main.py) | Restoration pipeline and batch processing. |
| [`driving_images/`](driving_images/) | 100 input images. |
| [`Results/`](Results/) | Included processed images. |
| [`classify.py`](classify.py) | Supplied evaluation script, credited to Amir Atapour Abarghouei in its header. |
| [`classifier.model`](classifier.model) | Supplied pretrained classifier loaded with OpenCV DNN. |
| [`cqst66 .pdf`](cqst66%20.pdf) | Report with processing rationale, comparisons and examples. |

<details>
<summary>Running the coursework pipeline</summary>

Install `opencv-python`, `numpy` and `matplotlib`, then run:

```bash
python main.py driving_images
python classify.py --data=Results --model=classifier.model
```

The first command writes images to `Results/`, replacing matching filenames. The second evaluates them with the supplied model.

For comparison, evaluate the inputs with:

```bash
python classify.py --data=driving_images --model=classifier.model
```

</details>
