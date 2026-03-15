const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api';

export interface SegmentationResult {
  imageUrl: string;
  segmentedImageUrl: string;
  chromocenterMask: string; // Base64 encoded mask image
  nucleiMask: string; // Base64 encoded mask image
  backgroundMask: string; // Base64 encoded mask image
  modelSource?: string;
}

export interface ProcessedImage {
  file: File;
  fileName: string;
  result: SegmentationResult;
}

/**
 * Upload and process an image through the segmentation model
 * @param file - The image file to process
 * @returns Promise with segmentation results
 */
export async function processImage(file: File): Promise<SegmentationResult> {
  const formData = new FormData();
  formData.append('file', file);

  const [localImageUrl, response] = await Promise.all([
    readFileAsDataUrl(file),
    fetch(`${BASE_URL}/segment`, {
      method: 'POST',
      body: formData,
    }),
  ]);

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || 'Segmentation failed');
  }

  const imageUrl = typeof data.original_base64 === 'string' && data.original_base64.length > 0
    ? ensureDataUrl(data.original_base64)
    : localImageUrl;

  // Prefer the 3-class semantic mask; fall back to legacy binary mask_base64
  const rawMask = typeof data.semantic_mask_base64 === 'string' && data.semantic_mask_base64.length > 0
    ? data.semantic_mask_base64
    : data.mask_base64;
  const semanticDataUrl = ensureDataUrl(rawMask);

  // Semantic mask pixel values:
  //   0   → background (outside nucleus)
  //   128 → nucleoplasm (inside nucleus, not chromocenter)
  //   255 → chromocenter
  //
  // When a grayscale PNG is drawn to a canvas, each pixel's R channel equals
  // the grayscale value, so we threshold on imageData.data[i] (the R channel).
  const chromocenterMask  = await colorizeMask(semanticDataUrl, [226, 72, 12],   (r) => r >= 200);
  const nucleiMask        = await createBlankMask(semanticDataUrl);
  const backgroundMask    = await createBlankMask(semanticDataUrl);
  const segmentedImageUrl = await overlayMaskOnImage(imageUrl, semanticDataUrl);

  return {
    imageUrl,
    segmentedImageUrl,
    chromocenterMask,
    nucleiMask,
    backgroundMask,
    modelSource: data.model_source,
  };
}

/**
 * Update/save edited masks back to the backend
 * @param fileName - Name of the file being edited
 * @param masks - Updated mask data
 */
export async function updateMasks(
  fileName: string,
  masks: {
    chromocenter: string;
    nuclei: string;
    background: string;
  }
): Promise<void> {
  // No backend persistence endpoint exists yet for edited masks.
  console.log('Saving masks for:', fileName, masks);
  return Promise.resolve();
}

/**
 * Process multiple images in batch
 * @param files - Array of image files
 * @param onProgress - Callback for progress updates
 */
export async function processBatch(
  files: File[],
  onProgress?: (current: number, total: number) => void
): Promise<ProcessedImage[]> {
  const results: ProcessedImage[] = [];
  
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const result = await processImage(file);
    results.push({
      file,
      fileName: file.name,
      result,
    });
    
    if (onProgress) {
      onProgress(i + 1, files.length);
    }
  }
  
  return results;
}

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error('Failed to read image file'));
    reader.readAsDataURL(file);
  });
}

function colorizeMask(maskDataUrl: string, color: [number, number, number], match: (r: number) => boolean): Promise<string> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = image.width;
      canvas.height = image.height;
      const context = canvas.getContext('2d');

      if (!context) {
        reject(new Error('Unable to create mask canvas'));
        return;
      }

      context.drawImage(image, 0, 0);
      const imageData = context.getImageData(0, 0, canvas.width, canvas.height);

      for (let index = 0; index < imageData.data.length; index += 4) {
        const shouldFill = match(imageData.data[index]);

        imageData.data[index]     = shouldFill ? color[0] : 0;
        imageData.data[index + 1] = shouldFill ? color[1] : 0;
        imageData.data[index + 2] = shouldFill ? color[2] : 0;
        imageData.data[index + 3] = shouldFill ? 180 : 0;
      }

      context.putImageData(imageData, 0, 0);
      resolve(canvas.toDataURL('image/png'));
    };
    image.onerror = () => reject(new Error('Unable to load returned mask image'));
    image.src = maskDataUrl;
  });
}

function ensureDataUrl(maskBase64OrDataUrl: string): string {
  if (maskBase64OrDataUrl.startsWith('data:image')) {
    return maskBase64OrDataUrl;
  }

  return `data:image/png;base64,${maskBase64OrDataUrl}`;
}

function overlayMaskOnImage(imageDataUrl: string, maskDataUrl: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const mask = new Image();

    let imageLoaded = false;
    let maskLoaded = false;

    const tryCompose = () => {
      if (!imageLoaded || !maskLoaded) {
        return;
      }

      const canvas = document.createElement('canvas');
      canvas.width = image.width;
      canvas.height = image.height;
      const context = canvas.getContext('2d');

      if (!context) {
        resolve(maskDataUrl);
        return;
      }

      context.drawImage(image, 0, 0);
      context.globalAlpha = 0.45;
      context.drawImage(mask, 0, 0, canvas.width, canvas.height);
      context.globalAlpha = 1;

      resolve(canvas.toDataURL('image/png'));
    };

    image.onload = () => {
      imageLoaded = true;
      tryCompose();
    };
    image.onerror = () => {
      // Do not fail processing if preview composition cannot be generated.
      resolve(maskDataUrl);
    };

    mask.onload = () => {
      maskLoaded = true;
      tryCompose();
    };
    mask.onerror = () => resolve(maskDataUrl);

    image.src = imageDataUrl;
    mask.src = maskDataUrl;
  });
}

function createBlankMask(referenceDataUrl: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = image.width;
      canvas.height = image.height;
      const context = canvas.getContext('2d');

      if (!context) {
        reject(new Error('Unable to create blank mask canvas'));
        return;
      }

      context.clearRect(0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL('image/png'));
    };
    image.onerror = () => reject(new Error('Unable to load reference mask image'));
    image.src = referenceDataUrl;
  });
}
