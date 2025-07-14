// src/utils/image-processor.ts
import { Tensor } from '../lib/tensor';

/**
 * Builds a vocabulary from a list of image labels.
 * @param labels Array of string labels.
 * @returns An object containing the list of unique labels and mappings.
 */
export function buildImageVocabulary(labels: string[]): { labels: string[]; labelToIndex: Map<string, number>; indexToLabel: Map<number, string>; numClasses: number } {
  const uniqueLabels = Array.from(new Set(labels)).sort();
  const labelToIndex = new Map(uniqueLabels.map((label, i) => [label, i]));
  const indexToLabel = new Map(uniqueLabels.map((label, i) => [i, label]));
  return { labels: uniqueLabels, labelToIndex, indexToLabel, numClasses: uniqueLabels.length };
}

/**
 * Converts a data URL of an image into a Tensor.
 * Resizes the image and normalizes pixel values.
 * @param dataUrl The data URL of the image.
 * @param width The target width.
 * @param height The target height.
 * @returns A promise that resolves to a Tensor of shape [1, channels, height, width].
 */
export function imageToTensor(dataUrl: string, width: number, height: number): Promise<Tensor> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.src = dataUrl;
    img.crossOrigin = 'Anonymous';
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        return reject(new Error('Could not get canvas context'));
      }
      ctx.drawImage(img, 0, 0, width, height);
      const imageData = ctx.getImageData(0, 0, width, height);
      const { data } = imageData; // data is a Uint8ClampedArray: [R, G, B, A, R, G, B, A, ...]

      const numChannels = 3; // RGB
      const tensorData = new Float32Array(numChannels * height * width);
      
      // Reshape from [H, W, C] to [C, H, W] and normalize
      for (let c = 0; c < numChannels; c++) {
        for (let h = 0; h < height; h++) {
          for (let w = 0; w < width; w++) {
            const sourceIndex = (h * width + w) * 4; // 4 because of RGBA
            const targetIndex = c * (height * width) + h * width + w;
            tensorData[targetIndex] = data[sourceIndex + c] / 255.0; // Normalize to [0, 1]
          }
        }
      }

      // The batch dimension is added when creating batches.
      resolve(new Tensor(tensorData, [numChannels, height, width]));
    };
    img.onerror = (err) => {
      reject(new Error('Failed to load image: ' + err));
    };
  });
}

/**
 * Creates batches from sequences of image and target tensors.
 * @param inputTensors Array of image tensors.
 * @param targetTensors Array of target tensors (one-hot).
 * @param batchSize The desired batch size.
 * @returns An array of batches.
 */
export function createImageBatches(inputTensors: Tensor[], targetTensors: Tensor[], batchSize: number): { inputs: Tensor, targets: Tensor }[] {
  if (inputTensors.length !== targetTensors.length) {
    throw new Error("Input and target tensors must have the same length.");
  }
  const batches = [];
  const totalItems = inputTensors.length;

  // Shuffle data
  const indices = Array.from({ length: totalItems }, (_, i) => i);
  for (let i = totalItems - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  for (let i = 0; i < totalItems; i += batchSize) {
    const batchIndices = indices.slice(i, i + batchSize);
    const actualBatchSize = batchIndices.length;

    if (actualBatchSize === 0) continue;

    const firstInput = inputTensors[batchIndices[0]];
    const firstTarget = targetTensors[batchIndices[0]];

    const [channels, height, width] = firstInput.shape;
    const numClasses = firstTarget.shape[1];

    const batchedInputData = new Float32Array(actualBatchSize * channels * height * width);
    const batchedTargetData = new Float32Array(actualBatchSize * numClasses);
    
    for (let j = 0; j < actualBatchSize; j++) {
      const originalIndex = batchIndices[j];
      const input = inputTensors[originalIndex];
      const target = targetTensors[originalIndex];
      
      batchedInputData.set(input.data, j * input.size);
      batchedTargetData.set(target.data, j * target.size);
    }
    
    const batchedInputTensor = new Tensor(batchedInputData, [actualBatchSize, channels, height, width]);
    const batchedTargetTensor = new Tensor(batchedTargetData, [actualBatchSize, numClasses]);

    batches.push({ inputs: batchedInputTensor, targets: batchedTargetTensor });
  }
  
  return batches;
}
