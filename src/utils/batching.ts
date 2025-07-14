// src/utils/batching.ts
import { Tensor } from '../lib/tensor';

// Plain object representation of a Tensor for worker communication
type PlainTensor = {
  data: Float32Array;
  shape: number[];
};

type Batch = {
  inputs: PlainTensor;
  targets: PlainTensor;
};

/**
 * Creates batches from sequences of text input and target tensors.
 * This function is designed to be self-contained and used within a worker.
 * @param inputTensors Array of input tensors.
 * @param targetTensors Array of target tensors.
 * @param batchSize The desired batch size.
 * @returns An array of batches (plain objects for worker compatibility).
 */
export function createTextBatches(inputTensors: Tensor[], targetTensors: Tensor[], batchSize: number): Batch[] {
  const batches = [];
  const totalSteps = inputTensors.length;

  for (let i = 0; i < totalSteps; i += batchSize) {
    const currentInputBatch = inputTensors.slice(i, i + batchSize);
    const currentTargetBatch = targetTensors.slice(i, i + batchSize);
    const actualBatchSize = currentInputBatch.length;

    if (actualBatchSize === 0) continue;

    const batchedInputData = new Float32Array(actualBatchSize);
    for (let j = 0; j < actualBatchSize; j++) {
      batchedInputData[j] = currentInputBatch[j].data[0];
    }

    const vocabSize = currentTargetBatch[0].shape[1];
    const batchedTargetData = new Float32Array(actualBatchSize * vocabSize);
    for (let j = 0; j < actualBatchSize; j++) {
      batchedTargetData.set(currentTargetBatch[j].data, j * vocabSize);
    }

    batches.push({
      inputs: { data: batchedInputData, shape: [actualBatchSize, 1] },
      targets: { data: batchedTargetData, shape: [actualBatchSize, vocabSize] },
    });
  }
  return batches;
}

/**
 * Creates batches from sequences of image and target tensors.
 * Shuffles the data before batching.
 * @param inputTensors Array of image tensors.
 * @param targetTensors Array of target tensors (one-hot).
 * @param batchSize The desired batch size.
 * @returns An array of batches (plain objects for worker compatibility).
 */
export function createImageBatches(inputTensors: Tensor[], targetTensors: Tensor[], batchSize: number): Batch[] {
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

    if (!firstInput || !firstTarget) continue;

    const [channels, height, width] = firstInput.shape;
    const numClasses = firstTarget.shape[1];

    const batchedInputData = new Float32Array(actualBatchSize * channels * height * width);
    const batchedTargetData = new Float32Array(actualBatchSize * numClasses);

    for (let j = 0; j < actualBatchSize; j++) {
      const originalIndex = batchIndices[j];
      const input = inputTensors[originalIndex];
      const target = targetTensors[originalIndex];

      if (input && target) {
        batchedInputData.set(input.data, j * input.size);
        batchedTargetData.set(target.data, j * target.size);
      }
    }

    batches.push({
      inputs: { data: batchedInputData, shape: [actualBatchSize, channels, height, width] },
      targets: { data: batchedTargetData, shape: [actualBatchSize, numClasses] },
    });
  }

  return batches;
}
